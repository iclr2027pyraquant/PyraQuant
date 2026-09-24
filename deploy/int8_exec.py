"""INT8 execution module for the deployment / latency measurements.

This is the executor behind the paper's practical-deployment table and the latency-decomposition
ladder.  It is a separate implementation from the QDQ quality simulator (flux/run_flux.py,
sdxl/run_sdxl.py): the token-level ``nn.Linear`` layers are replaced by ``Int8Linear`` and run
through real INT8 kernels.  It is used for **latency and memory only**; its numerics are not the
QDQ simulation and no quality claim is made from it.

  * Weights   : per-output-channel symmetric absmax INT8, computed from the (bf16/fp16) weights at
                conversion time.  In the latency table the W3 and W4 regions run the same INT8 kernel:
                W3 vs W4 differs in storage / BOPs, not in wall-clock (stated as such in the paper).
  * Activations: per-token (row) symmetric absmax INT8.  A Triton kernel computes the absmax and
                quantizes in one pass (replacing the unfused ``x.float().abs().amax`` upper bound).
  * GEMM      : ``triton`` backend (default): one fused Triton kernel, INT8 x INT8 -> int32 accumulate
                in registers, per-row/per-column dequantization (+bias) in the epilogue, written
                directly as bf16/fp16.
                ``cublas`` backend: ``torch._int_mm(x_i8 [M,K], w_i8.t() [K,N]) -> int32`` (cuBLASLt,
                TN layout) followed by a Triton epilogue that reads the int32 once and writes bf16
                (no fp32 intermediate).  The int32 round trip makes this path slower than bf16 at 1K.
  * ``torch._int_mm`` requires M > 16: for M <= 16 the input is zero-padded to 32 rows (only
                happens for tiny inputs; the whitelisted layers are never quantized anyway).
  * Low rank  : an optional 16-bit low-rank branch (SVDQuant-style ``y += (x @ A) @ B``) is executed
                for real: the down-projection ``t = x @ A`` is a small separate GEMM and the
                up-projection ``t @ B`` is folded into the epilogue of the fused INT8 GEMM.

Whitelist (kept in bf16; executed once per image / tiny): x_embedder, context_embedder,
time_text_embed.*, norm_out.linear, proj_out (top level), and every adaLN modulation projection
``*.norm*.linear`` (transformer_blocks.N.norm1[_context].linear, single_transformer_blocks.N.norm.linear).

Backend selection: ``DEFAULT_BACKEND`` ("triton") or ``set_backend("triton" | "cublas")`` from a
driver's ``--int8-backend`` flag.  Compiled Triton kernels are cached in the standard
``TRITON_CACHE_DIR`` location.

Self-test / micro-benchmark:  python deploy/int8_exec.py [--int8-backend triton] [kernel_bench.json]
(sums all token-level FLUX Linear shapes for each M in {4352, 16640, 33744, 65792} and compares
against bf16 ``F.linear``).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:  # triton 3.x
    from triton.language.extra import libdevice as _libdevice
except Exception:  # pragma: no cover
    from triton.language.extra.cuda import libdevice as _libdevice  # type: ignore

_INT_MM_MIN_M = 17          # torch._int_mm: self.size(0) needs to be greater than 16
_PAD_M = 32

# --------------------------------------------------------------------------- backend selection
BACKENDS = ("triton", "cublas")
DEFAULT_BACKEND = "triton"   # triton: single fused INT8 GEMM + epilogue kernel (faster than bf16 at every M on an L40S)
                             # cublas: torch._int_mm + Triton epilogue (int32 round trip; slower than bf16 at 1K)
BACKEND = DEFAULT_BACKEND    # current selection; change it with set_backend()


def set_backend(name: str) -> str:
    """Select the INT8 GEMM backend for every subsequent Int8Linear forward / int8_linear_2d call."""
    global BACKEND
    if name not in BACKENDS:
        raise ValueError(f"unknown INT8 backend {name!r} (expected one of {BACKENDS})")
    BACKEND = name
    return BACKEND


def get_backend() -> str:
    return BACKEND


# --------------------------------------------------------------------------- Triton kernels
@triton.jit
def _quant_rows_kernel(x_ptr, q_ptr, s_ptr, K, stride_xm,
                       BLOCK_K: tl.constexpr):
    """One program per row: two sweeps over K (first absmax, second quantize; the second hits L2).
    scale = absmax / 127 (clamped at 1e-8)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    xrow = x_ptr + row.to(tl.int64) * stride_xm
    amax = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        m = (offs + k0) < K
        v = tl.load(xrow + k0 + offs, mask=m, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(v))
    a = tl.max(amax, axis=0)
    a = tl.maximum(a, 1e-8)
    scale = a / 127.0
    qrow = q_ptr + row.to(tl.int64) * K
    for k0 in range(0, K, BLOCK_K):
        m = (offs + k0) < K
        v = tl.load(xrow + k0 + offs, mask=m, other=0.0).to(tl.float32)
        q = _libdevice.rint(_libdevice.div_rn(v, scale))   # IEEE round-to-nearest division: element-wise identical to torch.round(x/scale) (Triton's / is the approximate div.full)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(qrow + k0 + offs, q.to(tl.int8), mask=m)
    tl.store(s_ptr + row, scale)


@triton.jit
def _dequant_epilogue_kernel(acc_ptr, out_ptr, srow_ptr, scol_ptr, bias_ptr, M, N,
                             HAS_BIAS: tl.constexpr, OUT_BF16: tl.constexpr,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """int32 [M,N] * s_row[M] * s_col[N] (+bias[N]) -> bf16/fp16 [M,N]; one read, one write."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = rm < M
    mn = rn < N
    sr = tl.load(srow_ptr + rm, mask=mm, other=0.0)
    sc = tl.load(scol_ptr + rn, mask=mn, other=0.0)
    offs = rm.to(tl.int64)[:, None] * N + rn[None, :]
    mask = mm[:, None] & mn[None, :]
    a = tl.load(acc_ptr + offs, mask=mask, other=0).to(tl.float32)
    y = a * (sr[:, None] * sc[None, :])
    if HAS_BIAS:
        b = tl.load(bias_ptr + rn, mask=mn, other=0.0)
        y = y + b[None, :]
    if OUT_BF16:
        tl.store(out_ptr + offs, y.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + offs, y.to(tl.float16), mask=mask)



def _gemm_configs():
    C = triton.Config
    return [
        C({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        C({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        C({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        C({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=3, num_warps=4),
        C({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        C({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=2, num_warps=8),
        C({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
    ]


@triton.jit
def _int8_gemm_fused_kernel_impl(a_ptr, w_ptr, c_ptr, srow_ptr, scol_ptr, bias_ptr, M, N, K, M_BUCKET,
                            stride_am, stride_wn, stride_cm,
                            HAS_BIAS: tl.constexpr, OUT_BF16: tl.constexpr,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """C[M,N] = (A[M,K] int8 @ W[N,K]^T int8) * s_row[M] * s_col[N] (+bias[N]) -> bf16; the int32 accumulator
    never leaves registers.  A and W are both K-contiguous (TN) and are fed to mma directly; requires
    K % BLOCK_K == 0 and N % BLOCK_N == 0 (all FLUX shapes satisfy this), M arbitrary."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rm_a = tl.where(rm < M, rm, 0)                       # out-of-range rows read row 0 (masked out at the store), so no row mask inside the K loop
    a_ptrs = a_ptr + rm_a.to(tl.int64)[:, None] * stride_am + rk[None, :]
    w_ptrs = w_ptr + rn.to(tl.int64)[:, None] * stride_wn + rk[None, :]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        w = tl.load(w_ptrs)
        acc = tl.dot(a, tl.trans(w), acc)
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    sr = tl.load(srow_ptr + rm, mask=rm < M, other=0.0)
    sc = tl.load(scol_ptr + rn)
    y = acc.to(tl.float32) * (sr[:, None] * sc[None, :])
    if HAS_BIAS:
        y = y + tl.load(bias_ptr + rn)[None, :]
    c_ptrs = c_ptr + rm.to(tl.int64)[:, None] * stride_cm + rn[None, :]
    if OUT_BF16:
        tl.store(c_ptrs, y.to(tl.bfloat16), mask=(rm < M)[:, None])
    else:
        tl.store(c_ptrs, y.to(tl.float16), mask=(rm < M)[:, None])


_FUSED_KERNEL = None


def _fused_kernel():
    """The autotune decorator needs a GPU driver at construction time, so the kernel is wrapped lazily
    on first use (the module then imports on CPU-only machines)."""
    global _FUSED_KERNEL
    if _FUSED_KERNEL is None:
        _FUSED_KERNEL = triton.autotune(configs=_gemm_configs(), key=["N", "K", "M_BUCKET"])(_int8_gemm_fused_kernel_impl)
    return _FUSED_KERNEL


def _m_bucket(M: int) -> int:
    return 0 if M < 8192 else (1 if M < 32768 else 2)


def int8_gemm_fused(xq: torch.Tensor, s_row: torch.Tensor, w_int8: torch.Tensor, w_scale: torch.Tensor,
                    bias: Optional[torch.Tensor], out_dtype: torch.dtype) -> torch.Tensor:
    M, K = xq.shape
    N = w_int8.shape[0]
    assert K % 64 == 0 and N % 256 == 0, (K, N)
    out = torch.empty((M, N), dtype=out_dtype, device=xq.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _fused_kernel()[grid](xq, w_int8, out, s_row, w_scale, bias if bias is not None else w_scale,
                                  M, N, K, _m_bucket(M), xq.stride(0), w_int8.stride(0), out.stride(0),
                                  HAS_BIAS=bias is not None, OUT_BF16=(out_dtype == torch.bfloat16))
    return out


# --------------------------------------------------------------------------- python wrappers
def quant_rows(x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """x2 [M,K] -> (int8 [M,K], fp32 scale [M]), per-row symmetric absmax.
    The kernel walks rows through stride_xm, so no copy is needed as long as the last dim is contiguous.
    Row-slice views (e.g. the [M, K'] slices handed in by split-input layers) would otherwise be copied
    twice per step on every split layer, about 9% of the INT8 Linear kernel time."""
    assert x2.dim() == 2 and x2.is_cuda
    if x2.stride(1) != 1:
        x2 = x2.contiguous()
    M, K = x2.shape
    q = torch.empty((M, K), dtype=torch.int8, device=x2.device)
    s = torch.empty((M,), dtype=torch.float32, device=x2.device)
    BLOCK_K = min(4096, triton.next_power_of_2(K))
    nw = 4 if BLOCK_K <= 2048 else 8
    _quant_rows_kernel[(M,)](x2, q, s, K, x2.stride(0), BLOCK_K=BLOCK_K, num_warps=nw)
    return q, s


def dequant_epilogue(acc: torch.Tensor, s_row: torch.Tensor, s_col: torch.Tensor,
                     bias: Optional[torch.Tensor], out_dtype: torch.dtype) -> torch.Tensor:
    assert acc.dtype == torch.int32 and acc.is_contiguous()
    M, N = acc.shape
    out = torch.empty((M, N), dtype=out_dtype, device=acc.device)
    BLOCK_M, BLOCK_N = 16, 256
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _dequant_epilogue_kernel[grid](acc, out, s_row, s_col, bias if bias is not None else s_col, M, N,
                                   HAS_BIAS=bias is not None, OUT_BF16=(out_dtype == torch.bfloat16),
                                   BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4)
    return out


def int8_linear_2d(x2: torch.Tensor, w_int8: torch.Tensor, w_scale: torch.Tensor,
                   bias: Optional[torch.Tensor], backend: Optional[str] = None) -> torch.Tensor:
    """x2 [M,K] bf16 -> [M,N] bf16.  w_int8 [N,K] row-major (passed as .t() to _int_mm = TN layout).
    ``backend`` overrides the module-level selection (see set_backend) for this call only."""
    if backend is None:
        backend = BACKEND
    M = x2.shape[0]
    xq, sx = quant_rows(x2)
    if backend == "triton":
        return int8_gemm_fused(xq, sx, w_int8, w_scale, bias, x2.dtype)
    if M < _INT_MM_MIN_M:                       # cuBLAS limit: zero-pad to 32 rows, then crop
        xq = F.pad(xq, (0, 0, 0, _PAD_M - M))
        sx = F.pad(sx, (0, _PAD_M - M))
    acc = torch._int_mm(xq, w_int8.t())         # int32 [M',N]
    y = dequant_epilogue(acc, sx, w_scale, bias, x2.dtype)
    return y[:M] if M < _INT_MM_MIN_M else y


# --------------------------------------------------------------------------- INT8 GEMM with the low-rank up-projection folded into the epilogue
# Written as `y = y + (x@A)@B`, the low-rank branch reads/writes the large [M,N] output four extra times.
# Measured on an L40S over all Linear layers of one FLUX image: bf16 12.222 s | INT8 GEMM only 6.709 s (1.82x) |
# INT8 + low rank as a separate matmul 10.446 s (1.17x) | INT8 + low rank fused into the epilogue 7.524 s (1.62x).
# The branch is only 1.56% of the main GEMM's FLOPs, yet its memory traffic drags 1.82x down to 1.17x.  Adding
# t@B inside the GEMM epilogue (while acc is still in registers) leaves 0.815 s of overhead, which is also what
# the SVDQuant / nunchaku kernels do.  Relative error stays at 2.35e-03 (bf16 rounding level).
# The down-projection t = x @ A remains a separate, cheap GEMM ([M,32]).
@triton.jit
def _int8_gemm_lr_impl(a_ptr, w_ptr, c_ptr, srow_ptr, scol_ptr, bias_ptr, t_ptr, lb_ptr,
                       M, N, K, M_BUCKET, stride_am, stride_wn, stride_cm, stride_tm, stride_bn,
                       R: tl.constexpr, HAS_BIAS: tl.constexpr, OUT_BF16: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """_int8_gemm_fused_kernel_impl + the SVDQuant up-projection folded into the epilogue:
    C = (A_i8 @ W_i8^T) * s_row * s_col + bias + t[M,R] @ Blr[R,N].  t = x @ A_lr is computed outside."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rm_a = tl.where(rm < M, rm, 0)
    a_ptrs = a_ptr + rm_a.to(tl.int64)[:, None] * stride_am + rk[None, :]
    w_ptrs = w_ptr + rn.to(tl.int64)[:, None] * stride_wn + rk[None, :]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        w = tl.load(w_ptrs)
        acc = tl.dot(a, tl.trans(w), acc)
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    sr = tl.load(srow_ptr + rm, mask=rm < M, other=0.0)
    sc = tl.load(scol_ptr + rn)
    y = acc.to(tl.float32) * (sr[:, None] * sc[None, :])
    rr = tl.arange(0, R)                                            # ---- low-rank up-projection, 8 lines
    t = tl.load(t_ptr + rm_a.to(tl.int64)[:, None] * stride_tm + rr[None, :])
    lb = tl.load(lb_ptr + rr.to(tl.int64)[:, None] * stride_bn + rn[None, :])
    y += tl.dot(t, lb, out_dtype=tl.float32)
    if HAS_BIAS:
        y = y + tl.load(bias_ptr + rn)[None, :]
    c_ptrs = c_ptr + rm.to(tl.int64)[:, None] * stride_cm + rn[None, :]
    if OUT_BF16:
        tl.store(c_ptrs, y.to(tl.bfloat16), mask=(rm < M)[:, None])
    else:
        tl.store(c_ptrs, y.to(tl.float16), mask=(rm < M)[:, None])


_LR_KERNEL = None


def _lr_kernel():
    global _LR_KERNEL
    if _LR_KERNEL is None:
        _LR_KERNEL = triton.autotune(configs=_gemm_configs(), key=["N", "K", "M_BUCKET"])(_int8_gemm_lr_impl)
    return _LR_KERNEL


def int8_gemm_lr(xq, s_row, w_i8, w_scale, bias, t, lb, out_dtype):
    M, K = xq.shape
    N = w_i8.shape[0]
    out = torch.empty((M, N), dtype=out_dtype, device=xq.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _lr_kernel()[grid](xq, w_i8, out, s_row, w_scale, bias if bias is not None else w_scale, t, lb,
                       M, N, K, _m_bucket(M), xq.stride(0), w_i8.stride(0), out.stride(0),
                       t.stride(0), lb.stride(0), R=t.shape[1],
                       HAS_BIAS=bias is not None, OUT_BF16=(out_dtype == torch.bfloat16))
    return out


# Attribute names under which a weight loader may attach a 16-bit low-rank branch to an nn.Linear
# before conversion (A: [in_features, r], B: [r, out_features], both in the weight dtype).
LOWRANK_A_ATTR = "_lowrank_A"
LOWRANK_B_ATTR = "_lowrank_B"


class Int8Linear(nn.Module):
    """INT8 stand-in for nn.Linear: per-channel INT8 weights (buffers) + per-token dynamic INT8
    activations, fused kernels.  Latency measurement only."""

    def __init__(self, lin: nn.Linear, device: Optional[torch.device] = None,
                 lowrank: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        super().__init__()
        w = lin.weight.detach()
        if device is not None:
            w = w.to(device, non_blocking=False)
        wf = w.float()
        s = wf.abs().amax(dim=1).clamp_min(1e-8) / 127.0                        # [N]
        q = torch.round(wf / s[:, None]).clamp_(-127, 127).to(torch.int8)       # [N,K]
        self.register_buffer("weight_int8", q.contiguous())
        self.register_buffer("w_scale", s.contiguous())
        if lin.bias is not None:
            b = lin.bias.detach().float()
            self.register_buffer("bias", (b.to(device) if device is not None else b).contiguous())
        else:
            self.bias = None
        self.in_features = int(lin.in_features)
        self.out_features = int(lin.out_features)
        # The rank-32 low-rank branch of an SVDQuant-style base must really be executed; folding it into the
        # weight would waive the cost that a real W4 kernel has to pay.  A weight loader attaches it to the
        # nn.Linear (LOWRANK_A_ATTR / LOWRANK_B_ATTR) or it is passed explicitly; it is moved into buffers here
        # and applied in forward (one addmm, or fused into the INT8 GEMM epilogue); the INT8 main path stays one pass.
        if lowrank is not None:
            A, B = lowrank
        else:
            A = getattr(lin, LOWRANK_A_ATTR, None); B = getattr(lin, LOWRANK_B_ATTR, None)
        if A is not None and B is not None:
            self.register_buffer("lr_A", (A.to(device) if device is not None else A).contiguous())
            self.register_buffer("lr_B", (B.to(device) if device is not None else B).contiguous())
        else:
            self.lr_A = None; self.lr_B = None
        del wf, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.stride(1) != 1:                    # row-slice views (split-input layers) need no copy: the kernel takes a row stride
            x2 = x2.contiguous()
        if self.lr_A is not None and BACKEND == "triton" and x2.shape[1] % 32 == 0:
            # fused path: the down-projection t = x@A is computed separately (small), the up-projection t@B
            # goes into the epilogue of the INT8 GEMM
            t = x2.to(self.lr_A.dtype) @ self.lr_A
            xq, sx = quant_rows(x2)
            return int8_gemm_lr(xq, sx, self.weight_int8, self.w_scale, self.bias,
                                t.contiguous(), self.lr_B, x2.dtype).reshape(*shape[:-1], self.out_features)
        y = int8_linear_2d(x2, self.weight_int8, self.w_scale, self.bias)
        if self.lr_A is not None:
            # y += (x @ A) @ B, the 16-bit low-rank parallel branch of SVDQuant.
            # Written as `y = y + (xA) @ B` this reads/writes the large [M,N] output four extra times
            # (one write for B's GEMM, two reads and one write for the add) while the branch itself is only
            # 1.56% of the main GEMM's FLOPs.  Measured on an L40S over all Linear layers of one FLUX image:
            # 4.345 s of low-rank overhead, of which only 1.65 s is the GEMM and 1.651 s is the addition.
            # An in-place addmm brings this down to 2.606 s (-40%), mathematically identical.
            # Folding the term into the INT8 GEMM epilogue (0.815 s overhead) is the fused path above.
            xa = x2.to(self.lr_A.dtype) @ self.lr_A
            if y.dtype == self.lr_B.dtype and y.is_contiguous():
                y.addmm_(xa, self.lr_B)          # in place: one fewer [M,N] allocation and two fewer passes
            else:
                y = torch.addmm(y, xa, self.lr_B.to(y.dtype))
        return y.reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int8 per-channel W / per-token A (latency-only)"


# --------------------------------------------------------------------------- model conversion
def is_bf16_whitelisted(name: str) -> bool:
    """SVDQuant-style whitelist + adaLN modulation projections: kept in bf16."""
    top = name.split(".")[0]
    if top in ("x_embedder", "context_embedder", "proj_out", "time_text_embed", "norm_out"):
        return True
    parts = name.split(".")
    return len(parts) >= 2 and parts[-1] == "linear" and parts[-2].startswith("norm")


def convert_linears_to_int8(model: nn.Module, device: Optional[str] = "cuda",
                            whitelist: Callable[[str], bool] = is_bf16_whitelisted) -> Dict[str, object]:
    """Replace the token-level nn.Linear layers by Int8Linear in place (the weight quantization is done on
    ``device`` layer by layer; the bf16 weight is released as soon as a layer is converted).
    Returns statistics: layer counts / parameter counts / whitelisted layer names."""
    dev = torch.device(device) if device is not None else None
    n_int8, n_keep, p_int8, p_keep, kept = 0, 0, 0, 0, []
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for name, lin in targets:
        if whitelist(name):
            n_keep += 1; p_keep += lin.weight.numel(); kept.append(name)
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        q = Int8Linear(lin, device=dev)
        setattr(parent, attr, q)
        n_int8 += 1; p_int8 += lin.weight.numel()
        del lin
    if dev is not None and dev.type == "cuda":
        torch.cuda.synchronize()
    return {"n_int8_linear": n_int8, "n_bf16_linear_kept": n_keep, "params_int8": p_int8, "params_bf16_linear_kept": p_keep,
            "bf16_kept_names": kept, "int8_weight_gib": p_int8 / 2**30, "bf16_equiv_gib": p_int8 * 2 / 2**30}


def warm_kernels(Ms=(4352, 16640, 65792), device="cuda", lowrank_R=32, warm_lowrank: bool = True):
    """Pre-compile the Triton kernels (the three BLOCK_K variants of the quantizer / the epilogue / the autotuned
    fused GEMM for every (N, K, M bucket)) so that compilation never lands inside a timed region.

    The low-rank kernel (SVDQuant up-projection fused into the epilogue) is a **separate** autotuned kernel:
    if it is not warmed, the first image pays tens of seconds of compilation, and because the M bucket is
    triggered on demand a later image that first enters another bucket compiles again -- inside the timed
    median.  It is therefore warmed here on the same (K, N) x M-bucket x HAS_BIAS x (M % 16) grid.
    ``lowrank_R`` must equal the actual rank (SVDQuant: r = 32).  ``warm_lowrank=False`` restores the
    main-GEMM-only warm-up (A/B comparison only)."""
    t0 = time.perf_counter()
    for (K, N) in FLUX_TOKEN_SHAPES:
        w = torch.randint(-127, 127, (N, K), device=device, dtype=torch.int8); s = torch.rand(N, device=device); b = torch.rand(N, device=device)
        lb = torch.randn(lowrank_R, N, device=device, dtype=torch.bfloat16).contiguous()
        for M in Ms:
            x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
            int8_linear_2d(x, w, s, b); int8_linear_2d(x[:8], w, s, b)
            for xx in ((x, x[:8], x[:M - 2]) if warm_lowrank else ()):   # three specialisations: M%16==0 / small M / same bucket with M%16!=0
                xq, sx = quant_rows(xx)
                t = torch.randn(xx.shape[0], lowrank_R, device=device, dtype=torch.bfloat16).contiguous()
                int8_gemm_lr(xq, sx, w, s, b, t, lb, torch.bfloat16)      # HAS_BIAS=True
                int8_gemm_lr(xq, sx, w, s, None, t, lb, torch.bfloat16)   # HAS_BIAS=False
                del xq, sx, t
            del x
        del w, s, b, lb
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    print(f"[int8_exec] kernels warmed (backend={BACKEND}, gemm+lowrank r={lowrank_R}) in {time.perf_counter()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- self-test / micro-benchmark
FLUX_TOKEN_SHAPES = {   # (K, N): count -- FLUX.1-schnell token-level Linear layers (19 dual-stream + 38 single-stream blocks), excluding whitelist/adaLN
    (3072, 3072): 19 * 8 + 38 * 3,
    (3072, 12288): 19 * 2 + 38,
    (12288, 3072): 19 * 2,
    (15360, 3072): 38,
}


def _bench(fn, iters=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters * 1e3


def selftest_and_bench(Ms=(4352, 16640, 33744, 65792), iters=10):
    dev = "cuda"
    torch.manual_seed(0)
    # ---- numerical self-test: fused path vs unfused reference (same int8 codes) / vs bf16 F.linear
    lin = nn.Linear(3072, 12288, bias=True).to(dev, torch.bfloat16)
    x = (torch.randn(4352, 3072, device=dev) * 2).to(torch.bfloat16)
    x[:, 5] *= 30.0                                   # outlier channel
    ref_bf16 = F.linear(x, lin.weight, lin.bias).float()
    q = Int8Linear(lin, device=torch.device(dev))
    y = q(x).float()
    xq, sx = quant_rows(x)
    ref_int8 = (xq.float() @ q.weight_int8.float().t()) * sx[:, None] * q.w_scale[None, :] + q.bias[None, :]
    # torch reference quantizer (same math as the QDQ simulator's per-token A8 fake-quant)
    xf = x.float(); sc = xf.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
    xq_ref = torch.round(xf / sc).clamp(-127, 127).to(torch.int8)
    ref_int8_bf16 = ref_int8.to(torch.bfloat16).float()          # the unfused reference is also rounded to bf16; only the fp32 multiplication order differs
    x_plain = torch.randn(4352, 3072, device=dev).to(torch.bfloat16)   # typical input without an outlier channel
    y_plain = q(x_plain).float(); ref_plain = F.linear(x_plain, lin.weight, lin.bias).float()
    rep = {
        "act_quant_mismatch_elems": int((xq != xq_ref).sum().item()),
        "act_scale_max_rel_err": float(((sx - sc[:, 0]).abs() / sc[:, 0]).max().item()),
        "fused_vs_unfused_int8_max_abs": float((y - ref_int8_bf16).abs().max().item()),
        "fused_vs_unfused_int8_rel": float(((y - ref_int8_bf16).norm() / ref_int8_bf16.norm()).item()),
        "int8_vs_bf16_rel_outlier30x": float(((y - ref_bf16).norm() / ref_bf16.norm()).item()),
        "int8_vs_bf16_rel_plain": float(((y_plain - ref_plain).norm() / ref_plain.norm()).item()),
    }
    # small-M padding path
    xs = x[:8]
    ys = q(xs).float(); rs = ((quant_rows(xs)[0].float() @ q.weight_int8.float().t()) * quant_rows(xs)[1][:, None] * q.w_scale[None, :] + q.bias[None, :]).to(torch.bfloat16).float()
    rep["smallM_fused_vs_unfused_rel"] = float(((ys - rs).norm() / rs.norm()).item())
    yt = int8_gemm_fused(xq, sx, q.weight_int8, q.w_scale, q.bias, torch.bfloat16).float()
    rep["triton_gemm_vs_cublas_path_max_abs"] = float((yt - y).abs().max().item())
    rep["triton_gemm_vs_unfused_rel"] = float(((yt - ref_int8_bf16).norm() / ref_int8_bf16.norm()).item())
    yts = int8_gemm_fused(*quant_rows(xs), q.weight_int8, q.w_scale, q.bias, torch.bfloat16).float()
    rep["triton_smallM_vs_unfused_rel"] = float(((yts - rs).norm() / rs.norm()).item())
    x3 = x[:2048].reshape(8, 256, 3072)
    rep["3d_input_ok"] = bool(torch.equal(q(x3).reshape(2048, -1), q(x[:2048])))
    print("[int8_exec] selftest:", rep, flush=True)
    rep["act_quant_mismatch_frac"] = rep["act_quant_mismatch_elems"] / x.numel()
    assert rep["act_quant_mismatch_frac"] <= 1e-4, rep          # rint(div_rn) should match torch.round(x/scale) element-wise; 1e-4 is a safety margin
    assert rep["fused_vs_unfused_int8_rel"] < 1e-3 and rep["int8_vs_bf16_rel_plain"] < 2e-2, rep
    assert rep["triton_gemm_vs_unfused_rel"] < 1e-3 and rep["triton_smallM_vs_unfused_rel"] < 1e-3, rep
    del lin, q, x, ref_bf16, ref_int8, y, x_plain, y_plain, ref_plain
    torch.cuda.empty_cache()
    # ---- micro-benchmark: for each M, weighted sum over all token-level shapes
    out = {"selftest": rep, "gpu": torch.cuda.get_device_name(), "per_M": {}}
    for M in Ms:
        tot = {"bf16_ms": 0.0, "int8_fused_ms": 0.0, "int8_gemm_only_ms": 0.0, "quant_ms": 0.0, "epilogue_ms": 0.0, "triton_fused_ms": 0.0}
        per_shape = {}
        for (K, N), cnt in FLUX_TOKEN_SHAPES.items():
            lin = nn.Linear(K, N, bias=True).to(dev, torch.bfloat16)
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            q = Int8Linear(lin, device=torch.device(dev))
            b = _bench(lambda: F.linear(x, lin.weight, lin.bias), iters)
            f = _bench(lambda: q(x), iters)
            xq, sx = quant_rows(x); wt = q.weight_int8.t()
            g = _bench(lambda: torch._int_mm(xq, wt), iters)
            qq = _bench(lambda: quant_rows(x), iters)
            acc = torch._int_mm(xq, wt)
            e = _bench(lambda: dequant_epilogue(acc, sx, q.w_scale, q.bias, torch.bfloat16), iters)
            tg = _bench(lambda: int8_gemm_fused(xq, sx, q.weight_int8, q.w_scale, q.bias, torch.bfloat16), iters)   # first call triggers the autotune
            per_shape[f"{K}x{N}"] = {"count": cnt, "bf16_ms": round(b, 3), "int8_fused_ms": round(f, 3), "gemm_ms": round(g, 3),
                                     "quant_ms": round(qq, 3), "epilogue_ms": round(e, 3), "speedup": round(b / f, 3),
                                     "triton_gemm_fused_ms": round(tg, 3), "speedup_triton": round(b / (tg + qq), 3)}
            tot["bf16_ms"] += b * cnt; tot["int8_fused_ms"] += f * cnt; tot["int8_gemm_only_ms"] += g * cnt
            tot["quant_ms"] += qq * cnt; tot["epilogue_ms"] += e * cnt; tot["triton_fused_ms"] += (tg + qq) * cnt
            del lin, x, q, xq, sx, acc, wt
            torch.cuda.empty_cache()
        tot = {k: round(v, 2) for k, v in tot.items()}
        tot["speedup_fused_vs_bf16"] = round(tot["bf16_ms"] / tot["int8_fused_ms"], 3)
        tot["speedup_triton_vs_bf16"] = round(tot["bf16_ms"] / tot["triton_fused_ms"], 3)
        tot["per_shape"] = per_shape
        out["per_M"][str(M)] = tot
        print(f"[int8_exec] M={M}: bf16 {tot['bf16_ms']} ms | cublas path {tot['int8_fused_ms']} ms "
              f"(gemm {tot['int8_gemm_only_ms']}, quant {tot['quant_ms']}, epilogue {tot['epilogue_ms']}) -> x{tot['speedup_fused_vs_bf16']}"
              f" | triton fused GEMM+quant {tot['triton_fused_ms']} ms -> x{tot['speedup_triton_vs_bf16']}", flush=True)
    return out


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="INT8 kernel self-test and micro-benchmark on the FLUX token-level Linear shapes (needs a CUDA GPU).")
    ap.add_argument("out", nargs="?", default=None, help="optional JSON path for the results (e.g. kernel_bench.json)")
    ap.add_argument("--int8-backend", choices=BACKENDS, default=DEFAULT_BACKEND,
                    help="INT8 GEMM backend used by the Int8Linear path of the benchmark")
    args = ap.parse_args()
    set_backend(args.int8_backend)
    res = selftest_and_bench()
    res["int8_backend"] = BACKEND
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"[int8_exec] saved {args.out}")
