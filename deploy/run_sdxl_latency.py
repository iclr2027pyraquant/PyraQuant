"""End-to-end latency ladder for PyraQuant on ScaleDiff-SDXL (1K -> 2K -> 4K, 50/20/20 steps, CFG 7.5)
on one GPU with all weights resident (no CPU offload).  This is the deployment executor behind the
SDXL rows of the paper's practical-deployment table and the appendix latency decomposition.

Five arms (--arm), same prompts / seed 42 / schedule:
  fp16_full            host as is: full canvas, fp16 Linears (official ScaleDiff-SDXL path; only wrapped
                       in SpatialMaskUNet for call counting)
  int8_full            full canvas; every token-level nn.Linear of the UNet -> Int8Linear (per-channel
                       INT8 weights + fused per-token INT8 activations + fused INT8 GEMM, deploy/int8_exec.py);
                       Conv2d stays fp16 (torch has no CUDA INT8 convolution), attention fp16; the tiny
                       per-image Linears (time_embedding / add_embedding / resnet time_emb_proj) stay fp16
  int8_routed_nocache  INT8 + routed tile executor with ALL 4K leaves active (no inheritance)
                       -> 4 full-row strips (128x512 latent) + halo; the cache mask is identically 0, so
                       only the "strip / tile execution + halo" effect remains, without cached reuse
  int8_routed_cache    INT8 + routed + spatial cache, relative-gap threshold tau (the executor of the final
                       SDXL recipe, configs/sdxl_pyraquant.json; the hi / lo (W8 | nested W4) dual pass is
                       collapsed into ONE INT8 pass)
  fp16_routed_cache    fp16 Linears + routed + cache (strategy only, no kernel)

Selection = the residual score + relative-gap rule of sdxl/run_sdxl.py (leaf_scores_residual, build_leaf_hi,
inner_split_median and soft_mask are imported, not re-implemented); the routing boxes (2x2-leaf tiles merged
per row) mirror make_spatial_hook of that driver line by line, with per-stage timers added.  The routed
executor and the cache path are sdxl/pipeline_sdxl.py as is.

Only latency (per stage and total) and per-stage peak memory are measured; quality is NOT (the INT8 numerics
differ from the QDQ simulation of sdxl/run_sdxl.py).  The 4K image of the warm-up prompt is saved as a
numerics sanity artifact (PSNR between arms is reported by --breakdown).

Reproducing the SDXL rows of the deployment table (one NVIDIA L40S; median over 20 prompts after 1 warm-up;
the runs were made with the standard PyTorch allocator setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True):
  # fp16 baseline
  python deploy/run_sdxl_latency.py --arm fp16_full --n 20 --warmup 1 --out-dir outputs/latency_sdxl/fp16_full
  # PyraQuant (final recipe executor: cascade at 4K, tau 0.10, 8-latent halo; --weight-source method writes the
  # RTN W8 g32 weights of the recipe into the Linears and per-output-channel RTN W8 into the Conv2d weights
  # before the INT8 codes are derived -- this only affects the warm-up image, not the timing)
  python deploy/run_sdxl_latency.py --arm int8_routed_cache --weight-source method --cascade --tau 0.10 \\
      --route-halo 8 --n 20 --warmup 1 --out-dir outputs/latency_sdxl/int8_routed_cache

Latency decomposition ladder (the other rungs, same protocol; then the summary over --out-dir/<arm>/rec.json):
  python deploy/run_sdxl_latency.py --arm int8_full --weight-source method --n 20 --warmup 1 --out-dir outputs/latency_sdxl/int8_full
  python deploy/run_sdxl_latency.py --arm int8_routed_nocache --weight-source method --cascade --tau 0.10 --route-halo 8 \\
      --n 20 --warmup 1 --out-dir outputs/latency_sdxl/int8_routed_nocache
  python deploy/run_sdxl_latency.py --arm fp16_routed_cache --cascade --tau 0.10 --route-halo 8 --n 20 --warmup 1 \\
      --out-dir outputs/latency_sdxl/fp16_routed_cache
  python deploy/run_sdxl_latency.py --breakdown --out-dir outputs/latency_sdxl --json outputs/latency_sdxl/latency_sdxl.json
    -> outputs/latency_sdxl/latency_sdxl.json + outputs/latency_sdxl/summary.md (ladder table, ratios, PSNR sanity)

Kernel self-test / micro-benchmark on the SDXL Linear shapes (writes --out-dir/kernel_bench_sdxl.json):
  python deploy/run_sdxl_latency.py --selftest --out-dir outputs/latency_sdxl

Compiled Triton kernels are cached in the standard TRITON_CACHE_DIR location.  The INT8 GEMM backend is
chosen with --int8-backend (triton, default | cublas), see deploy/int8_exec.py.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sdxl"))
sys.path.insert(0, str(HERE))

import triton                                                     # noqa: E402
import int8_exec                                                  # noqa: E402  (deploy/int8_exec.py)
from run_sdxl import (LEAF_LATENT, MASK_EDGE, MODEL_ID, MODEL_REVISION, NEGATIVE_PROMPT,   # noqa: E402
                      SpatialMaskUNet, build_leaf_hi, inner_split_median, leaf_scores_residual, soft_mask)
from pipeline_sdxl import CustomStableDiffusionXLPipeline        # noqa: E402  (release pipeline with the routed executor)

_PIPELINE_FILE = (ROOT / "sdxl" / "pipeline_sdxl.py").resolve()
if Path(inspect.getfile(CustomStableDiffusionXLPipeline)).resolve() != _PIPELINE_FILE:
    raise RuntimeError(f"loaded the wrong pipeline module: {inspect.getfile(CustomStableDiffusionXLPipeline)}")

STAGE_PX = (1024, 2048, 4096)
NEGATIVE = NEGATIVE_PROMPT                    # same negative prompt as sdxl/run_sdxl.py
SEED = 42                                     # every prompt uses seed 42, as in the quality runs

ARMS = {   # arm -> (Linear numerics, 4K executor, no_cache, description)
    "fp16_full": ("fp16", "full", False, "fp16 Linears, full canvas (official ScaleDiff-SDXL path)"),
    "int8_full": ("int8", "full", False, "INT8 Linears (Conv/attention fp16), full canvas"),
    "int8_routed_nocache": ("int8", "routed", True, "INT8, routed strips, ALL 4K leaves computed (+halo 8), no cache reuse"),
    "int8_routed_cache": ("int8", "routed", False, "INT8, routed tiles + spatial cache, relgap tau (final SDXL recipe executor)"),
    "fp16_routed_cache": ("fp16", "routed", False, "fp16 Linears, routed tiles + spatial cache, relgap tau (strategy only, no kernel)"),
}
LADDER = ("fp16_full", "int8_full", "int8_routed_nocache", "int8_routed_cache", "fp16_routed_cache")


def gib(x):
    return x / 2**30


# --------------------------------------------------------------------------- INT8 Linear for the SDXL shapes
# int8_exec.int8_gemm_fused asserts N % 256 == 0 (FLUX shapes).  The 640-channel SDXL layers (down/up_blocks.1)
# have N = 640 = 5 x 128 and use the same Triton kernel, but with a separate autotune set of BLOCK_N in {128, 64}
# (the kernel requires N % BLOCK_N == 0 and K % BLOCK_K == 0; an out-of-range config would read/write out of
# bounds, so the two config sets cannot share one autotuner).
_NARROW_KERNEL = None


def _narrow_configs():
    C = triton.Config
    return [
        C({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        C({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=3, num_warps=4),
        C({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        C({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        C({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        C({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=3, num_warps=8),
    ]


def _narrow_kernel():
    global _NARROW_KERNEL
    if _NARROW_KERNEL is None:   # the autotune decorator needs the GPU driver at construction -> wrap on first call (as int8_exec._fused_kernel)
        _NARROW_KERNEL = triton.autotune(configs=_narrow_configs(), key=["N", "K", "M_BUCKET"])(int8_exec._int8_gemm_fused_kernel_impl)
    return _NARROW_KERNEL


def _gemm_fused_narrow(xq, s_row, w_int8, w_scale, bias, out_dtype):
    M, K = xq.shape
    N = w_int8.shape[0]
    assert K % 128 == 0 and N % 128 == 0, (K, N)
    out = torch.empty((M, N), dtype=out_dtype, device=xq.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _narrow_kernel()[grid](xq, w_int8, out, s_row, w_scale, bias if bias is not None else w_scale,
                           M, N, K, int8_exec._m_bucket(M), xq.stride(0), w_int8.stride(0), out.stride(0),
                           HAS_BIAS=bias is not None, OUT_BF16=(out_dtype == torch.bfloat16))
    return out


# ---- direct launch of the compiled kernels (bypassing JITFunction.run's Python binding + the autotuner dispatch) ----
# Measured on an L40S / Triton 3.2: per-token quantization kernel 75 us -> 8 us, fused GEMM 100 us -> 11 us per call
# (bit-identical results, checked at M = 2048).  The SDXL 1K-stage Linears (M = 2048 / 8192) take only 30-60 us
# themselves, so without the bypass INT8 is 2x SLOWER than fp16 at 1K -- that is Python launch overhead, not the
# kernel.  The cache key covers Triton's specialisation inputs: 16-byte pointer alignment, whether integers are
# divisible by 16 / equal to 1, dtype, constexprs; the config is still the one the autotuner picks on the first call
# (best_config).  FAST_LAUNCH is switched off with --no-fast-launch.
FAST_LAUNCH = True
_FAST = {"quant": {}, "gemm": {}, "cfg": {}}


def _cur_stream():
    from triton.runtime import driver
    return driver.active.get_current_stream(driver.active.get_current_device())


def fast_quant_rows(x2):
    M, K = x2.shape
    q = torch.empty((M, K), dtype=torch.int8, device=x2.device)
    s = torch.empty((M,), dtype=torch.float32, device=x2.device)
    BLOCK_K = min(4096, triton.next_power_of_2(K)); nw = 4 if BLOCK_K <= 2048 else 8
    key = (K, x2.dtype, int(x2.stride(0)), (x2.data_ptr() & 15) == 0)
    kern = _FAST["quant"].get(key)
    if kern is None:
        kern = int8_exec._quant_rows_kernel.run(x2, q, s, K, x2.stride(0), BLOCK_K=BLOCK_K, num_warps=nw, grid=(M,), warmup=True)
        kern._init_handles(); _FAST["quant"][key] = kern
    kern.run(M, 1, 1, _cur_stream(), kern.function, kern.packed_metadata, None, None, None, x2, q, s, K, x2.stride(0))
    return q, s


def fast_gemm_fused(xq, s_row, w_int8, w_scale, bias, out_dtype):
    M, K = xq.shape; N = w_int8.shape[0]
    mb = int8_exec._m_bucket(M)
    ckey = (N, K, mb, bias is not None, out_dtype)
    cfg = _FAST["cfg"].get(ckey)
    if cfg is None:                                   # first call: go through the autotuner (benchmarks every config, launches once), remember the chosen config
        narrow = (N % 256 != 0)
        y = (_gemm_fused_narrow if narrow else int8_exec.int8_gemm_fused)(xq, s_row, w_int8, w_scale, bias, out_dtype)
        _FAST["cfg"][ckey] = (_narrow_kernel() if narrow else int8_exec._fused_kernel()).best_config
        return y
    out = torch.empty((M, N), dtype=out_dtype, device=xq.device)
    kw = cfg.kwargs
    g0 = triton.cdiv(M, kw["BLOCK_M"]) * triton.cdiv(N, kw["BLOCK_N"])
    b = bias if bias is not None else w_scale
    kkey = ckey + (M % 16 == 0, M == 1, (xq.data_ptr() & 15) == 0)
    kern = _FAST["gemm"].get(kkey)
    if kern is None:
        kern = int8_exec._int8_gemm_fused_kernel_impl.run(xq, w_int8, out, s_row, w_scale, b, M, N, K, mb, xq.stride(0), w_int8.stride(0), out.stride(0),
                                                          HAS_BIAS=bias is not None, OUT_BF16=(out_dtype == torch.bfloat16), grid=(g0,), warmup=True, **cfg.all_kwargs())
        kern._init_handles(); _FAST["gemm"][kkey] = kern
    kern.run(g0, 1, 1, _cur_stream(), kern.function, kern.packed_metadata, None, None, None,
             xq, w_int8, out, s_row, w_scale, b, M, N, K, mb, xq.stride(0), w_int8.stride(0), out.stride(0))
    return out


def sdxl_int8_linear_2d(x2, w_int8, w_scale, bias):
    """x2 [M,K] fp16 -> [M,N] fp16.  N % 256 == 0 uses the fused kernel of int8_exec, otherwise the narrow-N config set
    of the same kernel; the cublas backend goes straight to int8_exec.
    FAST_LAUNCH (default on): both kernels are launched directly from the cached compiled kernel (bit-identical
    numerics, only the Python launch overhead is saved)."""
    if int8_exec.BACKEND != "triton":
        return int8_exec.int8_linear_2d(x2, w_int8, w_scale, bias)
    if FAST_LAUNCH:
        xq, sx = fast_quant_rows(x2)
        return fast_gemm_fused(xq, sx, w_int8, w_scale, bias, x2.dtype)
    xq, sx = int8_exec.quant_rows(x2)
    if w_int8.shape[0] % 256 == 0:
        return int8_exec.int8_gemm_fused(xq, sx, w_int8, w_scale, bias, x2.dtype)
    return _gemm_fused_narrow(xq, sx, w_int8, w_scale, bias, x2.dtype)


class SdxlInt8Linear(int8_exec.Int8Linear):
    """SDXL flavour of Int8Linear: the same per-channel INT8 weights + per-token INT8 activations, the GEMM dispatch
    only adds the narrow-N (640) config set."""

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        y = sdxl_int8_linear_2d(x2, self.weight_int8, self.w_scale, self.bias)
        return y.reshape(*shape[:-1], self.out_features)


def is_fp16_whitelisted_sdxl(name: str) -> bool:
    """Tiny per-image Linears (M = batch) stay fp16: time_embedding.*, add_embedding.*, resnets.*.time_emb_proj."""
    return ("time_embedding" in name) or ("add_embedding" in name) or name.endswith("time_emb_proj")


def convert_unet_linears_to_int8(unet: nn.Module, device="cuda"):
    dev = torch.device(device)
    n_int8, n_keep, p_int8, p_keep, kept = 0, 0, 0, 0, []
    shapes = defaultdict(int)
    for name, lin in [(n, m) for n, m in unet.named_modules() if isinstance(m, nn.Linear)]:
        if is_fp16_whitelisted_sdxl(name):
            n_keep += 1; p_keep += lin.weight.numel(); kept.append(name)
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = unet.get_submodule(parent_name) if parent_name else unet
        setattr(parent, attr, SdxlInt8Linear(lin, device=dev))
        shapes[f"{lin.in_features}x{lin.out_features}"] += 1
        n_int8 += 1; p_int8 += lin.weight.numel()
        del lin
    torch.cuda.synchronize()
    return {"n_int8_linear": n_int8, "n_fp16_linear_kept": n_keep, "params_int8": p_int8, "params_fp16_linear_kept": p_keep,
            "fp16_kept_names": kept, "int8_weight_gib": p_int8 / 2**30, "fp16_equiv_gib": p_int8 * 2 / 2**30,
            "shapes_KxN": dict(sorted(shapes.items()))}


# --------------------------------------------------------------------------- weight source: the recipe's RTN W8 weights
def load_method_weights(unet: nn.Module, w_bits: int = 8, group: int = 32, conv_bits: int = 8,
                        conv_group: Optional[int] = None) -> dict:
    """--weight-source method: write the RTN W{w_bits} group-{group} fake-quantized weights of the recipe back into
    the UNet Linears (pyraquant.quantizers_weight.qdq_weight, the base quantizer of configs/sdxl_pyraquant.json; the
    nested W4 state is derived from these W8 codes and adds no weights, so the INT8 codes of the timed pass are derived
    from the W8 weights) and RTN W{conv_bits} into the Conv2d weights (pyraquant.quantizers_conv.rtn_quantize_conv_;
    ``conv_group=None`` = one scale per output channel, as in the paper's timed runs -- the quality recipe groups the
    conv weights by 128 along (c, kh, kw); the convolutions execute in fp16 either way).  Weights only, no activation
    hooks; the timing does not depend on the weight values."""
    from pyraquant.quantizers_weight import qdq_weight
    from pyraquant.quantizers_conv import rtn_quantize_conv_
    n = 0
    for name, m in unet.named_modules():
        if not isinstance(m, nn.Linear) or is_fp16_whitelisted_sdxl(name):
            continue                                     # same fp16 whitelist as the INT8 conversion
        m.weight.data.copy_(qdq_weight(m.weight.data, w_bits, group))
        n += 1
    n_conv = 0
    if conv_bits and conv_bits < 16:
        n_conv = sum(1 for _, c in unet.named_modules() if isinstance(c, nn.Conv2d))
        rtn_quantize_conv_(unet, conv_bits, conv_group)  # in place, every Conv2d of the UNet
    conv_txt = f"g{conv_group}" if conv_group else "per-output-channel"
    return {"method": f"RTN W{w_bits} g{group} (Linear) + RTN W{conv_bits} {conv_txt} (Conv2d)", "n_layers": n, "n_conv": n_conv,
            "note": f"weights = dequantized RTN W{w_bits} g{group} on the Linears and {conv_txt} RTN W{conv_bits} on the Conv2d (the "
                    "convolutions still compute in fp16: torch has no CUDA INT8 conv); the INT8 codes are derived from THESE weights"}


SDXL_TOKEN_SHAPES = [(640, 640), (2048, 640), (640, 5120), (2560, 640), (1280, 1280), (2048, 1280), (1280, 10240), (5120, 1280)]   # (K, N)


def warm_sdxl_kernels(Ms=(2048, 8192, 65536, 154, 8202, 32778), device="cuda"):
    """Pre-compile / autotune every SDXL (K, N) x M bucket (the autotuner picks the config on the first M of each bucket:
    2048 / 8192 / 65536), then the M % 16 != 0 specialisations of each bucket (154 = text K/V, 8202 / 32778 = odd
    crops), so that no compilation lands inside the timed prompts."""
    t0 = time.perf_counter()
    for (K, N) in SDXL_TOKEN_SHAPES:
        w = torch.randint(-127, 127, (N, K), device=device, dtype=torch.int8); s = torch.rand(N, device=device); b = torch.rand(N, device=device)
        for M in Ms:
            x = torch.randn(M, K, device=device, dtype=torch.float16)
            sdxl_int8_linear_2d(x, w, s, b)
            sdxl_int8_linear_2d(x, w, s, None)        # the SDXL attention to_q/k/v have no bias -> HAS_BIAS=False is another autotune key / compiled variant
            del x
        del w, s, b
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    print(f"[sdxl-int8] kernels warmed (backend={int8_exec.BACKEND}, fast_launch={FAST_LAUNCH}; {len(_FAST['gemm'])} gemm + {len(_FAST['quant'])} quant compiled variants) "
          f"in {time.perf_counter()-t0:.0f}s", flush=True)


def selftest_and_bench(out_dir: Path, Ms=(2048, 8192, 32768, 131072), iters=10):
    """Numerical self-test on the SDXL shapes (fused kernel vs an unfused reference on the same INT8 codes / vs fp16
    F.linear) and a per-shape micro-benchmark."""
    dev = "cuda"; torch.manual_seed(0)
    rep = {"gpu": torch.cuda.get_device_name(), "backend": int8_exec.BACKEND, "selftest": {}, "per_M": {}}
    for (K, N) in SDXL_TOKEN_SHAPES:
        lin = nn.Linear(K, N, bias=True).to(dev, torch.float16)
        x = (torch.randn(4096, K, device=dev) * 2).to(torch.float16); x[:, 5] *= 30.0
        q = SdxlInt8Linear(lin, device=torch.device(dev)); y = q(x).float()
        xq, sx = int8_exec.quant_rows(x)
        ref = ((xq.float() @ q.weight_int8.float().t()) * sx[:, None] * q.w_scale[None, :] + q.bias[None, :]).to(torch.float16).float()
        ref16 = F.linear(x, lin.weight, lin.bias).float()
        xs = x[:154]; ys = q(xs).float()
        xqs, sxs = int8_exec.quant_rows(xs)
        refs = ((xqs.float() @ q.weight_int8.float().t()) * sxs[:, None] * q.w_scale[None, :] + q.bias[None, :]).to(torch.float16).float()
        r = {"fused_vs_unfused_rel": float(((y - ref).norm() / ref.norm()).item()), "int8_vs_fp16_rel": float(((y - ref16).norm() / ref16.norm()).item()),
             "smallM154_fused_vs_unfused_rel": float(((ys - refs).norm() / refs.norm()).item()), "path": ("fused256" if N % 256 == 0 else "narrow128")}
        assert r["fused_vs_unfused_rel"] < 2e-3 and r["smallM154_fused_vs_unfused_rel"] < 2e-3, (K, N, r)
        if FAST_LAUNCH:                                   # direct launch vs the autotuner / JIT path: must be bit-identical (both the M % 16 == 0 and != 0 specialisations)
            for xx in (x, xs, x[:2312]):
                xq_, sx_ = int8_exec.quant_rows(xx); slow = (int8_exec.int8_gemm_fused if N % 256 == 0 else _gemm_fused_narrow)(xq_, sx_, q.weight_int8, q.w_scale, q.bias, torch.float16)
                assert torch.equal(q(xx), slow), ("fast launch mismatch", K, N, xx.shape)
            r["fast_launch_bitwise"] = True
        rep["selftest"][f"{K}x{N}"] = r
        del lin, x, q, y, ref, ref16
    torch.cuda.empty_cache()
    print("[sdxl-int8] selftest:", json.dumps(rep["selftest"]), flush=True)
    for M in Ms:
        per = {}; tb = tq = 0.0
        for (K, N) in SDXL_TOKEN_SHAPES:
            lin = nn.Linear(K, N, bias=True).to(dev, torch.float16)
            x = torch.randn(M, K, device=dev, dtype=torch.float16)
            q = SdxlInt8Linear(lin, device=torch.device(dev))
            b = int8_exec._bench(lambda: F.linear(x, lin.weight, lin.bias), iters)
            f = int8_exec._bench(lambda: q(x), iters)
            per[f"{K}x{N}"] = {"fp16_ms": round(b, 3), "int8_ms": round(f, 3), "speedup": round(b / f, 3)}
            tb += b; tq += f
            del lin, x, q; torch.cuda.empty_cache()
        rep["per_M"][str(M)] = {"fp16_ms_sum": round(tb, 2), "int8_ms_sum": round(tq, 2), "speedup_unweighted": round(tb / tq, 3), "per_shape": per}
        print(f"[sdxl-int8] M={M}: " + ", ".join(f"{k} x{v['speedup']:.2f}" for k, v in per.items()) + f" | unweighted sum x{tb/tq:.2f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out_dir / "kernel_bench_sdxl.json", "w"), indent=1)
    print(f"[sdxl-int8] saved {out_dir / 'kernel_bench_sdxl.json'}")


# --------------------------------------------------------------------------- plan / hook
def make_plan(executor: str, tau: float, vname: str, no_cache: bool, cascade: bool = False):
    if executor == "full":
        return {w: {"mode": "uniform", "v": vname} for w in (128, 256, 512)}
    # stage structure of configs/sdxl_pyraquant.json with hi / lo collapsed into the same variant (INT8 or fp16)
    plan = {128: {"mode": "uniform", "v": vname},
            256: {"mode": "spatial", "hi": vname, "lo": vname, "score": "residual", "thresh": tau, "norm": "relgap"},
            512: {"mode": "spatial", "hi": vname, "lo": vname, "score": "residual", "inherit": True, "thresh": tau, "norm": "relgap",
                  "inner_thresh": 1.0, "inner_norm": "median", "cache_bg": True}}
    if cascade:
        # final recipe: 4K cascade (inherit_hi) -- the children of the 2K high-precision leaves are all computed (median split
        # inside), the children of the 2K low-precision leaves are cached; no relative-gap test at 4K
        plan[512] = {"mode": "spatial", "hi": vname, "lo": vname, "score": "residual", "inherit_hi": True,
                     "inner_thresh": 1.0, "inner_norm": "median", "cache_bg": True}
    if no_cache:
        # rung 3 of the ladder: every 4K leaf active (fixed quota of 100%, no inheritance) -> 4 full-row strips + cache mask
        # identically 0.  cache_bg=False cannot be used for this (the routing boxes would not be armed and the pipeline
        # would silently fall back to the full canvas).  The 2K stage is the same as in the final recipe (hi == lo, one
        # full-canvas pass; the selection does not change the output).
        plan[512] = {"mode": "spatial", "hi": vname, "lo": vname, "score": "residual", "all_leaves": True, "inherit": False, "cache_bg": True}
    return plan


def make_hook(pipe, wrapper, plan, state, timeline, route: dict):
    """Mirror of make_spatial_hook in sdxl/run_sdxl.py (scores / selection / masks / cache / routing boxes) with per-stage
    timers; the score and selection rules are imported from that driver."""
    executor = route["executor"]; route_min_leaf = int(route["min_leaf"]); route_batch = int(route["batch"]); route_halo = int(route["halo_latent"])

    def hook(p, latents_LU, latents_RU, latents_LFM, image_RU):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        width = int(latents_LFM.shape[-1]); px = width * 8
        wrapper.ensure_stage(width)
        st = plan.get(width, {})
        if st.get("mode") != "spatial":
            pipe._route_boxes = None; pipe._cache_mask = None; pipe._x0_cache = None
            torch.cuda.synchronize(); timeline.setdefault("select", {})[px] = time.perf_counter() - t0; timeline.setdefault("hook_end", {})[px] = time.perf_counter()
            return
        n_leaf = width // LEAF_LATENT
        assert st.get("score", "residual") == "residual"
        sc = leaf_scores_residual(latents_LFM, latents_LU, n_leaf)                           # residual score of the driver
        if st.get("inherit_hi"):                                       # cascade: candidates = children of the previous stage's high-precision leaves, all computed
            par = state["hi_map"][width // 2]
            sel = par.repeat_interleave(2, 0).repeat_interleave(2, 1)
        elif st.get("all_leaves"):                                     # no-cache rung: every leaf active
            sel = torch.ones_like(sc)
        else:
            parent = state["leaf_hi"].get(width // 2) if st.get("inherit") else None
            sel = build_leaf_hi(sc, parent, float(st["thresh"]), st.get("norm", "relgap"))   # relative-gap rule of the driver
        inner_th = st.get("inner_thresh")
        if inner_th is not None:                                        # W8 / W4 split inside the active set (median rule) -- bookkeeping only, hi == lo in the latency table
            assert st.get("inner_norm", "median") == "median"
            hi_map = inner_split_median(sc, sel, float(inner_th))
        else:
            hi_map = sel
        state["leaf_hi"][width] = sel; state["hi_map"][width] = hi_map
        wrapper.masks[width] = soft_mask(hi_map, width, MASK_EDGE)
        if st.get("cache_bg"):
            pipe._cache_mask = 1.0 - soft_mask(sel, width, MASK_EDGE)
            pipe._x0_cache = latents_LFM.detach().clone()
        else:
            pipe._cache_mask = None; pipe._x0_cache = None
        if st.get("cache_bg") and executor == "routed":
            # --- line-by-line mirror of the driver: 2x2-leaf tiles (a tile is computed if any of its leaves is active) + adjacent tiles in a row merged into one crop ---
            L = LEAF_LATENT; N = int(sel.shape[0]); T = max(1, int(route_min_leaf))
            sm = (sel.detach().cpu().numpy() > 0.5)
            if N % T:
                raise ValueError(f"leaf grid {N} is not divisible by the tile size {T}")
            nt = N // T
            tm = sm.reshape(nt, T, nt, T).any(axis=(1, 3))
            rects = []
            for i in range(nt):
                j = 0
                while j < nt:
                    if tm[i, j]:
                        j0 = j
                        while j < nt and tm[i, j]: j += 1
                        rects.append((i * T * L, (i + 1) * T * L, j0 * T * L, j * T * L))
                    else:
                        j += 1
            pipe._route_boxes = rects
            pipe._route_batch = int(route_batch); pipe._route_halo_lat = int(route_halo)
            _S = N * L
            frac_halo = sum((min(_S, y1 + route_halo) - max(0, y0 - route_halo)) * (min(_S, x1 + route_halo) - max(0, x0 - route_halo)) for (y0, y1, x0, x1) in rects) / float(_S ** 2)
            frac_tiles = sum((y1 - y0) * (x1 - x0) for (y0, y1, x0, x1) in rects) / float(_S ** 2)
            state["route_frac"][width] = frac_halo; state["tile_frac"][width] = frac_tiles; state["n_boxes"][width] = len(rects)
        else:
            pipe._route_boxes = None
        f_hi = float(hi_map.mean().item()); f_sel = float(sel.mean().item())
        state["coverage"][width] = ({"hi": f_hi, "lo": f_sel - f_hi, "cache": 1.0 - f_sel} if st.get("cache_bg") else {"hi": f_hi, "lo": 1.0 - f_hi, "cache": 0.0})
        torch.cuda.synchronize(); timeline.setdefault("select", {})[px] = time.perf_counter() - t0; timeline.setdefault("hook_end", {})[px] = time.perf_counter()
    return hook


def read_prompts(path: Path, n: int, offset: int = 0):
    out = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if not line.strip() or i < offset:
                continue
            if len(out) >= n:
                break
            d = json.loads(line)
            out.append((d["name"], d["prompt"]))
    if len(out) < n:
        raise SystemExit(f"wanted {n} prompts, got {len(out)}")
    return out


# --------------------------------------------------------------------------- run one arm
def run(args):
    arm = args.arm
    linear, executor, no_cache, desc = ARMS[arm]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda *a: print(f"[sdxl-e2e:{arm}]", *a, flush=True)
    log(f"gpu={torch.cuda.get_device_name()} torch={torch.__version__} triton={triton.__version__} int8_backend={int8_exec.BACKEND} "
        f"leaf_latent={LEAF_LATENT} tau={args.tau} linear={linear} executor={executor} no_cache={no_cache} halo={args.route_halo} batch={args.route_batch} min_leaf={args.route_min_leaf}")

    t_load = time.perf_counter()
    if args.model_path:
        pipe = CustomStableDiffusionXLPipeline.from_pretrained(str(args.model_path), torch_dtype=torch.float16, variant="fp16",
                                                               use_safetensors=True, local_files_only=True)
    else:
        pipe = CustomStableDiffusionXLPipeline.from_pretrained(args.model_id, revision=args.model_revision, torch_dtype=torch.float16, variant="fp16",
                                                               use_safetensors=True, local_files_only=args.local_files_only)
    pipe = pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    unet = pipe.unet
    fp16_bytes = sum(p.numel() * p.element_size() for p in unet.parameters())
    # The timed model is ours: the weights are stored as RTN W8 g32 (nested W4 derived from the same codes) and the INT8
    # kernel computes on the codes derived from them.
    wsrc_report = None
    if args.weight_source == "method":
        t0 = time.perf_counter()
        wsrc_report = load_method_weights(unet, w_bits=8, group=32)
        wsrc_report["load_seconds"] = time.perf_counter() - t0
        print(f"[sdxl-bench] weight source = method ({wsrc_report['method']}): {wsrc_report['n_layers']} Linear, {wsrc_report['n_conv']} Conv2d in {wsrc_report['load_seconds']:.0f}s", flush=True)
    int8_report = None
    if linear == "int8":
        t0 = time.perf_counter()
        int8_report = convert_unet_linears_to_int8(unet, device="cuda")
        int8_report["convert_seconds"] = time.perf_counter() - t0
        torch.cuda.empty_cache()
        log(f"INT8 conversion: {int8_report['n_int8_linear']} Linear -> Int8Linear ({int8_report['int8_weight_gib']:.2f} GiB int8, shapes {int8_report['shapes_KxN']}), "
            f"{int8_report['n_fp16_linear_kept']} kept fp16 (per-image embeddings, {int8_report['params_fp16_linear_kept']/1e6:.1f}M params) in {int8_report['convert_seconds']:.0f}s")
        warm_sdxl_kernels()
    # Conv2d executes in fp16 in every arm (--conv fp16 is the only released option).
    vname = "fp16" if linear == "fp16" else "w8a8"
    plan = make_plan(executor, args.tau, vname, no_cache, cascade=bool(args.cascade))
    wrapper = SpatialMaskUNet({vname: unet}, plan, base_name=vname)
    pipe.unet = wrapper
    wrapper.ensure_stage(128)
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    unet_bytes = sum(p.numel() * p.element_size() for p in unet.parameters()) + sum(b.numel() * b.element_size() for b in unet.buffers())
    load_seconds = time.perf_counter() - t_load
    log(f"resident: unet {gib(unet_bytes):.2f} GiB (fp16 params {gib(fp16_bytes):.2f}), total allocated {gib(resident):.2f} GiB (text encoders + VAE resident), "
        f"free {gib(torch.cuda.mem_get_info()[0]):.2f} GiB, load {load_seconds:.0f}s")

    # ---- instance-level timing patches: text encoding / VAE encode (LFM) / VAE decode (stage end; also records the per-stage peak) ----
    timeline = {}
    orig_encode_prompt, orig_vae_decode, orig_vae_encode = pipe.encode_prompt, pipe.vae.decode, pipe.vae.encode

    def timed_encode_prompt(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig_encode_prompt(*a, **k)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        timeline["text_encode"] = t1 - t0; timeline["text_end"] = t1
        return r

    def timed_vae_decode(z, *a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        px = int(z.shape[-1]) * 8
        timeline.setdefault("decode_start", {})[px] = t0
        r = orig_vae_decode(z, *a, **k)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        timeline.setdefault("decode", {})[px] = t1 - t0
        timeline.setdefault("stage_end", {})[px] = t1
        timeline.setdefault("peak_alloc", {})[px] = gib(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        return r

    def timed_vae_encode(img, *a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig_vae_encode(img, *a, **k)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        timeline.setdefault("encode", {})[int(img.shape[-1])] = t1 - t0
        return r
    pipe.encode_prompt = timed_encode_prompt
    pipe.vae.decode = timed_vae_decode
    pipe.vae.encode = timed_vae_encode

    route = {"executor": executor, "min_leaf": args.route_min_leaf, "batch": args.route_batch, "halo_latent": args.route_halo}
    state = {"leaf_hi": {}, "hi_map": {}, "coverage": {}, "route_frac": {}, "tile_frac": {}, "n_boxes": {}}
    pipe._spatial_hook = make_hook(pipe, wrapper, plan, state, timeline, route)

    prompts = read_prompts(Path(args.prompts_file), args.n, args.offset)
    records = []
    rec_path = out_dir / "rec.json"
    meta = {"arm": arm, "description": desc, "linear": linear, "executor": executor, "no_cache": no_cache, "tau": args.tau,
            "route": ({"min_leaf": args.route_min_leaf, "batch": args.route_batch, "halo_latent": args.route_halo} if executor == "routed" else None),
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
            "weight_source": args.weight_source, "weight_source_report": wsrc_report,
            "int8_backend": (int8_exec.BACKEND if linear == "int8" else None), "fast_launch": FAST_LAUNCH,
            "conv": args.conv, "conv_report": None,
            "int8_report": ({k: v for k, v in int8_report.items() if k != "fp16_kept_names"} if int8_report else None),
            "fp16_kept_names": (int8_report["fp16_kept_names"] if int8_report else None),
            "plan": {str(k): v for k, v in plan.items()}, "leaf_latent": LEAF_LATENT, "mask_edge": MASK_EDGE,
            "steps": f"{args.steps} steps at 1K, restart_ratio 0.4 at 2K and 4K (50/20/20 for the default), CFG 7.5, scale_factor 0.125, upsample_stage 2", "seed": SEED,
            "model": (str(args.model_path) if args.model_path else f"{args.model_id}@{args.model_revision}"),
            "resident_alloc_gib": gib(resident), "unet_bytes_gib": gib(unet_bytes), "unet_fp16_params_gib": gib(fp16_bytes),
            "load_seconds_excluded": load_seconds, "prompts_file": str(args.prompts_file), "offset": args.offset, "n_timed": args.n, "warmup": args.warmup,
            "notes": ["weights resident on one GPU (no CPU offload); text encoders (CLIP-L + OpenCLIP-G) resident and their encoding is INSIDE the timed total (identical across arms, ~0.1 s, reported separately)",
                      "attention (SDPA, window attention of ScaleDiff-SDXL) stays fp16 in all arms; Conv2d stays fp16 in all arms (torch has no CUDA INT8 conv); no torch.compile / CUDA graphs",
                      "int8 arms: every token-level nn.Linear of the UNet -> INT8 (per-channel W, fused per-token A8, Triton fused GEMM); time_embedding/add_embedding/time_emb_proj stay fp16",
                      "routed arms: hi/lo (W8/W4 nested) dual pass collapsed to ONE pass of the same numerics -- the W4 tier is a storage/BOPs effect only, not a wall-clock effect",
                      "routed arms 2K stage: relgap selection computed (inherited by 4K) but full canvas, no cache at 2K (as in the recipe) -> same work as the full-canvas arm",
                      "quality NOT measured here (INT8 numerics differ from the QDQ simulation); the warm-up 4K PNG is a numerics sanity artifact only"]
                     + (["no_cache: all 64 4K leaves selected (fixed quota of 100%, no inheritance) -> 4 full-row strips (128x512 latent) + halo; cache mask identically 0"] if no_cache else [])}
    schedule = [(True, prompts[0])] * int(args.warmup) + [(False, p) for p in prompts]
    for k, (is_warm, (name, prompt)) in enumerate(schedule):
        for key in ("leaf_hi", "hi_map", "coverage", "route_frac", "tile_frac", "n_boxes"):
            state[key].clear()
        pipe._cache_mask = None; pipe._x0_cache = None; pipe._route_boxes = None; pipe._route_stats = []
        wrapper.masks.clear(); wrapper.call_stats.clear(); timeline.clear()
        wrapper.ensure_stage(128)
        gen = torch.Generator(device="cuda").manual_seed(SEED)
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t_start = time.perf_counter()
        images = pipe(prompt, negative_prompt=NEGATIVE, height=1024, width=1024, generator=gen, num_inference_steps=args.steps,
                      guidance_scale=7.5, restart_ratio=0.4, scale_factor=0.125, upsample_stage=2)
        torch.cuda.synchronize(); t_end = time.perf_counter()
        total = t_end - t_start
        if images[-1].size != (4096, 4096):
            raise RuntimeError(f"unexpected final size {images[-1].size}")
        if executor == "routed" and not pipe._route_stats:
            raise RuntimeError("routed arm but no routed 4K step recorded")
        ends = timeline.get("stage_end", {}); dstart = timeline.get("decode_start", {}); hend = timeline.get("hook_end", {})
        stage_wall, prev = {}, t_start
        for px in STAGE_PX:
            e = ends.get(px, t_end)
            stage_wall[px] = e - prev; prev = e
        stage_wall[4096] += t_end - ends.get(4096, t_end)                # the 4K post-processing counts towards 4K
        denoise = {1024: dstart.get(1024, t_end) - timeline.get("text_end", t_start)}
        for px in (2048, 4096):
            denoise[px] = dstart.get(px, t_end) - hend.get(px, dstart.get(px, t_end))
        calls = defaultdict(int)
        for key_, c in wrapper.call_stats.items():
            calls[int(key_.split("|")[0]) * 8] += int(c)
        rs = pipe._route_stats or []
        r4 = [x for x in rs if x["resolution"] == 4096]
        exec_frac = (sum(x["routed_px"] for x in r4) / max(1, sum(x["full_px"] for x in r4))) if r4 else 1.0
        cov4 = state["coverage"].get(512); cov2 = state["coverage"].get(256)
        rec = {"prompt_name": name, "seed": SEED, "warmup": is_warm, "total_s": total, "text_encode_s_included": timeline.get("text_encode"),
               "stage_wall_s": {str(px): stage_wall[px] for px in STAGE_PX},
               "denoise_s": {str(px): denoise[px] for px in STAGE_PX},
               "vae_decode_s": {str(px): timeline.get("decode", {}).get(px) for px in STAGE_PX},
               "vae_encode_s": {str(px): timeline.get("encode", {}).get(px) for px in STAGE_PX[1:]},
               "select_s": {str(px): timeline.get("select", {}).get(px) for px in STAGE_PX[1:]},
               "peak_alloc_gib": {str(px): timeline.get("peak_alloc", {}).get(px) for px in STAGE_PX},
               "unet_calls": {str(px): int(calls.get(px, 0)) for px in STAGE_PX},
               "routed_px_frac_4096": (exec_frac if executor == "routed" else 1.0),
               "tile_frac_4096": (state["tile_frac"].get(512) if executor == "routed" else 1.0),
               "selected_leaf_frac_4096": ((cov4["hi"] + cov4["lo"]) if cov4 else 1.0),
               "hi_leaf_frac_4096": (cov4["hi"] if cov4 else 1.0),
               "selected_leaf_frac_2048": ((cov2["hi"] + cov2["lo"]) if cov2 else 1.0),
               "n_boxes_4096": state["n_boxes"].get(512),
               "coverage": {str(k2 * 8): v for k2, v in state["coverage"].items()},
               "route_steps": rs, "call_stats": dict(wrapper.call_stats)}
        records.append(rec)
        if is_warm:
            images[-1].save(out_dir / f"warmup_{arm}_{name}_4096.png")   # only the 4K image; numerics sanity artifact
        json.dump({"meta": meta, "records": records}, open(rec_path, "w"), indent=1)
        log(f"{k+1}/{len(schedule)} {name}{' [warm-up]' if is_warm else ''}: total {total:.1f}s | 1K {stage_wall[1024]:.1f} 2K {stage_wall[2048]:.1f} 4K {stage_wall[4096]:.1f} | "
            f"denoise {denoise[1024]:.1f}/{denoise[2048]:.1f}/{denoise[4096]:.1f} | vae dec {timeline.get('decode', {}).get(4096, 0):.1f} enc {timeline.get('encode', {}).get(4096, 0):.1f} | "
            f"unet calls {rec['unet_calls']} | peak {rec['peak_alloc_gib']['1024']:.1f}/{rec['peak_alloc_gib']['2048']:.1f}/{rec['peak_alloc_gib']['4096']:.1f} GiB"
            + (f" | 4K area {exec_frac:.3f} (tiles {rec['tile_frac_4096']:.3f}, leaves {rec['selected_leaf_frac_4096']:.3f}, hi {rec['hi_leaf_frac_4096']:.3f}, boxes {rec['n_boxes_4096']})" if executor == "routed" else ""))
    log(f"done -> {rec_path}")


# --------------------------------------------------------------------------- summary / breakdown
def _stats(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
    if v.size == 0:
        return None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"median": float(med), "iqr": float(q3 - q1), "q1": float(q1), "q3": float(q3), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def _psnr(a_path, b_path):
    from PIL import Image
    a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    return (10.0 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf")


def _arm_stats(path: Path):
    d = json.load(open(path)); meta, recs = d["meta"], [r for r in d["records"] if not r["warmup"]]
    S = lambda key: _stats([r[key] for r in recs])
    e = {"rec_path": str(path), "n": len(recs), "n_warmup": sum(1 for r in d["records"] if r["warmup"]),
         **{k: meta.get(k) for k in ("arm", "description", "linear", "executor", "no_cache", "tau", "route", "plan", "int8_backend", "int8_report", "weight_source", "conv", "gpu",
                                     "resident_alloc_gib", "unet_bytes_gib", "unet_fp16_params_gib", "steps", "seed", "notes")},
         "total_s": S("total_s"), "text_encode_s_included": S("text_encode_s_included"),
         "stage_wall_s": {px: _stats([r["stage_wall_s"][px] for r in recs]) for px in ("1024", "2048", "4096")},
         "denoise_s": {px: _stats([r["denoise_s"][px] for r in recs]) for px in ("1024", "2048", "4096")},
         "vae_decode_s": {px: _stats([r["vae_decode_s"][px] for r in recs]) for px in ("1024", "2048", "4096")},
         "vae_encode_s": {px: _stats([r["vae_encode_s"][px] for r in recs]) for px in ("2048", "4096")},
         "select_s": {px: _stats([r["select_s"][px] for r in recs]) for px in ("2048", "4096")},
         "peak_alloc_gib": {px: _stats([r["peak_alloc_gib"][px] for r in recs]) for px in ("1024", "2048", "4096")},
         "unet_calls": {px: _stats([r["unet_calls"][px] for r in recs]) for px in ("1024", "2048", "4096")},
         "routed_px_frac_4096": S("routed_px_frac_4096"), "tile_frac_4096": S("tile_frac_4096"),
         "selected_leaf_frac_4096": S("selected_leaf_frac_4096"), "hi_leaf_frac_4096": S("hi_leaf_frac_4096"), "selected_leaf_frac_2048": S("selected_leaf_frac_2048"),
         "n_boxes_4096": S("n_boxes_4096"),
         "per_prompt": [{"prompt_name": r["prompt_name"], "total_s": r["total_s"], "stage_wall_s": r["stage_wall_s"], "denoise_4096": r["denoise_s"]["4096"],
                         "peak4k": r["peak_alloc_gib"]["4096"], "area4k": r["routed_px_frac_4096"], "tiles4k": r["tile_frac_4096"], "leaves4k": r["selected_leaf_frac_4096"],
                         "unet_calls_4096": r["unet_calls"]["4096"]} for r in recs]}
    return e


def breakdown(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.json) if args.json else out_dir / "latency_sdxl.json"
    arms, missing = {}, []
    for arm in LADDER:
        p = out_dir / arm / "rec.json"
        if not p.exists():
            missing.append((arm, str(p))); continue
        arms[arm] = _arm_stats(p)
    if missing:
        print("[breakdown] missing arms: " + ", ".join(f"{a} ({p})" for a, p in missing))
    if not arms:
        raise SystemExit(f"no arm records found under {out_dir}/<arm>/rec.json")
    med = lambda a: arms[a]["total_s"]["median"]
    st4 = lambda a: arms[a]["stage_wall_s"]["4096"]["median"]
    dn4 = lambda a: arms[a]["denoise_s"]["4096"]["median"]

    def ratio(num, den, f):
        return (f(num) / f(den)) if (num in arms and den in arms) else None
    pairs = {"kernel_gain_t1_over_t2": ("fp16_full", "int8_full"), "tiles_effect_t2_over_t3": ("int8_full", "int8_routed_nocache"),
             "cache_gain_t3_over_t4": ("int8_routed_nocache", "int8_routed_cache"), "total_t1_over_t4": ("fp16_full", "int8_routed_cache"),
             "strategy_only_fp16_t1_over_t5": ("fp16_full", "fp16_routed_cache"), "kernel_on_top_of_strategy_t5_over_t4": ("fp16_routed_cache", "int8_routed_cache"),
             "strategy_on_top_of_kernel_t2_over_t4": ("int8_full", "int8_routed_cache")}
    decomp = {k: {"total": ratio(a, b, med), "stage_4096": ratio(a, b, st4), "denoise_4096": ratio(a, b, dn4)} for k, (a, b) in pairs.items()}
    # PSNR between the warm-up 4K images (numerics sanity, not a quality claim)
    warm = {a: sorted((out_dir / a).glob(f"warmup_{a}_*_4096.png")) for a in LADDER}
    psnr = {}
    for a, b in (("int8_full", "fp16_full"), ("int8_routed_nocache", "int8_full"), ("int8_routed_cache", "int8_full"), ("fp16_routed_cache", "fp16_full"), ("int8_routed_cache", "fp16_full")):
        if warm.get(a) and warm.get(b):
            psnr[f"{a}_vs_{b}_4096_db"] = _psnr(warm[a][0], warm[b][0])
    kb = out_dir / "kernel_bench_sdxl.json"
    kernel_bench = json.load(open(kb)) if kb.exists() else None
    tau = next((arms[a]["tau"] for a in ("int8_routed_cache", "fp16_routed_cache") if a in arms), None)
    res = {"gpu": next(iter(arms.values()))["gpu"], "ladder": list(LADDER), "arms": arms, "decomposition": decomp, "psnr_sanity_4096": psnr,
           "kernel_bench_sdxl": ({"per_M": {m: {k: v for k, v in d.items() if k != "per_shape"} for m, d in kernel_bench["per_M"].items()}, "selftest": kernel_bench["selftest"]} if kernel_bench else None),
           "missing": missing,
           "definitions": {"t1": "fp16_full", "t2": "int8_full", "t3": "int8_routed_nocache", "t4": f"int8_routed_cache (tau={tau})", "t5": f"fp16_routed_cache (tau={tau})",
                           "ratios": "median total wall-clock (1K+2K+4K incl. text encoding and VAE) of the numerator arm over the denominator arm; stage_4096 / denoise_4096 = same ratio on the 4K stage / the 4K denoise loop only",
                           "routed_px_frac_4096": "sum over the 20 4K steps of executed crop area (tiles/strips + 8-latent halo, clamped to the canvas) / (20 x 512^2 latent); full-canvas arms = 1.00",
                           "tile_frac_4096": "area of the routed 2x2-leaf tiles / strips without halo / canvas", "selected_leaf_frac_4096": "fraction of the 64 4K leaves active (residual score + relative-gap rule, inherited from 2K)",
                           "hi_leaf_frac_4096": "fraction of leaves in the W8 (hi) tier of the recipe (inner median split); the rest of the active leaves would be W4 -- collapsed to the same pass here"}}
    res["caveats"] = [
        f"All arms: ScaleDiff-SDXL 1K->2K->4K (50/20/20 steps, restart_ratio 0.4, CFG 7.5, scale_factor 0.125, ScaleDiff window attention) on one {res['gpu']} with all weights resident; the same 20 prompts (the first 20 of prompts/eval_ultrahr_2000.jsonl, seed 42 as in the quality runs) + 1 warm-up; text encoding (~0.1 s) is inside the total; no torch.compile / CUDA graphs.",
        "INT8 is Linear-only: every token-level nn.Linear of the UNet runs as per-channel-INT8-W / per-token-INT8-A Triton fused GEMM (deploy/int8_exec.py kernels; the 640-channel layers use a BLOCK_N=128 autotune set). Conv2d stays fp16 in every arm because torch has no CUDA INT8 convolution -- Conv is ~24% of SDXL MACs, so the kernel rung leaves that share untouched; attention (SDPA) is fp16; time_embedding / add_embedding / time_emb_proj Linears (per-image, M=batch) stay fp16.",
        "The routed arms collapse the recipe's hi/lo (W8 | nested W4, g32) dual pass into ONE pass of the same numerics: the W4 tier is a storage/BOPs effect only and contributes nothing to wall-clock here; the kernel gain is credited entirely to the INT8 Linear path.",
        "The INT8 numerics (per-channel absmax requantisation of the fp16 weights + dynamic per-token absmax activations, int32 accumulate, fp32 epilogue) differ from the QDQ simulation (RTN W8 g32 + A8 fake-quant) that carries the quality tables; no quality claim is made from these runs -- the warm-up 4K PSNR is a numerics sanity check only.",
        "int8_routed_nocache = all 64 4K leaves active (fixed quota of 100%, no 2K inheritance): the driver logic builds 4 full-row strips (128x512 latent) which with the 8-latent halo become 136/144 x 512 crops (executed area 1.09x canvas), grouped by size into 2 UNet calls per step; the cache mask is identically zero. It is NOT cache_bg=False (that would silently fall back to the full canvas). Its 1K/2K stages are the same work as int8_full.",
        "Executed 4K area of the cache arms is content-dependent (relative-gap tau on the inherited candidate set, 2x2-leaf tiles merged per row + 8-latent halo); the per-prompt spread is the IQR. The 2K stage of the routed arms runs the full canvas (no cache at 2K in the recipe) and the 4K VAE encode/decode (fp32, tiled) is host-inherent, so end-to-end ratios are bounded by those fixed costs; the 4K-stage and 4K-denoise columns show the executor effect without them.",
        "fp16 and INT8 cache arms select blocks per prompt on different trajectories (residual scores differ), so their executed fractions are not paired; compare medians only.",
    ]
    json.dump(res, open(json_path, "w"), indent=1)
    f = lambda s, d=1: (f"{s['median']:.{d}f} [{s['iqr']:.{d}f}]" if s else "-")
    lines = [f"## E2E latency ladder, ScaleDiff-SDXL 1K->2K->4K (50/20/20 steps, CFG 7.5), 1x {res['gpu']}, weights resident, relgap tau={tau} (median [IQR] over n timed prompts)", "",
             "| # | arm | Linear | 4K executor | n | 1K (s) | 2K (s) | 4K (s) | total (s) | 4K denoise (s) | 4K exec. area (incl. halo) | 4K tiles (no halo) | 4K leaves selected | 4K UNet calls | 4K peak (GiB) | vs fp16_full |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    execs = {"fp16_full": "full canvas", "int8_full": "full canvas", "int8_routed_nocache": "routed strips, all leaves, no cache",
             "int8_routed_cache": "routed tiles + spatial cache", "fp16_routed_cache": "routed tiles + spatial cache"}
    for i, arm in enumerate(LADDER, 1):
        if arm not in arms:
            lines.append(f"| {i} | {arm} | - | - | - | (missing) | | | | | | | | | | |"); continue
        e = arms[arm]; full = e["executor"] != "routed"
        sp = med("fp16_full") / med(arm) if "fp16_full" in arms else float("nan")
        lines.append(f"| {i} | {arm} | {e['linear']} | {execs[arm]} | {e['n']} | {f(e['stage_wall_s']['1024'])} | {f(e['stage_wall_s']['2048'])} | {f(e['stage_wall_s']['4096'])} | "
                     f"{f(e['total_s'])} | {f(e['denoise_s']['4096'])} | {'1.00' if full else f(e['routed_px_frac_4096'], 2)} | {'1.00' if full else f(e['tile_frac_4096'], 2)} | "
                     f"{'1.00' if full else f(e['selected_leaf_frac_4096'], 2)} | {f(e['unet_calls']['4096'], 0)} | {e['peak_alloc_gib']['4096']['median']:.1f} (max {e['peak_alloc_gib']['4096']['max']:.1f}) | {sp:.2f}x |")
    lines += ["", "Decomposition (ratio of median wall-clock; total / 4K stage / 4K denoise loop):", "", "| factor | ratio | total | 4K stage | 4K denoise |", "|---|---|---|---|---|"]
    names = {"kernel_gain_t1_over_t2": "INT8 Linear kernel", "tiles_effect_t2_over_t3": "tile/strip execution + halo (no cache)", "cache_gain_t3_over_t4": "spatial cache reuse",
             "total_t1_over_t4": "total (final recipe executor)", "strategy_only_fp16_t1_over_t5": "strategy only at fp16 (no kernel)", "kernel_on_top_of_strategy_t5_over_t4": "kernel on top of strategy",
             "strategy_on_top_of_kernel_t2_over_t4": "strategy on top of kernel"}
    g = lambda v: (f"{v:.2f}x" if v else "-")
    for k, (a, b) in pairs.items():
        d = decomp[k]
        lines.append(f"| {names[k]} | {a} / {b} | {g(d['total'])} | {g(d['stage_4096'])} | {g(d['denoise_4096'])} |")
    lines += ["", "denoise-loop only (s, median 1K/2K/4K): " + "; ".join(f"{a}: " + "/".join(f"{e['denoise_s'][px]['median']:.1f}" for px in ("1024", "2048", "4096")) for a, e in arms.items()),
              "VAE (s, median; fp32 tiled, host-inherent): " + "; ".join(f"{a}: dec " + "/".join(f"{e['vae_decode_s'][px]['median']:.1f}" for px in ("1024", "2048", "4096")) + ", enc " + "/".join(f"{e['vae_encode_s'][px]['median']:.1f}" for px in ("2048", "4096")) for a, e in arms.items()),
              "peak memory per stage (GiB, median 1K/2K/4K): " + "; ".join(f"{a}: " + "/".join(f"{e['peak_alloc_gib'][px]['median']:.1f}" for px in ("1024", "2048", "4096")) for a, e in arms.items())]
    if any(a in arms for a in ("int8_routed_cache", "fp16_routed_cache")):
        lines.append("4K selection (median [IQR], range): " + "; ".join(
            f"{a}: leaves {f(e['selected_leaf_frac_4096'], 2)} ({e['selected_leaf_frac_4096']['min']:.2f}-{e['selected_leaf_frac_4096']['max']:.2f}), W8-tier {f(e['hi_leaf_frac_4096'], 2)}, "
            f"tiles {f(e['tile_frac_4096'], 2)}, exec. area {f(e['routed_px_frac_4096'], 2)} ({e['routed_px_frac_4096']['min']:.2f}-{e['routed_px_frac_4096']['max']:.2f}), boxes/step {f(e['n_boxes_4096'], 1)}"
            for a, e in arms.items() if e["executor"] == "routed"))
    if all(a in arms for a in ("int8_full", "int8_routed_nocache")):
        r = decomp["tiles_effect_t2_over_t3"]
        lines.append(f"Strip execution without cache is {'SLOWER' if r['total'] < 1 else 'faster'} than the INT8 full canvas ({r['total']:.2f}x total; 4K stage {r['stage_4096']:.2f}x, 4K denoise {r['denoise_4096']:.2f}x) "
                     f"while executing {arms['int8_routed_nocache']['routed_px_frac_4096']['median']:.2f}x the canvas area (halo); per executed latent px the strip executor is "
                     f"{r['denoise_4096'] * arms['int8_routed_nocache']['routed_px_frac_4096']['median']:.2f}x the speed of the host full-canvas step.")
    if psnr:
        lines.append("PSNR sanity (warm-up prompt, 4K PNG): " + ", ".join(f"{k}={v:.1f} dB" for k, v in psnr.items()))
    if kernel_bench:
        lines.append("INT8 vs fp16 Linear kernel micro-bench on the SDXL token shapes (unweighted sum over the 8 shapes): " + ", ".join(f"M={m} x{d['speedup_unweighted']:.2f}" for m, d in kernel_bench["per_M"].items()))
    md = "\n".join(lines)
    print(md)
    (out_dir / "summary.md").write_text(md + "\n\nCaveats:\n" + "\n".join("- " + c for c in res["caveats"]) + "\n")
    print(f"[breakdown] saved {json_path} and {out_dir / 'summary.md'}")


def main():
    global FAST_LAUNCH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=tuple(ARMS), default="fp16_full")
    ap.add_argument("--n", type=int, default=20, help="number of timed prompts")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1, help="untimed warm-up generations of the first prompt (its 4K image is saved)")
    ap.add_argument("--steps", type=int, default=50, help="1K steps; 2K and 4K run restart_ratio 0.4 of them (50/20/20)")
    ap.add_argument("--tau", type=float, default=0.10, help="relative-gap threshold at 2K (and at 4K without --cascade); the paper's final recipe uses 0.10 (configs/sdxl_pyraquant.json)")
    ap.add_argument("--route-halo", type=int, default=8, help="halo around each routed crop, in latent pixels (multiple of 8)")
    ap.add_argument("--cascade", action="store_true", help="4K rule of the final recipe (inherit_hi): the children of the 2K high-precision leaves stay active (median split inside), the children of the 2K low-precision leaves are cached")
    ap.add_argument("--route-batch", type=int, default=4, help="max crops per UNet call in the routed executor")
    ap.add_argument("--route-min-leaf", type=int, default=2, help="routing tile edge in leaves (2 = 2x2-leaf tiles)")
    ap.add_argument("--weight-source", choices=("host", "method"), default="host",
                    help="host: derive the INT8 codes from the stock fp16 weights. method: write the recipe's RTN W8 g32 weights "
                         "(per-output-channel W8 on Conv2d) back first and derive the INT8 codes from those (the paper's INT8 rows; timing is unaffected)")
    ap.add_argument("--conv", choices=("fp16",), default="fp16", help="execution precision of Conv2d (fp16 / cuDNN in every arm)")
    ap.add_argument("--int8-backend", choices=("triton", "cublas"), default="triton", help="INT8 GEMM backend of deploy/int8_exec.py")
    ap.add_argument("--no-fast-launch", action="store_true", help="launch the Triton kernels through the JIT / autotuner dispatch instead of the cached compiled kernels (same numerics, more Python overhead)")
    ap.add_argument("--prompts-file", default=str(ROOT / "prompts" / "eval_ultrahr_2000.jsonl"))
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--model-revision", default=MODEL_REVISION)
    ap.add_argument("--model-path", type=Path, default=None, help="local diffusers directory of SDXL base 1.0 (fp16 variant) instead of --model-id / --model-revision")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "latency_sdxl"), help="run: where rec.json and the warm-up PNG go; --breakdown: the directory holding <arm>/rec.json")
    ap.add_argument("--json", default=None, help="--breakdown: output JSON (default <out-dir>/latency_sdxl.json)")
    ap.add_argument("--breakdown", action="store_true", help="summarise the five-arm ladder (--out-dir/<arm>/rec.json) -> --json + --out-dir/summary.md")
    ap.add_argument("--selftest", action="store_true", help="kernel self-test + micro-benchmark on the SDXL shapes -> --out-dir/kernel_bench_sdxl.json")
    args = ap.parse_args()
    int8_exec.set_backend(args.int8_backend)
    FAST_LAUNCH = not args.no_fast_launch
    if args.breakdown:
        breakdown(args)
    elif args.selftest:
        selftest_and_bench(Path(args.out_dir))
    else:
        run(args)


if __name__ == "__main__":
    main()
