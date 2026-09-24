"""Conv2d weight quantization for PyraQuant -- RTN / AWQ / GPTQ on the im2col view.

Why this file exists
--------------------
``quantizers_weight.py`` only touches ``nn.Linear``.  On the SDXL UNet that leaves
0.333 B of the 2.567 B parameters (13.0 %) in fp16, so a run advertised as "W4" is
really running at ``(2.234*4 + 0.333*16)/2.567 = 5.56`` bit.  This module closes that
gap without touching a single line of the existing code.

The whole idea in one equation
------------------------------
A ``Conv2d`` is a matmul over patches.  With ``W`` of shape ``[O, C, kh, kw]`` and
``im2col(x)`` of shape ``[N*Ho*Wo, C*kh*kw]`` (rows = receptive fields, columns
ordered ``(c, i, j)`` with ``c`` slowest -- exactly ``torch.nn.functional.unfold``'s
layout)::

    conv2d(x, W)[n, o, h, w]  ==  ( im2col(x) @ W.reshape(O, -1).T )[n*Ho*Wo + h*Wo + w, o]

So ``W.reshape(O, C*kh*kw)`` *is* a linear layer's weight matrix, and RTN / GPTQ /
AWQ apply verbatim -- this is precisely what the original GPTQ release does for conv
(``gptq/modelutils.py`` folds conv weights to 2-D before quantizing).  Every routine
below therefore builds a **zero-copy ``nn.Linear`` proxy that shares storage with the
conv weight** and hands it to the already-tested ``quantizers_weight`` implementations.
No quantization math is reimplemented here.

Group granularity runs along the *flattened* input dim ``C*kh*kw``.  Consequences for
SDXL (``group_size`` must divide it or ``effective_group_size`` silently falls back to
per-out-channel, which is much worse at W2/W3)::

    conv 3x3, C=320  -> 2880   2880 % 128 = 64  -> FALLBACK      2880 % 64 = 0  -> ok
    conv 3x3, C=640  -> 5760   5760 % 128 = 0   -> ok
    conv 3x3, C=1280 -> 11520  11520 % 128 = 0  -> ok
    conv 3x3, C=1920 -> 17280  17280 % 128 = 0  -> ok
    conv 3x3, C=2560 -> 23040  23040 % 128 = 0  -> ok
    conv 1x1, C=320  -> 320    320 % 128 = 64   -> FALLBACK      320 % 64 = 0   -> ok
    conv_in 3x3, C=3 -> 27     divides nothing  -> per-row (8.6 k params, ignorable)

**Use ``group_size=64`` for conv, same as the recommendation for Linear.**
``conv_group_size_report`` prints the fallback set for any model/group_size.

Activation-side transforms (SmoothQuant / rotation) on Conv2d
-------------------------------------------------------------
See the module-level notes in ``SMOOTHQUANT_ON_CONV`` / ``ROTATION_ON_CONV`` and the
docstring of ``awq_quantize_conv_``.  Short version:

  * per-input-**channel** diagonal scaling (SmoothQuant, and AWQ's internal scale)
    commutes with Conv2d exactly -- including through zero/reflect/replicate padding;
  * a general orthogonal rotation over the flattened ``C*kh*kw`` axis is **not**
    realizable as an op on the feature map (the rows are overlapping patches);
  * a rotation restricted to the **channel** axis *is* exact, at the cost of an extra
    1x1 conv that GroupNorm prevents you from folding away.

Only ``torch`` + ``pyraquant.quantizers_weight`` are imported.  Nothing here
mutates any other project file.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # package import
    from .quantizers_weight import (
        _EPS,
        _qmax,
        W8_GROUPED,
        act_channel_absmax_from_inputs,
        awq_quantize_,
        effective_group_size,
        gptq_quantize_,
        qdq_weight,
        rtn_quantize_,
    )
except ImportError:  # running this file directly (python quantizers_conv.py)
    from quantizers_weight import (  # type: ignore[no-redef]
        _EPS,
        _qmax,
        W8_GROUPED,
        act_channel_absmax_from_inputs,
        awq_quantize_,
        effective_group_size,
        gptq_quantize_,
        qdq_weight,
        rtn_quantize_,
    )

__all__ = [
    # --- shape helpers ------------------------------------------------------
    "conv_flat_in_dim",
    "flatten_conv_weight",
    "conv_output_hw",
    "unfold_conv_input",
    "conv_group_size_report",
    # --- quantizers ---------------------------------------------------------
    "rtn_quantize_conv_",
    "gptq_quantize_conv_",
    "awq_quantize_conv_",
    "quantize_conv_weights_",
    "conv_quant_report",
    # --- calibration --------------------------------------------------------
    "collect_conv_layer_inputs",
    "collect_conv_act_channel_absmax",
    "conv_act_channel_absmax_from_inputs",
    # --- accounting ---------------------------------------------------------
    "conv_param_share",
    "average_bits_from_counts",
    "average_bits",
    # --- notes --------------------------------------------------------------
    "SMOOTHQUANT_ON_CONV",
    "ROTATION_ON_CONV",
]


# ===========================================================================
# 1. shape / im2col helpers
# ===========================================================================
def _pair(v) -> Tuple[int, int]:
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


_PAD_MODE = {"zeros": "constant", "reflect": "reflect", "replicate": "replicate",
             "circular": "circular"}


def _pad_lrtb(conv: nn.Conv2d) -> Tuple[int, int, int, int]:
    """(left, right, top, bottom) padding, resolving the ``'same'``/``'valid'`` strings."""
    p = conv.padding
    kh, kw = _pair(conv.kernel_size)
    dh, dw = _pair(conv.dilation)
    if isinstance(p, str):
        if p == "valid":
            return 0, 0, 0, 0
        if p == "same":
            th, tw = dh * (kh - 1), dw * (kw - 1)
            return tw // 2, tw - tw // 2, th // 2, th - th // 2
        raise ValueError(f"unsupported padding string {p!r}")
    ph, pw = _pair(p)
    return pw, pw, ph, ph


def conv_flat_in_dim(conv: nn.Conv2d, per_group: bool = True) -> int:
    """Length of the flattened weight row.

    ``per_group=True``  -> ``(C/groups) * kh * kw``  (what one weight row actually has)
    ``per_group=False`` -> ``C * kh * kw``           (what ``unfold`` produces)
    """
    kh, kw = _pair(conv.kernel_size)
    c = conv.in_channels // (conv.groups if per_group else 1)
    return int(c * kh * kw)


def flatten_conv_weight(conv: nn.Conv2d) -> torch.Tensor:
    """``[O, C/groups, kh, kw] -> [O, (C/groups)*kh*kw]`` as a **view** (shares storage)."""
    w = conv.weight.data
    return w.reshape(w.shape[0], -1)


def conv_output_hw(conv: nn.Conv2d, h: int, w: int) -> Tuple[int, int]:
    kh, kw = _pair(conv.kernel_size)
    sh, sw = _pair(conv.stride)
    dh, dw = _pair(conv.dilation)
    pl, pr, pt, pb = _pad_lrtb(conv)
    ho = (h + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    wo = (w + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    return int(ho), int(wo)


def unfold_conv_input(
    x: torch.Tensor,
    conv: nn.Conv2d,
    *,
    max_rows: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """im2col of ``x`` for ``conv``: ``[N, C, H, W] -> [M, C*kh*kw]``.

    Row ``n*Ho*Wo + h*Wo + w`` is the receptive field of output pixel ``(n, h, w)``;
    columns are ordered ``(c, i, j)`` with ``c`` slowest -- **bit-identical to
    ``F.unfold(x, ...).transpose(1, 2).reshape(-1, C*kh*kw)``** and therefore aligned
    with ``flatten_conv_weight``.  Verified numerically in ``_test_im2col_identity``.

    Memory: the full im2col is ``kh*kw`` times bigger than ``x`` (9x for a 3x3), which
    at 4K is tens of GB.  This function therefore never materializes it: the patches
    are taken as a **strided view** (``Tensor.unfold`` twice, plus a strided slice for
    dilation) and only the ``max_rows`` sampled rows are copied out.  Peak extra memory
    is ``M * C*kh*kw`` floats plus one padded copy of ``x``.

    ``max_rows`` sampling is without replacement (``randperm``) when it would keep more
    than ~1/8 of the candidates, and with replacement (``randint``) otherwise -- a full
    ``randperm`` over the 4M output pixels of a 4K conv costs more than the sample
    itself, and at ``M=4096`` out of 4M the expected number of duplicate rows is < 0.2 %.
    """
    if x.dim() != 4:
        raise ValueError(f"expected [N, C, H, W], got {tuple(x.shape)}")
    if x.shape[1] != conv.in_channels:
        raise ValueError(f"x has {x.shape[1]} channels, conv wants {conv.in_channels}")
    kh, kw = _pair(conv.kernel_size)
    sh, sw = _pair(conv.stride)
    dh, dw = _pair(conv.dilation)
    pl, pr, pt, pb = _pad_lrtb(conv)

    mode = _PAD_MODE.get(conv.padding_mode, "constant")
    if (pl or pr or pt or pb):
        xp = F.pad(x, (pl, pr, pt, pb), mode=mode) if mode != "constant" \
            else F.pad(x, (pl, pr, pt, pb))
    else:
        xp = x
    eh, ew = dh * (kh - 1) + 1, dw * (kw - 1) + 1
    if xp.shape[2] < eh or xp.shape[3] < ew:
        raise ValueError(
            f"padded input {tuple(xp.shape)} smaller than the dilated kernel ({eh}, {ew})")

    v = xp.unfold(2, eh, sh).unfold(3, ew, sw)          # [N, C, Ho, Wo, eh, ew]  (view)
    if dh > 1:
        v = v[..., ::dh, :]
    if dw > 1:
        v = v[..., ::dw]                                # [N, C, Ho, Wo, kh, kw]
    n, c, ho, wo = v.shape[0], v.shape[1], v.shape[2], v.shape[3]
    total = n * ho * wo

    if max_rows is None or int(max_rows) >= total:
        # contiguous, ordered path: permute then reshape (one copy of the sampled rows)
        rows = v.permute(0, 2, 3, 1, 4, 5).reshape(total, c * kh * kw)
        return rows
    m = int(max_rows)
    if m * 8 >= total:
        idx = torch.randperm(total, generator=generator)[:m]
    else:
        idx = torch.randint(0, total, (m,), generator=generator)
    idx = idx.to(v.device)
    ni = idx // (ho * wo)
    rem = idx % (ho * wo)
    hi = rem // wo
    wi = rem % wo
    # advanced indices separated by a slice -> the sampled axis comes first: [M, C, kh, kw]
    picked = v[ni, :, hi, wi]
    return picked.reshape(m, c * kh * kw)


def _iter_convs(
    model: nn.Module,
    name_filter: Optional[Callable[[str, nn.Conv2d], bool]] = None,
):
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        if type(mod.weight).__name__ != "Parameter":
            continue  # already swapped for a torchao tensor subclass -> not ours
        if name_filter is not None and not name_filter(name, mod):
            continue
        yield name, mod


def conv_group_size_report(
    model: nn.Module,
    w_bits: int,
    group_size: Optional[int] = 128,
    *,
    name_filter: Optional[Callable[[str, nn.Conv2d], bool]] = None,
) -> Dict[str, Any]:
    """Which Conv2d would silently fall back to per-out-channel scales, and how much
    of the conv parameter mass that is.  Run this before committing to a group_size."""
    rows, fb_params, tot_params = [], 0, 0
    for name, conv in _iter_convs(model, name_filter):
        in_f = conv_flat_in_dim(conv)
        g = effective_group_size(in_f, w_bits, group_size)
        p = conv.weight.numel()
        tot_params += p
        # a fallback is a layer that *asked* for grouping and did not get it
        if (group_size or 0) > 0 and (int(w_bits) < 8 or W8_GROUPED) and in_f % int(group_size) != 0:
            fb_params += p
            rows.append({"name": name, "in_flat": in_f, "group": g, "params": p})
    return {
        "group_size": group_size,
        "w_bits": int(w_bits),
        "n_fallback": len(rows),
        "fallback_params": int(fb_params),
        "conv_weight_params": int(tot_params),
        "fallback_share": (fb_params / tot_params) if tot_params else 0.0,
        "fallback_layers": rows,
    }


# ===========================================================================
# 2. the zero-copy Linear proxy -- how every algorithm gets reused verbatim
# ===========================================================================
def _linear_proxy(w2d: torch.Tensor) -> nn.Linear:
    """Wrap a 2-D tensor in an ``nn.Linear`` **sharing its storage** (no copy).

    ``nn.Parameter`` goes through ``Tensor._make_subclass``, which aliases the given
    storage, and every routine in ``quantizers_weight`` writes results with
    ``mod.weight.data.copy_(...)`` -- so quantizing the proxy quantizes the conv weight
    in place.  ``_run_on_proxy`` still verifies the aliasing and copies back if a future
    implementation ever rebinds ``.weight`` instead of writing through it.
    """
    out_f, in_f = int(w2d.shape[0]), int(w2d.shape[1])
    lin = nn.Linear(1, 1, bias=False)
    lin.in_features, lin.out_features = in_f, out_f
    lin.weight = nn.Parameter(w2d, requires_grad=False)
    return lin


def _conv_blocks(conv: nn.Conv2d) -> List[Tuple[int, torch.Tensor, slice]]:
    """Split a conv into ``groups`` independent linear problems.

    Returns ``[(g_index, W2d, unfold_col_slice), ...]`` where ``W2d`` is a *view* of the
    group's weight rows flattened to ``[O/groups, (C/groups)*kh*kw]`` and
    ``unfold_col_slice`` selects that group's columns out of a full
    ``[N, C*kh*kw]`` im2col (contiguous, because unfold orders columns channel-major).
    For ``groups == 1`` this is a single block covering everything.
    """
    g = int(conv.groups)
    w = conv.weight.data
    o = w.shape[0]
    per_in = conv_flat_in_dim(conv, per_group=True)
    if g == 1:
        return [(0, w.reshape(o, -1), slice(0, per_in))]
    if o % g:
        raise ValueError(f"out_channels={o} not divisible by groups={g}")
    op = o // g
    return [(k, w[k * op:(k + 1) * op].reshape(op, -1), slice(k * per_in, (k + 1) * per_in))
            for k in range(g)]


def _run_on_proxy(w2d: torch.Tensor, fn: Callable[[nn.Linear], None]) -> Dict[str, Any]:
    """Apply a ``quantizers_weight`` routine to a 2-D weight view; return its metadata."""
    proxy = _linear_proxy(w2d)
    fn(proxy)
    out = proxy.weight.data
    if out.data_ptr() != w2d.data_ptr():  # defensive: routine rebound instead of copy_
        w2d.copy_(out)
    return dict(getattr(proxy, "_wq_meta", {}))


def _mark_conv(conv: nn.Conv2d, metas: List[Dict[str, Any]], **extra) -> None:
    m = dict(metas[0]) if metas else {}
    m.update(extra)
    if len(metas) > 1:
        m["n_groups"] = len(metas)
        m["per_group_meta"] = metas
    conv._wq_meta = m  # type: ignore[attr-defined]


def conv_quant_report(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Per-Conv2d metadata left behind by the ``*_quantize_conv_`` calls."""
    return {n: dict(getattr(m, "_wq_meta"))
            for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d) and hasattr(m, "_wq_meta")}


# ===========================================================================
# 3. RTN
# ===========================================================================
def rtn_quantize_conv_(
    model: nn.Module,
    w_bits: int,
    group_size: Optional[int] = 128,
    *,
    name_filter: Optional[Callable[[str, nn.Conv2d], bool]] = None,
    verbose: bool = False,
) -> None:
    """Round-to-nearest QDQ of every ``nn.Conv2d.weight``, in place.

    Identical convention to ``quantizers_weight.rtn_quantize_``, applied to
    ``W.reshape(O, -1)``: symmetric absmax, ``qmax = 2**(w_bits-1)-1``, groups of
    ``group_size`` **along the flattened ``(c, i, j)`` axis** (falling back to
    per-out-channel when ``group_size`` does not divide ``C*kh*kw`` -- see
    ``conv_group_size_report``; ``w_bits >= 8`` is grouped too while ``W8_GROUPED``).

    Grouped convs are quantized group by group, which is what the weight layout means
    anyway (row ``o`` of group ``k`` only sees that group's ``C/groups`` channels).
    """
    t0 = time.perf_counter()
    n = 0
    for _, conv in _iter_convs(model, name_filter):
        metas = [_run_on_proxy(w2d, lambda p: rtn_quantize_(p, w_bits, group_size))
                 for _, w2d, _ in _conv_blocks(conv)]
        _mark_conv(conv, metas, kind="conv2d", flat_in=conv_flat_in_dim(conv),
                   kernel=tuple(_pair(conv.kernel_size)), groups=int(conv.groups))
        n += 1
    if verbose:
        print(f"[quantizers_conv] rtn w{w_bits} over {n} Conv2d in {time.perf_counter()-t0:.1f}s")


# ===========================================================================
# 4. calibration
# ===========================================================================
def _call_forward_fn(run_forward_fn: Callable, model: nn.Module) -> None:
    import inspect

    try:
        n_params = len(inspect.signature(run_forward_fn).parameters)
    except (TypeError, ValueError):
        n_params = 1
    with torch.no_grad():
        run_forward_fn(model) if n_params >= 1 else run_forward_fn()


def collect_conv_layer_inputs(
    model: nn.Module,
    run_forward_fn: Callable,
    max_tokens: int = 4096,
    *,
    name_filter: Optional[Callable[[str, nn.Conv2d], bool]] = None,
    store_device: str = "cpu",
    store_dtype: torch.dtype = torch.float32,
    max_rows_per_call: Optional[int] = None,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """Hook every ``nn.Conv2d`` and collect its im2col rows as ``[N, C*kh*kw]``.

    Twin of ``quantizers_weight.collect_layer_inputs``; the only difference is that a
    conv "token" is one *receptive field patch* rather than one sequence position, so
    each forward contributes ``batch * Ho * Wo`` candidate rows and each row is
    ``kh*kw`` times wider than the feature map's channel count.

    Memory is the whole problem here.  One 4K SDXL forward through a 3x3 conv at
    C=1280 offers ~4.2e6 patches x 11520 columns = 190 GB if materialized.  Two guards,
    both applied **before** any copy happens (see ``unfold_conv_input``):

      * ``max_rows_per_call`` -- random patches per individual forward, so a long
        denoising trajectory contributes uniformly instead of the first step eating the
        whole budget.  **Always set this for SDXL** (256-1024 is a good range).
      * ``max_tokens``        -- hard per-layer cap; the hook stops recording after it.

    Storage cost is ``max_tokens * C*kh*kw * 4`` bytes per layer on ``store_device``
    (cpu/fp32 by default): 4096 x 11520 x 4 = 189 MB for one C=1280 3x3 conv, 377 MB at
    C=2560.  Budget accordingly, or calibrate a subset via ``name_filter``.

    Returns ``{layer_name: Tensor[N, C*kh*kw]}``; layers never executed are absent.
    For a grouped conv the rows carry **all** ``C*kh*kw`` columns -- ``gptq_quantize_conv_``
    slices the per-group column block out itself.
    """
    max_tokens = int(max_tokens)
    gen = torch.Generator().manual_seed(int(seed))
    bufs: Dict[str, List[torch.Tensor]] = {}
    counts: Dict[str, int] = {}
    handles = []

    def _make_hook(name: str):
        def hook(module, args):
            done = counts.get(name, 0)
            if done >= max_tokens:
                return None
            if not args:
                return None
            x = args[0]
            if not torch.is_tensor(x) or not x.is_floating_point() or x.dim() != 4:
                return None
            budget = max_tokens - done
            cap = budget if max_rows_per_call is None else min(budget, int(max_rows_per_call))
            rows = unfold_conv_input(x.detach(), module, max_rows=cap, generator=gen)
            if rows.shape[0] > budget:
                rows = rows[:budget]
            bufs.setdefault(name, []).append(rows.to(store_device, store_dtype))
            counts[name] = done + rows.shape[0]
            return None

        return hook

    for name, conv in _iter_convs(model, name_filter):
        handles.append(conv.register_forward_pre_hook(_make_hook(name)))
    try:
        _call_forward_fn(run_forward_fn, model)
    finally:
        for h in handles:
            h.remove()
    out = {k: torch.cat(v, dim=0) for k, v in bufs.items() if v}
    if verbose:
        tot = sum(v.numel() * v.element_size() for v in out.values())
        print(f"[quantizers_conv] collected {len(out)} Conv2d, "
              f"{sum(v.shape[0] for v in out.values())} rows, {tot/2**20:.0f} MiB")
    return out


def collect_conv_act_channel_absmax(
    model: nn.Module,
    run_forward_fn: Callable,
    *,
    name_filter: Optional[Callable[[str, nn.Conv2d], bool]] = None,
    store_device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Streaming per-**input-channel** ``|x|`` max for every Conv2d -- O(C) memory.

    Reduces over ``(N, H, W)``, i.e. one number per input channel, which is the
    granularity AWQ/SmoothQuant scaling actually has on a conv (see
    ``awq_quantize_conv_``).  Use this when the im2col rows would not fit.
    """
    stats: Dict[str, torch.Tensor] = {}
    handles = []

    def _make_hook(name: str):
        def hook(module, args):
            if not args:
                return None
            x = args[0]
            if not torch.is_tensor(x) or not x.is_floating_point() or x.dim() != 4:
                return None
            a = x.detach().float().abs().amax(dim=3).amax(dim=2).amax(dim=0).to(store_device)
            prev = stats.get(name)
            stats[name] = a if prev is None else torch.maximum(prev, a)
            return None

        return hook

    for name, conv in _iter_convs(model, name_filter):
        handles.append(conv.register_forward_pre_hook(_make_hook(name)))
    try:
        _call_forward_fn(run_forward_fn, model)
    finally:
        for h in handles:
            h.remove()
    return stats


def conv_act_channel_absmax_from_inputs(
    layer_inputs: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """Per-input-channel AWQ statistic ``mean|x|`` from already-collected im2col rows.

    Rows are ``[N, C*kh*kw]``; this reshapes to ``[N, C, kh*kw]`` and averages ``|x|``
    over both the sample axis and the ``kh*kw`` replicas, giving ``[C]``.  Averaging the
    replicas is the right move: the ``kh*kw`` columns of one channel are shifted copies
    of the same feature map, so they share a single activation magnitude up to boundary
    effects, and the mean is the lower-variance estimator of it.
    """
    convs = {n: m for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}
    out: Dict[str, torch.Tensor] = {}
    for name, rows in layer_inputs.items():
        conv = convs.get(name)
        if conv is None:
            continue
        kh, kw = _pair(conv.kernel_size)
        k = kh * kw
        c = rows.shape[1] // k
        out[name] = rows.float().abs().reshape(-1, c, k).mean(dim=(0, 2))
    return out


# ===========================================================================
# 5. GPTQ
# ===========================================================================
def gptq_quantize_conv_(
    model: nn.Module,
    layer_inputs: Dict[str, torch.Tensor],
    w_bits: int,
    group_size: Optional[int] = 128,
    percdamp: float = 0.01,
    blocksize: int = 128,
    *,
    act_order: bool = True,
    mse_scale_grid: int = 0,
    max_tokens: int = 4096,
    hessian_chunk: int = 1024,
    name_filter: Optional[Callable[[str, nn.Conv2d], bool]] = None,
    fallback_rtn: bool = True,
    verbose: bool = False,
) -> None:
    """GPTQ on the im2col view -- exactly ``quantizers_weight.gptq_quantize_`` on
    ``W.reshape(O, -1)`` with ``H = (2/N) X^T X`` over im2col rows.

    ``layer_inputs`` comes from ``collect_conv_layer_inputs`` and is keyed by the same
    module names; a grouped conv's rows are sliced per group before the Hessian is
    formed (group ``k`` owns the contiguous column block
    ``[k*(C/g)*kh*kw, (k+1)*(C/g)*kh*kw)``, because unfold orders columns channel-major).

    Two conv-specific facts worth knowing before reading the numbers:

    * **The Hessian is big.** ``in_flat = C*kh*kw``, so ``H`` is ``in_flat^2`` fp32:
      530 MB at C=1280 3x3 and 2.1 GB at C=2560 3x3, versus 100 MB for the widest SDXL
      Linear (5120).  Run these on GPU one layer at a time, or move the layer to CPU.
    * **The Hessian is structurally rank-deficient.** The ``kh*kw`` columns belonging to
      one channel are shifted copies of the same feature map and are therefore strongly
      correlated (near-collinear on smooth activations), on top of the usual
      ``N < in_flat`` deficiency.  Effective rank is closer to ``C`` than to ``C*kh*kw``.
      ``_cholesky_inv_upper``'s escalating damping absorbs it, but this is the same
      failure mode already seen for W2 GPTQ on Linear here (in_features=5120 with 2048
      calibration rows -> divergence), and conv makes the ratio ``kh*kw`` times worse.

    Measured on a real SDXL conv shape (C=320 3x3, ``in_flat=2880``, W3 g64, spatially
    correlated + channel-heteroscedastic inputs; held-out conv output MSE, RTN = 11.12)::

        N rows   N/in_flat    awq    gptq   gptq+mse(10)   gptq percdamp=0.05
          1024       0.36    5.76    8.05        5.64             5.96
          2880       1.00    5.75    5.95        4.02             4.94
          5760       2.00    5.74    5.33        3.69             4.62
         11520       4.00    5.76    4.94        3.52             4.51
         23040       8.00    5.76    4.75        3.47             4.40

      i.e. **GPTQ needs ``N >~ 2 * in_flat`` just to match AWQ**, and ``mse_scale_grid=10``
      is worth more than any amount of extra data.  AWQ is flat in ``N`` (it only needs a
      per-channel statistic) and is the safe default for the wide convs: at C=2560 3x3,
      matching ``N = 2 * in_flat`` would mean 46 k rows x 23040 columns = 4.2 GB of
      calibration for a single layer, plus a 2.1 GB Hessian.
    """
    t0 = time.perf_counter()
    n_gptq = n_rtn = 0
    for name, conv in _iter_convs(model, name_filter):
        X_full = layer_inputs.get(name)
        blocks = _conv_blocks(conv)
        in_full = conv_flat_in_dim(conv, per_group=False)
        if X_full is None or X_full.numel() == 0:
            if fallback_rtn:
                metas = [_run_on_proxy(w2d, lambda p: rtn_quantize_(p, w_bits, group_size))
                         for _, w2d, _ in blocks]
                _mark_conv(conv, metas, kind="conv2d", method="gptq->rtn(no inputs)")
                n_rtn += 1
            continue
        X_full = X_full.detach().reshape(-1, X_full.shape[-1])
        if X_full.shape[1] != in_full:
            raise ValueError(f"{name}: layer_inputs has {X_full.shape[1]} columns, "
                             f"conv im2col width is {in_full}")
        if verbose and X_full.shape[0] < conv_flat_in_dim(conv):
            print(f"[quantizers_conv] warn {name}: {X_full.shape[0]} rows for "
                  f"in_flat={conv_flat_in_dim(conv)} -> rank-deficient Hessian (damped)")
        metas = []
        for _, w2d, cols in blocks:
            Xg = X_full[:, cols]
            metas.append(_run_on_proxy(w2d, lambda p, _X=Xg: gptq_quantize_(
                p, {"": _X}, w_bits, group_size, percdamp, blocksize,
                act_order=act_order, mse_scale_grid=mse_scale_grid,
                max_tokens=max_tokens, hessian_chunk=hessian_chunk,
                fallback_rtn=fallback_rtn, verbose=False)))
        _mark_conv(conv, metas, kind="conv2d", flat_in=conv_flat_in_dim(conv),
                   kernel=tuple(_pair(conv.kernel_size)), groups=int(conv.groups))
        n_gptq += 1
    if verbose:
        print(f"[quantizers_conv] gptq w{w_bits} over {n_gptq} Conv2d "
              f"({n_rtn} rtn-fallback) in {time.perf_counter()-t0:.1f}s")


# ===========================================================================
# 6. AWQ
# ===========================================================================
SMOOTHQUANT_ON_CONV = """\
SmoothQuant on Conv2d: VALID, exactly, with s indexed by INPUT CHANNEL.

    y[o,u] = sum_{c,i,j} W[o,c,i,j] * x[c, u+i, u+j]

Put x'[c,.] = x[c,.] / s_c and W'[o,c,i,j] = W[o,c,i,j] * s_c (i.e.
`W *= s.view(1, -1, 1, 1)`); every term picks up s_c/s_c = 1, so conv2d(x', W') ==
conv2d(x, W) for any s > 0.  The bias is untouched.

Padding does not break it.  'zeros' pads with 0 and 0/s_c = 0; 'reflect'/'replicate'/
'circular' copy already-scaled values.  (A nonzero *constant* pad would break it -- and
that is exactly why an additive/affine transform is inadmissible here while a diagonal
multiplicative one is fine.)

What is NOT valid: a scale that varies within a channel's kh*kw block, i.e. per
flattened column (c,i,j).  There is no per-pixel-of-kernel factor you can apply to a
feature map, because each input pixel is read by kh*kw different kernel taps at
different output positions.  So a *runtime* transform must be block-constant.
(AWQ escapes this rule -- see awq_quantize_conv_ -- because it folds s back into W.)

Deployment bonus: SDXL conv inputs come out of GroupNorm(+SiLU), and 1/s_c can be
folded into GroupNorm's per-channel affine weight and bias, so the transform is free at
inference.  Fold into the *affine*, not before the normalization -- GroupNorm's
statistics are computed per channel-group and would otherwise change.
"""

ROTATION_ON_CONV = """\
Orthogonal rotation on Conv2d: PARTIALLY valid -- channel-axis only.

(1) Rotation over the full flattened axis (R of size C*kh*kw): NOT USABLE at runtime.
    The im2col rows are overlapping receptive fields; one input pixel appears in kh*kw
    different rows at different positions.  A dense R mixing the (c,i,j) coordinates is
    therefore not expressible as any operator on the feature map x -- you would have to
    materialize im2col(x), rotate it, and run an explicit GEMM.  That is exact (and it
    is a legitimate deployment choice if you already run conv as explicit im2col+GEMM),
    but it forfeits cuDNN's implicit-GEMM/Winograd kernels and blows memory up kh*kw x.
    For a training-free PTQ study on SDXL: do not.

(2) Rotation over the CHANNEL axis (R orthogonal, size C): EXACT and usable.
        x'[c',u] = sum_c R[c,c'] x[c,u]                (a 1x1 conv with weights R^T)
        W'[o,c',i,j] = sum_c W[o,c,i,j] R[c,c']        (W rotated per kernel tap)
    =>  sum_{c'} W'[o,c',i,j] x'[c',u]
          = sum_{c,d} W[o,c,i,j] (R R^T)[c,d] x[d,u] = sum_c W[o,c,i,j] x[c,u].
    Channel mixing and spatial convolution act on different axes and commute; both are
    linear, so the identity is exact.  Zero padding survives it (R^T 0 = 0), and so does
    circular padding; reflect/replicate also survive (they copy whole channel vectors).
    Cost: an extra C*C*H*W 1x1 conv, i.e. 1/(kh*kw) of the conv it protects for a 3x3.
    It canNOT be folded into the preceding GroupNorm -- normalization statistics are
    per channel-group and a cross-channel rotation changes them -- so unlike
    SmoothQuant it is a real runtime op.  It can be folded into a preceding 1x1 conv or
    Linear when there is one.
    Reach: it spreads outliers over C directions, not over C*kh*kw.  That is the right
    target anyway -- activation outliers in a UNet are channel-structured, not
    tap-structured -- but it means the incoherence bound is sqrt(2 ln C), not
    sqrt(2 ln (C*kh*kw)).

(3) Output-channel rotation: exact only if the consumer is rotated too; in a UNet the
    consumer is a GroupNorm or a residual add, so no.  Skip it.

Practical recommendation for the conv half of the PyraQuant bit-map:
  * W-only conv quantization (this file): needs no transform at all.  AWQ's scaling is
    folded back into W, so W4/W3 conv works with zero runtime change.
  * A-side conv quantization (per-token/per-pixel QDQ on the feature map): use
    SmoothQuant on the input channels first (free, folds into GroupNorm), then add the
    channel rotation from (2) only if the per-pixel absmax/rms is still far above
    sqrt(2 ln C).  Never try (1).
"""


def awq_quantize_conv_(
    model: nn.Module,
    act_channel_absmax: Dict[str, torch.Tensor],
    w_bits: int,
    group_size: Optional[int] = 128,
    grid: int = 20,
    *,
    layer_inputs: Optional[Dict[str, torch.Tensor]] = None,
    max_calib_tokens: int = 512,
    clip_grid: int = 0,
    channel_scope: str = "channel",
    name_filter: Optional[Callable[[str, nn.Conv2d], bool]] = None,
    fallback_rtn: bool = True,
    verbose: bool = False,
) -> None:
    """AWQ for Conv2d.  Reuses ``quantizers_weight.awq_quantize_`` on the im2col view.

    How the ``kh*kw`` replication of a channel scale is handled
    ----------------------------------------------------------
    AWQ searches a per-input-channel scale ``s`` and quantizes ``Q(W diag(s)) diag(s)^-1``.
    On a conv the *activation* only has ``C`` channels, but the flattened weight has
    ``C*kh*kw`` columns -- column ``c*kh*kw + i*kw + j`` is tap ``(i,j)`` reading channel
    ``c``.  So a per-channel scale is lifted to the flattened axis by

        ``s_flat = s.repeat_interleave(kh*kw)``     (block-constant within each channel)

    and passed straight into the existing Linear implementation, which then does the
    right thing everywhere: the alpha grid search, the optional clipping search and the
    fold-back ``W/s_flat`` are all elementwise along the flattened axis and preserve
    block-constancy.  Equivalently, in conv coordinates the update is
    ``W <- Q(W * s.view(1,-1,1,1)) / s.view(1,-1,1,1)``.

    ``channel_scope``
        ``"channel"`` (default, paper-faithful) -- block-constant scale as above; it is
            the only structure a *runtime* transform could have (see
            ``SMOOTHQUANT_ON_CONV``), so it stays interpretable as difficulty migration
            and could be folded into GroupNorm if you ever wanted to.
        ``"column"`` -- one free scale per flattened column ``(c,i,j)``, derived from
            ``layer_inputs`` (mean ``|x|`` per column).  This is admissible **only for
            AWQ**, and the reason is worth stating: AWQ never applies ``s`` at runtime,
            it folds it back into ``W`` analytically, so ``s`` is nothing but a
            reparameterization of the rounding grid and *any* positive vector is legal.
            SmoothQuant/rotation cannot do this because their ``s`` has to survive as an
            op on the feature map.  ``"column"`` has kh*kw times more freedom and scores
            slightly better on the calibration loss; it is off by default because it
            drops the "salient channel" interpretation.

    Args:
        act_channel_absmax: ``{conv_name: Tensor}`` of length ``C`` (from
            ``collect_conv_act_channel_absmax`` / ``conv_act_channel_absmax_from_inputs``)
            or already of length ``C*kh*kw``; both are accepted and the short form is
            repeat-interleaved.  Missing layers get RTN when ``fallback_rtn``.
        layer_inputs: optional im2col rows for the exact output-MSE selection criterion;
            without them AWQ falls back to the diagonal proxy loss, which needs only the
            channel statistic.
    """
    scope = str(channel_scope).lower()
    if scope not in ("channel", "column"):
        raise ValueError(f"channel_scope must be 'channel' or 'column', got {channel_scope!r}")
    t0 = time.perf_counter()
    n_awq = n_rtn = 0
    for name, conv in _iter_convs(model, name_filter):
        blocks = _conv_blocks(conv)
        kh, kw = _pair(conv.kernel_size)
        k = kh * kw
        in_full = conv_flat_in_dim(conv, per_group=False)
        X_full = None
        if layer_inputs is not None and name in layer_inputs:
            X_full = layer_inputs[name].detach().reshape(-1, layer_inputs[name].shape[-1])

        if scope == "column":
            if X_full is None:
                raise ValueError(f"{name}: channel_scope='column' needs layer_inputs")
            chan_flat = X_full.float().abs().mean(dim=0)
        else:
            chan = act_channel_absmax.get(name)
            if chan is None:
                if fallback_rtn:
                    metas = [_run_on_proxy(w2d, lambda p: rtn_quantize_(p, w_bits, group_size))
                             for _, w2d, _ in blocks]
                    _mark_conv(conv, metas, kind="conv2d", method="awq->rtn(no stats)")
                    n_rtn += 1
                continue
            chan = chan.detach().float().reshape(-1)
            if chan.numel() == conv.in_channels:
                chan_flat = chan.repeat_interleave(k)      # <-- the kh*kw lift
            elif chan.numel() == in_full:
                chan_flat = chan
            else:
                raise ValueError(f"{name}: act stat has {chan.numel()} entries, expected "
                                 f"{conv.in_channels} (per channel) or {in_full} (flattened)")

        metas = []
        for _, w2d, cols in blocks:
            cg = chan_flat[cols].to(w2d.device)
            kwargs: Dict[str, Any] = {}
            if X_full is not None:
                kwargs["layer_inputs"] = {"": X_full[:, cols]}
            metas.append(_run_on_proxy(w2d, lambda p, _c=cg, _k=kwargs: awq_quantize_(
                p, {"": _c}, w_bits, group_size, grid,
                max_calib_tokens=max_calib_tokens, clip_grid=clip_grid,
                fallback_rtn=fallback_rtn, verbose=False, **_k)))
        _mark_conv(conv, metas, kind="conv2d", flat_in=conv_flat_in_dim(conv),
                   kernel=(kh, kw), groups=int(conv.groups), channel_scope=scope)
        n_awq += 1
    if verbose:
        print(f"[quantizers_conv] awq w{w_bits} ({scope}) over {n_awq} Conv2d "
              f"({n_rtn} rtn-fallback) in {time.perf_counter()-t0:.1f}s")


# ===========================================================================
# 7. dispatcher
# ===========================================================================
def quantize_conv_weights_(
    model: nn.Module,
    method: str,
    w_bits: int,
    group_size: Optional[int] = 128,
    *,
    layer_inputs: Optional[Dict[str, torch.Tensor]] = None,
    act_channel_absmax: Optional[Dict[str, torch.Tensor]] = None,
    **kwargs,
) -> None:
    """``method in {'rtn','awq','gptq','none'}`` -- Conv2d twin of
    ``quantizers_weight.quantize_weights_``, same argument names, so a sweep config can
    drive both sides from one field.

    ``awq`` derives the per-channel statistic from ``layer_inputs`` when
    ``act_channel_absmax`` is not given.
    """
    m = str(method).lower()
    if m in ("none", "fp16", "off"):
        return
    if m == "rtn":
        rtn_quantize_conv_(model, w_bits, group_size, **kwargs)
    elif m == "awq":
        stats = act_channel_absmax
        if stats is None:
            if layer_inputs is None:
                raise ValueError("awq needs act_channel_absmax or layer_inputs")
            stats = conv_act_channel_absmax_from_inputs(layer_inputs, model)
        awq_quantize_conv_(model, stats, w_bits, group_size,
                           layer_inputs=layer_inputs, **kwargs)
    elif m == "gptq":
        if layer_inputs is None:
            raise ValueError("gptq needs layer_inputs")
        gptq_quantize_conv_(model, layer_inputs, w_bits, group_size, **kwargs)
    else:
        raise ValueError(f"unknown weight-quant method {method!r}")


# ===========================================================================
# 8. parameter accounting / true average bit-width
# ===========================================================================
def conv_param_share(model: nn.Module) -> Dict[str, Any]:
    """Split the parameter count into Conv2d / Linear / everything else.

    Returns (all counts are raw parameter numbers)::

        {"total": T,
         "conv":   {"n_modules", "weight", "bias", "params", "share"},
         "linear": {"n_modules", "weight", "bias", "params", "share"},
         "other":  {"params", "share"},                     # norms, embeddings, biases
         "quantizable_weight": conv.weight + linear.weight,
         "conv_share_of_quantizable": ...,
         "conv_kinds": {(kh,kw,groups): {"n","params"}}}     # 1x1 vs 3x3 breakdown

    ``other`` is what stays fp16 no matter what: it is dominated by norm affines and
    biases and is ~0.1 % of an SDXL UNet, but the average-bit formula should carry it
    explicitly rather than pretend it is zero.
    """
    total = sum(p.numel() for p in model.parameters())
    cw = cb = lw = lb = 0
    nc = nl = 0
    kinds: Dict[Any, Dict[str, int]] = {}
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            nc += 1
            cw += mod.weight.numel()
            cb += 0 if mod.bias is None else mod.bias.numel()
            key = (*_pair(mod.kernel_size), int(mod.groups))
            e = kinds.setdefault(key, {"n": 0, "params": 0})
            e["n"] += 1
            e["params"] += mod.weight.numel()
        elif isinstance(mod, nn.Linear):
            nl += 1
            lw += mod.weight.numel()
            lb += 0 if mod.bias is None else mod.bias.numel()
    conv_p, lin_p = cw + cb, lw + lb
    other = total - conv_p - lin_p
    qw = cw + lw
    return {
        "total": int(total),
        "conv": {"n_modules": nc, "weight": int(cw), "bias": int(cb),
                 "params": int(conv_p), "share": conv_p / total if total else 0.0},
        "linear": {"n_modules": nl, "weight": int(lw), "bias": int(lb),
                   "params": int(lin_p), "share": lin_p / total if total else 0.0},
        "other": {"params": int(other), "share": other / total if total else 0.0},
        "quantizable_weight": int(qw),
        "conv_share_of_quantizable": (cw / qw) if qw else 0.0,
        "conv_kinds": {f"k{a}x{b}_g{g}": v for (a, b, g), v in sorted(kinds.items())},
    }


def average_bits_from_counts(
    n_linear_w: int,
    n_conv_w: int,
    n_other: int = 0,
    *,
    b_linear: float = 16.0,
    b_conv: float = 16.0,
    b_other: float = 16.0,
    scale_bits_linear: float = 0.0,
    scale_bits_conv: float = 0.0,
) -> Dict[str, Any]:
    """True average weight bit-width from raw parameter counts.

        avg = (P_lin*b_lin + P_conv*b_conv + P_other*b_other) / (P_lin + P_conv + P_other)

    ``scale_bits_*`` is the *amortized* group-scale overhead ``16/group_size`` bits per
    weight (a group-64 W4 tensor really costs 4 + 16/64 = 4.25 bit); pass 0 to quote the
    headline number the way the literature does.  ``average_bits`` computes it for you.
    """
    p = float(n_linear_w) + float(n_conv_w) + float(n_other)
    if p <= 0:
        raise ValueError("no parameters")
    eff_l = float(b_linear) + float(scale_bits_linear)
    eff_c = float(b_conv) + float(scale_bits_conv)
    num = n_linear_w * eff_l + n_conv_w * eff_c + n_other * float(b_other)
    qw = float(n_linear_w) + float(n_conv_w)
    num_q = n_linear_w * eff_l + n_conv_w * eff_c
    return {
        "avg_bits": num / p,
        "avg_bits_quantizable_only": (num_q / qw) if qw else float("nan"),
        "b_linear_effective": eff_l,
        "b_conv_effective": eff_c,
        "params": {"linear_w": int(n_linear_w), "conv_w": int(n_conv_w),
                   "other": int(n_other), "total": int(p)},
        "formula": (f"({n_linear_w}*{eff_l:g} + {n_conv_w}*{eff_c:g} + "
                    f"{n_other}*{b_other:g}) / {int(p)} = {num/p:.4f} bit"),
    }


def average_bits(
    model: nn.Module,
    w_bits_linear: float = 16.0,
    w_bits_conv: float = 16.0,
    *,
    group_size: Optional[int] = None,
    include_scale_overhead: bool = False,
    share: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """``conv_param_share`` + ``average_bits_from_counts`` in one call.

    ``include_scale_overhead=True`` adds ``16/group_size`` bit per weight to whichever
    side is quantized below 16 bit (fp16 scales, one per group along the input axis).
    """
    sh = share or conv_param_share(model)
    ov = 0.0
    if include_scale_overhead and group_size:
        ov = 16.0 / float(group_size)
    out = average_bits_from_counts(
        sh["linear"]["weight"], sh["conv"]["weight"],
        sh["total"] - sh["linear"]["weight"] - sh["conv"]["weight"],
        b_linear=w_bits_linear, b_conv=w_bits_conv,
        scale_bits_linear=ov if w_bits_linear < 16 else 0.0,
        scale_bits_conv=ov if w_bits_conv < 16 else 0.0,
    )
    out["share"] = sh
    return out


# ===========================================================================
# CPU unit tests
# ===========================================================================
def _blurred_field(n, c, h, w, gen, chan_scale=None):
    """Spatially correlated, channel-heteroscedastic feature map.

    Real conv inputs are smooth in space (so the kh*kw taps of one channel are strongly
    correlated -> the near-singular Hessian GPTQ has to survive) and have a few
    high-magnitude channels (what AWQ exploits).  White noise would test neither.
    """
    x = torch.randn(n, c, h + 4, w + 4, generator=gen)
    ker = torch.ones(c, 1, 5, 5) / 25.0
    x = F.conv2d(x, ker, groups=c)
    x = x / x.std()
    if chan_scale is None:
        chan_scale = torch.exp(torch.randn(c, generator=gen) * 1.2)
    return x * chan_scale.view(1, -1, 1, 1), chan_scale


def _conv_mse(conv: nn.Conv2d, ref_w: torch.Tensor, x: torch.Tensor) -> float:
    with torch.no_grad():
        cur = conv.weight.data.clone()
        y = F.conv2d(x, cur, None, conv.stride, conv.padding, conv.dilation, conv.groups)
        r = F.conv2d(x, ref_w, None, conv.stride, conv.padding, conv.dilation, conv.groups)
        return float((y - r).pow(2).mean())


def _test_im2col_identity():
    """THE load-bearing test: flatten(W) @ im2col(x)^T must equal conv2d(x, W)."""
    print("\n--- test 1: im2col identity  flatten(W) @ unfold(x)^T == conv2d(x, W) ---")
    cases = [
        dict(cin=6, cout=5, k=3, s=1, p=1, d=1, g=1, hw=(9, 11)),
        dict(cin=4, cout=8, k=1, s=1, p=0, d=1, g=1, hw=(7, 7)),
        dict(cin=6, cout=4, k=3, s=2, p=1, d=1, g=1, hw=(10, 12)),
        dict(cin=4, cout=6, k=3, s=1, p=2, d=2, g=1, hw=(9, 9)),
        dict(cin=5, cout=7, k=(3, 1), s=1, p=(1, 0), d=1, g=1, hw=(8, 8)),
        dict(cin=6, cout=6, k=3, s=1, p=1, d=1, g=3, hw=(8, 8)),   # grouped
        dict(cin=4, cout=4, k=3, s=1, p=1, d=1, g=4, hw=(8, 8)),   # depthwise
    ]
    torch.manual_seed(0)
    worst32 = worst64 = 0.0
    for cs in cases:
        conv = nn.Conv2d(cs["cin"], cs["cout"], cs["k"], cs["s"], cs["p"],
                         cs["d"], cs["g"], bias=True).double()
        x = torch.randn(2, cs["cin"], *cs["hw"], dtype=torch.float64)
        with torch.no_grad():
            y_ref = conv(x)
        n, ho, wo = x.shape[0], *conv_output_hw(conv, *cs["hw"])
        assert (ho, wo) == tuple(y_ref.shape[2:]), f"conv_output_hw wrong: {(ho,wo)} vs {y_ref.shape}"

        rows = unfold_conv_input(x, conv)
        assert rows.shape == (n * ho * wo, cs["cin"] * _pair(cs["k"])[0] * _pair(cs["k"])[1]), rows.shape
        # (a) column order matches F.unfold exactly (bit-identical, not just close)
        if not isinstance(conv.padding, str):
            ref_rows = F.unfold(x, conv.kernel_size, conv.dilation, conv.padding,
                                conv.stride).transpose(1, 2).reshape(rows.shape)
            assert torch.equal(rows, ref_rows), "column order differs from F.unfold"
        # (b) the actual matmul identity, per conv group
        y = torch.zeros(n * ho * wo, cs["cout"], dtype=torch.float64)
        for gi, (_, w2d, cols) in enumerate(_conv_blocks(conv)):
            op = w2d.shape[0]
            y[:, gi * op:(gi + 1) * op] = rows[:, cols] @ w2d.t()
        y = y.reshape(n, ho, wo, cs["cout"]).permute(0, 3, 1, 2) + conv.bias.view(1, -1, 1, 1)
        d64 = float((y - y_ref).abs().max())
        # and again in the fp32 the pipeline actually runs in
        conv32, x32 = conv.float(), x.float()
        rows32 = unfold_conv_input(x32, conv32)
        y32 = torch.zeros(n * ho * wo, cs["cout"])
        for gi, (_, w2d, cols) in enumerate(_conv_blocks(conv32)):
            op = w2d.shape[0]
            y32[:, gi * op:(gi + 1) * op] = rows32[:, cols] @ w2d.t()
        y32 = y32.reshape(n, ho, wo, cs["cout"]).permute(0, 3, 1, 2) + conv32.bias.view(1, -1, 1, 1)
        with torch.no_grad():
            d32 = float((y32 - conv32(x32)).abs().max())
        worst64, worst32 = max(worst64, d64), max(worst32, d32)
        print(f"  cin={cs['cin']} cout={cs['cout']} k={cs['k']} s={cs['s']} p={cs['p']} "
              f"d={cs['d']} g={cs['g']}: rows{tuple(rows.shape)} "
              f"max|diff| fp64={d64:.2e} fp32={d32:.2e}")
        assert d64 < 1e-10 and d32 < 1e-4, f"im2col identity broken: {d64:.2e}/{d32:.2e}"
    # 'same' string padding
    conv = nn.Conv2d(3, 4, 3, padding="same", bias=False).double()
    x = torch.randn(1, 3, 6, 6, dtype=torch.float64)
    rows = unfold_conv_input(x, conv)
    y = (rows @ flatten_conv_weight(conv).t()).reshape(1, 6, 6, 4).permute(0, 3, 1, 2)
    with torch.no_grad():
        d = float((y - conv(x)).abs().max())
    print(f"  padding='same': max|diff| fp64={d:.2e}")
    assert d < 1e-10
    print(f"  ok (worst fp64 {worst64:.2e}, worst fp32 {worst32:.2e}, threshold 1e-4)")


def _test_rtn_grid_and_groups():
    print("\n--- test 2: RTN lands on the grid, group boundaries follow (c,i,j) ---")
    torch.manual_seed(0)
    cin, cout, k, bits = 8, 6, 3, 4
    conv = nn.Conv2d(cin, cout, k, padding=1, bias=True)
    with torch.no_grad():  # make each input channel live on its own magnitude decade
        for c in range(cin):
            conv.weight.data[:, c] *= 10.0 ** (c - 4)
    b_ref = conv.bias.data.clone()
    w_ref = conv.weight.data.clone()
    in_flat = conv_flat_in_dim(conv)
    assert in_flat == cin * k * k == 72

    # the zero-copy claim: the Linear proxy aliases the conv weight's storage, so the
    # reused quantizers_weight routines write straight into the conv (no 2x memory)
    prox = _linear_proxy(flatten_conv_weight(conv))
    assert prox.weight.data_ptr() == conv.weight.data_ptr(), "proxy is not zero-copy"
    assert (prox.in_features, prox.out_features) == (in_flat, cout)
    with torch.no_grad():
        prox.weight.data[0, 0] = 12345.0
    assert float(conv.weight.data[0, 0, 0, 0]) == 12345.0, "proxy write did not alias"
    with torch.no_grad():
        conv.weight.data.copy_(w_ref)

    # group_size = kh*kw = 9  ->  exactly one group per input channel
    rtn_quantize_conv_(conv, bits, 9)
    wq = conv.weight.data
    assert torch.equal(conv.bias.data, b_ref), "bias must not be touched"
    qmax = _qmax(bits)
    # (a) bit-exact agreement with the Linear-side reference on the flattened view
    ref = qdq_weight(w_ref.reshape(cout, -1), bits, 9).reshape(w_ref.shape)
    assert torch.equal(wq, ref), "conv RTN must equal qdq_weight on the flattened matrix"
    # (b) every value is an integer multiple of its own group's scale, |q| <= qmax
    wg = w_ref.reshape(cout, in_flat // 9, 9)
    qg = wq.reshape(cout, in_flat // 9, 9)
    scale = wg.abs().amax(-1, keepdim=True).clamp_min(_EPS) / qmax
    lev = qg / scale
    assert (lev - lev.round()).abs().max() < 1e-4, "values off the grid"
    assert lev.abs().max() <= qmax + 1e-6, f"level {lev.abs().max()} exceeds qmax={qmax}"
    n_lev = int(torch.unique(lev.round()).numel())
    # (c) the group absmax element round-trips exactly -> the group owns its own scale
    am = wg.abs().argmax(-1, keepdim=True)
    assert torch.allclose(torch.gather(qg, -1, am), torch.gather(wg, -1, am), rtol=1e-5, atol=1e-9)
    # (d) group k really is input channel k: per-channel relative error stays flat
    #     across 8 decades, while a single shared row scale annihilates the small ones
    per_row = qdq_weight(w_ref.reshape(cout, -1), bits, None).reshape(w_ref.shape)
    rel_g, rel_r = [], []
    for c in range(cin):
        e = w_ref[:, c].pow(2).mean()
        rel_g.append(float((wq[:, c] - w_ref[:, c]).pow(2).mean() / e))
        rel_r.append(float((per_row[:, c] - w_ref[:, c]).pow(2).mean() / e))
    print(f"  levels used={n_lev} (grid {2*qmax+1}); per-channel rel MSE, group=9 (=kh*kw):")
    print("    " + " ".join(f"{v:.1e}" for v in rel_g))
    print("    per-row scale for comparison:")
    print("    " + " ".join(f"{v:.1e}" for v in rel_r))
    assert max(rel_g) < 5e-3, f"group-wise rel err too high: {max(rel_g):.2e}"
    assert rel_r[0] > 0.99, "per-row scale should annihilate the 1e-4x channel"
    assert float(per_row[:, 0].abs().max()) == 0.0 and float(wq[:, 0].abs().max()) > 0

    # (e) the documented fallback rules on the flattened dim
    assert effective_group_size(72, 4, 128) == 72, "72 % 128 != 0 -> per-row"
    assert effective_group_size(2880, 4, 128) == 2880, "SDXL 3x3 C=320 falls back at g=128"
    assert effective_group_size(2880, 4, 64) == 64, "...but is fine at g=64"
    assert effective_group_size(11520, 4, 128) == 128
    assert effective_group_size(72, 8, 9) == (9 if W8_GROUPED else 72)
    rep = conv_group_size_report(nn.Sequential(nn.Conv2d(320, 320, 3), nn.Conv2d(640, 640, 3)), 4, 128)
    assert rep["n_fallback"] == 1 and rep["fallback_layers"][0]["in_flat"] == 2880, rep
    rep64 = conv_group_size_report(nn.Sequential(nn.Conv2d(320, 320, 3), nn.Conv2d(640, 640, 3)), 4, 64)
    assert rep64["n_fallback"] == 0, rep64
    print(f"  group_size=128 fallback report on [C320 3x3, C640 3x3]: "
          f"{rep['n_fallback']}/2 layers, {rep['fallback_share']:.0%} of conv params; "
          f"at 64: {rep64['n_fallback']}/2")

    # (f) grouped conv: each group quantized independently, storage aliasing works
    gc = nn.Conv2d(6, 6, 3, padding=1, groups=3, bias=False)
    w0 = gc.weight.data.clone()
    rtn_quantize_conv_(gc, 3, 9)
    assert not torch.equal(gc.weight.data, w0), "grouped conv untouched"
    ref_g = torch.cat([qdq_weight(w0[k * 2:(k + 1) * 2].reshape(2, -1), 3, 9)
                       for k in range(3)]).reshape(w0.shape)
    assert torch.equal(gc.weight.data, ref_g), "grouped conv must match per-group reference"
    print("  grouped conv (g=3) quantized per group, matches reference")
    print("  ok")


def _test_collect_rows():
    print("\n--- test 3: collect_conv_layer_inputs shape / count / ordering ---")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.SiLU(),
                        nn.Conv2d(8, 8, 3, stride=2, padding=1), nn.SiLU(),
                        nn.Conv2d(8, 3, 1)).eval()
    gen = torch.Generator().manual_seed(3)
    batches = [_blurred_field(2, 4, 12, 12, gen)[0] for _ in range(4)]

    def run(m):
        for b in batches:
            m(b)

    li = collect_conv_layer_inputs(net, run, max_tokens=100000)
    assert set(li) == {"0", "2", "4"}, sorted(li)
    # layer 0: 4 forwards x 2 imgs x 12x12 out = 1152 rows, 4*3*3 = 36 columns
    assert li["0"].shape == (4 * 2 * 12 * 12, 36), li["0"].shape
    # layer 2: stride 2 on 12x12 -> 6x6
    assert li["2"].shape == (4 * 2 * 6 * 6, 8 * 9), li["2"].shape
    # layer 4: 1x1 conv on 6x6 -> columns == channels
    assert li["4"].shape == (4 * 2 * 6 * 6, 8), li["4"].shape
    for v in li.values():
        assert v.dtype == torch.float32 and v.device.type == "cpu"
    # rows are in unfold order and verbatim: first 288 rows == unfold of batch 0
    assert torch.equal(li["0"][:2 * 12 * 12], unfold_conv_input(batches[0], net[0])), \
        "collected rows must be the verbatim im2col of the first forward"
    print(f"  full collect: {[tuple(li[k].shape) for k in ('0','2','4')]} (rows = calls*N*Ho*Wo)")

    # per-call subsample spreads the budget over the trajectory
    spread = collect_conv_layer_inputs(net, run, max_tokens=100000, max_rows_per_call=17, seed=1)
    assert all(v.shape[0] == 4 * 17 for v in spread.values()), \
        {k: tuple(v.shape) for k, v in spread.items()}
    # hard cap stops mid-trajectory (200 < 288, the smallest layer's total supply)
    small = collect_conv_layer_inputs(net, run, max_tokens=200)
    assert all(v.shape[0] == 200 for v in small.values()), \
        {k: tuple(v.shape) for k, v in small.items()}
    # subsampled rows are still genuine patches (each equals some row of the full im2col)
    full0 = unfold_conv_input(batches[0], net[0])
    s1 = collect_conv_layer_inputs(net, lambda m: m(batches[0]), max_tokens=8, seed=5)["0"]
    d = torch.cdist(s1, full0).min(dim=1).values.max()
    assert s1.shape == (8, 36) and float(d) < 1e-6, f"sampled rows are not real patches ({d})"
    print(f"  max_rows_per_call=17 -> {tuple(spread['0'].shape)}; max_tokens=200 -> "
          f"{tuple(small['0'].shape)}; sampled rows match real patches (max dist {float(d):.1e})")

    # streaming channel stat + the from-rows variant, both length C
    st = collect_conv_act_channel_absmax(net, run)
    assert st["0"].shape == (4,) and st["2"].shape == (8,), {k: v.shape for k, v in st.items()}
    st2 = conv_act_channel_absmax_from_inputs(li, net)
    assert st2["0"].shape == (4,) and st2["2"].shape == (8,)
    ratio = (st["0"] / st2["0"]).tolist()
    print(f"  channel stats: absmax {st['0'].tolist()[:2]}... mean|x| {st2['0'].tolist()[:2]}... "
          f"(absmax/mean ratio {[round(r,2) for r in ratio]})")
    print("  ok")


def _test_beats_rtn():
    print("\n--- test 4: AWQ / GPTQ beat RTN on Conv2d (held-out output MSE) ---")
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(11)
    cin, cout = 16, 24
    conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
    with torch.no_grad():  # a few big weights, like a real kernel
        w = torch.randn(cout, cin, 3, 3, generator=gen) * 0.05
        w.view(-1)[torch.randperm(w.numel(), generator=gen)[: w.numel() // 100]] *= 6.0
        conv.weight.copy_(w)
    ref_w = conv.weight.data.clone()
    xc, cs = _blurred_field(6, cin, 16, 16, gen)          # calib: 6*256 = 1536 patches
    xt, _ = _blurred_field(4, cin, 16, 16, gen, cs)       # held out
    li = {"": unfold_conv_input(xc, conv)}
    # the bare conv is its own model here, so the module name is "" -- same key convention
    # quantizers_weight's own tests use
    stats = conv_act_channel_absmax_from_inputs(li, conv)
    assert stats[""].shape == (cin,), stats[""].shape
    print(f"  calib rows {tuple(li[''].shape)} (in_flat={conv_flat_in_dim(conv)}), "
          f"channel absmax spread {float(stats[''].max()/stats[''].min()):.1f}x")
    import copy as _copy
    ok = True
    cols = ("rtn", "awq", "awq-col", "gptq", "gptq+mse")
    print(f"  {'bits':>5} " + " ".join(f"{c:>10}" for c in cols))
    for bits in (2, 3, 4):
        row = []
        for meth in cols:
            m = _copy.deepcopy(conv)
            if meth == "rtn":
                rtn_quantize_conv_(m, bits, 72)
            elif meth == "awq":
                awq_quantize_conv_(m, stats, bits, 72, layer_inputs=li, max_calib_tokens=512)
            elif meth == "awq-col":
                awq_quantize_conv_(m, stats, bits, 72, layer_inputs=li,
                                   max_calib_tokens=512, channel_scope="column")
            elif meth == "gptq":
                gptq_quantize_conv_(m, li, bits, 72)
            else:
                gptq_quantize_conv_(m, li, bits, 72, mse_scale_grid=10)
            row.append(_conv_mse(m, ref_w, xt))
        print(f"  w{bits:<4} " + " ".join(f"{v:10.6f}" for v in row) + "   " +
              "  ".join(f"{c} {v/row[0]:.3f}x" for c, v in zip(cols[1:], row[1:])))
        for c, v in zip(cols[1:], row[1:]):
            if not v < row[0]:
                print(f"    FAIL: {c} w{bits} MSE {v:.6f} not below RTN {row[0]:.6f}")
                ok = False
    assert ok, "AWQ/GPTQ must beat RTN on conv at equal w_bits"
    ma = _copy.deepcopy(conv)
    awq_quantize_conv_(ma, stats, 3, 72, layer_inputs=li)
    a = conv_quant_report(ma)[""]
    mg = _copy.deepcopy(conv)
    gptq_quantize_conv_(mg, li, 3, 72)
    g = conv_quant_report(mg)[""]
    print(f"  awq  meta: alpha={a['alpha']:.2f} group={a['group']} scope={a['channel_scope']} "
          f"flat_in={a['flat_in']} kernel={a['kernel']}")
    print(f"  gptq meta: group={g['group']} act_order={g['act_order']} "
          f"calib_tokens={g['calib_tokens']} dead_cols={g['dead_cols']}")
    print("  ok")


def _test_end_to_end_and_robustness():
    print("\n--- test 5: end-to-end net, dispatcher, degenerate inputs ---")
    import copy as _copy
    torch.manual_seed(4)
    net = nn.Sequential(nn.Conv2d(6, 12, 3, padding=1), nn.SiLU(),
                        nn.Conv2d(12, 12, 3, padding=1), nn.SiLU(),
                        nn.Conv2d(12, 6, 1)).eval()
    gen = torch.Generator().manual_seed(6)
    calib = [_blurred_field(3, 6, 14, 14, gen)[0] for _ in range(4)]
    test_x = _blurred_field(2, 6, 14, 14, gen)[0]

    def run(m):
        for b in calib:
            m(b)

    li = collect_conv_layer_inputs(net, run, max_tokens=4096, max_rows_per_call=512, seed=2)
    with torch.no_grad():
        ref = net(test_x)
    out = {}
    for meth in ("rtn", "awq", "gptq"):
        m = _copy.deepcopy(net)
        quantize_conv_weights_(m, meth, 3, 108, layer_inputs=li)
        with torch.no_grad():
            out[meth] = float((m(test_x) - ref).pow(2).mean())
    print(f"  3-conv net w3 g108 end-to-end MSE: rtn {out['rtn']:.6f} | "
          f"awq {out['awq']:.6f} ({out['awq']/out['rtn']:.3f}x) | "
          f"gptq {out['gptq']:.6f} ({out['gptq']/out['rtn']:.3f}x)")
    assert out["awq"] < out["rtn"] and out["gptq"] < out["rtn"]
    rep = conv_quant_report(m)
    assert set(rep) == {"0", "2", "4"} and rep["0"]["method"] == "gptq"
    print(f"  conv_quant_report()['0']: method={rep['0']['method']} w_bits={rep['0']['w_bits']} "
          f"group={rep['0']['group']} flat_in={rep['0']['flat_in']} kernel={rep['0']['kernel']} "
          f"calib_tokens={rep['0']['calib_tokens']} proxy_err={rep['0']['proxy_err']:.3e}")

    # dispatcher: none is a no-op; awq without stats derives them from layer_inputs
    m0 = _copy.deepcopy(net)
    quantize_conv_weights_(m0, "none", 4)
    assert torch.equal(m0[0].weight.data, net[0].weight.data)
    m1 = _copy.deepcopy(net)
    quantize_conv_weights_(m1, "awq", 4, 108, layer_inputs=li)
    assert conv_quant_report(m1)["0"]["method"] == "awq"

    # degenerate: dead channel, collinear columns, zero weight, fp16, missing calib
    x = torch.randn(64, 6, 8, 8)
    x[:, 2] = 0.0
    x[:, 3] = x[:, 4]
    c = nn.Conv2d(6, 4, 3, padding=1, bias=False)
    rows = {"": unfold_conv_input(x, c)}
    m = _copy.deepcopy(c)
    gptq_quantize_conv_(m, rows, 4, 54)
    assert torch.isfinite(m.weight.data).all()
    assert float(m.weight.data[:, 2].abs().sum()) == 0.0, "dead channel must be zeroed"
    z = nn.Conv2d(6, 4, 3, padding=1, bias=False)
    with torch.no_grad():
        z.weight.zero_()
    for meth, kw in (("rtn", {}), ("gptq", {"layer_inputs": rows}), ("awq", {"layer_inputs": rows})):
        zz = _copy.deepcopy(z)
        quantize_conv_weights_(zz, meth, 4, 54, **kw)
        assert torch.isfinite(zz.weight.data).all() and float(zz.weight.data.abs().max()) == 0.0
    h32 = nn.Conv2d(8, 4, 3, padding=1, bias=False)      # NB: Module.float() is in place
    hr = {"": unfold_conv_input(torch.randn(32, 8, 8, 8), h32)}
    h = _copy.deepcopy(h32).to(torch.float16)
    # im2col of an fp16 feature map by an fp16 conv (the real SDXL storage dtype)
    h16_rows = unfold_conv_input(torch.randn(4, 8, 8, 8).half(), h)
    assert h16_rows.dtype == torch.float16 and h16_rows.shape == (4 * 64, 72)
    for meth, kw in (("rtn", {}), ("gptq", {"layer_inputs": hr}), ("awq", {"layer_inputs": hr})):
        hh = _copy.deepcopy(h)
        quantize_conv_weights_(hh, meth, 4, 72, **kw)
        assert hh.weight.dtype == torch.float16 and torch.isfinite(hh.weight.data).all()
    miss = nn.Conv2d(6, 4, 3, bias=False)
    gptq_quantize_conv_(miss, {}, 4, 54)
    assert "rtn" in conv_quant_report(miss)[""]["method"]
    awq_quantize_conv_(miss, {}, 4, 54)
    assert "rtn" in conv_quant_report(miss)[""]["method"]
    print("  dispatcher/none/fp16/dead-channel/zero-weight/missing-calib all handled")
    print("  ok")


def _test_param_accounting():
    print("\n--- test 6: conv_param_share + true average bit-width ---")
    net = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.GroupNorm(4, 8),
                        nn.Conv2d(8, 8, 1), nn.Flatten(), nn.Linear(128, 64), nn.Linear(64, 8))
    sh = conv_param_share(net)
    exp_cw = 8 * 4 * 9 + 8 * 8 * 1
    exp_lw = 128 * 64 + 64 * 8
    assert sh["conv"]["weight"] == exp_cw and sh["linear"]["weight"] == exp_lw, sh
    assert sh["conv"]["bias"] == 16 and sh["linear"]["bias"] == 72
    assert sh["other"]["params"] == 16, sh["other"]  # GroupNorm affine
    assert sh["total"] == sum(p.numel() for p in net.parameters())
    print(f"  toy net: conv w={sh['conv']['weight']} lin w={sh['linear']['weight']} "
          f"other={sh['other']['params']} total={sh['total']} "
          f"conv share of quantizable={sh['conv_share_of_quantizable']:.3f}")
    print(f"  conv kinds: {sh['conv_kinds']}")

    # reproduce the measured SDXL UNet accounting
    LIN, CONV, TOT = 2_234_000_000, 333_000_000, 2_567_000_000
    a = average_bits_from_counts(LIN, CONV, TOT - LIN - CONV, b_linear=4, b_conv=16)
    print(f"  SDXL, Linear-only W4:  {a['formula']}")
    assert abs(a["avg_bits"] - 5.5566) < 2e-3, a["avg_bits"]
    b = average_bits_from_counts(LIN, CONV, TOT - LIN - CONV, b_linear=4, b_conv=4)
    print(f"  SDXL, Linear+Conv W4:  {b['formula']}")
    assert abs(b["avg_bits"] - 4.0) < 1e-6
    for bl, bc in ((4, 8), (3, 4), (2, 4), (4, 4)):
        r = average_bits_from_counts(LIN, CONV, TOT - LIN - CONV, b_linear=bl, b_conv=bc)
        r2 = average_bits_from_counts(LIN, CONV, TOT - LIN - CONV, b_linear=bl, b_conv=bc,
                                      scale_bits_linear=16 / 64, scale_bits_conv=16 / 64)
        print(f"  W{bl}(lin)/W{bc}(conv): {r['avg_bits']:.4f} bit  "
              f"({r2['avg_bits']:.4f} incl. fp16 group-64 scales)")
    m = average_bits(net, 4, 4, group_size=64, include_scale_overhead=True)
    assert m["b_conv_effective"] == 4.25 and m["b_linear_effective"] == 4.25
    print(f"  toy net W4/W4 g64 with scale overhead: {m['avg_bits']:.4f} bit "
          f"(quantizable only {m['avg_bits_quantizable_only']:.4f})")
    print("  ok")


def _test_notes_are_true():
    """The SmoothQuant / rotation claims in the module notes, checked numerically."""
    print("\n--- test 7: the conv transform claims (SmoothQuant exact, channel rotation exact) ---")
    torch.manual_seed(8)
    conv = nn.Conv2d(8, 6, 3, stride=2, padding=1, bias=True).double()
    x = torch.randn(2, 8, 11, 11, dtype=torch.float64) * torch.exp(
        torch.randn(8, dtype=torch.float64) * 1.5).view(1, -1, 1, 1)
    with torch.no_grad():
        y0 = conv(x)

    # (1) per-input-channel SmoothQuant scaling: exact, through zero padding
    s = torch.exp(torch.randn(8, dtype=torch.float64) * 0.8)
    c1 = nn.Conv2d(8, 6, 3, stride=2, padding=1, bias=True).double()
    with torch.no_grad():
        c1.weight.copy_(conv.weight * s.view(1, -1, 1, 1))
        c1.bias.copy_(conv.bias)
        d_sq = float((c1(x / s.view(1, -1, 1, 1)) - y0).abs().max())
    print(f"  SmoothQuant  (diag s per in-channel, zeros padding): max|dy|={d_sq:.2e}")
    assert d_sq < 1e-12, d_sq
    for mode in ("reflect", "replicate", "circular"):
        cm = nn.Conv2d(8, 6, 3, stride=2, padding=1, bias=True, padding_mode=mode).double()
        with torch.no_grad():
            cm.weight.copy_(conv.weight)
            cm.bias.copy_(conv.bias)
            yr = cm(x).clone()
            cm.weight.copy_(conv.weight * s.view(1, -1, 1, 1))
            d = float((cm(x / s.view(1, -1, 1, 1)) - yr).abs().max())
        assert d < 1e-12, (mode, d)
    print("  SmoothQuant  survives reflect/replicate/circular padding too (all < 1e-12)")

    # (2) channel-axis rotation: exact (x @ R along channels, W @ R per tap)
    R, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
    xr = torch.einsum("nchw,cd->ndhw", x, R)
    c2 = nn.Conv2d(8, 6, 3, stride=2, padding=1, bias=True).double()
    with torch.no_grad():
        c2.weight.copy_(torch.einsum("ocij,cd->odij", conv.weight, R))
        c2.bias.copy_(conv.bias)
        d_rot = float((c2(xr) - y0).abs().max())
    print(f"  Rotation(C)  (orthogonal over in-channels, zeros padding): max|dy|={d_rot:.2e}")
    assert d_rot < 1e-12, d_rot

    # (3) rotation over the full flattened axis: exact ONLY as explicit im2col+GEMM,
    #     and there is no feature-map operator that realizes it
    Rf, _ = torch.linalg.qr(torch.randn(72, 72, dtype=torch.float64))
    rows = unfold_conv_input(x, conv)
    ho, wo = conv_output_hw(conv, 11, 11)
    y_gemm = ((rows @ Rf) @ (flatten_conv_weight(conv) @ Rf).t())
    y_gemm = y_gemm.reshape(2, ho, wo, 6).permute(0, 3, 1, 2) + conv.bias.view(1, -1, 1, 1)
    d_flat = float((y_gemm - y0).abs().max())
    print(f"  Rotation(C*kh*kw) as explicit im2col+GEMM: max|dy|={d_flat:.2e} (exact, but "
          f"needs the {rows.shape[1]}-wide im2col materialized -> {rows.shape[1]//8}x memory)")
    assert d_flat < 1e-12
    # proof by counterexample that no per-pixel channel op can realize Rf: after the
    # flattened rotation the 9 taps of one channel carry different scales, and a feature
    # map has one value per (c, pixel) that every tap must share.
    off = Rf - torch.block_diag(*[Rf[a * 9:(a + 1) * 9, a * 9:(a + 1) * 9] for a in range(8)])
    print(f"  ...and Rf mixes taps within a channel (off-block-diag energy "
          f"{float(off.pow(2).sum()/Rf.pow(2).sum()):.2f} of total) -> not a feature-map op")
    print("  ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(8)
    t0 = time.perf_counter()
    _test_im2col_identity()
    _test_rtn_grid_and_groups()
    _test_collect_rows()
    _test_beats_rtn()
    _test_end_to_end_and_robustness()
    _test_param_accounting()
    _test_notes_are_true()
    print(f"\nALL TESTS PASSED in {time.perf_counter()-t0:.1f}s")
