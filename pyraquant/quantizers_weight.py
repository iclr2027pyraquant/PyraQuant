"""Weight-side quantizers for PyraQuant: RTN, nested (W3 from W4 codes, W4 from W8 codes),
AWQ and GPTQ.

All of them write **fake-quantized (QDQ) fp weights back in place** into the model's
existing ``nn.Linear.weight`` tensors, so a model prepared here stays a plain fp16/fp32
module and drops straight into the simulated-variant plumbing
(``pyraquant.sim_recipes.build_recipe_variant`` / ``PathDispatchUNet``).
Nothing in this file imports from or mutates the rest of the package.

One scale convention is shared by every method so they are comparable:

    * symmetric absmax, ``qmax = 2**(w_bits-1) - 1``   (W2 -> ternary {-1,0,+1})
    * group-``group_size`` scales along in_features; this includes ``w_bits >= 8``
      while ``W8_GROUPED`` is True (nested W4 must share the groups of its W8 codes)
    * ``in_features % group_size != 0`` -> falls back to per-row

Note for SDXL: ``in_features`` takes values {320, 640, 1280, 2048, 2560, 2816, 5120}.
``group_size=128`` makes the 320-wide layers fall back to per-row (much worse at W2/W3);
``group_size=64`` and ``32`` divide **every** SDXL width.

Only ``torch`` and the standard library are imported.  Everything runs on CPU or GPU;
calibration tensors are kept on CPU by default and streamed to the layer's device.

References
----------
AWQ : Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and
      Acceleration", 2023.
GPTQ: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative
      Pre-trained Transformers", 2022.
"""
from __future__ import annotations

import inspect
import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "effective_group_size",
    "qdq_weight",
    "rtn_quantize_",
    "nested3_qdq_weight",
    "nested3_quantize_",
    "nested3_from_scale",
    "nested4_qdq_weight",
    "nested4_quantize_",
    "W8_GROUPED",
    "awq_quantize_",
    "gptq_quantize_",
    "collect_layer_inputs",
    "collect_act_channel_absmax",
    "act_channel_absmax_from_inputs",
    "quantize_weights_",
    "quant_report",
]

_EPS = 1e-8

# 8-bit weights use the same group-wise scales as the lower bit-widths (instead of one
# scale per output row).  Required by the nested W4 variant, whose 4-bit codes are
# derived from the 8-bit codes and therefore share their groups.
W8_GROUPED = True


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def _qmax(w_bits: int) -> int:
    q = (1 << (int(w_bits) - 1)) - 1
    if q < 1:
        raise ValueError(f"w_bits={w_bits} leaves no symmetric levels (need >= 2)")
    return q


def effective_group_size(in_features: int, w_bits: int, group_size: Optional[int]) -> int:
    """Group length actually used along in_features.

    Returns ``in_features`` (i.e. per-out-row scales) when ``group_size`` is None/<=0,
    when ``in_features`` is not divisible by ``group_size``, or when ``w_bits >= 8``
    and ``W8_GROUPED`` is False.
    """
    in_features = int(in_features)
    if group_size is None or int(group_size) <= 0:
        return in_features
    if int(w_bits) >= 8 and not W8_GROUPED:
        return in_features
    g = int(group_size)
    if in_features % g != 0:
        return in_features
    return min(g, in_features)


def qdq_weight(w: torch.Tensor, w_bits: int, group_size: Optional[int] = 128) -> torch.Tensor:
    """Symmetric absmax quantize-dequantize of a ``[out, in]`` weight matrix.

    Pure function (returns a new tensor, fp32 math, output dtype == input dtype).
    """
    if w.dim() != 2:
        raise ValueError(f"expected [out_features, in_features], got {tuple(w.shape)}")
    out_f, in_f = w.shape
    qmax = _qmax(w_bits)
    g = effective_group_size(in_f, w_bits, group_size)
    wg = w.detach().float().reshape(out_f, in_f // g, g)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp_min(_EPS) / qmax
    wq = (wg / scale).round().clamp_(-qmax, qmax) * scale
    return wq.reshape(out_f, in_f).to(w.dtype)


def _iter_linears(model: nn.Module, name_filter: Optional[Callable[[str, nn.Linear], bool]]):
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if type(mod.weight).__name__ != "Parameter":
            # already replaced by a torchao quantized tensor subclass -> not ours
            continue
        if name_filter is not None and not name_filter(name, mod):
            continue
        yield name, mod


def _mark(mod: nn.Linear, **meta) -> None:
    mod._wq_meta = dict(meta)  # type: ignore[attr-defined]


def quant_report(model: nn.Module) -> Dict[str, Dict]:
    """Per-layer metadata left behind by the ``*_quantize_`` calls."""
    return {n: dict(getattr(m, "_wq_meta")) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and hasattr(m, "_wq_meta")}


# ---------------------------------------------------------------------------
# (C) RTN baseline
# ---------------------------------------------------------------------------
def rtn_quantize_(
    model: nn.Module,
    w_bits: int,
    group_size: Optional[int] = 128,
    *,
    name_filter: Optional[Callable[[str, nn.Linear], bool]] = None,
    verbose: bool = False,
) -> None:
    """Round-to-nearest baseline: in-place QDQ of every ``nn.Linear.weight``
    (``qdq_weight`` convention; the control arm for AWQ / GPTQ).
    """
    t0 = time.perf_counter()
    n = 0
    for name, mod in _iter_linears(model, name_filter):
        w = mod.weight.data
        g = effective_group_size(w.shape[1], w_bits, group_size)
        mod.weight.data.copy_(qdq_weight(w, w_bits, group_size))
        _mark(mod, method="rtn", w_bits=int(w_bits), group=int(g))
        n += 1
    if verbose:
        print(f"[quantizers_weight] rtn w{w_bits} over {n} Linear in {time.perf_counter()-t0:.1f}s")


def nested3_qdq_weight(w: torch.Tensor, group_size: Optional[int] = 128) -> torch.Tensor:
    """Nested W3: only the W4 integer codes q4 (symmetric absmax, qmax=7) and their group
    scales s4 are stored; the W3 values are derived from q4 by an element-wise lookup,
    without storing a second weight tensor:
        |q4|: 0 -> 0, {1,2} -> 1, {3,4} -> 2, {5,6,7} -> 3   (3-bit code c = sign(q4) * level; 7 levels, exact zero)
        dequantize: W3 = T[c] * s4,  T = {0, +-1.5, +-3.5, +-6}  (bin midpoints, fixed constant table)
    On synthetic weights its MSE is about 0.7x that of an independent RTN-W3 (absmax/3 grid).
    """
    if w.dim() != 2:
        raise ValueError(f"expected [out_features, in_features], got {tuple(w.shape)}")
    out_f, in_f = w.shape
    qmax4 = _qmax(4)
    g = effective_group_size(in_f, 4, group_size)      # same grouping as the W4 variant
    wg = w.detach().float().reshape(out_f, in_f // g, g)
    s4 = wg.abs().amax(dim=-1, keepdim=True).clamp_min(_EPS) / qmax4
    q4 = (wg / s4).round().clamp_(-qmax4, qmax4)          # the stored 4-bit codes
    a = q4.abs()
    c = torch.where(a == 0, torch.zeros_like(a), (a + 1).div(2, rounding_mode="floor")).clamp_(max=3) * q4.sign()
    table = torch.tensor([-6.0, -3.5, -1.5, 0.0, 1.5, 3.5, 6.0], device=w.device)
    return (table[(c + 3).long()] * s4).reshape(out_f, in_f).to(w.dtype)


def nested4_qdq_weight(w: torch.Tensor, group_size: Optional[int] = 128) -> torch.Tensor:
    """Nested W4 derived from the W8 codes: only the W8 integer codes q8 (symmetric absmax,
    qmax=127) and their group scales s8 are stored.  The W4 code is
    c = round(q8 * 7/127) in [-7, 7] and dequantizes as W4 = c * (127/7) * s8 = c * s4,
    i.e. the RTN-W4 grid (absmax/7; zero and absmax exact) obtained by re-rounding the
    W8 codes, without storing a second weight tensor.  Its MSE ratio to an independent
    RTN-W4 is ~1.0 (the double-rounding error is negligible).
    """
    if w.dim() != 2:
        raise ValueError(f"expected [out_features, in_features], got {tuple(w.shape)}")
    out_f, in_f = w.shape
    qmax8, qmax4 = _qmax(8), _qmax(4)
    g = effective_group_size(in_f, 8, group_size)      # same grouping as the W8 variant
    wg = w.detach().float().reshape(out_f, in_f // g, g)
    s8 = wg.abs().amax(dim=-1, keepdim=True).clamp_min(_EPS) / qmax8
    q8 = (wg / s8).round().clamp_(-qmax8, qmax8)          # the stored 8-bit codes
    c = (q8 * (qmax4 / qmax8)).round().clamp_(-qmax4, qmax4)   # the 4-bit codes derived on the fly
    return (c * (qmax8 / qmax4) * s8).reshape(out_f, in_f).to(w.dtype)

def nested4_quantize_(
    model: nn.Module,
    w_bits: int = 4,
    group_size: Optional[int] = 128,
    *,
    name_filter: Optional[Callable[[str, nn.Linear], bool]] = None,
    verbose: bool = False,
) -> None:
    """Nested W4 variant: every Linear weight is replaced by the 4-bit values derived from
    its W8 codes (see ``nested4_qdq_weight``)."""
    if int(w_bits) != 4:
        raise ValueError(f"nested4 is only defined for w_bits=4 (W4 from W8 codes), got {w_bits}")
    t0 = time.perf_counter()
    n = 0
    for name, mod in _iter_linears(model, name_filter):
        w = mod.weight.data
        g = effective_group_size(w.shape[1], 8, group_size)
        mod.weight.data.copy_(nested4_qdq_weight(w, group_size))
        _mark(mod, method="nested4", w_bits=4, group=int(g))
        n += 1
    if verbose:
        print(f"[quantizers_weight] nested4 (W4 from W8 codes) over {n} Linear in {time.perf_counter()-t0:.1f}s")


def nested3_from_scale(w: torch.Tensor, s4: torch.Tensor, code_min: int = -8, code_max: int = 7) -> torch.Tensor:
    """Nested W3 whose W4 codes come from an external quantizer (e.g. an SVDQuant package:
    codes in [-8, 7], group scales s4 given by the package rather than absmax/7).
    q4 = round(w / s4), where w is the package's dequantized weight and must lie on the
    code grid (otherwise an error is raised); W3 then follows the same rule as nested3:
    |q4|: 0 -> 0, {1,2} -> 1, {3,4} -> 2, {5,6,7,8} -> 3;  W3 = T[c] * s4, T = {0, +-1.5, +-3.5, +-6}.
    Low-rank branches, smoothing and activation quantization stay as in the hi variant.
    """
    if w.dim() != 2 or s4.dim() != 2:
        raise ValueError(f"expected w [o,i], s4 [o,i/g]; got {tuple(w.shape)}, {tuple(s4.shape)}")
    o, i = w.shape; g = i // s4.shape[1]
    wg = w.detach().float().reshape(o, -1, g); s = s4.to(w.device).float().clamp_min(_EPS)[..., None]
    r = wg / s; q4 = r.round()
    dev = (r - q4).abs().max().item()
    if dev > 0.05:
        raise ValueError(f"nested3_from_scale: weight/scale mismatch (max |w/s - round| = {dev:.3f}); package weights are not on the code grid")
    q4.clamp_(code_min, code_max)
    a = q4.abs().clamp_(max=7)
    c = torch.where(a == 0, torch.zeros_like(a), (a + 1).div(2, rounding_mode="floor")).clamp_(max=3) * q4.sign()
    table = torch.tensor([-6.0, -3.5, -1.5, 0.0, 1.5, 3.5, 6.0], device=w.device)
    return (table[(c + 3).long()] * s).reshape(o, i).to(w.dtype)


def nested3_quantize_(
    model: nn.Module,
    w_bits: int = 3,
    group_size: Optional[int] = 128,
    *,
    name_filter: Optional[Callable[[str, nn.Linear], bool]] = None,
    verbose: bool = False,
) -> None:
    """Nested W3 variant: every Linear weight is replaced by the 3-bit values derived from
    its W4 codes (see ``nested3_qdq_weight``)."""
    if int(w_bits) != 3:
        raise ValueError(f"nested3 is only defined for w_bits=3 (W3 from W4 codes), got {w_bits}")
    t0 = time.perf_counter()
    n = 0
    for name, mod in _iter_linears(model, name_filter):
        w = mod.weight.data
        g = effective_group_size(w.shape[1], 4, group_size)
        mod.weight.data.copy_(nested3_qdq_weight(w, group_size))
        _mark(mod, method="nested3", w_bits=3, group=int(g))
        n += 1
    if verbose:
        print(f"[quantizers_weight] nested3 (W3 from W4 codes) over {n} Linear in {time.perf_counter()-t0:.1f}s")


# ---------------------------------------------------------------------------
# (D) calibration collection
# ---------------------------------------------------------------------------
def _call_forward_fn(run_forward_fn: Callable, model: nn.Module) -> None:
    try:
        n_params = len(inspect.signature(run_forward_fn).parameters)
    except (TypeError, ValueError):
        n_params = 1
    with torch.no_grad():
        run_forward_fn(model) if n_params >= 1 else run_forward_fn()


def collect_layer_inputs(
    model: nn.Module,
    run_forward_fn: Callable,
    max_tokens: int = 4096,
    *,
    name_filter: Optional[Callable[[str, nn.Linear], bool]] = None,
    store_device: str = "cpu",
    store_dtype: torch.dtype = torch.float32,
    max_rows_per_call: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Hook every ``nn.Linear`` and collect its flattened inputs as ``[N, in_features]``.

    ``run_forward_fn`` is invoked once under ``torch.no_grad()``; it is called as
    ``run_forward_fn(model)`` if it takes at least one positional parameter, else
    ``run_forward_fn()``.  It may run any number of forwards (e.g. a whole diffusion
    trajectory) -- rows accumulate across calls.

    Memory control (SDXL at 4K produces >1e6 tokens per Linear):
      * ``max_tokens``        hard cap on stored rows **per layer**; once reached the
                              layer's hook stops recording.
      * ``max_rows_per_call`` random row subsample applied to each individual forward,
                              so the budget is spread over the trajectory instead of
                              being eaten by the first call.
      * rows are moved to ``store_device`` / ``store_dtype`` (CPU fp32 by default).

    Returns ``{layer_name: Tensor[N, in_features]}`` (layers never hit are absent).
    """
    max_tokens = int(max_tokens)
    gen = torch.Generator().manual_seed(int(seed))
    bufs: Dict[str, List[torch.Tensor]] = {}
    counts: Dict[str, int] = {}
    handles = []

    def _make_hook(name: str):
        def hook(module, args):
            if counts.get(name, 0) >= max_tokens:
                return None
            if not args:
                return None
            x = args[0]
            if not torch.is_tensor(x) or not x.is_floating_point():
                return None
            rows = x.detach().reshape(-1, x.shape[-1])
            if max_rows_per_call is not None and rows.shape[0] > int(max_rows_per_call):
                idx = torch.randperm(rows.shape[0], generator=gen)[: int(max_rows_per_call)]
                rows = rows[idx.to(rows.device)]
            remaining = max_tokens - counts.get(name, 0)
            if rows.shape[0] > remaining:
                idx = torch.randperm(rows.shape[0], generator=gen)[:remaining]
                rows = rows[idx.to(rows.device)]
            bufs.setdefault(name, []).append(rows.to(store_device, store_dtype))
            counts[name] = counts.get(name, 0) + rows.shape[0]
            return None

        return hook

    for name, mod in _iter_linears(model, name_filter):
        handles.append(mod.register_forward_pre_hook(_make_hook(name)))
    try:
        _call_forward_fn(run_forward_fn, model)
    finally:
        for h in handles:
            h.remove()
    return {k: torch.cat(v, dim=0) for k, v in bufs.items() if v}


def collect_act_channel_absmax(
    model: nn.Module,
    run_forward_fn: Callable,
    *,
    name_filter: Optional[Callable[[str, nn.Linear], bool]] = None,
    store_device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Streaming per-input-channel |x| max for every Linear -- O(in_features) memory.

    This is the statistic AWQ needs; use it instead of ``collect_layer_inputs`` when the
    token budget would not fit (it never stores tokens).
    """
    stats: Dict[str, torch.Tensor] = {}
    handles = []

    def _make_hook(name: str):
        def hook(module, args):
            if not args:
                return None
            x = args[0]
            if not torch.is_tensor(x) or not x.is_floating_point():
                return None
            a = x.detach().float().reshape(-1, x.shape[-1]).abs().amax(dim=0).to(store_device)
            prev = stats.get(name)
            stats[name] = a if prev is None else torch.maximum(prev, a)
            return None

        return hook

    for name, mod in _iter_linears(model, name_filter):
        handles.append(mod.register_forward_pre_hook(_make_hook(name)))
    try:
        _call_forward_fn(run_forward_fn, model)
    finally:
        for h in handles:
            h.remove()
    return stats


def act_channel_absmax_from_inputs(layer_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Derive the AWQ per-channel statistic from already-collected token samples.

    Uses the RMS-style mean(|x|) per channel (AWQ's ``s_x``), which is far less noisy
    than the raw max on a few thousand tokens.
    """
    return {k: v.float().abs().mean(dim=0) for k, v in layer_inputs.items()}


# ---------------------------------------------------------------------------
# (A) AWQ
# ---------------------------------------------------------------------------
def _awq_loss(
    W: torch.Tensor,
    Wq: torch.Tensor,
    X: Optional[torch.Tensor],
    ref: Optional[torch.Tensor],
    chan: torch.Tensor,
) -> torch.Tensor:
    if X is not None and ref is not None:
        return (X @ Wq.t() - ref).pow(2).mean()
    # diagonal proxy: E[((Wq-W)x)^2] ~= sum_j dW_ij^2 * E[x_j^2], E[x_j^2] ~ chan_j^2
    return ((Wq - W) * chan.unsqueeze(0)).pow(2).mean()


def _awq_search_scale(
    W: torch.Tensor,
    chan: torch.Tensor,
    X: Optional[torch.Tensor],
    w_bits: int,
    group_size: Optional[int],
    grid: int,
) -> Tuple[torch.Tensor, float, float]:
    """Grid-search alpha in [0,1] for s_j = chan_j**alpha (AWQ eq. 6)."""
    ref = None if X is None else X @ W.t()
    chan = chan.clamp_min(1e-5)
    best_s = torch.ones_like(chan)
    best_loss = float("inf")
    best_alpha = 0.0
    for i in range(int(grid) + 1):
        alpha = i / float(grid)
        s = chan.pow(alpha)
        s = s / (s.max() * s.min()).sqrt().clamp_min(_EPS)  # keep s ~ O(1)
        s = s.clamp(1e-4, 1e4)
        Wq = qdq_weight(W * s.unsqueeze(0), w_bits, group_size) / s.unsqueeze(0)
        loss = _awq_loss(W, Wq, X, ref, chan)
        if torch.isfinite(loss) and loss.item() < best_loss:
            best_loss, best_s, best_alpha = loss.item(), s, alpha
    return best_s, best_loss, best_alpha


def _awq_search_clip(
    W: torch.Tensor,
    s: torch.Tensor,
    X: Optional[torch.Tensor],
    ref: Optional[torch.Tensor],
    chan: torch.Tensor,
    w_bits: int,
    group_size: Optional[int],
    clip_grid: int,
    max_shrink: float = 0.5,
) -> float:
    """Search a global shrink factor on the absmax clipping range (AWQ sec. 4.3)."""
    Ws = W * s.unsqueeze(0)
    out_f, in_f = Ws.shape
    g = effective_group_size(in_f, w_bits, group_size)
    qmax = _qmax(w_bits)
    wg = Ws.reshape(out_f, in_f // g, g)
    amax = wg.abs().amax(dim=-1, keepdim=True).clamp_min(_EPS)
    best_ratio, best_loss = 1.0, float("inf")
    for i in range(int(clip_grid) + 1):
        ratio = 1.0 - max_shrink * (i / float(clip_grid))
        scale = (amax * ratio) / qmax
        wq = ((wg / scale).round().clamp(-qmax, qmax) * scale).reshape(out_f, in_f)
        Wq = wq / s.unsqueeze(0)
        loss = _awq_loss(W, Wq, X, ref, chan)
        if torch.isfinite(loss) and loss.item() < best_loss:
            best_loss, best_ratio = loss.item(), ratio
    return best_ratio


def awq_quantize_(
    model: nn.Module,
    act_channel_absmax: Dict[str, torch.Tensor],
    w_bits: int,
    group_size: Optional[int] = 128,
    grid: int = 20,
    *,
    layer_inputs: Optional[Dict[str, torch.Tensor]] = None,
    max_calib_tokens: int = 512,
    clip_grid: int = 0,
    name_filter: Optional[Callable[[str, nn.Linear], bool]] = None,
    fallback_rtn: bool = True,
    verbose: bool = False,
) -> None:
    """AWQ-style activation-aware weight quantization, in place.

    For every Linear it grid-searches ``alpha in {0, 1/grid, ..., 1}`` for the
    per-input-channel scale ``s_j = act_channel_absmax[name]_j ** alpha`` (normalized to
    ``s / sqrt(max(s)*min(s))``), quantizes ``W diag(s)`` group-wise and folds the scale
    back out:  ``W <- Q(W diag(s)) diag(s)^-1``.  That identity makes the transform
    exact in a QDQ/simulated setting, so **no activation-side change and no edit to any
    other module is required** -- salient (large-activation) channels simply end up with
    a smaller relative rounding error.

    Selection criterion is the layer output MSE ``|| (Q(W diag s) diag(s)^-1 - W) X^T ||^2``
    estimated on up to ``max_calib_tokens`` rows of ``layer_inputs[name]``.  If
    ``layer_inputs`` is None (or misses a layer), it falls back to the diagonal proxy
    ``sum_j (dW_ij * chan_j)^2``, which needs only ``act_channel_absmax``.

    Args:
        act_channel_absmax: ``{layer_name: Tensor[in_features]}`` per-input-channel
            activation magnitude (from ``collect_act_channel_absmax`` or
            ``act_channel_absmax_from_inputs``).  Layers missing from this dict get
            plain RTN when ``fallback_rtn`` else are left untouched.
        grid: number of alpha steps (grid+1 points; ``alpha=0`` *is* RTN, so on the
            calibration data AWQ can never score worse than RTN).
        clip_grid: >0 additionally searches a global shrink of the clipping range
            (helps at W2/W3); 0 disables it (pure scale search).
    """
    t0 = time.perf_counter()
    n_awq = n_rtn = 0
    for name, mod in _iter_linears(model, name_filter):
        W_orig = mod.weight.data
        dev = W_orig.device
        chan = act_channel_absmax.get(name)
        if chan is None:
            if fallback_rtn:
                mod.weight.data.copy_(qdq_weight(W_orig, w_bits, group_size))
                _mark(mod, method="awq->rtn(no stats)", w_bits=int(w_bits),
                      group=int(effective_group_size(W_orig.shape[1], w_bits, group_size)))
                n_rtn += 1
            continue
        chan = chan.detach().to(dev, torch.float32).reshape(-1)
        if chan.numel() != W_orig.shape[1]:
            raise ValueError(f"{name}: act_channel_absmax has {chan.numel()} entries, "
                             f"in_features={W_orig.shape[1]}")
        W = W_orig.detach().to(torch.float32)
        X = None
        if layer_inputs is not None and name in layer_inputs:
            X = layer_inputs[name].detach()
            if X.shape[0] > int(max_calib_tokens):
                X = X[:: max(1, X.shape[0] // int(max_calib_tokens))][: int(max_calib_tokens)]
            X = X.to(dev, torch.float32)
        s, loss, alpha = _awq_search_scale(W, chan, X, w_bits, group_size, grid)
        ratio = 1.0
        if int(clip_grid) > 0:
            ref = None if X is None else X @ W.t()
            ratio = _awq_search_clip(W, s, X, ref, chan, w_bits, group_size, clip_grid)
        Ws = W * s.unsqueeze(0)
        if ratio != 1.0:
            out_f, in_f = Ws.shape
            g = effective_group_size(in_f, w_bits, group_size)
            qmax = _qmax(w_bits)
            wg = Ws.reshape(out_f, in_f // g, g)
            sc = (wg.abs().amax(dim=-1, keepdim=True).clamp_min(_EPS) * ratio) / qmax
            Wq = ((wg / sc).round().clamp(-qmax, qmax) * sc).reshape(out_f, in_f)
        else:
            Wq = qdq_weight(Ws, w_bits, group_size).float()
        mod.weight.data.copy_((Wq / s.unsqueeze(0)).to(W_orig.dtype))
        _mark(mod, method="awq", w_bits=int(w_bits),
              group=int(effective_group_size(W_orig.shape[1], w_bits, group_size)),
              alpha=float(alpha), clip_ratio=float(ratio), calib_tokens=(0 if X is None else int(X.shape[0])),
              search_loss=float(loss))
        n_awq += 1
    if verbose:
        print(f"[quantizers_weight] awq w{w_bits} over {n_awq} Linear "
              f"({n_rtn} rtn-fallback) in {time.perf_counter()-t0:.1f}s")


# ---------------------------------------------------------------------------
# (B) GPTQ
# ---------------------------------------------------------------------------
def _cholesky_inv_upper(H: torch.Tensor, percdamp: float, tries: int = 6) -> torch.Tensor:
    """Upper-Cholesky factor of H^-1, with escalating diagonal damping.

    ``H`` is modified in place (already dead-column patched by the caller).
    """
    n = H.shape[0]
    idx = torch.arange(n, device=H.device)
    mean_diag = torch.diagonal(H).mean().clamp_min(_EPS)
    damp = float(percdamp) * float(mean_diag)
    if damp <= 0:
        damp = float(mean_diag) * 1e-4
    H[idx, idx] += damp
    applied = damp
    for k in range(tries):
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
            return torch.linalg.cholesky(Hinv, upper=True)
        except Exception:
            extra = applied * 9.0  # 10x total each round
            H[idx, idx] += extra
            applied += extra
    raise torch.linalg.LinAlgError(
        f"GPTQ Hessian stayed non-PD after {tries} damping escalations (final damp={applied:.3e})"
    )


def _find_scale(block: torch.Tensor, qmax: int, mse_grid: int = 0, max_shrink: float = 0.5) -> torch.Tensor:
    """Per-out-row symmetric scale for one group block ``[out, g]``.

    ``mse_grid == 0`` -> plain absmax (the convention ``quant_unet`` and ``rtn_quantize_``
    use).  ``mse_grid > 0`` -> additionally search a per-row shrink of the clipping range
    (GPTQ's ``Quantizer(mse=True)`` option); trades outlier clipping for finer steps,
    which is what makes W2 usable.
    """
    s = (block.abs().amax(dim=1) / qmax).clamp_min(_EPS)
    if int(mse_grid) <= 0:
        return s
    best_s, best_e = s.clone(), None
    for k in range(int(mse_grid) + 1):
        ratio = 1.0 - float(max_shrink) * (k / float(mse_grid))
        sc = (s * ratio).clamp_min(_EPS)
        q = (block / sc.unsqueeze(1)).round().clamp(-qmax, qmax) * sc.unsqueeze(1)
        e = (q - block).pow(2).sum(dim=1)
        if best_e is None:
            best_e = e
        else:
            better = e < best_e
            best_e = torch.where(better, e, best_e)
            best_s = torch.where(better, sc, best_s)
    return best_s


def gptq_quantize_(
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
    name_filter: Optional[Callable[[str, nn.Linear], bool]] = None,
    fallback_rtn: bool = True,
    verbose: bool = False,
) -> None:
    """GPTQ: column-by-column quantization with Hessian-inverse error compensation.

    For each Linear with samples in ``layer_inputs``:
      1. ``H = (2/N) X^T X`` over up to ``max_tokens`` rows (streamed in
         ``hessian_chunk`` pieces); zero-variance ("dead") columns are patched to
         ``H_ii = 1, W[:, i] = 0``.
      2. Damped by ``percdamp * mean(diag(H))`` on the diagonal, escalating x10 until
         Cholesky succeeds (handles the singular / rank-deficient Hessian you get when
         ``N < in_features`` or when input channels are collinear).
      3. ``Hinv_chol = cholesky(cholesky_inverse(cholesky(H)), upper=True)``.  Columns
         are quantized left to right; after each column the residual
         ``err = (w - q) / Hinv[i, i]`` is subtracted from every not-yet-quantized
         column (``W[:, i:] -= err (x) Hinv[i, i:]``), and the whole trailing block is
         corrected with ``W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]``.  Group scales are
         re-derived lazily at each ``group_size`` boundary from the **already
         error-compensated** working weights (slightly better than the reference
         implementation, which reads the pre-block values).

    Args:
        layer_inputs: ``{layer_name: Tensor[N, in_features]}`` from
            ``collect_layer_inputs``.  ``N >> in_features`` is strongly recommended;
            ``verbose`` prints a warning when ``N < in_features``.
        blocksize: snapped to a multiple of the effective group size so a group never
            straddles a block boundary.
        act_order: quantize columns in descending ``diag(H)`` order (the ``--act-order``
            variant), then permute back.  Large-activation channels get quantized while
            the compensation budget is still intact; measured 2-4x lower output MSE
            here and it is what keeps W2 from diverging.  Since the weights are written
            back as fp (QDQ), the permutation is undone in place and **no ``g_idx`` is
            needed** -- a real packed-int4 kernel would need one.
        mse_scale_grid: >0 turns on the per-row MSE-optimal clipping search inside
            ``_find_scale`` (off by default so the scale convention stays identical to
            ``rtn_quantize_`` / ``quant_unet``; 10 is a good value at W2/W3).
        fallback_rtn: layers absent from ``layer_inputs`` get plain RTN instead of
            being skipped.
    """
    t0 = time.perf_counter()
    n_gptq = n_rtn = 0
    for name, mod in _iter_linears(model, name_filter):
        W_param = mod.weight.data
        out_f, in_f = W_param.shape
        dev = W_param.device
        X = layer_inputs.get(name)
        if X is None or X.numel() == 0:
            if fallback_rtn:
                mod.weight.data.copy_(qdq_weight(W_param, w_bits, group_size))
                _mark(mod, method="gptq->rtn(no inputs)", w_bits=int(w_bits),
                      group=int(effective_group_size(in_f, w_bits, group_size)))
                n_rtn += 1
            continue
        X = X.detach().reshape(-1, X.shape[-1])
        if X.shape[1] != in_f:
            raise ValueError(f"{name}: layer_inputs has in_features={X.shape[1]}, layer wants {in_f}")
        if X.shape[0] > int(max_tokens):
            X = X[:: max(1, X.shape[0] // int(max_tokens))][: int(max_tokens)]
        if X.shape[0] < in_f and verbose:
            print(f"[quantizers_weight] warn {name}: only {X.shape[0]} tokens for "
                  f"in_features={in_f} -> rank-deficient Hessian (damped)")

        qmax = _qmax(w_bits)
        g = effective_group_size(in_f, w_bits, group_size)
        per_group = g < in_f
        bs = max(1, int(blocksize))
        if per_group:
            bs = g if bs < g else (bs // g) * g

        # ---- Hessian ----
        H = torch.zeros(in_f, in_f, device=dev, dtype=torch.float32)
        for i in range(0, X.shape[0], int(hessian_chunk)):
            xc = X[i: i + int(hessian_chunk)].to(dev, torch.float32)
            H.addmm_(xc.t(), xc)
        H *= 2.0 / max(1, X.shape[0])

        W = W_param.detach().to(dev, torch.float32).clone()
        dead = torch.diagonal(H) == 0
        if bool(dead.any()):
            H[dead, dead] = 1.0
            W[:, dead] = 0.0
        inv_perm = None
        if act_order:
            perm = torch.argsort(torch.diagonal(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            inv_perm = torch.argsort(perm)
        Hinv = _cholesky_inv_upper(H, percdamp)
        del H

        Q = torch.zeros_like(W)
        # per-row mode has no group boundaries -> one scale fixed up front
        scale = None if per_group else _find_scale(W, qmax, mse_scale_grid)
        proxy = 0.0
        for i1 in range(0, in_f, bs):
            i2 = min(i1 + bs, in_f)
            cnt = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            E1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(cnt):
                col = i1 + i
                w = W1[:, i]
                d = Hinv1[i, i].clamp_min(_EPS)
                if per_group and col % g == 0:
                    # bs is a multiple of g, so the whole group lives in this block and
                    # W1 already carries the compensation from earlier columns
                    scale = _find_scale(W1[:, i: i + g], qmax, mse_scale_grid)
                q = (w / scale).round().clamp_(-qmax, qmax) * scale
                Q1[:, i] = q
                err = (w - q) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                E1[:, i] = err
                proxy += float(err.pow(2).sum())
            Q[:, i1:i2] = Q1
            if i2 < in_f:
                W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
        if inv_perm is not None:
            Q = Q[:, inv_perm]
        mod.weight.data.copy_(Q.to(W_param.dtype))
        _mark(mod, method="gptq", w_bits=int(w_bits), group=int(g), blocksize=int(bs),
              percdamp=float(percdamp), act_order=bool(act_order),
              mse_scale_grid=int(mse_scale_grid), calib_tokens=int(X.shape[0]),
              dead_cols=int(dead.sum()), proxy_err=float(proxy))
        n_gptq += 1
        del W, Q, Hinv
    if verbose:
        print(f"[quantizers_weight] gptq w{w_bits} over {n_gptq} Linear "
              f"({n_rtn} rtn-fallback) in {time.perf_counter()-t0:.1f}s")


# ---------------------------------------------------------------------------
# one-call dispatcher (fair-comparison harness)
# ---------------------------------------------------------------------------
def quantize_weights_(
    model: nn.Module,
    method: str,
    w_bits: int,
    group_size: Optional[int] = 128,
    *,
    layer_inputs: Optional[Dict[str, torch.Tensor]] = None,
    act_channel_absmax: Optional[Dict[str, torch.Tensor]] = None,
    **kwargs,
) -> None:
    """``method in {'rtn','nested3','nested4','awq','gptq','none'}`` -- single entry point
    so every arm can be swept from one config field."""
    m = str(method).lower()
    if m in ("none", "fp16", "off"):
        return
    if m == "rtn":
        rtn_quantize_(model, w_bits, group_size, **kwargs)
    elif m == "nested3":
        nested3_quantize_(model, w_bits, group_size, **kwargs)
    elif m == "nested4":
        nested4_quantize_(model, w_bits, group_size, **kwargs)
    elif m == "awq":
        stats = act_channel_absmax
        if stats is None:
            if layer_inputs is None:
                raise ValueError("awq needs act_channel_absmax or layer_inputs")
            stats = act_channel_absmax_from_inputs(layer_inputs)
        awq_quantize_(model, stats, w_bits, group_size, layer_inputs=layer_inputs, **kwargs)
    elif m == "gptq":
        if layer_inputs is None:
            raise ValueError("gptq needs layer_inputs")
        gptq_quantize_(model, layer_inputs, w_bits, group_size, **kwargs)
    else:
        raise ValueError(f"unknown weight-quant method {method!r}")


# ---------------------------------------------------------------------------
# CPU unit tests
# ---------------------------------------------------------------------------
def _make_case(out_f: int, in_f: int, n_calib: int, n_test: int, seed: int = 0):
    """Linear + correlated, per-channel-heteroscedastic inputs (the regime AWQ/GPTQ
    are designed for: a few salient input channels + cross-channel correlation)."""
    gen = torch.Generator().manual_seed(seed)
    lin = nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        w = torch.randn(out_f, in_f, generator=gen) * 0.05
        w.view(-1)[torch.randperm(out_f * in_f, generator=gen)[: (out_f * in_f) // 200]] *= 6.0
        lin.weight.copy_(w)
    mix = torch.randn(in_f, in_f, generator=gen) / math.sqrt(in_f)
    chan = torch.exp(torch.randn(in_f, generator=gen) * 1.2)

    def sample(n):
        return (torch.randn(n, in_f, generator=gen) @ mix) * chan

    return lin, sample(n_calib), sample(n_test)


def _mse(lin: nn.Linear, ref_w: torch.Tensor, X: torch.Tensor) -> float:
    with torch.no_grad():
        return float((X @ lin.weight.t() - X @ ref_w.t()).pow(2).mean())


def _test_beats_rtn():
    import copy as _copy
    print("\n--- test 1: AWQ / GPTQ vs RTN, held-out output MSE ---")
    out_f, in_f, gs = 96, 256, 64
    lin, Xc, Xt = _make_case(out_f, in_f, n_calib=2048, n_test=1024, seed=1)
    ref_w = lin.weight.data.clone()
    li = {"": Xc}
    stats = act_channel_absmax_from_inputs(li)
    ok = True
    for bits in (2, 3, 4):
        res = {}
        for meth in ("rtn", "awq", "gptq"):
            m = _copy.deepcopy(lin)
            if meth == "rtn":
                rtn_quantize_(m, bits, gs)
            elif meth == "awq":
                awq_quantize_(m, stats, bits, gs, layer_inputs=li, max_calib_tokens=512)
            else:
                gptq_quantize_(m, li, bits, gs)
            res[meth] = (_mse(m, ref_w, Xc), _mse(m, ref_w, Xt))
        r = res["rtn"][1]
        print(f"  w{bits} g{gs}: rtn calib/test = {res['rtn'][0]:.5f}/{res['rtn'][1]:.5f} | "
              f"awq {res['awq'][1]:.5f} ({res['awq'][1]/r:.3f}x) | "
              f"gptq {res['gptq'][1]:.5f} ({res['gptq'][1]/r:.3f}x)")
        for meth in ("awq", "gptq"):
            if not res[meth][1] < res["rtn"][1]:
                print(f"    FAIL: {meth} w{bits} held-out MSE not below RTN")
                ok = False
            if not res[meth][0] < res["rtn"][0]:
                print(f"    FAIL: {meth} w{bits} calib MSE not below RTN")
                ok = False
    assert ok, "AWQ/GPTQ must beat RTN at equal w_bits"
    print("  ok")


def _test_per_row_and_stats_only():
    import copy as _copy
    print("\n--- test 2: per-row (group_size=None) + AWQ with stats only (no tokens) ---")
    lin, Xc, Xt = _make_case(64, 192, 1536, 768, seed=2)
    ref_w = lin.weight.data.clone()
    li = {"": Xc}
    stats = act_channel_absmax_from_inputs(li)
    row = {}
    for meth in ("rtn", "awq", "gptq"):
        m = _copy.deepcopy(lin)
        if meth == "rtn":
            rtn_quantize_(m, 3, None)
        elif meth == "awq":
            awq_quantize_(m, stats, 3, None, layer_inputs=li)
        else:
            gptq_quantize_(m, li, 3, None)
        row[meth] = _mse(m, ref_w, Xt)
    print(f"  w3 per-row: rtn {row['rtn']:.5f} | awq {row['awq']:.5f} | gptq {row['gptq']:.5f}")
    assert row["awq"] < row["rtn"] and row["gptq"] < row["rtn"], "per-row arm regressed"
    m = _copy.deepcopy(lin)
    awq_quantize_(m, stats, 3, 64)  # no layer_inputs -> diagonal proxy loss
    proxy = _mse(m, ref_w, Xt)
    m2 = _copy.deepcopy(lin)
    rtn_quantize_(m2, 3, 64)
    print(f"  w3 g64 stats-only AWQ (diag proxy): {proxy:.5f} vs rtn {_mse(m2, ref_w, Xt):.5f}")
    assert proxy < _mse(m2, ref_w, Xt), "stats-only AWQ must still beat RTN"
    print("  ok")


def _test_group_boundaries():
    print("\n--- test 3: group_size boundary handling ---")
    assert effective_group_size(256, 4, 128) == 128
    assert effective_group_size(256, 4, 64) == 64
    assert effective_group_size(250, 4, 64) == 250, "non-divisible must fall back to per-row"
    assert effective_group_size(2816, 4, 128) == 128
    assert effective_group_size(2560, 4, 128) == 128
    assert effective_group_size(320, 4, 128) == 320, "320 % 128 != 0 -> per-row"
    assert effective_group_size(256, 8, 64) == (64 if W8_GROUPED else 256)
    assert effective_group_size(256, 4, None) == 256
    # every group must own an independent scale: the group absmax element round-trips
    # exactly (it maps onto +-qmax), and per-group magnitudes differ by 1e3
    out_f, in_f, g, bits = 5, 256, 64, 4
    torch.manual_seed(0)
    w = torch.randn(out_f, in_f)
    for k in range(in_f // g):
        w[:, k * g:(k + 1) * g] *= 10.0 ** k
    wq = qdq_weight(w, bits, g)
    wg, wqg = w.view(out_f, -1, g), wq.view(out_f, -1, g)
    am = wg.abs().argmax(dim=-1, keepdim=True)
    got = torch.gather(wqg, -1, am)
    want = torch.gather(wg, -1, am)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-7), "group absmax element must round-trip"
    # and the values inside one group live on exactly (2*qmax+1) levels
    lev = torch.unique(torch.round(wqg[0, 2] / (wg[0, 2].abs().max() / _qmax(bits)))).numel()
    assert lev <= 2 * _qmax(bits) + 1, f"{lev} levels > grid"
    # one shared row scale would be set by the 1e3-larger group and annihilate the
    # small ones; per-group scales must keep each group's *own* relative error
    per_row = qdq_weight(w, bits, None).view(out_f, -1, g)
    for k in range(in_f // g):
        e_grp = (wg[:, k] - wqg[:, k]).pow(2).mean()
        e_row = (wg[:, k] - per_row[:, k]).pow(2).mean()
        rel = float(e_grp / wg[:, k].pow(2).mean())
        # uniform-quantizer bound: (amax/qmax)^2/12 with amax ~ 2.8 sigma over 64 draws
        assert rel < 3e-2, f"group {k} relative error {rel:.2e} too high"
        if k < in_f // g - 1:
            assert e_grp * 20 < e_row, f"group {k}: {float(e_grp):.3e} vs per-row {float(e_row):.3e}"
    # the 1e3-smaller first group is annihilated outright by a single shared scale
    assert float(per_row[:, 0].abs().max()) == 0.0 and float(wqg[:, 0].abs().max()) > 0
    # x-check against the in-repo reference implementation (bit-exact convention)
    ref = _reference_fake_quant(w.clone(), bits, g)
    assert torch.equal(ref, wq), "must match quant_unet._fake_quant_weight_ exactly"
    print(f"  group scales independent, absmax exact, {lev} levels, matches quant_unet ref")
    print("  ok")


def _reference_fake_quant(w: torch.Tensor, w_bits: int, group_size: int) -> torch.Tensor:
    """Verbatim math of quant_unet._fake_quant_weight_ (kept local; nothing imported)."""
    qmax = (1 << (int(w_bits) - 1)) - 1
    out_f, in_f = w.shape
    g = in_f if int(w_bits) >= 8 else (group_size if in_f % group_size == 0 else in_f)
    wg = w.float().view(out_f, in_f // g, g)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    return ((wg / scale).round().clamp(-qmax, qmax) * scale).view(out_f, in_f).to(w.dtype)


def _test_indivisible_fallback():
    import copy as _copy
    print("\n--- test 4: in_features not divisible by group_size (250 % 128) ---")
    lin, Xc, Xt = _make_case(48, 250, 1600, 800, seed=3)
    ref_w = lin.weight.data.clone()
    li = {"": Xc}
    stats = act_channel_absmax_from_inputs(li)
    res = {}
    for meth in ("rtn", "awq", "gptq"):
        m = _copy.deepcopy(lin)
        if meth == "rtn":
            rtn_quantize_(m, 4, 128)
        elif meth == "awq":
            awq_quantize_(m, stats, 4, 128, layer_inputs=li)
        else:
            gptq_quantize_(m, li, 4, 128, blocksize=128)
        assert quant_report(m)[""]["group"] == 250, "must report per-row fallback"
        assert torch.isfinite(m.weight.data).all()
        res[meth] = _mse(m, ref_w, Xt)
    print(f"  w4 in_f=250: rtn {res['rtn']:.6f} | awq {res['awq']:.6f} | gptq {res['gptq']:.6f}")
    assert res["awq"] < res["rtn"] and res["gptq"] < res["rtn"]
    print("  ok")


def _test_sequential_end_to_end():
    import copy as _copy
    print("\n--- test 5: nn.Sequential end-to-end + collect_layer_inputs ---")
    torch.manual_seed(7)
    net = nn.Sequential(nn.Linear(128, 256, bias=True), nn.GELU(),
                        nn.Linear(256, 256, bias=False), nn.GELU(),
                        nn.Linear(256, 64, bias=True))
    net.eval()
    gen = torch.Generator().manual_seed(11)
    mix = torch.randn(128, 128, generator=gen) / math.sqrt(128)
    chan = torch.exp(torch.randn(128, generator=gen) * 1.0)
    batches = [((torch.randn(8, 16, 128, generator=gen) @ mix) * chan) for _ in range(6)]
    test_x = (torch.randn(4, 16, 128, generator=gen) @ mix) * chan

    def run(model):
        for b in batches:
            model(b)

    # 6 forwards x (8*16) tokens = 768 rows per layer, flattened over the leading dims
    li = collect_layer_inputs(net, run, max_tokens=4096)
    assert set(li.keys()) == {"0", "2", "4"}, li.keys()
    for k, v in li.items():
        assert v.dim() == 2 and v.shape[0] == 6 * 8 * 16, (k, v.shape)
        assert v.dtype == torch.float32 and v.device.type == "cpu"
    assert li["0"].shape[1] == 128 and li["2"].shape[1] == 256 and li["4"].shape[1] == 256
    assert torch.equal(li["0"][:128], batches[0].reshape(-1, 128)), "rows must be verbatim inputs"
    # per-call subsample spreads the budget over the trajectory
    spread = collect_layer_inputs(net, run, max_tokens=4096, max_rows_per_call=64)
    assert all(v.shape[0] == 6 * 64 for v in spread.values()), \
        {k: v.shape for k, v in spread.items()}
    # hard cap stops recording mid-trajectory
    small = collect_layer_inputs(net, run, max_tokens=300)
    assert all(v.shape[0] == 300 for v in small.values()), "max_tokens cap not enforced"
    stats_stream = collect_act_channel_absmax(net, run)
    assert set(stats_stream) == {"0", "2", "4"} and stats_stream["0"].shape == (128,)

    with torch.no_grad():
        ref = net(test_x)
    out = {}
    for meth in ("rtn", "awq", "gptq"):
        m = _copy.deepcopy(net)
        quantize_weights_(m, meth, 3, 128, layer_inputs=li)
        with torch.no_grad():
            out[meth] = float((m(test_x) - ref).pow(2).mean())
    print(f"  3-layer MLP w3 g128 end-to-end MSE: rtn {out['rtn']:.5f} | "
          f"awq {out['awq']:.5f} ({out['awq']/out['rtn']:.3f}x) | "
          f"gptq {out['gptq']:.5f} ({out['gptq']/out['rtn']:.3f}x)")
    assert out["awq"] < out["rtn"] and out["gptq"] < out["rtn"]
    rep = quant_report(m)
    assert set(rep) == {"0", "2", "4"} and rep["0"]["method"] == "gptq"
    print(f"  gptq report[layer 0]: {rep['0']}")
    print("  ok")


def _test_robustness():
    import copy as _copy
    print("\n--- test 6: degenerate inputs (dead channels, singular Hessian, dtypes) ---")
    torch.manual_seed(5)
    lin = nn.Linear(64, 32, bias=False)
    X = torch.randn(96, 64)
    X[:, 7] = 0.0            # dead channel -> H_ii == 0
    X[:, 8] = X[:, 9]        # exactly collinear -> singular H
    m = _copy.deepcopy(lin)
    gptq_quantize_(m, {"": X}, 4, 32)           # N=96 < in_f=64? no: 96 > 64, still rank-deficient
    assert torch.isfinite(m.weight.data).all() and float(m.weight.data[:, 7].abs().sum()) == 0.0
    m2 = _copy.deepcopy(lin)
    gptq_quantize_(m2, {"": X[:20]}, 4, 32)     # N << in_f -> heavily rank-deficient
    assert torch.isfinite(m2.weight.data).all()
    zero = nn.Linear(64, 8, bias=False)
    with torch.no_grad():
        zero.weight.zero_()
    for meth, kw in (("rtn", {}), ("gptq", {"layer_inputs": {"": X}}),
                     ("awq", {"layer_inputs": {"": X}})):
        z = _copy.deepcopy(zero)
        quantize_weights_(z, meth, 4, 32, **kw)
        assert torch.isfinite(z.weight.data).all() and float(z.weight.data.abs().max()) == 0.0
    h = nn.Linear(128, 32, bias=False).to(torch.float16)   # fp16 storage round-trip
    li = {"": torch.randn(512, 128)}
    for meth, kw in (("rtn", {}), ("gptq", {"layer_inputs": li}), ("awq", {"layer_inputs": li})):
        q = _copy.deepcopy(h)
        quantize_weights_(q, meth, 4, 128, **kw)
        assert q.weight.dtype == torch.float16 and torch.isfinite(q.weight.data).all()
    # missing calibration entry -> documented RTN fallback
    miss = nn.Linear(64, 8, bias=False)
    gptq_quantize_(miss, {}, 4, 32)
    assert "rtn" in quant_report(miss)[""]["method"]
    print("  dead/collinear/zero/fp16/missing-stats all handled")
    print("  ok")


def _test_bit_ladder():
    import copy as _copy
    print("\n--- test 7: bit ladder on an SDXL-shaped layer (in_f=1280, g=128) ---")
    lin, Xc, Xt = _make_case(640, 1280, 4096, 1024, seed=9)
    ref_w = lin.weight.data.clone()
    li = {"": Xc}
    stats = act_channel_absmax_from_inputs(li)
    cols = ("rtn", "awq", "awq+clip", "gptq", "gptq+mse")
    print("  bits " + " ".join(f"{c:>10}" for c in cols) + "   (held-out output MSE)")
    for bits in (2, 3, 4):
        row = []
        for meth in cols:
            m = _copy.deepcopy(lin)
            if meth == "rtn":
                rtn_quantize_(m, bits, 128)
            elif meth == "awq":
                awq_quantize_(m, stats, bits, 128, layer_inputs=li, max_calib_tokens=512)
            elif meth == "awq+clip":
                awq_quantize_(m, stats, bits, 128, layer_inputs=li, max_calib_tokens=512, clip_grid=8)
            elif meth == "gptq":
                gptq_quantize_(m, li, bits, 128)
            else:
                gptq_quantize_(m, li, bits, 128, mse_scale_grid=10)
            row.append(_mse(m, ref_w, Xt))
        print(f"  w{bits:<4}" + " ".join(f"{v:10.5f}" for v in row)
              + "   " + "  ".join(f"{c} {v/row[0]:.3f}x" for c, v in zip(cols[1:], row[1:])))
        for c, v in zip(cols[1:], row[1:]):
            assert v < row[0], f"w{bits} {c} regression vs RTN"
    print("  ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(8)
    t0 = time.perf_counter()
    _test_beats_rtn()
    _test_per_row_and_stats_only()
    _test_group_boundaries()
    _test_indivisible_fallback()
    _test_sequential_end_to_end()
    _test_robustness()
    _test_bit_ladder()
    print(f"\nALL TESTS PASSED in {time.perf_counter()-t0:.1f}s")
