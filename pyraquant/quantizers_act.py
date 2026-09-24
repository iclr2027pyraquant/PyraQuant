"""Activation-side transforms for PyraQuant (nn.Linear only).

Two *mathematically exact* input transforms that shrink the dynamic range the
per-token activation quantizer has to cover.  Both compose with the existing
``quant_unet._ActFakeQuant`` per-token dynamic symmetric QDQ hook; neither
changes the function computed by the network in fp32/fp64 arithmetic.

(A) SmoothQuant (Xiao et al., ICML 2023) -- calibration-based difficulty migration
    For ``y = x W^T + b`` with per-input-channel positive scales ``s``::

        x' = x / s        W' = W * s      (broadcast over out-rows: W'[:, j] = W[:, j] * s[j])
        x' W'^T = sum_j (x_j / s_j) (W_ij s_j) = x W^T          <-- exact for any s > 0

    with ``s_j = act_absmax_j^alpha / weight_absmax_j^(1 - alpha)``.  Outliers move
    from the activation (per-token quantized, cannot afford them) to the weight
    (per-channel / group quantized, can).

(B) Rotation (QuaRot / SpinQuant style) -- data-free incoherence processing
    For an orthogonal ``R`` of size ``in_features``::

        x' = x @ R        W' = W @ R
        x' W'^T = x R R^T W^T = x W^T                           <-- exact for R R^T = I

    Rotation mixes every input channel into every other one, so a few huge
    channels are spread over the whole width and the per-token ``absmax`` drops
    towards the RMS of the row.  ``kind="hadamard"`` builds a randomized
    Kronecker Hadamard (Sylvester 2^k block (x) dense orthogonal odd-m block),
    which is the incoherence-optimal choice; ``kind="orthogonal"`` draws a Haar
    random orthogonal matrix by QR.

Both are applied in place: the weight is rewritten once, and a
``forward_pre_hook`` applies the matching input transform at run time.  The
transform hook MUST run before the QDQ hook -- see ``build_act_quant_model``.

Numerical contract (measured on CPU)
    With fp32 storage both transforms are exact to ~1e-6 max abs diff, well inside
    the 1e-4 requirement; in fp64 (``compute_dtype=rot_dtype=float64``) rotation is
    exact to 1e-14.  In **fp16** storage the round trip cannot be exact to 1e-4 in
    absolute terms -- rewriting ``W`` densifies it and rounds back to fp16.  What
    matters is that this stays at the dtype's own noise floor: on a 1280x1280 Linear
    with outlier inputs, relative error vs an fp64 reference is 3.50e-4 with no
    transform, 4.90e-4 after rotation and 4.95e-4 after SmoothQuant -- i.e. the
    transform costs ~1.4x of fp16's *existing* rounding noise, three orders of
    magnitude below the A4 quantization damage it is there to remove.

Which transform to reach for
    They attack different statistics and are not interchangeable (see
    ``test_transforms_attack_different_statistics``):
      * rotation is orthogonal, so it preserves ||x|| **exactly**. It can only fix
        the peak-to-RMS ratio, and bottoms out at the Gaussian floor
        absmax/rms ~ sqrt(2 ln n) (~2.4 at n=320). It is data-free -- no calibration.
      * SmoothQuant moves energy out of ``x`` into ``W``, shrinking rms(x) itself
        (measured 6.94 -> 0.14 on the synthetic bench). That is the only lever when
        the outlier channels' energy is *discarded* by ``W`` -- the regime where
        naive per-token quantization actually collapses. It needs calibration.
    On the synthetic bench at A4, with outlier channels the weights discard, relative
    error goes 0.99 (baseline) -> 0.12 (SmoothQuant) -> 0.68 (rotation).  Where the
    outliers do drive the output, the two are comparable (0.15 -> 0.09 -> 0.10).
    Try SmoothQuant first on SDXL, and stack rotation on top via
    ``method="rotation+smoothquant"``.

Only torch / numpy / stdlib are used.  Nothing in this module imports or mutates
any other project file (``quant_unet.py`` is imported lazily and optionally, and
falls back to local equivalents).
"""

from __future__ import annotations

import contextlib
import inspect
import math
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch
from torch import nn

__all__ = [
    # --- SmoothQuant -------------------------------------------------------
    "collect_act_channel_absmax",
    "apply_smoothquant_",
    # --- Rotation ----------------------------------------------------------
    "apply_rotation_",
    "build_rotation_matrix",
    "rotation_plan",
    "clear_rotation_cache",
    # --- shared plumbing ---------------------------------------------------
    "remove_act_transforms_",
    "act_transform_report",
    "act_range_stats",
    "PerTokenActFakeQuant",
    "build_act_quant_model",
]

# Attribute stamped on every Linear we touch, so the transform is auditable and
# double-application can be refused.
_STATE_ATTR = "_pq_act_transform"
_HANDLE_ATTR = "_pq_act_transform_handle"


# =============================================================================
# small utilities
# =============================================================================
@contextlib.contextmanager
def _exact_matmul():
    """Disable TF32 while we rewrite weights.

    A TF32 matmul carries only ~10 mantissa bits; the exactness contract of both
    transforms (max abs diff < 1e-4) would be violated on Ampere if TF32 were on.
    """
    prev_cuda = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_cuda
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def _iter_linears(
    model: nn.Module,
    predicate: Optional[Callable[[str, nn.Linear], bool]] = None,
) -> Iterator[Tuple[str, nn.Linear]]:
    """Yield ``(qualified_name, module)`` for every nn.Linear passing ``predicate``."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if predicate is None or predicate(name, mod):
                yield name, mod


def _first_float_tensor(args: Tuple[Any, ...]) -> Optional[torch.Tensor]:
    if not args:
        return None
    x = args[0]
    if not torch.is_tensor(x) or not x.is_floating_point():
        return None
    return x


def act_range_stats(x: torch.Tensor) -> Dict[str, float]:
    """Diagnostics for how hard ``x`` is for a *per-token* (last-dim) quantizer.

    Returns
        ``absmax_over_meanabs``  mean over tokens of ``amax(|x_t|) / mean(|x_t|)``.
            This is exactly the factor by which per-token absmax scaling wastes
            quantization levels: 1.0 is ideal, a Gaussian row of width n sits at
            ~sqrt(2*ln n)/0.798, an outlier channel drives it to tens/hundreds.
        ``absmax_over_rms``      same with the RMS of the row in the denominator.
        ``channel_absmax_ratio`` max over channels of absmax_j / median_j absmax_j
            (how spiky the *channel* profile is -- what SmoothQuant attacks).
    """
    xf = x.detach().float().reshape(-1, x.shape[-1])
    a = xf.abs()
    amax = a.amax(dim=-1)
    mean = a.mean(dim=-1).clamp_min(1e-12)
    rms = xf.pow(2).mean(dim=-1).sqrt().clamp_min(1e-12)
    ch = a.amax(dim=0)
    med = ch.median().clamp_min(1e-12)
    return {
        "absmax_over_meanabs": float((amax / mean).mean()),
        "absmax_over_rms": float((amax / rms).mean()),
        "channel_absmax_ratio": float(ch.amax() / med),
    }


def _mark(mod: nn.Linear, state: Dict[str, Any]) -> None:
    setattr(mod, _STATE_ATTR, state)


def _get_state(mod: nn.Module) -> Optional[Dict[str, Any]]:
    return getattr(mod, _STATE_ATTR, None)


def act_transform_report(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Audit which Linears carry which transform. ``{name: {kind, ...}}``."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, mod in _iter_linears(model):
        st = _get_state(mod)
        if st is not None:
            out[name] = {k: v for k, v in st.items() if k != "_tensor"}
    return out


# =============================================================================
# (A) SmoothQuant
# =============================================================================
class _ActAbsmaxCollector:
    """forward_pre_hook: running per-input-channel absmax over all dims but last."""

    __slots__ = ("stat", "n_calls", "n_tokens")

    def __init__(self) -> None:
        self.stat: Optional[torch.Tensor] = None
        self.n_calls = 0
        self.n_tokens = 0

    def __call__(self, module, args):
        x = _first_float_tensor(args)
        if x is None:
            return None
        xf = x.detach().float().reshape(-1, x.shape[-1]).abs()
        cur = xf.amax(dim=0).cpu()
        self.stat = cur if self.stat is None else torch.maximum(self.stat, cur)
        self.n_calls += 1
        self.n_tokens += int(xf.shape[0])
        return None


def _call_run_forward(run_forward_fn: Callable[..., Any], model: nn.Module) -> Any:
    """Accept either ``fn(model)`` or ``fn()`` as the calibration driver."""
    try:
        sig = inspect.signature(run_forward_fn)
        n_required = sum(
            1
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
        )
        takes_varargs = any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values())
    except (TypeError, ValueError):
        n_required, takes_varargs = 1, False
    if n_required >= 1 or takes_varargs:
        return run_forward_fn(model)
    return run_forward_fn()


def collect_act_channel_absmax(
    model: nn.Module,
    run_forward_fn: Callable[..., Any],
    *,
    predicate: Optional[Callable[[str, nn.Linear], bool]] = None,
    return_meta: bool = False,
) -> Dict[str, torch.Tensor]:
    """Calibrate: per-``in_features``-channel absolute maximum of every Linear input.

    ``run_forward_fn`` is called once, as ``run_forward_fn(model)`` if it takes a
    positional argument else ``run_forward_fn()``; it must drive whatever forward
    passes constitute the calibration set (for SDXL: a handful of denoising steps
    at the stage resolutions you care about).  Everything runs under ``no_grad``;
    the caller owns train/eval mode.

    Returns ``{module_name: FloatTensor[in_features]}`` on CPU.  Layers that were
    never invoked (e.g. a branch the calibration prompt did not exercise) are
    absent from the dict -- ``apply_smoothquant_`` skips them.

    With ``return_meta=True`` the returned dict additionally carries the key
    ``"__meta__"`` -> ``{name: {"n_calls": int, "n_tokens": int}}``.
    """
    collectors: Dict[str, _ActAbsmaxCollector] = {}
    handles = []
    for name, mod in _iter_linears(model, predicate):
        c = _ActAbsmaxCollector()
        collectors[name] = c
        handles.append(mod.register_forward_pre_hook(c))
    try:
        with torch.no_grad():
            _call_run_forward(run_forward_fn, model)
    finally:
        for h in handles:
            h.remove()

    stats: Dict[str, torch.Tensor] = {}
    meta: Dict[str, Dict[str, int]] = {}
    for name, c in collectors.items():
        if c.stat is None:
            continue
        stats[name] = c.stat.contiguous()
        meta[name] = {"n_calls": c.n_calls, "n_tokens": c.n_tokens}
    if return_meta:
        stats["__meta__"] = meta  # type: ignore[assignment]
    return stats


class _SmoothScaleHook:
    """forward_pre_hook: ``x -> x * inv_s`` (per input channel)."""

    __slots__ = ("inv_scale", "_cache")

    def __init__(self, inv_scale: torch.Tensor):
        self.inv_scale = inv_scale.detach().float().cpu().contiguous()
        self._cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _get(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.device, x.dtype)
        t = self._cache.get(key)
        if t is None:
            t = self.inv_scale.to(device=x.device, dtype=x.dtype)
            self._cache[key] = t
        return t

    def __call__(self, module, args):
        x = _first_float_tensor(args)
        if x is None:
            return None
        return (x * self._get(x),) + tuple(args[1:])


def apply_smoothquant_(
    model: nn.Module,
    stats: Dict[str, torch.Tensor],
    alpha: float = 0.5,
    *,
    eps: float = 1e-5,
    max_scale: float = 1e4,
    predicate: Optional[Callable[[str, nn.Linear], bool]] = None,
    prepend: bool = False,
    strict: bool = False,
    allow_restack: bool = False,
) -> None:
    """In-place SmoothQuant difficulty migration. Mathematically exact (no quantization).

    For every Linear that has an entry in ``stats``:
      ``s_j = clamp(act_absmax_j^alpha / weight_absmax_j^(1-alpha), 1/max_scale, max_scale)``
    then ``W[:, j] *= s_j`` and a pre-hook divides the input by ``s``.  Channels whose
    activation or weight statistic is <= ``eps`` get ``s_j = 1`` (dead channel: migrating
    nothing, and 0^alpha would blow up the ratio).  The identity holds for *any*
    positive ``s``, so all of these guards are numerically free.

    alpha        migration strength. 0 -> no-op, 1 -> flatten activations completely
                 (and wreck the weights). 0.5 is the paper default; 0.6-0.8 is the
                 usual range for hard-to-quantize activations.
    prepend      register the transform hook at the *front* of the pre-hook list.
                 Needed only if a QDQ hook is already registered on the module -- the
                 smooth transform must run first.
    strict       raise if a targeted Linear has no calibration stats (default: skip).
    allow_restack  permit applying a second transform on top of an existing one.

    Returns None. Use ``act_transform_report(model)`` to audit what was applied, and
    ``remove_act_transforms_(model)`` to undo it exactly.
    """
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    stats = {k: v for k, v in stats.items() if k != "__meta__"}

    with torch.no_grad(), _exact_matmul():
        for name, mod in _iter_linears(model, predicate):
            act = stats.get(name)
            if act is None:
                if strict:
                    raise KeyError(f"no calibration stats for Linear {name!r}")
                continue
            if _get_state(mod) is not None and not allow_restack:
                raise RuntimeError(
                    f"Linear {name!r} already carries transform "
                    f"{_get_state(mod)['kind']!r}; pass allow_restack=True to stack."
                )
            w = mod.weight.data
            in_f = w.shape[1]
            if act.numel() != in_f:
                raise ValueError(
                    f"stats[{name!r}] has {act.numel()} entries but in_features={in_f}"
                )
            act = act.detach().to(device=w.device, dtype=torch.float64).abs()
            wmax = w.detach().to(torch.float64).abs().amax(dim=0)

            live = (act > eps) & (wmax > eps)
            s = torch.ones(in_f, dtype=torch.float64, device=w.device)
            s[live] = act[live].pow(alpha) / wmax[live].pow(1.0 - alpha)
            s = s.clamp(1.0 / max_scale, max_scale)

            mod.weight.data.copy_((w.to(torch.float64) * s.unsqueeze(0)).to(w.dtype))
            hook = _SmoothScaleHook(s.reciprocal().float())
            handle = mod.register_forward_pre_hook(hook, prepend=prepend)
            setattr(mod, _HANDLE_ATTR, handle)
            _mark(
                mod,
                {
                    "kind": "smoothquant",
                    "alpha": alpha,
                    "in_features": int(in_f),
                    "scale_min": float(s.min()),
                    "scale_max": float(s.max()),
                    "n_live_channels": int(live.sum()),
                    "_tensor": s.float().cpu(),  # kept for exact removal
                },
            )


# =============================================================================
# (B) Rotation (QuaRot / SpinQuant style, data-free)
# =============================================================================
_ROT_CACHE: Dict[Tuple[int, str, int, bool, str], torch.Tensor] = {}


def clear_rotation_cache() -> None:
    """Drop cached rotation matrices (they are shared across same-width Linears)."""
    _ROT_CACHE.clear()


def _sylvester_hadamard(k: int, dtype: torch.dtype) -> torch.Tensor:
    """Unnormalized +-1 Sylvester Hadamard of order 2^k (exact in any float dtype)."""
    h = torch.ones(1, 1, dtype=dtype)
    for _ in range(int(k)):
        top = torch.cat([h, h], dim=1)
        bot = torch.cat([h, -h], dim=1)
        h = torch.cat([top, bot], dim=0)
    return h


def _dht_block(m: int) -> torch.Tensor:
    """Normalized discrete Hartley transform of order m: real, symmetric, involutory.

    ``H[j,k] = cas(2*pi*j*k/m)/sqrt(m)`` with ``cas = cos + sin``.  ``H @ H = I`` so
    ``H`` is orthogonal, and every entry has magnitude <= sqrt(2/m) -- within a
    factor sqrt(2) of the Hadamard incoherence bound 1/sqrt(m).  Defined for every
    m (unlike a true Hadamard matrix), which is what makes the Kronecker
    construction total over the SDXL widths.
    """
    j = torch.arange(m, dtype=torch.float64)
    ang = torch.outer(j, j) * (2.0 * math.pi / m)
    h = (torch.cos(ang) + torch.sin(ang)) / math.sqrt(m)
    return h


def _haar_orthogonal(n: int, seed: int) -> torch.Tensor:
    """Haar-distributed random orthogonal matrix via sign-corrected QR (float64)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    a = torch.randn(n, n, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    d = torch.sign(torch.diagonal(r))
    d = torch.where(d == 0, torch.ones_like(d), d)
    return q * d.unsqueeze(0)


def rotation_plan(n: int, *, max_odd_block: int = 4096) -> Dict[str, Any]:
    """Describe how ``build_rotation_matrix(n, kind='hadamard')`` will factor ``n``.

    ``n = 2^k * m`` with ``m`` odd.  Constructible as a Kronecker product iff
    ``m <= max_odd_block``; otherwise the builder falls back to a random
    orthogonal matrix (``fallback=True``).
    """
    m, k = int(n), 0
    while m % 2 == 0 and m > 1:
        m //= 2
        k += 1
    ok = m <= int(max_odd_block)
    return {
        "n": int(n),
        "pow2": 1 << k,
        "k": k,
        "odd": m,
        "constructible": bool(ok),
        "fallback": not ok,
        "structure": (f"sylvester(2^{k})" if m == 1 else f"sylvester(2^{k}) (x) dht({m})")
        if ok
        else "haar_orthogonal(qr)",
    }


def build_rotation_matrix(
    n: int,
    *,
    kind: str = "hadamard",
    seed: int = 0,
    random_sign: bool = True,
    dtype: torch.dtype = torch.float32,
    max_odd_block: int = 4096,
    cache: bool = True,
) -> torch.Tensor:
    """Return an orthogonal ``R`` of shape ``[n, n]`` on CPU.

    kind="hadamard"   ``R = D @ (H_{2^k} (x) B_m)`` where ``H`` is Sylvester,
        ``B`` the normalized DHT block for the odd cofactor, and ``D`` a random
        +-1 diagonal (``random_sign``) that makes it a *randomized* Hadamard
        transform -- without it the transform is deterministic and can align
        badly with structured activations.  Every entry of the Hadamard part has
        magnitude <= sqrt(2/n), which is what bounds the post-rotation absmax.
        Falls back to ``kind="orthogonal"`` if the odd cofactor exceeds
        ``max_odd_block``.
    kind="orthogonal" Haar random orthogonal via QR. Exact but dense and
        O(n^3) in float64 to build -- slow past n~2048; prefer "hadamard".

    Matrices are cached by ``(n, kind, seed, random_sign, dtype)``, so all Linears
    of the same width share one ``R`` (188 MB total in fp32 for the seven SDXL
    widths, versus ~100 GB if each of the 743 layers had its own).
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    kind = str(kind).lower()
    if kind not in ("hadamard", "orthogonal"):
        raise ValueError(f"kind must be 'hadamard' or 'orthogonal', got {kind!r}")

    key = (n, kind, int(seed), bool(random_sign), str(dtype))
    if cache and key in _ROT_CACHE:
        return _ROT_CACHE[key]

    if kind == "orthogonal":
        r = _haar_orthogonal(n, seed).to(dtype)
    else:
        plan = rotation_plan(n, max_odd_block=max_odd_block)
        if plan["fallback"]:
            r = _haar_orthogonal(n, seed).to(dtype)
        else:
            k, m = plan["k"], plan["odd"]
            # build the +-1 part in the target dtype (entries are exact), scale once
            h = _sylvester_hadamard(k, dtype)
            if m == 1:
                r = h * (1.0 / math.sqrt(1 << k))
            else:
                b = _dht_block(m).to(dtype)
                r = torch.kron(h, b) * (1.0 / math.sqrt(1 << k))
            if random_sign:
                g = torch.Generator(device="cpu").manual_seed(int(seed) * 1000003 + n)
                d = torch.randint(0, 2, (n,), generator=g, dtype=torch.int64) * 2 - 1
                r = r * d.to(dtype).unsqueeze(1)  # R <- diag(d) @ R, still orthogonal
    r = r.contiguous()
    if cache:
        _ROT_CACHE[key] = r
    return r


# Global device-side cache of rotation matrices: _ROT_CACHE shares the CPU matrices by
# size, this shares the GPU copies by data_ptr.  One copy per hook would cost ~6 GiB per
# variant over the 743 SDXL Linears.
_ROT_DEV_CACHE: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def clear_rotation_device_cache() -> None:
    _ROT_DEV_CACHE.clear()


class _RotationHook:
    """forward_pre_hook: ``x -> x @ R`` (GPU copies are shared per matrix, not per layer)."""

    __slots__ = ("rot",)

    def __init__(self, rot: torch.Tensor):
        self.rot = rot

    def _get(self, x: torch.Tensor) -> torch.Tensor:
        key = (self.rot.data_ptr(), x.device, x.dtype)
        t = _ROT_DEV_CACHE.get(key)
        if t is None:
            t = self.rot.to(device=x.device, dtype=x.dtype).contiguous()
            _ROT_DEV_CACHE[key] = t
        return t

    def __call__(self, module, args):
        x = _first_float_tensor(args)
        if x is None:
            return None
        return (x @ self._get(x),) + tuple(args[1:])


def apply_rotation_(
    model: nn.Module,
    seed: int = 0,
    kind: str = "hadamard",
    *,
    predicate: Optional[Callable[[str, nn.Linear], bool]] = None,
    prepend: bool = False,
    random_sign: bool = True,
    compute_dtype: torch.dtype = torch.float32,
    rot_dtype: torch.dtype = torch.float32,
    max_odd_block: int = 4096,
    allow_restack: bool = False,
) -> None:
    """In-place orthogonal rotation of every targeted Linear. Data-free, exact.

    Rewrites ``W <- W @ R`` (computed in ``compute_dtype``, TF32 forced off) and
    registers a pre-hook doing ``x <- x @ R``.  Bias is untouched: the transform
    only reparameterizes the input space.

    seed / kind / random_sign / max_odd_block -> ``build_rotation_matrix``.
    compute_dtype  precision of the one-off ``W @ R``. float32 is enough for the
        1e-4 exactness contract; float64 is available but ~30x slower on Ampere.
    rot_dtype      storage dtype of the cached ``R`` (fp32 recommended even for an
        fp16 model -- the hook casts per call site and caches the cast).
    prepend        put the rotation hook first (needed if QDQ is already attached).

    Returns None.
    """
    with torch.no_grad(), _exact_matmul():
        for name, mod in _iter_linears(model, predicate):
            if _get_state(mod) is not None and not allow_restack:
                raise RuntimeError(
                    f"Linear {name!r} already carries transform "
                    f"{_get_state(mod)['kind']!r}; pass allow_restack=True to stack."
                )
            w = mod.weight.data
            in_f = int(w.shape[1])
            r = build_rotation_matrix(
                in_f,
                kind=kind,
                seed=seed,
                random_sign=random_sign,
                dtype=rot_dtype,
                max_odd_block=max_odd_block,
            )
            r_dev = r.to(device=w.device, dtype=compute_dtype)
            mod.weight.data.copy_((w.to(compute_dtype) @ r_dev).to(w.dtype))
            handle = mod.register_forward_pre_hook(_RotationHook(r), prepend=prepend)
            setattr(mod, _HANDLE_ATTR, handle)
            plan = rotation_plan(in_f, max_odd_block=max_odd_block)
            _mark(
                mod,
                {
                    "kind": "rotation",
                    "rotation_kind": "orthogonal" if (kind == "orthogonal" or plan["fallback"]) else "hadamard",
                    "in_features": in_f,
                    "seed": int(seed),
                    "random_sign": bool(random_sign),
                    "structure": plan["structure"] if kind == "hadamard" else "haar_orthogonal(qr)",
                    "_tensor": r,  # shared, kept for exact removal
                },
            )


# =============================================================================
# undo
# =============================================================================
def remove_act_transforms_(
    model: nn.Module,
    *,
    restore_weights: bool = True,
    compute_dtype: torch.dtype = torch.float32,
) -> int:
    """Remove transform hooks and (optionally) invert the weight rewrite in place.

    Inverse is exact in exact arithmetic (``W / s`` and ``W @ R^T``); in float the
    round trip leaves ~1e-6 relative error, and it cannot recover anything if the
    weights were quantized in between.  Returns the number of Linears restored.
    """
    n = 0
    with torch.no_grad(), _exact_matmul():
        for _, mod in _iter_linears(model):
            st = _get_state(mod)
            if st is None:
                continue
            handle = getattr(mod, _HANDLE_ATTR, None)
            if handle is not None:
                handle.remove()
                delattr(mod, _HANDLE_ATTR)
            if restore_weights:
                w = mod.weight.data
                t = st["_tensor"]
                if st["kind"] == "smoothquant":
                    s = t.to(device=w.device, dtype=torch.float64)
                    mod.weight.data.copy_((w.to(torch.float64) / s.unsqueeze(0)).to(w.dtype))
                elif st["kind"] == "rotation":
                    r = t.to(device=w.device, dtype=compute_dtype)
                    mod.weight.data.copy_((w.to(compute_dtype) @ r.t()).to(w.dtype))
            delattr(mod, _STATE_ATTR)
            n += 1
    return n


# =============================================================================
# composition with the existing per-token QDQ
# =============================================================================
class PerTokenActFakeQuant:
    """Standalone twin of ``quant_unet._ActFakeQuant`` (per-token dynamic symmetric QDQ).

    Kept local so this module has no hard dependency on quant_unet; ``build_act_quant_model``
    prefers the real class when it can import it, so the two never drift.
    """

    __slots__ = ("qmax",)

    def __init__(self, a_bits: int):
        self.qmax = (1 << (int(a_bits) - 1)) - 1

    def __call__(self, module, args):
        x = _first_float_tensor(args)
        if x is None:
            return None
        xf = x.float()
        scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / self.qmax
        xq = ((xf / scale).round().clamp(-self.qmax, self.qmax) * scale).to(x.dtype)
        return (xq,) + tuple(args[1:])


def _local_fake_quant_weight_(linear: nn.Linear, w_bits: int, group_size: int = 128) -> None:
    """Twin of ``quant_unet._fake_quant_weight_`` (per-channel W>=8, group-128 W<=4)."""
    w = linear.weight.data
    qmax = (1 << (int(w_bits) - 1)) - 1
    out_f, in_f = w.shape
    g = in_f if int(w_bits) >= 8 else (group_size if in_f % group_size == 0 else in_f)
    wg = w.float().view(out_f, in_f // g, g)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    wq = (wg / scale).round().clamp(-qmax, qmax) * scale
    linear.weight.data.copy_(wq.view(out_f, in_f).to(w.dtype))


def _resolve_backends(prefer_project: bool = True):
    """Return ``(act_quant_cls, weight_quant_fn)``, preferring quant_unet's own."""
    if prefer_project:
        try:  # package-relative; fails when this file runs as __main__
            from .quant_unet import _ActFakeQuant, _fake_quant_weight_  # type: ignore

            return _ActFakeQuant, _fake_quant_weight_
        except Exception:
            pass
    return PerTokenActFakeQuant, _local_fake_quant_weight_


def build_act_quant_model(
    base_model: nn.Module,
    method: str = "rotation",
    a_bits: int = 8,
    *,
    w_bits: int = 16,
    stats: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.5,
    seed: int = 0,
    kind: str = "hadamard",
    random_sign: bool = True,
    predicate: Optional[Callable[[str, nn.Linear], bool]] = None,
    inplace: bool = False,
    prefer_project_backends: bool = True,
) -> nn.Module:
    """Build a simulated-quantization model with an activation-side transform in front.

    This is the composition helper: it fixes the **hook order**, which is the only
    subtle part of combining these transforms with the existing QDQ.
    ``nn.Module`` runs ``forward_pre_hooks`` in registration order, so the pipeline
    per Linear is::

        x --[hook 1: x/s  or  x@R]--> x' --[hook 2: per-token QDQ]--> x'_q --> Linear(W')

    i.e. **transform first, quantize second**.  Quantizing before the transform
    would defeat the point entirely (the outliers would already have been clipped
    into the grid).  Weight quantization likewise happens *after* the transform,
    because the transform rewrites ``W``.

    method
        ``"none"``          plain per-token QDQ (the current baseline)
        ``"smoothquant"``   requires ``stats`` from ``collect_act_channel_absmax``
        ``"rotation"``      data-free; ``kind`` in {"hadamard", "orthogonal"}
        ``"rotation+smoothquant"``  rotate, then smooth in the rotated basis.
            ``stats`` must then have been collected *after* rotation (rotation
            changes the input basis, so pre-rotation channel statistics are
            meaningless); stacking is enabled via ``allow_restack``.
    a_bits / w_bits
        ``>= 16`` disables that side.  Applied to every Linear passing ``predicate``.
    inplace
        ``False`` (default) deep-copies ``base_model`` first, matching
        ``quant_unet.build_sim_variant``'s contract.

    Returns the model.  It carries ``.act_quant_spec`` (e.g. ``"rotation-hadamard_w16a4_sim"``)
    for census/logging, mirroring ``sim_variant_spec``.

    Manual equivalent, if you would rather drive it yourself::

        stats = collect_act_channel_absmax(m, run_calib)     # (A) only
        apply_smoothquant_(m, stats, alpha=0.5)              # or apply_rotation_(m, kind="hadamard")
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                _fake_quant_weight_(mod, w_bits)             # after the transform
                mod.register_forward_pre_hook(_ActFakeQuant(a_bits))   # after the transform hook

    If a QDQ hook is somehow already attached, pass ``prepend=True`` to
    ``apply_smoothquant_`` / ``apply_rotation_`` to force the transform to the front.
    """
    import copy as _copy

    method = str(method).lower().replace(" ", "")
    valid = ("none", "smoothquant", "rotation", "rotation+smoothquant")
    if method not in valid:
        raise ValueError(f"method must be one of {valid}, got {method!r}")
    if "smoothquant" in method and stats is None:
        raise ValueError("method requires `stats` from collect_act_channel_absmax(...)")

    act_cls, wq_fn = _resolve_backends(prefer_project_backends)
    m = base_model if inplace else _copy.deepcopy(base_model)

    # 1) exact input-space transform (rewrites W, registers transform hook)
    if method.startswith("rotation"):
        apply_rotation_(m, seed=seed, kind=kind, random_sign=random_sign, predicate=predicate)
    if "smoothquant" in method:
        apply_smoothquant_(
            m, stats or {}, alpha=alpha, predicate=predicate,
            allow_restack=method.startswith("rotation"),
        )

    # 2) weight QDQ on the transformed weights, then 3) activation QDQ hook (runs last)
    n_lin = 0
    for _, mod in _iter_linears(m, predicate):
        if int(w_bits) < 16:
            wq_fn(mod, int(w_bits))
        if int(a_bits) < 16:
            mod.register_forward_pre_hook(act_cls(int(a_bits)))
        n_lin += 1

    tag = method if method != "rotation" else f"rotation-{kind}"
    m.act_quant_spec = f"{tag}_w{w_bits}a{a_bits}_sim"
    m.act_quant_n_linear = n_lin
    m.eval()
    return m


# =============================================================================
# unit tests  (CPU only, no project imports)
# =============================================================================
if __name__ == "__main__":
    import sys
    import traceback

    torch.manual_seed(0)
    _FAILURES = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    def run(fn):
        name = fn.__name__
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            _FAILURES.append(name)
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()
        return fn

    SDXL_WIDTHS = (320, 640, 1280, 2048, 2560, 2816, 5120)

    def make_outlier_input(tokens: int, ch: int, n_out: int = 3, mag: float = 50.0, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(tokens, ch, generator=g)
        idx = torch.randperm(ch, generator=g)[:n_out]
        x[:, idx] *= mag
        return x, idx

    def make_model(dims=(64, 128, 32), seed: int = 0):
        torch.manual_seed(seed)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        return nn.Sequential(*layers).eval()

    def rel_err(a, b):
        return float((a - b).norm() / b.norm().clamp_min(1e-12))

    # -- rotation matrix properties -----------------------------------------
    @run
    def test_rotation_matrix_orthogonality():
        for n in SDXL_WIDTHS:
            plan = rotation_plan(n)
            check(plan["constructible"], f"n={n} not constructible: {plan}")
            r = build_rotation_matrix(n, kind="hadamard", seed=0)
            if n <= 1280:  # full R^T R for the small widths
                err = (r.t() @ r - torch.eye(n)).abs().max().item()
                check(err < 1e-4, f"n={n} orthogonality err {err:.2e}")
            # norm preservation probe for every width (cheap)
            v = torch.randn(8, n)
            d = ((v @ r).norm(dim=1) - v.norm(dim=1)).abs().max().item()
            coh = r.abs().max().item() * math.sqrt(n)
            print(f"        n={n:5d} {plan['structure']:<28s} |norm diff|={d:.2e} coherence*sqrt(n)={coh:.3f}")
            check(d < 1e-3, f"n={n} norm not preserved: {d:.2e}")
            check(coh < 1.5, f"n={n} coherence too high: {coh:.3f}")
        clear_rotation_cache()

    @run
    def test_rotation_fallback_and_haar():
        # force the odd cofactor to be rejected -> QR fallback path
        plan = rotation_plan(320, max_odd_block=1)
        check(plan["fallback"], f"expected fallback, got {plan}")
        r = build_rotation_matrix(320, kind="hadamard", seed=1, max_odd_block=1, cache=False)
        err = (r.t() @ r - torch.eye(320)).abs().max().item()
        check(err < 1e-5, f"fallback R not orthogonal: {err:.2e}")
        r2 = build_rotation_matrix(256, kind="orthogonal", seed=3, cache=False)
        err2 = (r2.t() @ r2 - torch.eye(256)).abs().max().item()
        check(err2 < 1e-5, f"haar R not orthogonal: {err2:.2e}")
        print(f"        fallback orth err={err:.2e}  haar orth err={err2:.2e}")

    # -- (A) SmoothQuant ------------------------------------------------------
    @run
    def test_smoothquant_exactness():
        m = make_model((64, 128, 32), seed=1)
        x, _ = make_outlier_input(16, 64, seed=2)
        with torch.no_grad():
            y_ref = m(x).clone()
        stats = collect_act_channel_absmax(m, lambda mm: mm(x), return_meta=True)
        meta = stats.pop("__meta__")
        check(len(stats) == 2, f"expected 2 calibrated Linears, got {sorted(stats)}")
        check(all(v["n_calls"] == 1 and v["n_tokens"] == 16 for v in meta.values()), f"bad meta {meta}")
        w_before = m[0].weight.data.clone()
        apply_smoothquant_(m, stats, alpha=0.5)
        with torch.no_grad():
            y = m(x)
        d = (y - y_ref).abs().max().item()
        check(not torch.allclose(w_before, m[0].weight.data), "weights unchanged -> transform is a no-op")
        rep = act_transform_report(m)
        print(f"        max|dy|={d:.3e}  rel={rel_err(y, y_ref):.3e}  "
              f"s range=[{rep['0']['scale_min']:.3f}, {rep['0']['scale_max']:.3f}]")
        check(d < 1e-4, f"SmoothQuant not equivalent: max abs diff {d:.3e}")

    @run
    def test_smoothquant_flattens_channels():
        m = make_model((320, 256, 64), seed=3)
        x, idx = make_outlier_input(64, 320, n_out=6, mag=80.0, seed=4)
        before = act_range_stats(x)
        stats = collect_act_channel_absmax(m, lambda mm: mm(x))
        apply_smoothquant_(m, stats, alpha=0.5)
        s = m[0]._pq_act_transform["_tensor"]
        after = act_range_stats(x / s)
        print(f"        channel_absmax_ratio {before['channel_absmax_ratio']:.1f} -> "
              f"{after['channel_absmax_ratio']:.1f} | absmax/mean|x| "
              f"{before['absmax_over_meanabs']:.1f} -> {after['absmax_over_meanabs']:.1f}")
        check(after["channel_absmax_ratio"] < 0.5 * before["channel_absmax_ratio"],
              "SmoothQuant did not flatten the channel profile")
        check(s[idx].min() > 1.0, "outlier channels should get scale > 1")

    @run
    def test_smoothquant_guards():
        m = make_model((8, 4), seed=5)
        with torch.no_grad():
            m[0].weight.data[:, 2] = 0.0  # dead weight column
        x = torch.randn(5, 8)
        x[:, 3] = 0.0  # dead activation channel
        with torch.no_grad():
            y_ref = m(x).clone()
        stats = collect_act_channel_absmax(m, lambda mm: mm(x))
        apply_smoothquant_(m, stats, alpha=0.9)
        s = m[0]._pq_act_transform["_tensor"]
        check(float(s[2]) == 1.0 and float(s[3]) == 1.0, f"dead channels must get s=1, got {s[2]}, {s[3]}")
        with torch.no_grad():
            d = (m(x) - y_ref).abs().max().item()
        check(torch.isfinite(s).all(), "non-finite scale")
        check(d < 1e-4, f"exactness broken with dead channels: {d:.3e}")
        print(f"        dead-channel scales = 1.0 ok, max|dy|={d:.3e}")

    @run
    def test_smoothquant_alpha_sweep_and_strict():
        m = make_model((32, 16), seed=6)
        x, _ = make_outlier_input(8, 32, seed=7)
        stats = collect_act_channel_absmax(m, lambda mm: mm(x))
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            mm = make_model((32, 16), seed=6)
            with torch.no_grad():
                y_ref = mm(x).clone()
            apply_smoothquant_(mm, stats, alpha=a)
            with torch.no_grad():
                d = (mm(x) - y_ref).abs().max().item()
            check(d < 1e-4, f"alpha={a} broke exactness: {d:.3e}")
        try:
            apply_smoothquant_(make_model((32, 16), seed=6), {}, strict=True)
            raise AssertionError("strict=True should have raised on missing stats")
        except KeyError:
            pass
        apply_smoothquant_(m, stats)          # first application
        try:
            apply_smoothquant_(m, stats)      # second must be refused
            raise AssertionError("double application should have raised")
        except RuntimeError:
            pass
        print("        alpha in {0,.25,.5,.75,1} exact; strict + restack guards fire")

    # -- (B) Rotation ---------------------------------------------------------
    @run
    def test_rotation_exactness():
        for kind, dims in (("hadamard", (320, 128, 64)), ("hadamard", (2816, 64)), ("orthogonal", (256, 64))):
            m = make_model(dims, seed=8)
            x, _ = make_outlier_input(12, dims[0], seed=9)
            with torch.no_grad():
                y_ref = m(x).clone()
            w_before = m[0].weight.data.clone()
            apply_rotation_(m, seed=0, kind=kind)
            with torch.no_grad():
                y = m(x)
            d = (y - y_ref).abs().max().item()
            check(not torch.allclose(w_before, m[0].weight.data), "weights unchanged")
            print(f"        kind={kind:<10s} dims={dims} max|dy|={d:.3e} rel={rel_err(y, y_ref):.3e}")
            check(d < 1e-4, f"rotation({kind}, {dims}) not equivalent: {d:.3e}")
        clear_rotation_cache()

    @run
    def test_rotation_exactness_fp64():
        m = make_model((320, 64), seed=10).double()
        x = make_outlier_input(8, 320, seed=11)[0].double()
        with torch.no_grad():
            y_ref = m(x).clone()
        apply_rotation_(m, seed=0, kind="hadamard", compute_dtype=torch.float64, rot_dtype=torch.float64)
        with torch.no_grad():
            d = (m(x) - y_ref).abs().max().item()
        print(f"        float64 end-to-end max|dy|={d:.3e}")
        check(d < 1e-10, f"fp64 rotation should be near machine exact: {d:.3e}")
        clear_rotation_cache()

    @run
    def test_rotation_kills_outliers():
        for n in (320, 1280, 2816):
            x, _ = make_outlier_input(64, n, n_out=4, mag=60.0, seed=12)
            r = build_rotation_matrix(n, kind="hadamard", seed=0)
            before = act_range_stats(x)
            after = act_range_stats(x @ r)
            drop = before["absmax_over_meanabs"] / after["absmax_over_meanabs"]
            print(f"        n={n:5d} absmax/mean|x| {before['absmax_over_meanabs']:7.2f} -> "
                  f"{after['absmax_over_meanabs']:5.2f}  ({drop:5.1f}x)   "
                  f"absmax/rms {before['absmax_over_rms']:6.2f} -> {after['absmax_over_rms']:.2f}")
            check(after["absmax_over_meanabs"] < 0.35 * before["absmax_over_meanabs"],
                  f"n={n}: rotation failed to shrink per-token range")
            check(after["absmax_over_rms"] < 6.0, f"n={n}: post-rotation absmax/rms {after['absmax_over_rms']:.2f}")
        clear_rotation_cache()

    # -- undo -----------------------------------------------------------------
    @run
    def test_remove_restores_model():
        for setup in ("smoothquant", "rotation"):
            m = make_model((320, 64), seed=13)
            x, _ = make_outlier_input(8, 320, seed=14)
            with torch.no_grad():
                y_ref = m(x).clone()
            if setup == "smoothquant":
                apply_smoothquant_(m, collect_act_channel_absmax(m, lambda mm: mm(x)), alpha=0.6)
            else:
                apply_rotation_(m, seed=2, kind="hadamard")
            n = remove_act_transforms_(m)
            check(n == 1, f"expected 1 restored Linear, got {n}")
            check(len(m[0]._forward_pre_hooks) == 0, "hook not removed")
            check(act_transform_report(m) == {}, "state not cleared")
            with torch.no_grad():
                d = (m(x) - y_ref).abs().max().item()
            print(f"        {setup:<12s} round-trip max|dy|={d:.3e}")
            check(d < 1e-4, f"{setup} round trip drifted: {d:.3e}")
        clear_rotation_cache()

    # -- composition with QDQ -------------------------------------------------
    @run
    def test_hook_order_and_composition():
        m = make_model((320, 64), seed=15)
        apply_rotation_(m, seed=0, kind="hadamard")
        m[0].register_forward_pre_hook(PerTokenActFakeQuant(4))
        hooks = list(m[0]._forward_pre_hooks.values())
        check(len(hooks) == 2, f"expected 2 pre-hooks, got {len(hooks)}")
        check(isinstance(hooks[0], _RotationHook), "rotation hook must run first")
        check(isinstance(hooks[1], PerTokenActFakeQuant), "QDQ hook must run second")
        # prepend=True puts the transform in front of an already-registered QDQ
        m2 = make_model((320, 64), seed=15)
        m2[0].register_forward_pre_hook(PerTokenActFakeQuant(4))
        apply_rotation_(m2, seed=0, kind="hadamard", prepend=True)
        h2 = list(m2[0]._forward_pre_hooks.values())
        check(isinstance(h2[0], _RotationHook), "prepend=True did not move the transform to the front")
        print("        order: [transform, QDQ] ok; prepend=True ok")
        clear_rotation_cache()

    @run
    def test_quant_error_improves():
        """The point of the exercise: same A-bits, lower error after the transform.

        Two regimes, because they behave very differently and the difference is the
        whole story:
          "outlier-dominant" -- the outlier channels also dominate the *output*, so a
              naive quantizer that crushes every other channel to zero still keeps most
              of the signal and looks deceptively fine.
          "outlier-discarded" -- the weight columns of the outlier channels are small
              (SmoothQuant's actual observation about transformer activations), so the
              output depends on the channels the naive quantizer destroys. This is the
              regime where activation quantization actually breaks.
        """
        import copy as _copy

        n, tokens, mag = 320, 128, 60.0
        x, idx = make_outlier_input(tokens, n, n_out=4, mag=mag, seed=16)

        def make_base(discard_outliers: bool):
            lin = nn.Linear(n, 256)
            torch.manual_seed(17)
            nn.init.uniform_(lin.weight, -1 / math.sqrt(n), 1 / math.sqrt(n))
            nn.init.zeros_(lin.bias)
            if discard_outliers:
                with torch.no_grad():
                    lin.weight.data[:, idx] /= mag
            return lin.eval()

        for regime, discard in (("outlier-dominant", False), ("outlier-discarded", True)):
            base = make_base(discard)
            with torch.no_grad():
                y_ref = base(x).clone()
            frac = float((x[:, idx] @ base.weight.data[:, idx].t()).norm() / y_ref.norm())
            stats = collect_act_channel_absmax(base, lambda mm: mm(x))
            rot = _copy.deepcopy(base)
            apply_rotation_(rot, seed=0, kind="hadamard")
            stats_rot = collect_act_channel_absmax(rot, lambda mm: mm(x))

            print(f"        [{regime}] outlier channels carry {frac:.1%} of the output energy")
            print(f"        {'a_bits':>6s} {'baseline':>9s} {'smooth.5':>9s} {'rot-had':>9s} {'rot+smooth':>11s}")
            for a_bits in (8, 6, 4):
                errs = {}
                for method in ("none", "smoothquant", "rotation", "rotation+smoothquant"):
                    st = stats_rot if method == "rotation+smoothquant" else stats
                    mm = build_act_quant_model(
                        _copy.deepcopy(base), method, a_bits, stats=st, alpha=0.5,
                        seed=0, kind="hadamard", prefer_project_backends=False, inplace=True,
                    )
                    with torch.no_grad():
                        errs[method] = rel_err(mm(x), y_ref)
                print(f"        {a_bits:6d} {errs['none']:9.4f} {errs['smoothquant']:9.4f} "
                      f"{errs['rotation']:9.4f} {errs['rotation+smoothquant']:11.4f}")
                for method in ("smoothquant", "rotation", "rotation+smoothquant"):
                    check(errs[method] < 0.8 * errs["none"],
                          f"[{regime}] a{a_bits}: {method} {errs[method]:.4f} "
                          f"did not beat baseline {errs['none']:.4f} by 20%")
                if discard:
                    # baseline is catastrophic here (rel err ~1 == signal destroyed);
                    # SmoothQuant must actually rescue it, not merely improve on it.
                    check(errs["smoothquant"] < 0.25,
                          f"[{regime}] a{a_bits}: smoothquant {errs['smoothquant']:.4f} did not rescue")
            clear_rotation_cache()

    @run
    def test_transforms_attack_different_statistics():
        """Why the two methods are complementary -- and where rotation alone cannot help.

        Rotation is orthogonal, so it preserves ||x|| exactly: it can only fix the
        peak-to-RMS ratio (absmax/rms), never the absolute magnitude. A per-token
        absmax quantizer's noise floor is ~absmax/2^(b-1), so once absmax/rms hits the
        Gaussian floor (~sqrt(2 ln n)) rotation has nothing left to give.
        SmoothQuant instead moves energy out of x and into W, shrinking rms(x) itself,
        which is the only lever when the outlier channels' energy is discarded by W.
        """
        n = 320
        x, idx = make_outlier_input(128, n, n_out=4, mag=60.0, seed=16)
        lin = nn.Linear(n, 256)
        torch.manual_seed(17)
        nn.init.uniform_(lin.weight, -1 / math.sqrt(n), 1 / math.sqrt(n))
        with torch.no_grad():
            lin.weight.data[:, idx] /= 60.0
        lin.eval()

        r = build_rotation_matrix(n, kind="hadamard", seed=0)
        m = nn.Sequential(lin)
        stats = collect_act_channel_absmax(m, lambda mm: mm(x))
        apply_smoothquant_(m, stats, alpha=0.5)
        s = m[0]._pq_act_transform["_tensor"]

        rms0 = float(x.pow(2).mean().sqrt())
        rms_rot = float((x @ r).pow(2).mean().sqrt())
        rms_sm = float((x / s).pow(2).mean().sqrt())
        st0, st_rot, st_sm = act_range_stats(x), act_range_stats(x @ r), act_range_stats(x / s)
        print(f"        {'':<12s} {'rms(x)':>9s} {'absmax/rms':>11s} {'absmax/mean|x|':>15s}")
        for tag, rms, st in (("identity", rms0, st0), ("rotation", rms_rot, st_rot), ("smoothquant", rms_sm, st_sm)):
            print(f"        {tag:<12s} {rms:9.3f} {st['absmax_over_rms']:11.2f} {st['absmax_over_meanabs']:15.2f}")

        check(abs(rms_rot - rms0) / rms0 < 1e-4, f"rotation must preserve rms: {rms0} -> {rms_rot}")
        check(rms_sm < 0.1 * rms0, f"smoothquant must shrink rms: {rms0} -> {rms_sm}")
        check(st_rot["absmax_over_rms"] < 0.3 * st0["absmax_over_rms"],
              "rotation must shrink peak-to-rms")
        gauss_floor = math.sqrt(2 * math.log(n))
        check(st_rot["absmax_over_rms"] < 2.0 * gauss_floor,
              f"post-rotation absmax/rms {st_rot['absmax_over_rms']:.2f} vs Gaussian floor {gauss_floor:.2f}")
        clear_rotation_cache()

    @run
    def test_build_act_quant_model_api():
        base = make_model((320, 64), seed=18)
        w_ref = base[0].weight.data.clone()
        m = build_act_quant_model(base, "rotation", a_bits=4, w_bits=4, kind="hadamard")
        check(torch.allclose(base[0].weight.data, w_ref), "inplace=False must not touch the base model")
        check(m.act_quant_spec == "rotation-hadamard_w4a4_sim", f"bad spec {m.act_quant_spec}")
        check(m.act_quant_n_linear == 1, f"bad linear count {m.act_quant_n_linear}")
        check(len(m[0]._forward_pre_hooks) == 2, "expected transform + QDQ hooks")
        # weight really was quantized, and after the rotation
        wq = m[0].weight.data
        check(len(torch.unique(wq[0])) <= 16, f"W4 row should have <=16 levels, got {len(torch.unique(wq[0]))}")
        # predicate targeting
        m2 = build_act_quant_model(make_model((320, 64, 8), seed=18), "rotation", 8,
                                   predicate=lambda nm, mod: nm == "0")
        check(m2.act_quant_n_linear == 1, f"predicate ignored: {m2.act_quant_n_linear}")
        check(len(m2[2]._forward_pre_hooks) == 0, "predicate leaked to excluded layer")
        # method validation
        for bad in ("smoothquant",):
            try:
                build_act_quant_model(make_model((32, 8)), bad, 8)
                raise AssertionError("missing stats should raise")
            except ValueError:
                pass
        print(f"        spec={m.act_quant_spec!r}, deepcopy/predicate/validation ok")
        clear_rotation_cache()

    @run
    def test_stacked_rotation_then_smooth():
        n = 320
        x, _ = make_outlier_input(64, n, n_out=4, mag=60.0, seed=19)
        m = make_model((n, 128), seed=20)
        with torch.no_grad():
            y_ref = m(x).clone()
        apply_rotation_(m, seed=0, kind="hadamard")
        stats = collect_act_channel_absmax(m, lambda mm: mm(x))  # stats in the ROTATED basis
        apply_smoothquant_(m, stats, alpha=0.5, allow_restack=True)
        with torch.no_grad():
            d = (m(x) - y_ref).abs().max().item()
        check(len(m[0]._forward_pre_hooks) == 2, "expected rotation + smooth hooks")
        print(f"        rotation+smoothquant max|dy|={d:.3e}")
        check(d < 1e-4, f"stacked transform not equivalent: {d:.3e}")
        clear_rotation_cache()

    @run
    def test_nested_model_and_non_float_args():
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(64, 64)
                self.b = nn.Linear(64, 32)

            def forward(self, x):
                return self.b(torch.relu(self.a(x)))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block(), Block()])
                self.head = nn.Linear(32, 8)

            def forward(self, x):
                for blk in self.blocks:
                    x = blk(x) if x.shape[-1] == 64 else x
                return self.head(x)

        torch.manual_seed(21)
        net = Net().eval()
        x = torch.randn(2, 5, 64)  # 3-D input: absmax must reduce over dims 0 and 1
        with torch.no_grad():
            y_ref = net(x).clone()
        stats = collect_act_channel_absmax(net, lambda mm: mm(x))
        check("blocks.0.a" in stats and stats["blocks.0.a"].shape == (64,),
              f"bad stat shape/keys: {sorted(stats)}")
        check("blocks.1.a" not in stats, "layer never called must be absent from stats")
        apply_smoothquant_(net, stats, alpha=0.5)
        with torch.no_grad():
            d = (net(x) - y_ref).abs().max().item()
        check(d < 1e-4, f"nested/3-D exactness broken: {d:.3e}")
        n_applied = len(act_transform_report(net))
        print(f"        nested 3-D input ok, max|dy|={d:.3e}, {n_applied}/5 Linears transformed (2 uncalibrated skipped)")
        check(n_applied == 3, f"expected 3 transformed Linears, got {n_applied}")

    print()
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)} test(s): {_FAILURES}")
        sys.exit(1)
    print("ALL TESTS PASSED")
