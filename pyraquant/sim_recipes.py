"""Quantization recipes: compose an activation-side transform (SmoothQuant / rotation)
with a weight-side quantizer (RTN / nested / AWQ / GPTQ) into one simulated variant.
The recipe name appears in the variant name: w{W}a{A}_{recipe}_sim.

The order is fixed: transform -> weight quantization -> activation QDQ hook.
A transform rewrites W and changes the input basis the Linear sees, so calibration
statistics must be transformed alongside.
"""
from __future__ import annotations

import copy
import re
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .quantizers_act import _RotationHook, _SmoothScaleHook, apply_rotation_, apply_smoothquant_
from .quantizers_weight import quantize_weights_

# recipe -> (activation transform, weight quantizer)
RECIPES: Dict[str, Dict[str, Optional[str]]] = {
    "rtn":     {"act": None,                    "w": "rtn"},
    "nested3": {"act": None,                    "w": "nested3"},   # W3 derived from the W4 codes (one stored weight)
    "nested4": {"act": None,                    "w": "nested4"},   # W4 derived from the W8 codes (one stored weight)
    "sq":      {"act": "smoothquant",           "w": "rtn"},
    "rot":     {"act": "rotation",              "w": "rtn"},
    "sqrot":   {"act": "rotation+smoothquant",  "w": "rtn"},
    "awq":     {"act": None,                    "w": "awq"},
    "gptq":    {"act": None,                    "w": "gptq"},
    "sqawq":   {"act": "smoothquant",           "w": "awq"},
    "sqgptq":  {"act": "smoothquant",           "w": "gptq"},
    "rotawq":  {"act": "rotation",              "w": "awq"},
    "rotgptq": {"act": "rotation",              "w": "gptq"},
}


def _linear_hooks(model: nn.Module):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            for h in mod._forward_pre_hooks.values():
                yield name, mod, h


def _transform_calib(model: nn.Module, stats: Optional[Dict[str, torch.Tensor]],
                     X: Optional[Dict[str, torch.Tensor]]):
    """Move calibration statistics into the transformed input basis (otherwise AWQ/GPTQ
    would calibrate against the wrong statistics)."""
    if stats is None and X is None:
        return stats, X
    stats = dict(stats) if stats else None
    X = dict(X) if X else None
    for name, _mod, h in _linear_hooks(model):
        if isinstance(h, _SmoothScaleHook):
            inv = h.inv_scale.float()
            if stats is not None and name in stats:
                stats[name] = stats[name].float() * inv
            if X is not None and name in X:
                X[name] = (X[name].float() * inv).to(X[name].dtype)
        elif isinstance(h, _RotationHook):
            R = h.rot.float().cpu()
            if X is not None and name in X:
                xr = X[name].float() @ R
                X[name] = xr.to(X[name].dtype)
                if stats is not None:
                    stats[name] = xr.abs().amax(dim=0)
            elif stats is not None and name in stats:
                # without token samples the statistic cannot be recomputed in the rotated basis
                stats.pop(name, None)
    return stats, X


def build_recipe_variant(
    base_unet: nn.Module,
    w_bits: int,
    a_bits: int,
    recipe: str = "rtn",
    calib: Optional[Dict[str, Any]] = None,
    *,
    alpha: float = 0.5,
    group_size: Optional[int] = 128,
    seed: int = 0,
    rotation_kind: str = "hadamard",
    weight_kwargs: Optional[Dict[str, Any]] = None,
    conv_bits: Optional[int] = None,
    conv_method: str = "rtn",
    conv_group_size: Optional[int] = 128,
    w_skip_re: str = "",
    a_skip_re: str = "",
) -> nn.Module:
    """Deep-copy ``base_unet`` and turn it into the simulated variant ``w{w_bits}a{a_bits}_{recipe}_sim``.

    ``w_skip_re`` / ``a_skip_re`` are optional regexes over module names: Linear layers
    matching ``w_skip_re`` keep their full-precision weights, those matching ``a_skip_re``
    get no activation-quantization hook.  Empty (default) means nothing is skipped.
    """
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; have {sorted(RECIPES)}")
    spec = RECIPES[recipe]
    t0 = time.perf_counter()
    m = copy.deepcopy(base_unet)
    calib = calib or {}
    stats = calib.get("act_absmax")
    X = calib.get("layer_inputs")

    # 1) activation-side transform (first: it rewrites W)
    act = spec["act"]
    if act in ("rotation", "rotation+smoothquant"):
        apply_rotation_(m, seed=seed, kind=rotation_kind)
        stats, X = _transform_calib(m, stats, X)
    if act in ("smoothquant", "rotation+smoothquant"):
        if not stats:
            raise ValueError(f"recipe {recipe!r} needs calibration statistics 'act_absmax' "
                             "(and 'layer_inputs' under a rotation recipe)")
        apply_smoothquant_(m, stats, alpha=alpha, allow_restack=(act == "rotation+smoothquant"))
        stats, X = _transform_calib(m, stats, X)

    # 2) weight quantization (after the transform)
    n_lin = sum(1 for mod in m.modules() if isinstance(mod, nn.Linear))
    if int(w_bits) < 16:
        quantize_weights_(m, spec["w"], int(w_bits), group_size,
                          layer_inputs=X, act_channel_absmax=stats, **(weight_kwargs or {}))

    # 2a) optional whitelist: Linear layers matching w_skip_re get their full-precision
    #     weights copied back from the base model.
    n_wskip = 0
    if w_skip_re and int(w_bits) < 16:
        _base = dict(base_unet.named_modules())
        for _n, _mod in m.named_modules():
            if isinstance(_mod, nn.Linear) and re.search(w_skip_re, _n):
                _mod.weight.data.copy_(_base[_n].weight.data)
                n_wskip += 1
        print(f"[sim_recipes] weight whitelist /{w_skip_re}/: {n_wskip} Linear kept at full precision", flush=True)
    # 2b) Conv2d weight quantization (optional).  Quantizing only the Linear layers leaves the
    #     Conv share of a UNet at fp16, so the true average bit-width would be higher than advertised.
    if conv_bits is not None and int(conv_bits) < 16:
        from .quantizers_conv import quantize_conv_weights_
        quantize_conv_weights_(m, conv_method, int(conv_bits), conv_group_size)

    # 3) activation QDQ (registered last: pre-hooks run in registration order)
    if int(a_bits) < 16:
        from .quant_unet import _ActFakeQuant, _ActFakeQuantConv
        n_conv_hooks = 0
        n_askip = 0
        for _n, mod in m.named_modules():
            if isinstance(mod, nn.Linear):
                if a_skip_re and re.search(a_skip_re, _n):
                    n_askip += 1; continue
                mod.register_forward_pre_hook(_ActFakeQuant(int(a_bits)))
            elif isinstance(mod, nn.Conv2d) and conv_bits is not None and int(conv_bits) < 16:
                # a Conv2d with quantized weights also gets its input quantized ("Conv W8A8" means both)
                mod.register_forward_pre_hook(_ActFakeQuantConv(int(a_bits))); n_conv_hooks += 1
        if n_conv_hooks:
            print(f"[sim_recipes] A{a_bits} hooks also on {n_conv_hooks} Conv2d (per-position over channels)", flush=True)
        if a_skip_re:
            print(f"[sim_recipes] activation whitelist /{a_skip_re}/: {n_askip} Linear left unquantized", flush=True)

    m.eval()
    m.sim_variant_spec = (f"w{w_bits}a{a_bits}_{recipe}_sim"
                          + (f"+conv{conv_bits}" if conv_bits and int(conv_bits) < 16 else "")
                          + (f"+wskip{n_wskip}" if w_skip_re else "") + (f"+askip" if a_skip_re else ""))
    torch.cuda.empty_cache()
    print(f"[sim_recipes] built {m.sim_variant_spec} over {n_lin} Linear "
          f"(act={act or 'none'}, w={spec['w']}, g={group_size}) in {time.perf_counter()-t0:.1f}s")
    return m
