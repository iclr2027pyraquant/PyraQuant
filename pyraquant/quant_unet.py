from __future__ import annotations

import copy
import re
import time
from typing import Any, Dict, Optional

import torch


# Simulated-variant naming: w{W}a{A}_sim = QDQ fake quant (fp16 storage AND
# compute, values rounded to the target grid). Quality-only — no speed/memory
# claims may ever cite a _sim variant.
_SIM_NAME_RE = re.compile(r"^w(\d+)a(\d+)(?:_([a-z0-9]+))?_sim$")


def _fake_quant_weight_(linear: torch.nn.Linear, w_bits: int, group_size: int = 128) -> None:
    """In-place quantize-dequantize of a Linear weight: group-wise symmetric absmax
    (group 128 along in_features, matching tinygemm's W4 convention; falls back to
    per-row when in_features isn't divisible)."""
    w = linear.weight.data
    qmax = (1 << (int(w_bits) - 1)) - 1
    out_f, in_f = w.shape
    # scale granularity mirrors the real kernels: per-channel (per out-row) for
    # W>=8 like torchao int8, group-128 for W<=4 like tinygemm (symmetric in both
    # cases — real int4 is asymmetric, so sim W4 reads slightly worse: conservative)
    if int(w_bits) >= 8:
        g = in_f
    else:
        g = group_size if in_f % group_size == 0 else in_f
    wg = w.float().view(out_f, in_f // g, g)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    wq = (wg / scale).round().clamp(-qmax, qmax) * scale
    linear.weight.data.copy_(wq.view(out_f, in_f).to(w.dtype))


class _ActFakeQuant:
    """forward_pre_hook: per-token (last dim) dynamic symmetric QDQ of the Linear
    input — same granularity torchao's int8 dynamic-activation path uses."""

    __slots__ = ("qmax",)

    def __init__(self, a_bits: int):
        self.qmax = (1 << (int(a_bits) - 1)) - 1

    def __call__(self, module, args):
        x = args[0]
        if not torch.is_tensor(x) or not x.is_floating_point():
            return None
        # fp32 math: in fp16 the clamp floor underflows to 0 -> NaN/inf on
        # all-zero / tiny token rows
        xf = x.float()
        scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / self.qmax
        xq = ((xf / scale).round().clamp(-self.qmax, self.qmax) * scale).to(x.dtype)
        return (xq,) + tuple(args[1:])


class _ActFakeQuantConv:
    """forward_pre_hook for nn.Conv2d: per-position (over channels) dynamic symmetric QDQ of the
    [B,C,H,W] input — the conv analogue of the per-token Linear hook (each pixel is a token)."""

    __slots__ = ("qmax",)

    def __init__(self, a_bits: int):
        self.qmax = (1 << (int(a_bits) - 1)) - 1

    def __call__(self, module, args):
        x = args[0]
        if not torch.is_tensor(x) or not x.is_floating_point() or x.ndim != 4:
            return None
        xf = x.float()
        scale = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / self.qmax
        xq = ((xf / scale).round().clamp(-self.qmax, self.qmax) * scale).to(x.dtype)
        return (xq,) + tuple(args[1:])


def build_sim_variant(base_unet: torch.nn.Module, w_bits: int, a_bits: int) -> torch.nn.Module:
    t0 = time.perf_counter()
    m = copy.deepcopy(base_unet)
    n_lin = 0
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear):
            if int(w_bits) < 16:
                _fake_quant_weight_(mod, int(w_bits))
            if int(a_bits) < 16:
                mod.register_forward_pre_hook(_ActFakeQuant(int(a_bits)))
            n_lin += 1
    m.eval()
    m.sim_variant_spec = f"w{w_bits}a{a_bits}_sim"  # census marker: weights stay Parameter
    torch.cuda.empty_cache()
    print(
        f"[quant_unet] built w{w_bits}a{a_bits}_sim (QDQ, fp16 storage) over {n_lin} Linear "
        f"in {time.perf_counter() - t0:.1f}s"
    )
    return m
