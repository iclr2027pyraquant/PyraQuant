"""Loader for an external PTQ package (used for the SVDQuant W4A4 base of PyraQuant on FLUX).

An external method calibrates/quantizes the original diffusers model with its own tooling and is
exported as a package directory <dir> with the following files:
  weights.safetensors   dequantized bf16/fp16 weights; keys = "<module>.weight" / "<module>.bias"
                        (only Linear/Conv modules that were quantized or rewritten)
  act_spec.json         per-module activation fake-quant spec (see ExtActQuant); optional
                        "smooth" / "lowrank_A" / "lowrank_B" fields reference aux tensors
  aux.safetensors       optional side tensors: smoothing vectors, low-rank branch A/B, per-group
                        weight scales ("<module>.wscale") used to derive the nested W3 variant
  weight_spec.json      optional per-module weight bits / group / scale bits (storage accounting;
                        defaults to meta w_bits / group), plus "wscale" and "code_range" keys
  meta.json             accounting: {"method", "host", "w_bits", "a_bits", "group",
                        "scale_dtype_bits": 16, "lowrank_rank": 0, "fp16_modules": [...],
                        "side_table_bits": 0, "split_modules": {...}, "notes"}
After loading, the ScaleDiff pipeline is untouched: same model class, attention processors,
step schedule, seeds and prompts. Storage bits per weight = w_bits + scale_bits/group
(+ low-rank + side tables), computed from meta/spec and written into each image record, both as
"quantized Linear only" and "whole backbone incl. fp16 whitelist" averages.
"""
from __future__ import annotations
import json, os
from typing import Any, Dict, List, Optional
import torch, torch.nn as nn


def _load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    return load_file(path) if os.path.exists(path) else {}


class ExtState:
    """Model-wide runtime registers: stage (0/1/2 = 1K/2K/4K), call index within the stage, and
    saturation counters. Maintained by a root-module pre-hook when meta "step_counter" is set."""

    def __init__(self):
        self.stage = 0; self.step = -1; self._last_key = None
        self.sat: Dict[str, List[float]] = {}          # "s{stage}/{name}" -> [clipped, total]
        self.n_calls: Dict[int, int] = {}

    def on_call(self, key: Any):
        """Called on every root forward; key identifies the stage (SDXL: latent width; FLUX: token count).
        A new key starts a new stage and resets the step counter."""
        if key != self._last_key:
            self._last_key = key
            self.stage = len(self.n_calls); self.n_calls[self.stage] = 0; self.step = -1
        self.step += 1; self.n_calls[self.stage] += 1

    def add_sat(self, name: str, clipped: int, total: int):
        k = f"s{self.stage}/{name}"
        v = self.sat.setdefault(k, [0.0, 0.0]); v[0] += float(clipped); v[1] += float(total)

    def sat_summary(self) -> Dict[str, float]:
        out: Dict[str, List[float]] = {}
        for k, (c, t) in self.sat.items():
            s = k.split("/")[0]; v = out.setdefault(s, [0.0, 0.0]); v[0] += c; v[1] += t
        return {s: (c / t if t else 0.0) for s, (c, t) in out.items()}


def _qrange(bits: int, unsigned: bool):
    return (0, (1 << bits) - 1) if unsigned else (-((1 << (bits - 1)) - 1), (1 << (bits - 1)) - 1)


_DYN_MODES = ("dyn_token", "dyn_group")


class ExtActQuant:
    """forward_pre_hook: activation fake-quant of an external method, processed in row chunks
    (at 4K the largest Linear input, [1, 65792, 15360], cannot be cast to fp32 as a whole).
    Spec fields:
      bits          bit width
      mode          dyn_token   per-token (last dim) dynamic absmax, symmetric
                    dyn_group   per-token, dynamic absmax over groups of `group` input channels (SVDQuant: 64)
      unsigned      True -> codes 0..2^b-1 (SVDQuant layers after GELU: absmax/15)
      shift         x <- x + shift before dividing by smooth (nunchaku kernel order; SVDQuant ShiftedLinear);
                    the bias is pre-compensated in the package, so it is not subtracted back (unshift=False)
      smooth        aux key: x <- x / s (the factor folded into the weight is handled by the exporter)
      lowrank_A/B   aux keys: 16-bit low-rank branch (SVDQuant); by default it consumes the smoothed,
                    unquantized input, h = x_s @ A computed per chunk inside this hook
      lowrank_raw   True -> the branch consumes the raw input (before shift/smoothing), as in the nunchaku
                    kernel where the released lora_down is already divided by smooth
      chunk_rows    rows per chunk (default 8192)
    Clipped-element fractions are accumulated into ExtState per call."""

    def __init__(self, name: str, sp: Dict[str, Any], aux: Dict[str, torch.Tensor], state: ExtState):
        self.name, self.state = name, state
        self.bits = int(sp["bits"]); self.mode = sp["mode"]; self.group = int(sp.get("group") or 0)
        self.unsigned = bool(sp.get("unsigned", False)); self.shift = float(sp.get("shift") or 0.0); self.unshift = bool(sp.get("unshift", False))
        self.chunk_rows = int(sp.get("chunk_rows") or 8192); self.lowrank_raw = bool(sp.get("lowrank_raw", False))
        self.qmin, self.qmax = _qrange(self.bits, self.unsigned)
        r = lambda k: (aux[k] if isinstance(k, str) else (torch.as_tensor(k, dtype=torch.float32) if k is not None else None))
        self.smooth = r(sp.get("smooth"))
        self.A = aux[sp["lowrank_A"]] if sp.get("lowrank_A") else None
        self.B = aux[sp["lowrank_B"]] if sp.get("lowrank_B") else None
        if (self.A is None) != (self.B is None):
            raise ValueError(f"{name}: lowrank_A/B must be given together")
        if self.mode not in _DYN_MODES:
            raise ValueError(f"{name}: unsupported act mode {self.mode!r} (expected one of {_DYN_MODES})")
        if self.mode == "dyn_group" and self.group <= 0:
            raise ValueError(f"{name}: dyn_group requires group")

    def _qdq_rows(self, xf: torch.Tensor) -> torch.Tensor:
        """xf: [rows, C] fp32 -> fake-quantized fp32 of the same shape."""
        if self.mode == "dyn_group":
            g = self.group; xg = xf.reshape(xf.shape[0], xf.shape[1] // g, g)
            amax = (xg.amax(dim=-1, keepdim=True) if self.unsigned else xg.abs().amax(dim=-1, keepdim=True)).clamp_min(1e-8)
            s = amax / self.qmax
            q = (xg / s).round_(); clipped = int(((q < self.qmin) | (q > self.qmax)).sum())
            out = (q.clamp_(self.qmin, self.qmax) * s).reshape(xf.shape)
        else:
            amax = (xf.amax(dim=-1, keepdim=True) if self.unsigned else xf.abs().amax(dim=-1, keepdim=True)).clamp_min(1e-8)
            s = amax / self.qmax
            q = (xf / s).round_(); clipped = int(((q < self.qmin) | (q > self.qmax)).sum())
            out = q.clamp_(self.qmin, self.qmax) * s
        self.state.add_sat(self.name, clipped, xf.numel())
        return out

    def __call__(self, module, args):
        x = args[0]
        if not torch.is_tensor(x) or not x.is_floating_point():
            return None
        shape = x.shape
        conv = (x.dim() == 4)
        x2 = x.permute(0, 2, 3, 1).reshape(-1, shape[1]) if conv else x.reshape(-1, shape[-1])   # rows = tokens/pixels, cols = channels
        out = torch.empty_like(x2)
        dev = x2.device
        hs = [] if self.A is not None else None
        sm = self.smooth.to(dev, torch.float32) if self.smooth is not None else None
        for r0 in range(0, x2.shape[0], self.chunk_rows):
            xf = x2[r0: r0 + self.chunk_rows].float()
            if hs is not None and self.lowrank_raw:                # low-rank branch on the raw input (nunchaku convention)
                hs.append((xf.to(self.A.dtype) @ self.A.to(dev)).to(torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float16))
            if self.shift:                                         # nunchaku kernel order: (x + shift) / smooth -> quantize
                xf = xf + self.shift
            if sm is not None:
                xf = xf / sm
            if hs is not None and not self.lowrank_raw:            # low-rank branch on the smoothed, unquantized input (deepcompressor convention)
                hs.append((xf.to(self.A.dtype) @ self.A.to(dev)).to(torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float16))
            xq = self._qdq_rows(xf)
            if self.shift and self.unshift:
                xq = xq - self.shift
            out[r0: r0 + self.chunk_rows] = xq.to(x.dtype)
        if hs is not None:
            module.__dict__["_ext_h"] = torch.cat(hs, 0)          # [rows, r]; at 4K 65792 x 32 bf16 = 4 MB
            module.__dict__["_ext_h_conv"] = conv
        if conv:
            out = out.reshape(shape[0], shape[2], shape[3], shape[1]).permute(0, 3, 1, 2).contiguous()
        else:
            out = out.reshape(shape)
        return (out,) + tuple(args[1:])


class LowRankBranch:
    """forward_hook: y <- y + (x_s @ A) @ B (SVDQuant 16-bit low-rank branch in parallel with the quantized
    main path). h = x_s @ A is computed per chunk by ExtActQuant."""
    __slots__ = ("B",)

    def __init__(self, B: torch.Tensor):
        self.B = B

    def __call__(self, module, args, output):
        h = module.__dict__.pop("_ext_h", None)
        if h is None:
            raise RuntimeError(f"LowRankBranch: no h stored by ExtActQuant on {type(module).__name__}; the activation hook must be registered first")
        y = (h.to(self.B.dtype) @ self.B.to(h.device)).to(output.dtype)
        if module.__dict__.pop("_ext_h_conv", False):
            n, _, hh, ww = output.shape
            y = y.reshape(n, hh, ww, -1).permute(0, 3, 1, 2)
        else:
            y = y.reshape(output.shape)
        return output + y


class SplitInLinear(nn.Module):
    """Split a Linear (in = in0 + in1) along the input dim (deepcompressor ConcatLinear):
    y = L0(x[..., :in0]) + L1(x[..., in0:]). In the FLUX single-stream block proj_out (6144 -> 3072), the attn
    half and the mlp half carry separate smoothing / low-rank / activation quantizers in SVDQuant, so the
    hooks must be attached separately."""

    def __init__(self, lin: nn.Linear, in0: int):
        super().__init__()
        W = lin.weight.data; b = lin.bias
        l0 = nn.Linear(in0, W.shape[0], bias=False, device=W.device, dtype=W.dtype)
        l1 = nn.Linear(W.shape[1] - in0, W.shape[0], bias=b is not None, device=W.device, dtype=W.dtype)
        l0.weight.data.copy_(W[:, :in0]); l1.weight.data.copy_(W[:, in0:])
        if b is not None:
            l1.bias.data.copy_(b.data)
        self.linears = nn.ModuleList([l0, l1]); self.in0 = int(in0)
        self.in_features, self.out_features = W.shape[1], W.shape[0]

    def forward(self, x):
        return self.linears[0](x[..., : self.in0]) + self.linears[1](x[..., self.in0:])


def _install_step_counter(model: nn.Module, state: ExtState):
    def _pre(mod, args, kwargs):
        x = args[0] if args else kwargs.get("hidden_states", kwargs.get("sample"))
        key = (int(x.shape[-1]) if x.dim() == 4 else int(x.shape[-2])) if torch.is_tensor(x) else None   # SDXL: latent width; FLUX: token count
        state.on_call(key)
    model.register_forward_pre_hook(_pre, with_kwargs=True)


def apply_external_quant(model: nn.Module, spec_dir: str, state: Optional[ExtState] = None) -> Dict[str, Any]:
    """Load an external-quant package into `model` in place. Returns the accounting summary (stored in the record)."""
    meta = json.load(open(os.path.join(spec_dir, "meta.json")))
    w = _load_safetensors(os.path.join(spec_dir, "weights.safetensors"))
    aux = _load_safetensors(os.path.join(spec_dir, "aux.safetensors"))
    p_act = os.path.join(spec_dir, "act_spec.json"); act = json.load(open(p_act)) if os.path.exists(p_act) else {}
    p_ws = os.path.join(spec_dir, "weight_spec.json"); wspec = json.load(open(p_ws)) if os.path.exists(p_ws) else {}
    state = state or ExtState()
    # ---- 1) module boundary patches (first, so later lookups use the new names) ----
    for name, in0 in (meta.get("split_modules") or {}).items():
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        lin = getattr(parent, child)
        if not isinstance(lin, nn.Linear):
            raise TypeError(f"{name}: split_modules only applies to nn.Linear, got {type(lin).__name__}")
        setattr(parent, child, SplitInLinear(lin, int(in0)))
    mods = dict(model.named_modules())
    # ---- 2) weights / biases ----
    n_w = 0; missing = []
    for k, t in w.items():
        if k.endswith(".weight"): name, attr = k[:-7], "weight"
        elif k.endswith(".bias"):  name, attr = k[:-5], "bias"
        else: name, attr = k, "weight"
        m = mods.get(name)
        if m is None or getattr(m, attr, None) is None: missing.append(k); continue
        p = getattr(m, attr)
        if tuple(p.shape) != tuple(t.shape): raise ValueError(f"{k}: model shape {tuple(p.shape)} vs package {tuple(t.shape)}")
        p.data.copy_(t.to(p.dtype))
        if attr == "weight": n_w += 1
    # ---- 3) activation hooks / low-rank branches ----
    n_a = n_lr = 0
    for name, sp in act.items():
        m = mods.get(name)
        if m is None: missing.append(name); continue
        q = ExtActQuant(name, sp, aux, state)
        m.register_forward_pre_hook(q)
        n_a += 1
        if q.B is not None:
            m.register_forward_hook(LowRankBranch(q.B)); n_lr += 1
    if meta.get("step_counter"):
        _install_step_counter(model, state)
    if missing:
        print(f"[external_quant] warning: {len(missing)} keys not found in the model, e.g. {missing[:3]}", flush=True)
    # ---- 4) accounting ----
    wb, g, sb = float(meta["w_bits"]), int(meta.get("group") or 0), float(meta.get("scale_dtype_bits", 16))
    fp16_names = set(meta.get("fp16_modules", []))
    q_params = 0.0; q_bits = 0.0
    quantized_names = set()
    for k, t in w.items():
        if not k.endswith(".weight"): continue
        name = k[:-7]; quantized_names.add(name)
        ws = wspec.get(name, {}) if isinstance(wspec, dict) else {}
        b = float(ws.get("bits", wb)); gg = int(ws.get("group", g) or 0); sbits = float(ws.get("scale_bits", sb))
        fan_in = t[0].numel() if t.dim() >= 2 else t.numel()
        per = b + (sbits / gg if gg else (sbits / fan_in if ws.get("per_channel_overhead", meta.get("per_channel_overhead", True)) else 0.0))
        q_params += t.numel(); q_bits += t.numel() * per
    lr_bits = 16.0 * sum(v.numel() for k, v in aux.items() if k.endswith(("_A", "_B")) or "lowrank" in k or ".lora" in k)
    side_bits = float(meta.get("side_table_bits") or 0.0)
    store_linear = (q_bits + lr_bits + side_bits) / max(q_params, 1)
    all_params = sum(p.numel() for n_, p in model.named_parameters() if n_.endswith(".weight"))
    fp_params = max(all_params - q_params, 0)
    store_all = (q_bits + lr_bits + side_bits + 16.0 * fp_params) / max(all_params, 1)
    r = int(meta.get("lowrank_rank") or 0)
    summary = {"method": meta.get("method"), "w_bits": wb, "a_bits": meta.get("a_bits"), "group": g,
               "store_bits": round(store_linear, 4), "store_bits_all": round(store_all, 4),
               "quantized_params": int(q_params), "backbone_params": int(all_params),
               "quantized_GiB": round((q_bits + lr_bits + side_bits) / 8 / 2**30, 3),
               "lowrank_rank": r, "lowrank_bits_per_param": round(lr_bits / max(q_params, 1), 4), "side_table_bits_per_param": round(side_bits / max(q_params, 1), 4),
               "n_weight_modules": n_w, "n_act_hooks": n_a, "n_lowrank": n_lr,
               "fp16_modules": sorted(fp16_names), "notes": meta.get("notes", ""), "calibration": meta.get("calibration", ""),
               "paper_label": meta.get("paper_label", "")}
    if "conv_quantized" in meta: summary["conv_quantized"] = bool(meta["conv_quantized"])   # SDXL accounting: whether Conv layers are quantized by this method
    print(f"[external_quant] {summary}", flush=True)
    model.ext_quant_summary = summary
    model.__dict__["_ext_state"] = state
    return summary


def build_extq_lo_variant(base: nn.Module, spec_dir: str) -> nn.Module:
    """Low-precision variant of PyraQuant on top of an external quantizer. `base` already has apply_external_quant
    applied (hooks / low-rank / fp16 whitelist). The model is deep-copied and every Linear with a "wscale" entry gets
    its weight replaced by the W3 derived from its W4 codes by nesting (quantizers_weight.nested3_from_scale);
    adaLN (scale_bits 32, W4A16) and whitelisted layers keep the hi weights. Low-rank branches, smoothing and the A4
    activation quantizers are inherited through the deep copy."""
    import copy
    from pyraquant.quantizers_weight import nested3_from_scale
    lo = copy.deepcopy(base)
    aux = _load_safetensors(os.path.join(spec_dir, "aux.safetensors"))
    wspec = json.load(open(os.path.join(spec_dir, "weight_spec.json")))
    mods = dict(lo.named_modules()); n = 0; skipped = 0
    for name, ws in wspec.items():
        if "wscale" not in ws or int(ws.get("scale_bits", 16)) == 32:
            skipped += 1; continue
        m = mods.get(name)
        if m is None or not isinstance(m, nn.Linear):
            raise KeyError(f"build_extq_lo_variant: {name} not an nn.Linear in model")
        cr = ws.get("code_range", [-8, 7])
        m.weight.data.copy_(nested3_from_scale(m.weight.data, aux[ws["wscale"]], int(cr[0]), int(cr[1])))
        m._wq_meta = {"method": "extq_nested3", "w_bits": 3, "group": int(ws.get("group", 64))}
        n += 1
    lo.sim_variant_spec = f"w3a4_extq_nested3(+{n} Linear, {skipped} kept hi)"
    print(f"[external_quant] lo variant: nested W3 from package codes on {n} Linear ({skipped} adaLN/whitelist kept at hi)", flush=True)
    return lo


def ext_runtime_report(model: nn.Module) -> Dict[str, Any]:
    """After generation: calls per stage and clipped fractions (stored in the record as external_quant.runtime)."""
    st = model.__dict__.get("_ext_state")
    if st is None:
        return {}
    return {"calls_per_stage": dict(st.n_calls), "clip_frac_per_stage": st.sat_summary()}
