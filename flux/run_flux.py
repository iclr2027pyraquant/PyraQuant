"""PyraQuant inference on ScaleDiff-FLUX.1-schnell (DiT pipeline, 1K -> 2K -> 4K).

Main configuration of the paper:
  * base quantizer  : released SVDQuant W4A4 checkpoint (loaded as an external-quant package)
  * low-bit state   : nested W3, derived from the stored W4 codes (no extra weights, no calibration)
  * 2K stage        : residual-guided relative-gap selection of high / low precision regions
  * 4K stage        : cascaded inheritance, relative-gap test inside inherited sibling groups,
                      median split of the active regions into high / low precision, cached reuse elsewhere
  * executor        : routed -- only active leaves (+halo) are sent to the transformer, cached leaves
                      reuse the stage-entry prediction

Quality is simulated with quantize-dequantize (QDQ) weights and activations, as in the paper.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
MODEL_REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"

LEAF_LATENT = 64      # leaf edge in latent pixels (64 latent = 512 image pixels)
MASK_EDGE = 16        # soft-mask feathering width in latent pixels (0 = hard edges)


# ---------------------------------------------------------------------------
def _rng_snapshot():
    return (random.getstate(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _rng_restore(snap) -> None:
    random.setstate(snap[0])
    torch.set_rng_state(snap[1])
    if snap[2] is not None:
        torch.cuda.set_rng_state_all(snap[2])


class SpatialMaskTransformer(nn.Module):
    """Replacement for ``pipe.transformer``: per-stage uniform dispatch, or a hi/lo dual pass whose
    outputs are blended on the packed token sequence with the per-leaf precision mask armed by the
    stage-start hook.  Pipeline-side attribute writes (NPAttn, jitter, init_NPA) are broadcast to
    every variant."""

    _BROADCAST_ATTRS = ("NPAttn", "query_random_jitter", "height", "width",
                        "base_height", "base_width", "base_q_patch_len", "base_kv_patch_len")

    def __init__(self, variants: Dict[str, nn.Module], plan: Dict[int, dict],
                 base_name: str = "bf16", sync_dual_pass_rng: bool = True):
        super().__init__()
        self.variants = nn.ModuleDict(variants)
        self.plan = plan                      # latent width -> {"mode", "v" | "hi", "lo", ...}
        self.base_name = base_name
        self.sync_dual_pass_rng = bool(sync_dual_pass_rng)
        self.masks: Dict[int, torch.Tensor] = {}   # latent width -> [1, seq, 1]
        self.call_stats: Dict[str, int] = defaultdict(int)

    @staticmethod
    def _latent_width_of(hidden_states: torch.Tensor) -> int:
        seq = int(hidden_states.shape[1])
        n = math.isqrt(seq)
        if n * n != seq:
            raise ValueError(f"non-square packed sequence (len {seq})")
        return 2 * n

    @staticmethod
    def _sample_of(out):
        return out[0] if isinstance(out, tuple) else out.sample

    def _blend(self, st, hidden_states, m, *args, **kwargs):
        # NPA patchify draws Python-RNG jitter per forward: snapshot before the hi pass and restore
        # before the lo pass so both passes see the same jitter.
        snap = _rng_snapshot() if self.sync_dual_pass_rng else None
        hi_out = self.variants[st["hi"]](hidden_states, *args, **kwargs)
        if snap is not None:
            _rng_restore(snap)
        lo_out = self.variants[st["lo"]](hidden_states, *args, **kwargs)
        hi, lo = self._sample_of(hi_out), self._sample_of(lo_out)
        m = m.to(dtype=hi.dtype, device=hi.device)
        blended = lo + m * (hi - lo)          # == m*hi + (1-m)*lo, exact when hi == lo
        if isinstance(hi_out, tuple):
            return (blended,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=blended)

    def _forward_local(self, hidden_states: torch.Tensor, *args, **kwargs):
        """Routed-executor call on cropped patches: hidden_states [K, N, C] with absolute token
        coordinates in img_ids [K, N, 3]; the per-token hi/lo weights are gathered from the
        full-canvas mask."""
        w = 2 * int(self.width)
        st = self.plan.get(w)
        if st is None:
            raise KeyError(f"no plan for latent width {w} (have {sorted(self.plan)})")
        if st["mode"] == "uniform":
            self.call_stats[f"{w}|{st['v']}|local"] += 1
            return self.variants[st["v"]](hidden_states, *args, **kwargs)
        m_full = self.masks.get(w)
        if m_full is None:
            raise RuntimeError(f"spatial stage {w} has no mask armed")
        ids = kwargs.get("img_ids")
        if ids is None:
            raise RuntimeError("routed local call without img_ids")
        idx = ids[..., 1].long() * int(self.width) + ids[..., 2].long()      # [K, N]
        m = m_full[0, :, 0].to(idx.device)[idx][..., None]                   # [K, N, 1]
        self.call_stats[f"{w}|{st['hi']}+{st['lo']}|local"] += 1
        return self._blend(st, hidden_states, m, *args, **kwargs)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        if getattr(self, "_routed_local", False):
            return self._forward_local(hidden_states, *args, **kwargs)
        w = self._latent_width_of(hidden_states)
        st = self.plan.get(w)
        if st is None:
            raise KeyError(f"no plan for latent width {w} (have {sorted(self.plan)})")
        if st["mode"] == "uniform":
            self.call_stats[f"{w}|{st['v']}"] += 1
            return self.variants[st["v"]](hidden_states, *args, **kwargs)
        m = self.masks.get(w)
        if m is None:
            raise RuntimeError(f"spatial stage {w} has no mask armed")
        self.call_stats[f"{w}|{st['hi']}+{st['lo']}"] += 1
        return self._blend(st, hidden_states, m, *args, **kwargs)

    def init_NPA(self, height, width):
        for v in self.variants.values():
            v.init_NPA(height, width)
        for name, val in (("NPAttn", True), ("height", int(height)), ("width", int(width))):
            object.__setattr__(self, name, val)

    def __setattr__(self, name, value):
        if name in SpatialMaskTransformer._BROADCAST_ATTRS:
            variants = self.__dict__.get("_modules", {}).get("variants")
            if variants is not None:
                for v in variants.values():
                    setattr(v, name, value)
            object.__setattr__(self, name, value)
            return
        super().__setattr__(name, value)

    @property
    def attn_processors(self):
        return self.variants[self.base_name].attn_processors

    def set_attn_processor(self, processor):
        for v in self.variants.values():
            if hasattr(v, "set_attn_processor"):
                v.set_attn_processor(dict(processor) if isinstance(processor, dict) else processor)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = super().__getattr__("variants")[self.__dict__.get("base_name", "bf16")]
            return getattr(base, name)


# ---------------------------------------------------------------------------
def leaf_scores_residual(latents_LFM: torch.Tensor, latents_LU: torch.Tensor, n_leaf: int) -> torch.Tensor:
    """Residual energy e(B): RMS of (L_FM - L_U) over each leaf -> [n, n]."""
    d = (latents_LFM.float() - latents_LU.float()).pow(2).mean(dim=1)[0]   # [H, W]
    b = d.shape[0] // n_leaf
    blocks = d.view(n_leaf, b, n_leaf, b).permute(0, 2, 1, 3).reshape(n_leaf, n_leaf, -1)
    return blocks.mean(-1).sqrt()


def _relgap_select(scores: torch.Tensor, cand: torch.Tensor, tau: float, pending_keep: bool = True) -> torch.Tensor:
    """Relative-gap rule on aligned 2x2 sibling groups inside the candidate set:
    q_i = e_i / max_j e_j, gap = 1 - min_i q_i.  A group is separated only when gap >= tau; then the
    children with q_i > 1 - tau are kept (at least the strongest one).  Groups that are not
    separated keep every child (pending_keep=True)."""
    n0, n1 = scores.shape
    sel = torch.zeros_like(cand)
    for i in range(0, n0, 2):
        for j in range(0, n1, 2):
            blk = cand[i:i + 2, j:j + 2]
            if not bool(blk.any()):
                continue
            sc = scores[i:i + 2, j:j + 2]
            v = sc[blk]
            emax = float(v.max().item())
            if emax <= 1e-12:
                keep = blk.clone() if pending_keep else torch.zeros_like(blk)
            else:
                gap = 1.0 - float(v.min().item()) / emax
                if gap + 1e-12 >= tau:
                    keep = (sc / emax > 1.0 - tau) & blk
                    if not bool(keep.any()):
                        keep = (sc >= emax) & blk
                else:
                    keep = blk.clone() if pending_keep else torch.zeros_like(blk)
            sel[i:i + 2, j:j + 2] = keep
    return sel


def build_leaf_hi(scores: torch.Tensor, parent_hi: Optional[torch.Tensor],
                  thresh: float, norm: str = "relgap") -> torch.Tensor:
    """Binary [n, n] selection over the candidate set (all leaves, or the children of the selected
    parents when inheriting).  norm="relgap": relative-gap rule with tau=thresh;
    norm="median": score >= thresh * median(candidate scores).  At least one leaf is kept."""
    if parent_hi is None:
        cand = torch.ones_like(scores, dtype=torch.bool)
    else:
        par = parent_hi.repeat_interleave(2, 0).repeat_interleave(2, 1)
        cand = par > 0.5
    if int(cand.sum().item()) == 0:
        return torch.zeros_like(scores)
    masked = torch.where(cand, scores, torch.full_like(scores, -1e30))
    if norm == "relgap":
        sel = _relgap_select(scores, cand, float(thresh), pending_keep=True)
    elif norm == "median":
        vals = scores[cand]
        ref = float(vals.median().item()) if vals.numel() > 1 else float(vals.max().item())
        sel = (scores >= float(thresh) * max(ref, 1e-12)) & cand
    else:
        raise ValueError(f"unknown selection norm {norm!r}")
    if int(sel.sum().item()) == 0:
        sel = (masked >= masked.max()) & cand
    return sel.float()


def inner_split_median(scores: torch.Tensor, sel: torch.Tensor, inner_thresh: float) -> torch.Tensor:
    """Split the active leaves into high / low precision: high if score >= inner_thresh * median
    of the active scores (at least one leaf stays high)."""
    cand = sel > 0.5
    v = scores[cand]
    if v.numel() == 0:
        return torch.zeros_like(scores)
    ref = float(v.median().item()) if v.numel() > 1 else float(v.max().item())
    hi_map = ((scores >= float(inner_thresh) * max(ref, 1e-12)) & cand).float()
    if float(hi_map.sum().item()) == 0:
        masked = torch.where(cand, scores, torch.full_like(scores, -1e30))
        hi_map = ((masked >= masked.max()) & cand).float()
    return hi_map


def packed_soft_mask(leaf_hi: torch.Tensor, latent_hw: int, edge_latent: int) -> torch.Tensor:
    """[n, n] leaf map -> [1, seq, 1] soft mask over packed tokens (token grid = latent/2)."""
    tok = latent_hw // 2
    m = Fn.interpolate(leaf_hi[None, None], size=(tok, tok), mode="nearest")
    e = max(0, int(edge_latent) // 2)
    if e > 0:
        k = e | 1
        m = Fn.avg_pool2d(Fn.pad(m, (k // 2,) * 4, mode="replicate"), k, stride=1)
    return m.clamp(0, 1).reshape(1, tok * tok, 1)


def latent_soft_mask(leaf_hi: torch.Tensor, latent_hw: int, edge_latent: int) -> torch.Tensor:
    """[n, n] leaf map -> [1, 1, H, W] soft mask over unpacked latents."""
    m = Fn.interpolate(leaf_hi[None, None], size=(latent_hw, latent_hw), mode="nearest")
    if edge_latent > 0:
        k = int(edge_latent) | 1
        m = Fn.avg_pool2d(Fn.pad(m, (k // 2,) * 4, mode="replicate"), k, stride=1)
    return m.clamp(0, 1)


def variant_bits(name: str) -> float:
    return 16.0 if name in ("fp16", "bf16") else float(name[1:].split("a")[0])


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
def build_variants(base: nn.Module, variant_map: Dict[str, str], svdquant_dir: str,
                   plan: Dict[int, dict]) -> Dict[str, nn.Module]:
    """variant_map values: "svdquant" (the base with the SVDQuant package applied) or
    "svdquant_nested3" (nested W3 derived from the stored W4 codes)."""
    from pyraquant.external_quant import build_extq_lo_variant
    variants: Dict[str, nn.Module] = {"bf16": base}
    for name, kind in variant_map.items():
        if kind == "svdquant":
            variants[name] = base
        elif kind == "svdquant_nested3":
            t0 = time.perf_counter()
            variants[name] = build_extq_lo_variant(base, svdquant_dir)
            print(f"[pyraquant] built {name} (nested W3 from the stored W4 codes) in {time.perf_counter() - t0:.0f}s", flush=True)
        else:
            raise ValueError(f"unknown variant kind {kind!r} for {name}")
    for st in plan.values():
        for key in ("v", "hi", "lo"):
            if key in st and st[key] not in variants:
                raise ValueError(f"plan references unknown variant {st[key]!r}")
    return variants


def make_spatial_hook(wrapper: SpatialMaskTransformer, plan: Dict[int, dict], state: dict, pipe, route: dict):
    """Stage-start hook called by the pipeline with (stage index, L_U, L_RU, L_FM, image_RU)."""

    def spatial_hook(p, latents_LU, latents_RU, latents_LFM, image_RU):
        width = int(latents_LFM.shape[-1])       # latent width: 256 (2K) / 512 (4K)
        st = plan.get(width, {})
        if st.get("mode") != "spatial":
            pipe._route_boxes = None
            return
        n_leaf = width // LEAF_LATENT
        sc = leaf_scores_residual(latents_LFM, latents_LU, n_leaf)
        if st.get("inherit"):
            parent = state["leaf_hi"].get(width // 2)
            if parent is None:
                raise RuntimeError(f"stage {width} inherits but the previous stage produced no map")
        else:
            parent = None
        sel = build_leaf_hi(sc, parent, float(st["thresh"]), st.get("norm", "relgap"))
        if st.get("inner_thresh") is not None:
            hi_map = inner_split_median(sc, sel, float(st["inner_thresh"]))
        else:
            hi_map = sel
        state["leaf_hi"][width] = sel
        state["hi_map"][width] = hi_map
        wrapper.masks[width] = packed_soft_mask(hi_map, width, MASK_EDGE)
        if st.get("cache_bg"):
            pipe._cache_mask = 1.0 - latent_soft_mask(sel, width, MASK_EDGE)
            pipe._x0_cache = latents_LFM.detach().clone()
        else:
            pipe._cache_mask = None
            pipe._x0_cache = None
        if route.get("executor") == "routed" and width == 512 and not st.get("cache_bg"):
            raise ValueError("the routed executor requires cache_bg at the 4K stage; use executor \"full\" for plans without caching")
        if route.get("executor") == "routed" and st.get("cache_bg"):
            # Row-wise runs of selected leaves form the routed boxes (latent coordinates).
            L = LEAF_LATENT
            boxes = []
            sm = sel.detach().cpu().numpy() > 0.5
            for i in range(sm.shape[0]):
                j = 0
                while j < sm.shape[1]:
                    if sm[i, j]:
                        j0 = j
                        while j < sm.shape[1] and sm[i, j]:
                            j += 1
                        boxes.append((i * L, (i + 1) * L, j0 * L, j * L))
                    else:
                        j += 1
            pipe._route_boxes = boxes
            pipe._route_halo = int(route["halo"])
            pipe._route_batch = int(route["batch"])
            pipe._route_beta = 1.0
            print(f"[pyraquant] stage {width}: routed executor with {len(boxes)} boxes from {int(sm.sum())} leaves, halo {pipe._route_halo} tokens", flush=True)
        else:
            pipe._route_boxes = None
        f_hi = float(hi_map.mean().item())
        f_sel = float(sel.mean().item())
        f_lo, f_cache = (f_sel - f_hi, 1.0 - f_sel) if st.get("cache_bg") else (1.0 - f_hi, 0.0)
        state["coverage"][width] = {"hi": f_hi, "lo": f_lo, "cache": f_cache}
        state["eff_bits"][width] = f_hi * variant_bits(st["hi"]) + f_lo * variant_bits(st["lo"])
        print(f"[pyraquant] stage {width}: active {int(sel.sum())}/{sel.numel()} leaves "
              f"(high {int(hi_map.sum())}), cache={bool(st.get('cache_bg'))}, "
              f"nominal W-bit {state['eff_bits'][width]:.2f}", flush=True)

    return spatial_hook


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "flux_pyraquant.json")
    ap.add_argument("--svdquant-dir", type=Path, required=True,
                    help="directory of the SVDQuant external-quant package (see README)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prompts-file", type=Path, default=ROOT / "prompts" / "examples.jsonl")
    ap.add_argument("--prompt", default="", help="generate a single prompt instead of --prompts-file")
    ap.add_argument("--n", type=int, default=2, help="number of prompts to generate")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--model-revision", default=MODEL_REVISION)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--save-masks", action="store_true", help="also save the per-stage leaf maps as PNG")
    args = ap.parse_args()

    global LEAF_LATENT, MASK_EDGE
    cfg = json.loads(Path(args.config).read_text())
    LEAF_LATENT = int(cfg.get("leaf_latent", LEAF_LATENT))
    MASK_EDGE = int(cfg.get("mask_edge", MASK_EDGE))
    plan = {int(k): v for k, v in cfg["plan"].items()}
    route = {"executor": cfg.get("executor", "routed"), "halo": int(cfg.get("route_halo", 2)),
             "batch": int(cfg.get("route_batch", 8))}
    steps = int(cfg.get("steps", 4))
    guidance = float(cfg.get("guidance_scale", 0.0))
    restart_ratio = float(cfg.get("restart_ratio", 0.5))

    from pipeline_flux import FluxPipeline
    from transformer_flux import FluxTransformer2DModel
    from pyraquant.external_quant import apply_external_quant

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prompt:
        prompts = [("prompt", args.prompt)]
    else:
        prompts = []
        with open(args.prompts_file) as fh:
            for i, line in enumerate(fh):
                if not line.strip() or i < args.offset:
                    continue
                if len(prompts) >= args.n:
                    break
                d = json.loads(line)
                prompts.append((d["name"], d["prompt"]))

    t_load = time.perf_counter()
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_id, revision=args.model_revision, subfolder="transformer",
        torch_dtype=torch.bfloat16, local_files_only=args.local_files_only)
    ext_summary = apply_external_quant(transformer, str(args.svdquant_dir))
    print(f"[pyraquant] base quantizer: {ext_summary.get('method')} W{ext_summary.get('w_bits')}A{ext_summary.get('a_bits')}", flush=True)
    variants = build_variants(transformer, cfg["variants"], str(args.svdquant_dir), plan)
    wrapper = SpatialMaskTransformer(variants, plan, base_name="bf16", sync_dual_pass_rng=True)

    pipe = FluxPipeline.from_pretrained(
        args.model_id, revision=args.model_revision, transformer=transformer,
        torch_dtype=torch.bfloat16, local_files_only=args.local_files_only)
    pipe.transformer = wrapper                  # must precede offload so the hooks land on every variant
    pipe.enable_sequential_cpu_offload(gpu_id=0, device="cuda")
    pipe.vae.enable_tiling()
    print(f"[pyraquant] models ready in {time.perf_counter() - t_load:.0f}s", flush=True)

    state = {"leaf_hi": {}, "hi_map": {}, "coverage": {}, "eff_bits": {}}
    pipe._spatial_hook = make_spatial_hook(wrapper, plan, state, pipe, route)
    records_path = args.output_dir / "records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []

    for idx, (pname, prompt) in enumerate(prompts):
        for k in ("leaf_hi", "hi_map", "coverage", "eff_bits"):
            state[k].clear()
        pipe._cache_mask = None; pipe._x0_cache = None
        pipe._route_boxes = None; pipe._route_stats = []
        wrapper.masks.clear(); wrapper.call_stats.clear()
        set_all_seeds(args.seed)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        images = pipe(prompt, height=1024, width=1024, guidance_scale=guidance,
                      num_inference_steps=steps, max_sequence_length=256, generator=generator,
                      restart_ratio=restart_ratio, scale_factor=0.25, upsample_stage=2,
                      query_random_jitter=True, t5_to_cpu=False, deterministic_stage_seed=args.seed)
        torch.cuda.synchronize()
        gen_seconds = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        final = images[-1]
        if final.size != (4096, 4096):
            raise RuntimeError(f"unexpected final size {final.size}")
        out_png = args.output_dir / f"{pname}_4096.png"
        final.save(out_png)
        if args.save_masks:
            from PIL import Image as PImage
            for width, leaf in state["leaf_hi"].items():
                PImage.fromarray((leaf.cpu().numpy() * 255).astype(np.uint8), "L").resize((256, 256), PImage.NEAREST).save(
                    args.output_dir / f"{pname}_active_{width * 8}.png")
            for width, leaf in state["hi_map"].items():
                PImage.fromarray((leaf.cpu().numpy() * 255).astype(np.uint8), "L").resize((256, 256), PImage.NEAREST).save(
                    args.output_dir / f"{pname}_high_{width * 8}.png")
        rs = list(getattr(pipe, "_route_stats", []) or [])
        routed_frac = ((sum(x["routed_tokens"] for x in rs if x["resolution"] == 4096)
                        / max(1, sum(x["full_tokens"] for x in rs if x["resolution"] == 4096))) if rs else None)
        records.append({
            "prompt_name": pname, "prompt": prompt, "seed": args.seed, "output": out_png.name,
            "coverage_per_stage": {str(k * 8): v for k, v in state["coverage"].items()},
            "nominal_weight_bits_per_stage": {str(k * 8): round(v, 4) for k, v in state["eff_bits"].items()},
            "routed_token_fraction_4096": routed_frac,
            "transformer_calls": dict(wrapper.call_stats),
            "generation_seconds": round(gen_seconds, 1), "peak_gpu_gib": round(peak, 2),
            "gpu": torch.cuda.get_device_name(0),
        })
        records_path.write_text(json.dumps(records, indent=1) + "\n")
        print(f"[pyraquant] {idx + 1}/{len(prompts)} {pname}: {gen_seconds:.0f}s, peak {peak:.1f} GiB -> {out_png}", flush=True)
    print(f"[pyraquant] done -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
