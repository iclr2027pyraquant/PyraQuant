"""PyraQuant inference on ScaleDiff-SDXL (UNet pipeline, 1K -> 2K -> 4K).

Main configuration of the paper:
  * base quantizer  : round-to-nearest W8A8 (group size 32 on linear layers; convolutions W8A8 with
                      group 128 along (c, kh, kw) and per-channel fallback), no calibration
  * low-bit state   : nested W4, re-rounded from the stored W8 codes (no extra weights)
  * 2K stage        : residual-guided relative-gap selection of high / low precision regions
  * 4K stage        : the children of the 2K high-precision regions stay active and are split at
                      the median into high / low precision; all other regions use cached reuse
  * executor        : routed -- 2x2-leaf tiles that contain an active leaf are sent to the UNet
                      (with an 8-latent halo), cached regions reuse the stage-entry prediction

Quality is simulated with quantize-dequantize (QDQ) weights and activations, as in the paper.
"""
from __future__ import annotations

import argparse
import json
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

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
NEGATIVE_PROMPT = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

LEAF_LATENT = 64      # leaf edge in latent pixels (64 latent = 512 image pixels)
MASK_EDGE = 16        # soft-mask feathering width in latent pixels


# ---------------------------------------------------------------------------
class SpatialMaskUNet(nn.Module):
    """Replacement for ``pipe.unet``: per-stage uniform dispatch, or a hi/lo dual pass blended with
    the per-leaf precision mask armed by the stage-start hook.  The routed executor passes crops
    whose width differs from the stage width; the pipeline announces the stage and the crop mask
    through ``_route_override`` before each call."""

    def __init__(self, variants: Dict[str, nn.Module], plan: Dict[int, dict], base_name: str = "fp16"):
        super().__init__()
        self.variants = nn.ModuleDict(variants)
        self.plan = plan
        self.base_name = base_name
        self.masks: Dict[int, torch.Tensor] = {}   # latent width -> [1, 1, H, W]
        self.call_stats: Dict[str, int] = defaultdict(int)
        self._on_gpu = {n for n, v in variants.items() if next(v.parameters()).is_cuda}

    def needed(self, width: int):
        st = self.plan[width]
        return {st["v"]} if st["mode"] == "uniform" else {st["hi"], st["lo"]}

    def ensure_stage(self, width: int, device="cuda"):
        """Keep only the variants of this stage resident on the GPU."""
        need = self.needed(width)
        for n, v in self.variants.items():
            if n in need and n not in self._on_gpu:
                v.to(device); self._on_gpu.add(n)
            elif n not in need and n in self._on_gpu:
                v.to("cpu"); self._on_gpu.discard(n)
                torch.cuda.empty_cache()

    def forward(self, sample, *args, **kwargs):
        ov = self.__dict__.get("_route_override")
        w = int(ov["stage"]) if ov is not None else int(sample.shape[-1])
        st = self.plan.get(w)
        if st is None:
            raise KeyError(f"no plan for latent width {w} (have {sorted(self.plan)})")
        if st["mode"] == "uniform":
            self.call_stats[f"{w}|{st['v']}"] += 1
            return self.variants[st["v"]](sample, *args, **kwargs)
        m = ov["mask"] if ov is not None else self.masks.get(w)
        if m is None:
            raise RuntimeError(f"spatial stage {w} has no mask armed")
        if m.shape[-2:] != sample.shape[-2:]:
            raise RuntimeError(f"mask {tuple(m.shape)} does not match the input {tuple(sample.shape)}")
        self.call_stats[f"{w}|{st['hi']}+{st['lo']}"] += 1
        if st["hi"] == st["lo"]:
            return self.variants[st["hi"]](sample, *args, **kwargs)
        hi = self.variants[st["hi"]](sample, *args, **kwargs)[0]
        lo = self.variants[st["lo"]](sample, *args, **kwargs)[0]
        m = m.to(dtype=hi.dtype, device=hi.device)
        return (lo + m * (hi - lo),)

    @property
    def attn_processors(self):
        return self.variants[self.base_name].attn_processors

    def set_attn_processor(self, processor):
        for v in self.variants.values():
            v.set_attn_processor(dict(processor) if isinstance(processor, dict) else processor)

    def set_default_attn_processor(self):
        for v in self.variants.values():
            v.set_default_attn_processor()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = super().__getattr__("variants")[self.__dict__.get("base_name", "fp16")]
            return getattr(base, name)


# ---------------------------------------------------------------------------
def leaf_scores_residual(latents_LFM: torch.Tensor, latents_LU: torch.Tensor, n_leaf: int) -> torch.Tensor:
    """Residual energy e(B): RMS of (L_FM - L_U) over each leaf -> [n, n]."""
    d = (latents_LFM.float() - latents_LU.float()).pow(2).mean(dim=1)[0]
    b = d.shape[0] // n_leaf
    blocks = d.view(n_leaf, b, n_leaf, b).permute(0, 2, 1, 3).reshape(n_leaf, n_leaf, -1)
    return blocks.mean(-1).sqrt()


def _relgap_select(scores: torch.Tensor, cand: torch.Tensor, tau: float, pending_keep: bool = True) -> torch.Tensor:
    """Relative-gap rule on aligned 2x2 sibling groups (see flux/run_flux.py)."""
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


def build_leaf_hi(scores: torch.Tensor, parent_hi: Optional[torch.Tensor], thresh: float, norm: str = "relgap") -> torch.Tensor:
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


def soft_mask(leaf_hi: torch.Tensor, latent_hw: int, edge: int) -> torch.Tensor:
    m = Fn.interpolate(leaf_hi[None, None], size=(latent_hw, latent_hw), mode="nearest")
    if edge > 0:
        k = edge | 1
        m = Fn.avg_pool2d(Fn.pad(m, (k // 2,) * 4, mode="replicate"), k, stride=1)
    return m.clamp(0, 1)


def variant_bits(name: str) -> float:
    return 16.0 if name in ("fp16", "bf16") else float(name[1:].split("a")[0])


# ---------------------------------------------------------------------------
def build_variants(unet: nn.Module, variant_map: Dict[str, str], conv_bits: int, plan: Dict[int, dict]) -> Dict[str, nn.Module]:
    """variant_map values look like "rtn:g32" (RTN weights with group size 32) or "nested4:g32"
    (nested W4 re-rounded from the W8 codes of the same group size)."""
    from pyraquant.sim_recipes import build_recipe_variant
    variants: Dict[str, nn.Module] = {"fp16": unet}
    for name, spec in variant_map.items():
        recipe, _, g = spec.partition(":g")
        gs = int(g) if g else 128
        wb, ab = name[1:].split("a")
        t0 = time.perf_counter()
        v = build_recipe_variant(unet, int(wb), int(ab), recipe, None, weight_kwargs=None,
                                 conv_bits=(conv_bits if conv_bits < 16 else None), conv_method="rtn", group_size=gs)
        v.to("cpu"); torch.cuda.empty_cache()
        variants[name] = v
        print(f"[pyraquant] built {name} ({recipe}, group {gs}, conv W{conv_bits}) in {time.perf_counter() - t0:.0f}s", flush=True)
    for st in plan.values():
        for key in ("v", "hi", "lo"):
            if key in st and st[key] not in variants:
                raise ValueError(f"plan references unknown variant {st[key]!r}")
    return variants


def make_spatial_hook(wrapper: SpatialMaskUNet, plan: Dict[int, dict], state: dict, pipe, route: dict):
    def spatial_hook(p, latents_LU, latents_RU, latents_LFM, image_RU):
        width = int(latents_LFM.shape[-1])
        wrapper.ensure_stage(width)
        st = plan.get(width, {})
        if st.get("mode") != "spatial":
            pipe._route_boxes = None
            return
        n_leaf = width // LEAF_LATENT
        sc = leaf_scores_residual(latents_LFM, latents_LU, n_leaf)
        if st.get("inherit_hi"):
            # Children of the previous stage's high-precision regions stay active; everything else is cached.
            par = state["hi_map"].get(width // 2)
            if par is None:
                raise RuntimeError(f"stage {width} inherits but the previous stage produced no map")
            sel = par.repeat_interleave(2, 0).repeat_interleave(2, 1)
        else:
            parent = state["leaf_hi"].get(width // 2) if st.get("inherit") else None
            sel = build_leaf_hi(sc, parent, float(st["thresh"]), st.get("norm", "relgap"))
        hi_map = inner_split_median(sc, sel, float(st["inner_thresh"])) if st.get("inner_thresh") is not None else sel
        state["leaf_hi"][width] = sel
        state["hi_map"][width] = hi_map
        wrapper.masks[width] = soft_mask(hi_map, width, MASK_EDGE)
        if st.get("cache_bg"):
            pipe._cache_mask = 1.0 - soft_mask(sel, width, MASK_EDGE)
            pipe._x0_cache = latents_LFM.detach().clone()
        else:
            pipe._cache_mask = None
            pipe._x0_cache = None
        if route.get("executor") == "routed" and width == 512 and not st.get("cache_bg"):
            raise ValueError("the routed executor requires cache_bg at the 4K stage; use executor \"full\" for plans without caching")
        if st.get("cache_bg") and route.get("executor") == "routed":
            # The window attention of the UNet host needs crops of >= 2x2 leaves, so routing works on
            # 2x2-leaf tiles: a tile is computed if any of its leaves is active; adjacent tiles in a
            # row are merged into one crop.  Crops receive a halo (computed, not pasted back).
            L = LEAF_LATENT; N = int(sel.shape[0]); T = max(1, int(route["min_leaf"]))
            sm = sel.detach().cpu().numpy() > 0.5
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
                        while j < nt and tm[i, j]:
                            j += 1
                        rects.append((i * T * L, (i + 1) * T * L, j0 * T * L, j * T * L))
                    else:
                        j += 1
            pipe._route_boxes = rects
            pipe._route_batch = int(route["batch"])
            pipe._route_halo_lat = int(route["halo_latent"])
            hl = int(route["halo_latent"]); S = N * L
            frac = sum((min(S, y1 + hl) - max(0, y0 - hl)) * (min(S, x1 + hl) - max(0, x0 - hl)) for (y0, y1, x0, x1) in rects) / float(S ** 2)
            state["route_frac"][width] = frac
            print(f"[pyraquant] stage {width}: routed executor with {len(rects)} crops (halo {hl} latent), "
                  f"computed area {frac:.1%} (active leaves {int(sel.sum())}/{sel.numel()})", flush=True)
        else:
            pipe._route_boxes = None
        f_hi = float(hi_map.mean().item()); f_sel = float(sel.mean().item())
        f_lo, f_cache = (f_sel - f_hi, 1.0 - f_sel) if st.get("cache_bg") else (1.0 - f_hi, 0.0)
        state["coverage"][width] = {"hi": f_hi, "lo": f_lo, "cache": f_cache}
        state["eff_bits"][width] = f_hi * variant_bits(st["hi"]) + f_lo * variant_bits(st["lo"])
        print(f"[pyraquant] stage {width}: active {int(sel.sum())}/{sel.numel()} leaves (high {int(hi_map.sum())}), "
              f"cache={bool(st.get('cache_bg'))}, nominal W-bit {state['eff_bits'][width]:.2f}", flush=True)

    return spatial_hook


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "sdxl_pyraquant.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prompts-file", type=Path, default=ROOT / "prompts" / "examples.jsonl")
    ap.add_argument("--prompt", default="", help="generate a single prompt instead of --prompts-file")
    ap.add_argument("--n", type=int, default=2)
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
    route = {"executor": cfg.get("executor", "routed"), "halo_latent": int(cfg.get("route_halo_latent", 8)),
             "batch": int(cfg.get("route_batch", 4)), "min_leaf": int(cfg.get("route_min_leaf", 2))}
    steps = int(cfg.get("steps", 50)); guidance = float(cfg.get("guidance_scale", 7.5))
    restart_ratio = float(cfg.get("restart_ratio", 0.4)); conv_bits = int(cfg.get("conv_bits", 8))

    from pipeline_sdxl import CustomStableDiffusionXLPipeline

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
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(
        args.model_id, revision=args.model_revision, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True, local_files_only=args.local_files_only).to("cuda")
    pipe.vae.enable_tiling()
    variants = build_variants(pipe.unet, cfg["variants"], conv_bits, plan)
    wrapper = SpatialMaskUNet(variants, plan)
    pipe.unet = wrapper
    wrapper.ensure_stage(128)
    print(f"[pyraquant] models ready in {time.perf_counter() - t_load:.0f}s", flush=True)

    state = {"leaf_hi": {}, "hi_map": {}, "coverage": {}, "eff_bits": {}, "route_frac": {}}
    pipe._spatial_hook = make_spatial_hook(wrapper, plan, state, pipe, route)
    records_path = args.output_dir / "records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []

    for idx, (pname, prompt) in enumerate(prompts):
        for k in ("leaf_hi", "hi_map", "coverage", "eff_bits", "route_frac"):
            state[k].clear()
        wrapper.masks.clear(); wrapper.call_stats.clear()
        pipe._cache_mask = None; pipe._x0_cache = None
        pipe._route_boxes = None; pipe._route_stats = []
        wrapper.ensure_stage(128)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        images = pipe(prompt, negative_prompt=NEGATIVE_PROMPT, height=1024, width=1024,
                      generator=generator, num_inference_steps=steps, guidance_scale=guidance,
                      restart_ratio=restart_ratio, scale_factor=0.125, upsample_stage=2)
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
        records.append({
            "prompt_name": pname, "prompt": prompt, "seed": args.seed, "output": out_png.name,
            "coverage_per_stage": {str(k * 8): v for k, v in state["coverage"].items()},
            "nominal_weight_bits_per_stage": {str(k * 8): round(v, 4) for k, v in state["eff_bits"].items()},
            "computed_area_fraction_4096": state["route_frac"].get(512),
            "unet_calls": dict(wrapper.call_stats),
            "generation_seconds": round(gen_seconds, 1), "peak_gpu_gib": round(peak, 2),
            "gpu": torch.cuda.get_device_name(0),
        })
        records_path.write_text(json.dumps(records, indent=1) + "\n")
        print(f"[pyraquant] {idx + 1}/{len(prompts)} {pname}: {gen_seconds:.0f}s, peak {peak:.1f} GiB -> {out_png}", flush=True)
    print(f"[pyraquant] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
