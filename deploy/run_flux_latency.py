"""End-to-end latency of ScaleDiff-FLUX.1-schnell (1K -> 2K -> 4K) with the INT8 deployment executor.

This is the driver behind the FLUX rows of the paper's deployment table and of the latency
decomposition ladder in the appendix.  It is a separate implementation from the QDQ quality
simulator in flux/run_flux.py: weights are packed to INT8 (deploy/int8_exec.py: per-channel INT8
weights, Triton per-token activation quantisation, fused INT8 GEMM with the dequant epilogue and
the SVDQuant low-rank up-projection in registers), the hi/lo (W4/W3) dual pass is collapsed into
ONE INT8 pass, and the routed executor of flux/pipeline_flux.py computes only the selected 4K
leaves (+halo) while the rest of the canvas comes from the spatial cache.  Quality is NOT measured
here (the INT8 numerics differ from the QDQ simulation); the PSNR of the warm-up prompt is a
numerics sanity check only.

Three executors, same token schedule (4/2/2 steps, NPA windows, same seeds):
  bf16 : host as is -- full canvas, bf16 Linears
  int8 : full canvas -- every token-level Linear replaced by Int8Linear; whitelist + adaLN stay bf16
  ours : the same INT8 Linears + routed executor (4K: relgap-selected leaves + halo, the rest from
         the spatial cache); W3 and W4 regions run the same INT8 kernel, so the W3-vs-W4 distinction
         is a storage/BOPs effect only and has no wall-clock effect.

Timed region: denoising + VAE of the three stages.  Text encoding (T5-XXL + CLIP) is identical
across executors: it is pre-computed once per prompt, reported separately and excluded.  All
weights stay resident on the GPU (no CPU offload).  Per prompt the script records the total and
per-stage wall-clock, denoise-loop time, VAE encode/decode, selection time, transformer forward
calls, the 4K executed-token fraction (incl. halo) and the per-stage peak allocated memory; the
summary reports the median [IQR] over the timed prompts.

Paper protocol (one NVIDIA L40S; the first 20 prompts of prompts/eval_ultrahr_2000.jsonl (the
default --prompts-file), plus one warm-up run of the first prompt that is excluded from the
statistics; ckpt/svdquant_flux_w4a4 is the SVDQuant package built by tools/svdquant_export.py, see
README):

  # bf16 baseline row
  python deploy/run_flux_latency.py --executor bf16 --n 20 --warmup 1 --out-dir outputs/latency_flux
  # PyraQuant row: weights = the SVDQuant W4 package (smoothing/shift folded into weight and bias,
  # rank-32 low-rank branch kept live), INT8 codes derived from those weights; tau = 0.10, 2-token halo
  python deploy/run_flux_latency.py --executor ours --weight-source method --tau 0.10 --route-halo 2 \\
      --n 20 --warmup 1 --svdquant-dir ckpt/svdquant_flux_w4a4 --out-dir outputs/latency_flux
  # summary (median [IQR] per stage, speed-ups, PSNR sanity) -> outputs/latency_flux.json + summary.md
  python deploy/run_flux_latency.py --summarize --out-dir outputs/latency_flux

Latency decomposition ladder (appendix): five rungs, each run into its own sub-directory of one
--out-dir, then summarised with --breakdown:
  bf16_full            --executor bf16
  int8_full            --executor int8
  int8_routed_nocache  --executor ours --no-cache     (all 64 4K leaves through the routed executor, cache weight 0)
  int8_routed_cache    --executor ours                (final recipe)
  bf16_routed_cache    --executor ours --linear bf16  (routing + cache on bf16 Linears: strategy only, no kernel)

  python deploy/run_flux_latency.py --executor ours --weight-source method --tau 0.10 --route-halo 2 \\
      --no-cache --n 20 --warmup 1 --svdquant-dir ckpt/svdquant_flux_w4a4 \\
      --out-dir outputs/latency_ladder/int8_routed_nocache
  ...
  python deploy/run_flux_latency.py --breakdown --out-dir outputs/latency_ladder

  Each rung is read from <out-dir>/<rung>/rec_<executor>.json (--ladder rung=path overrides one file).
  --no-cache selects all 64 4K leaves (threshold 0 on the median rule, no inheritance) instead of
  switching the cache off, because the routed executor requires the cache path to be armed (with
  cache_bg=false the driver would fall back to the host's full canvas).

Other modes: --forward-error (one 1K forward: INT8 kernels vs bf16 vs the A8 QDQ simulation, numerics
sanity), --profile-4k (torch.profiler over the 4K denoise loop of one prompt, CUDA time by kernel
class), --int8-backend triton|cublas (the paper used triton).  The script reads no environment
variables.  The Triton autotuning cache can be kept across runs with the standard TRITON_CACHE_DIR
variable, and the paper's runs were launched with the standard PyTorch allocator setting
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the shell environment.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent          # <repo>/deploy
ROOT = HERE.parent                              # <repo>
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "flux"))
sys.path.insert(0, str(HERE))

from run_flux import (MODEL_ID, MODEL_REVISION, LEAF_LATENT, MASK_EDGE,        # noqa: E402
                      SpatialMaskTransformer, make_spatial_hook, set_all_seeds)
import int8_exec                                                                # noqa: E402

DEFAULT_PROMPTS = ROOT / "prompts" / "eval_ultrahr_2000.jsonl"      # the paper timed its first 20 prompts
STAGE_PX = (1024, 2048, 4096)
# nn.Linear attributes on which the live SVDQuant low-rank branch is parked for Int8Linear to pick up
# (names published by deploy/int8_exec.py).
LOWRANK_ATTRS = (int8_exec.LOWRANK_A_ATTR, int8_exec.LOWRANK_B_ATTR)


# --------------------------------------------------------------------------- single-pass wrapper
class SinglePassMaskTransformer(SpatialMaskTransformer):
    """Spatial stages with hi == lo run ONE pass (deployment: W3 and W4 regions execute on the same
    INT8 kernel, no dual-pass blend).  The masks are still armed by the driver's stage-start hook
    (the cache and the route boxes need them) but the forward does not use them."""

    def _pick(self, st):
        if st["mode"] == "uniform":
            return st["v"]
        if st["hi"] != st["lo"]:
            raise ValueError(f"single-pass executor needs hi==lo, got {st}")
        return st["hi"]

    def _forward_local(self, hidden_states, *args, **kwargs):
        w = 2 * int(self.width)
        st = self.plan[w]
        v = self._pick(st)
        self.call_stats[f"{w}|{v}|local"] += 1
        return self.variants[v](hidden_states, *args, **kwargs)

    def forward(self, hidden_states, *args, **kwargs):
        if getattr(self, "_routed_local", False):
            return self._forward_local(hidden_states, *args, **kwargs)
        w = self._latent_width_of(hidden_states)
        st = self.plan[w]
        v = self._pick(st)
        self.call_stats[f"{w}|{v}"] += 1
        return self.variants[v](hidden_states, *args, **kwargs)


def make_plan(executor: str, tau: float, vname: str = "w8a8", no_cache: bool = False):
    if executor == "bf16":
        return {w: {"mode": "uniform", "v": "bf16"} for w in (128, 256, 512)}
    if executor == "int8":
        return {w: {"mode": "uniform", "v": "w8a8"} for w in (128, 256, 512)}
    # ours: the stage structure of the paper's FLUX recipe (configs/flux_pyraquant.json) with hi/lo
    # collapsed into the same variant (INT8 by default; bf16 with --linear bf16)
    plan = {
        128: {"mode": "uniform", "v": vname},
        256: {"mode": "spatial", "hi": vname, "lo": vname, "score": "residual", "thresh": tau, "norm": "relgap"},
        512: {"mode": "spatial", "hi": vname, "lo": vname, "score": "residual", "thresh": tau, "norm": "relgap",
              "inherit": True, "inner_thresh": 1.0, "inner_norm": "median", "cache_bg": True},
    }
    if no_cache:
        # Ladder rung 3: at 4K, threshold 0 on the median rule without inheritance selects all 64 leaves
        # (every residual score is >= 0), so the driver arms the route boxes as usual (8 full-row boxes)
        # and a cache mask that is identically 0 (1 - soft_mask(all ones) == 0, replicate padding keeps
        # the border at 1): every leaf goes through the routed patch executor and no token comes from
        # the cache.  The 2K stage is the same as in the final recipe (hi == lo single pass on the full
        # canvas, the mask does not affect the output), so the 4K input is identical to rungs 2 and 4.
        plan[512] = {"mode": "spatial", "hi": vname, "lo": vname, "score": "residual", "thresh": 0.0, "norm": "median",
                     "inherit": False, "cache_bg": True}
    return plan


def arm_name(executor: str, linear: str, no_cache: bool, tau: float) -> str:
    if executor != "ours":
        return f"{linear}_full"
    return f"{linear}_routed_nocache" if no_cache else f"{linear}_routed_cache_tau{tau:.2f}"


def resolve_linear(args) -> str:
    """--linear defaults to the executor's numerics: bf16 executor -> bf16 Linears, int8/ours -> INT8.
    Only ours may be switched to bf16 explicitly (the strategy-only rung)."""
    lin = args.linear or ("bf16" if args.executor == "bf16" else "int8")
    if args.executor == "bf16" and lin != "bf16":
        raise SystemExit("--executor bf16 is the bf16 full-canvas arm; use --executor int8 for INT8 full canvas")
    if args.executor == "int8" and lin != "int8":
        raise SystemExit("--executor int8 --linear bf16 is identical to --executor bf16; refusing to duplicate the arm")
    if args.no_cache and args.executor != "ours":
        raise SystemExit("--no-cache only applies to the routed executor (--executor ours)")
    return lin


def read_prompts(path: Path, n: int, offset: int = 0):
    out = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if not line.strip() or i < offset:
                continue
            if len(out) >= n:
                break
            d = json.loads(line)
            out.append((d["name"], d["prompt"], int(d.get("seed", 42))))
    if len(out) < n:
        raise SystemExit(f"wanted {n} prompts, got {len(out)}")
    return out


def gib(x):
    return x / 2**30


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- method weights (SVDQuant package)
def _load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    return load_file(path)


def _fold_one(lin: nn.Linear, sp: dict, aux: Dict[str, torch.Tensor], fold_lowrank: bool = False) -> None:
    """Fold this layer's smoothing / shift into lin.weight and lin.bias in place (pure algebra, no
    run-time cost saved):  y = QDQ((x + shift) / smooth) @ W^T + b  ->  W_eff = W / smooth,
    b_eff = b + W_eff @ shift.  The low-rank branch is either folded into the weight (16-bit path,
    fold_lowrank=True) or parked on the Linear for Int8Linear to run as a live parallel branch."""
    W = lin.weight.data
    dt, dev = W.dtype, W.device
    Wf = W.float()
    smooth = sp.get("smooth")
    if smooth is not None:
        s = aux[smooth].float().to(dev) if isinstance(smooth, str) else torch.as_tensor(smooth, dtype=torch.float32, device=dev)
        if s.numel() == 1:
            Wf = Wf / s.reshape(())                    # scalar smoothing
        elif s.numel() == Wf.shape[1]:
            Wf = Wf / s.reshape(1, -1)                 # x/smooth @ W^T  ==  x @ (W/smooth)^T
        else:
            raise ValueError(f"smooth length {s.numel()} is neither 1 nor in_features {Wf.shape[1]}")
    shift = sp.get("shift")
    if shift is not None:
        sh = aux[shift].float().to(dev) if isinstance(shift, str) else torch.as_tensor(shift, dtype=torch.float32, device=dev)
        # (x + shift) / smooth -> the constant term goes into the bias; shift is a scalar or per input channel
        if sh.numel() == 1:
            add = Wf.sum(dim=1) * sh.reshape(())
        elif sh.numel() == Wf.shape[1]:
            add = Wf @ sh.reshape(-1)
        else:
            raise ValueError(f"shift length {sh.numel()} is neither 1 nor in_features {Wf.shape[1]}")
        if lin.bias is None:
            lin.bias = nn.Parameter(add.to(dt))
        else:
            lin.bias.data.add_(add.to(lin.bias.dtype))
    lin.weight.data.copy_(Wf.to(dt))
    # The low-rank branch is NOT folded into the INT8 weight: folding is numerically equivalent but
    # would drop a cost that a real W4 kernel has to pay, and the BOPs/storage tables account for the
    # branch.  Int8Linear moves A/B into buffers and runs them in parallel; the INT8 main path stays one pass.
    a_key, b_key = sp.get("lowrank_A"), sp.get("lowrank_B")
    if a_key is not None and b_key is not None:
        A = aux[a_key].to(dev); B = aux[b_key].to(dev)                 # [in, r] / [r, out]
        if A.shape[0] != lin.weight.shape[1] or B.shape[1] != lin.weight.shape[0]:
            raise ValueError(f"low-rank shapes A{tuple(A.shape)} B{tuple(B.shape)} do not match W{tuple(lin.weight.shape)}")
        if fold_lowrank:                              # 16-bit path: no Int8Linear to run the branch -> fold it into the weight (one full-precision GEMM, as a real deployment would)
            lin.weight.data.copy_((lin.weight.data.float() + (A.float() @ B.float()).t()).to(dt))
        else:
            setattr(lin, LOWRANK_ATTRS[0], A.to(dt)); setattr(lin, LOWRANK_ATTRS[1], B.to(dt))


def load_method_weights(model: nn.Module, pkg_dir: str, verbose: bool = True, fold_lowrank: bool = False) -> dict:
    """Put the SVDQuant W4 package (the same directory flux/run_flux.py reads with --svdquant-dir)
    into the transformer as ONE effective bf16 weight per Linear: dequantised W4 values with
    smoothing/shift folded into weight and bias, no activation hooks.  The caller then derives the
    INT8 codes from THESE weights.  The nested W3 is derived from the W4 codes and stored nowhere
    else, so the timed model is the W4-derived INT8 one; the low bit-width only affects storage/BOPs."""
    from pyraquant.external_quant import SplitInLinear
    meta = json.load(open(os.path.join(pkg_dir, "meta.json")))
    if meta.get("impl"):
        raise SystemExit(f"package uses a custom implementation impl={meta['impl']}; this loader handles plain SVDQuant packages only")
    w = _load_safetensors(os.path.join(pkg_dir, "weights.safetensors"))
    aux = _load_safetensors(os.path.join(pkg_dir, "aux.safetensors"))
    act = json.load(open(os.path.join(pkg_dir, "act_spec.json")))

    # 1) split modules first, so the later names resolve
    n_split = 0
    for name, in0 in (meta.get("split_modules") or {}).items():
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        lin = getattr(parent, child)
        if isinstance(lin, nn.Linear):
            setattr(parent, child, SplitInLinear(lin, int(in0))); n_split += 1

    # 2) write the package weights/bias in (these are the dequantized W4 values)
    mods = dict(model.named_modules())
    n_w = 0
    missing = []
    for k, t in w.items():
        if k.endswith(".weight"):
            name, attr = k[:-7], "weight"
        elif k.endswith(".bias"):
            name, attr = k[:-5], "bias"
        else:
            name, attr = k, "weight"
        m = mods.get(name)
        if m is None or getattr(m, attr, None) is None:
            missing.append(k); continue
        p = getattr(m, attr)
        if tuple(p.shape) != tuple(t.shape):
            raise ValueError(f"{k}: model shape {tuple(p.shape)} vs package {tuple(t.shape)}")
        p.data.copy_(t.to(p.dtype))
        n_w += attr == "weight"

    # 3) fold smooth / shift (and, on the 16-bit path, the low-rank branch) so one GEMM reproduces the layer
    n_fold = 0
    skipped = []
    for name, sp in act.items():
        base = name.split("@")[0]
        if sp.get("impl") or sp.get("where") == "output":
            skipped.append(name); continue
        m = mods.get(base)
        if not isinstance(m, nn.Linear):
            skipped.append(name); continue
        _fold_one(m, sp, aux, fold_lowrank=fold_lowrank)
        n_fold += 1
    n_lowrank = sum(1 for _, m in model.named_modules() if getattr(m, LOWRANK_ATTRS[0], None) is not None)
    rep = {"method": meta.get("method"), "w_bits": meta.get("w_bits"), "a_bits_simulated_away": meta.get("a_bits"), "n_lowrank_branches_live": n_lowrank,
           "group": meta.get("group"), "lowrank_rank": meta.get("lowrank_rank"),
           "n_weights_written": n_w, "n_layers_folded": n_fold, "n_split_modules": n_split,
           "n_missing_keys": len(missing), "n_specs_skipped": len(skipped),
           "note": "weights = dequantized W4 package with smooth/shift folded into weight+bias; the rank-32 low-rank branch is kept LIVE as a parallel 16-bit matmul (not folded); the INT8 codes are derived from THESE weights"}
    if verbose:
        print(f"[weight-source] FLUX method weights: {meta.get('method')} W{meta.get('w_bits')} group {meta.get('group')} low-rank r={meta.get('lowrank_rank')}; "
              f"wrote {n_w} weights, folded {n_fold} layers ({n_lowrank} live low-rank branches), split {n_split} layers, "
              f"{len(missing)} missing keys, {len(skipped)} specs skipped", flush=True)
    return rep


def _check_release_modules(*classes) -> None:
    """Refuse to time anything but the release pipeline / transformer under <repo>/flux."""
    want = (ROOT / "flux").resolve()
    for cls in classes:
        got = Path(inspect.getfile(cls)).resolve().parent
        if got != want:
            raise RuntimeError(f"loaded {cls.__name__} from {got}, expected {want}")


# --------------------------------------------------------------------------- run one executor
def run(args):
    from pipeline_flux import FluxPipeline                              # noqa: E402  (release pipeline with the routed executor)
    from transformer_flux import FluxTransformer2DModel                 # noqa: E402
    _check_release_modules(FluxPipeline, FluxTransformer2DModel)
    ex = args.executor
    linear = resolve_linear(args)                                       # bf16 | int8
    no_cache = bool(args.no_cache)
    arm = arm_name(ex, linear, no_cache, args.tau)
    if args.weight_source == "method" and args.svdquant_dir is None:
        raise SystemExit("--weight-source method needs --svdquant-dir (the SVDQuant package directory, see README)")
    int8_exec.set_backend(args.int8_backend)                            # module-level switch of deploy/int8_exec, read at call time
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda *a: print(f"[e2e:{ex}]", *a, flush=True)
    log(f"gpu={torch.cuda.get_device_name()} torch={torch.__version__} int8_backend={int8_exec.BACKEND} leaf_latent={LEAF_LATENT} tau={args.tau} "
        f"linear={linear} no_cache={no_cache} arm={arm}")

    t_load = time.perf_counter()
    transformer = FluxTransformer2DModel.from_pretrained(args.model_id, revision=args.model_revision, subfolder="transformer",
                                                         torch_dtype=torch.bfloat16, local_files_only=args.local_files_only)
    # The timed model must be OUR model: weights stored as W4 (nested W3), computed with the INT8 kernels.
    # --weight-source method folds the SVDQuant W4 package into the weights first and derives the INT8 codes
    # from them; host keeps the stock behaviour (INT8 codes from the original bf16 weights).
    wsrc_report = None
    if args.weight_source == "method":
        t0 = time.perf_counter()
        wsrc_report = load_method_weights(transformer, str(args.svdquant_dir), fold_lowrank=(linear != "int8"))
        wsrc_report["load_seconds"] = time.perf_counter() - t0
        log(f"weight source = method ({wsrc_report['method']} W{wsrc_report['w_bits']}): "
            f"folded {wsrc_report['n_layers_folded']} layers in {wsrc_report['load_seconds']:.0f}s")
    int8_report = None
    if linear == "int8":
        t0 = time.perf_counter()
        int8_report = int8_exec.convert_linears_to_int8(transformer, device="cuda")
        int8_report["convert_seconds"] = time.perf_counter() - t0
        log(f"INT8 conversion: {int8_report['n_int8_linear']} Linear -> Int8Linear ({int8_report['int8_weight_gib']:.2f} GiB int8), "
            f"{int8_report['n_bf16_linear_kept']} kept bf16 (whitelist+adaLN, {int8_report['params_bf16_linear_kept']/1e9:.2f}B params) in {int8_report['convert_seconds']:.0f}s")
        if wsrc_report is not None:                                     # the live low-rank branches must have been picked up by Int8Linear
            n_live = sum(1 for m in transformer.modules() if isinstance(m, int8_exec.Int8Linear) and getattr(m, "lr_A", None) is not None)
            if n_live != wsrc_report["n_lowrank_branches_live"]:
                raise RuntimeError(f"{n_live} Int8Linear carry a low-rank branch but the package has {wsrc_report['n_lowrank_branches_live']} "
                                   f"(attribute names {LOWRANK_ATTRS} not read by int8_exec.Int8Linear?)")
        int8_exec.warm_kernels()                                        # pre-compile / autotune every FLUX shape x M bucket outside the timed region
    vname = "bf16" if linear == "bf16" else "w8a8"
    plan = make_plan(ex, args.tau, vname=vname, no_cache=no_cache)
    wrapper = SinglePassMaskTransformer({vname: transformer}, plan, base_name=vname, sync_dual_pass_rng=False)
    pipe = FluxPipeline.from_pretrained(args.model_id, revision=args.model_revision, transformer=transformer,
                                        torch_dtype=torch.bfloat16, local_files_only=args.local_files_only)
    pipe.transformer = wrapper
    pipe.set_progress_bar_config(disable=True)

    # ---- text encoding: identical for the three executors; computed up front and timed (T5-XXL on the GPU), then the encoders are released ----
    prompts = read_prompts(Path(args.prompts_file), args.n, args.offset)
    pipe.text_encoder.to("cuda"); pipe.text_encoder_2.to("cuda")
    embeds, text_seconds = {}, {}
    for name, prompt, _ in prompts:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():                                           # otherwise the embeds keep an autograd graph whose saved tensors pin the T5 weights (8.9 GiB) + activations
            pe, pp, _ids = pipe.encode_prompt(prompt=prompt, prompt_2=None, device="cuda", num_images_per_prompt=1, max_sequence_length=256)
        torch.cuda.synchronize(); text_seconds[name] = time.perf_counter() - t0
        embeds[name] = (pe.detach(), pp.detach())
    for _n in ("text_encoder", "text_encoder_2"):                    # move to CPU first, then drop (stray references cannot hold GPU memory)
        _m = getattr(pipe, _n, None)
        if _m is not None:
            _m.to("cpu")
        setattr(pipe, _n, None)
    gc.collect(); torch.cuda.empty_cache()
    log(f"text encoding cached for {len(embeds)} prompts (median {np.median(list(text_seconds.values())):.2f}s, excluded from the timed region); "
        f"text encoders released, allocated now {gib(torch.cuda.memory_allocated()):.2f} GiB")

    pipe.to("cuda")                                                     # weights resident, no sequential offload
    pipe.vae.enable_tiling()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    tf_bytes = sum(p.numel() * p.element_size() for p in transformer.parameters()) + sum(b.numel() * b.element_size() for b in transformer.buffers())
    load_seconds = time.perf_counter() - t_load
    log(f"resident: transformer {gib(tf_bytes):.2f} GiB, total allocated {gib(resident):.2f} GiB, free {gib(torch.cuda.mem_get_info()[0]):.2f} GiB, load {load_seconds:.0f}s")
    assert gib(resident) < 30.0, f"weights must be resident with headroom for 4K activations (got {gib(resident):.1f} GiB; text encoders must be released)"

    # ---- instance-level timing patches: VAE decode (stage end) / encode (LFM) ----
    timeline = {}
    orig_decode, orig_encode = pipe.decode_latents, pipe.encode_image

    prof_box = {"prof": None, "active": False}

    def timed_decode(latents):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if prof_box["active"] and int(latents.shape[-1]) * 8 == 4096:      # --profile-4k: stop at the end of the 4K denoise loop (before decode)
            prof_box["prof"].stop(); prof_box["active"] = False
        img = orig_decode(latents)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        px = int(latents.shape[-1]) * 8
        timeline.setdefault("decode", {})[px] = t1 - t0
        timeline.setdefault("stage_end", {})[px] = t1
        timeline.setdefault("peak_alloc", {})[px] = max(gib(torch.cuda.max_memory_allocated()), timeline.get("peak_pre_hook", {}).get(px, 0.0))
        torch.cuda.reset_peak_memory_stats()
        return img

    def timed_encode(image):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig_encode(image)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        timeline.setdefault("encode", {})[int(image.shape[-1])] = t1 - t0
        return r
    pipe.decode_latents = timed_decode
    pipe.encode_image = timed_encode

    state = {"leaf_hi": {}, "hi_map": {}, "coverage": {}, "eff_bits": {}}
    route = {"executor": ("routed" if ex == "ours" else "full"), "halo": args.route_halo, "batch": args.route_batch, "beta": 1.0}
    base_hook = make_spatial_hook(wrapper, plan, state, pipe, route)

    def hook(p, latents_LU, latents_RU, latents_LFM, image_RU):
        px = int(latents_LFM.shape[-1]) * 8
        timeline.setdefault("peak_pre_hook", {})[px] = gib(torch.cuda.max_memory_allocated())   # includes this stage's LFM encode
        torch.cuda.reset_peak_memory_stats()                            # the peak of the denoise loop is read at decode time
        torch.cuda.synchronize(); t0 = time.perf_counter()
        base_hook(p, latents_LU, latents_RU, latents_LFM, image_RU)
        torch.cuda.synchronize(); timeline.setdefault("select", {})[px] = time.perf_counter() - t0
        if prof_box["prof"] is not None and px == 4096 and not prof_box["active"]:   # --profile-4k: start of the 4K denoise loop
            prof_box["prof"].start(); prof_box["active"] = True
    pipe._spatial_hook = hook

    records = []
    rec_path = out_dir / f"rec_{ex}.json"
    meta = {"executor": ex, "arm": arm, "linear": linear, "no_cache": no_cache,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": __import__("triton").__version__,
            "int8_backend": (int8_exec.BACKEND if linear == "int8" else None), "int8_report": ({k: v for k, v in int8_report.items() if k != "bf16_kept_names"} if int8_report else None),
            "bf16_kept_names": (int8_report["bf16_kept_names"] if int8_report else None),
            "weight_source": args.weight_source, "weight_source_report": wsrc_report,
            "plan": {str(k): v for k, v in plan.items()}, "leaf_latent": LEAF_LATENT, "mask_edge": MASK_EDGE, "tau": args.tau,
            "route": dict(route) if ex == "ours" else None,
            "steps": "4/2/2 (schnell, restart_ratio 0.5)", "resident_alloc_gib": gib(resident), "transformer_bytes_gib": gib(tf_bytes),
            "load_seconds_excluded": load_seconds, "text_encode_seconds": text_seconds, "prompts_file": str(args.prompts_file),
            "n_timed": args.n, "warmup": args.warmup,
            "notes": ["weights resident on one GPU (no CPU offload); text encoding excluded (identical across executors)",
                      "attention (SDPA) stays bf16 in all executors; no torch.compile / CUDA graphs anywhere",
                      "int8/ours: token-level Linears INT8 (per-channel W, fused per-token A); whitelist + adaLN modulation Linears stay bf16",
                      "ours: hi/lo (W4/W3) dual pass collapsed to ONE INT8 pass -- W3 vs W4 is a storage/BOPs effect only, not a wall-clock effect",
                      "ours 2K stage: relgap selection but full canvas (no cache at 2K in the recipe) -> same work as int8 uniform",
                      "quality NOT measured here (numerics differ from the QDQ simulation); PSNR is a sanity check only"]
                     + ([f"--no-cache: all 64 4K leaves selected (threshold 0 on the median rule, no inheritance) -> every leaf runs through the routed patch executor "
                         f"(8 full-row boxes + {args.route_halo}-token halo); cache mask is identically 0, so no token is reused"] if no_cache else [])
                     + (["--linear bf16: no INT8 conversion; routed+cache strategy on bf16 Linears (strategy-only arm)"] if (ex == "ours" and linear == "bf16") else [])}
    schedule = [(True, prompts[0])] * int(args.warmup) + [(False, p) for p in prompts]
    if args.profile_4k:
        schedule = [(True, prompts[0]), (False, prompts[0])]
    for k, (is_warm, (name, prompt, seed)) in enumerate(schedule):
        if args.profile_4k and not is_warm:
            from torch.profiler import ProfilerActivity, profile as _tprofile
            prof_box["prof"] = _tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, profile_memory=False, with_stack=False)
        wrapper.plan = plan
        for key in ("leaf_hi", "hi_map", "coverage", "eff_bits"):
            state[key].clear()
        pipe._cache_mask = None; pipe._x0_cache = None; pipe._route_boxes = None; pipe._route_stats = []
        wrapper.masks.clear(); wrapper.call_stats.clear(); timeline.clear()
        pe, pp = embeds[name]
        set_all_seeds(seed)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t_start = time.perf_counter()
        images = pipe(prompt=None, prompt_embeds=pe, pooled_prompt_embeds=pp, height=1024, width=1024, guidance_scale=0.0,
                      num_inference_steps=4, max_sequence_length=256, generator=gen, restart_ratio=0.5, scale_factor=0.25,
                      upsample_stage=2, query_random_jitter=True, t5_to_cpu=False, deterministic_stage_seed=seed)
        torch.cuda.synchronize(); t_end = time.perf_counter()
        total = t_end - t_start
        if images[-1].size != (4096, 4096):
            raise RuntimeError(f"unexpected final size {images[-1].size}")
        if ex == "ours" and not pipe._route_stats:
            raise RuntimeError("ours executor but no routed 4K step recorded")
        tr = pipe.scalediff_trace["stages"]
        denoise = {int(s["resolution"]): float(s["denoise_loop_seconds"]) for s in tr}
        fwd_calls = {int(s["resolution"]): int(s["transformer_forward_calls"]) for s in tr}
        ends = timeline.get("stage_end", {})
        stage_wall, prev = {}, t_start
        for px in STAGE_PX:
            e = ends.get(px, t_end)
            stage_wall[px] = e - prev; prev = e
        stage_wall[4096] += t_end - ends.get(4096, t_end)                # the 4K post-processing is counted in the 4K stage
        rs = pipe._route_stats or []
        r4 = [x for x in rs if x["resolution"] == 4096]
        exec_frac = (sum(x["routed_tokens"] for x in r4) / max(1, sum(x["full_tokens"] for x in r4))) if r4 else 1.0
        rec = {"prompt_name": name, "seed": seed, "warmup": is_warm, "total_s": total,
               "stage_wall_s": {str(px): stage_wall[px] for px in STAGE_PX},
               "denoise_s": {str(px): denoise.get(px) for px in STAGE_PX},
               "vae_decode_s": {str(px): timeline.get("decode", {}).get(px) for px in STAGE_PX},
               "vae_encode_s": {str(px): timeline.get("encode", {}).get(px) for px in STAGE_PX[1:]},
               "select_s": {str(px): timeline.get("select", {}).get(px) for px in STAGE_PX[1:]},
               "peak_alloc_gib": {str(px): timeline.get("peak_alloc", {}).get(px) for px in STAGE_PX},
               "transformer_forward_calls": {str(k2): v for k2, v in fwd_calls.items()},
               "executed_token_frac_4096_incl_halo": exec_frac,
               "coverage": {str(k2): v for k2, v in state["coverage"].items()},
               "route_steps": rs, "call_stats": dict(wrapper.call_stats), "text_encode_s_excluded": text_seconds[name]}
        records.append(rec)
        if is_warm:                                                     # the warm-up prompt keeps all stage images (1K/2K/4K) for the PSNR sanity check
            for _im in images:
                _im.save(out_dir / f"warmup_{ex}_{name}_{_im.size[0]}.png")
        json.dump({"meta": meta, "records": records}, open(rec_path, "w"), indent=1)
        log(f"{k+1}/{len(schedule)} {name}{' [warm-up]' if is_warm else ''}: total {total:.1f}s | "
            f"1K {stage_wall[1024]:.1f} 2K {stage_wall[2048]:.1f} 4K {stage_wall[4096]:.1f} | denoise "
            f"{denoise.get(1024, 0):.1f}/{denoise.get(2048, 0):.1f}/{denoise.get(4096, 0):.1f} | peak4K {rec['peak_alloc_gib']['4096']:.1f} GiB"
            + (f" | exec4K {exec_frac:.3f} (sel {state['coverage'].get(512, {}).get('hi', 0) + state['coverage'].get(512, {}).get('lo', 0):.3f})" if ex == "ours" else ""))
    if args.profile_4k:
        _profile_report(prof_box["prof"], ex, out_dir, records[-1], log)
    log(f"done -> {rec_path}")


def _profile_report(prof, ex, out_dir, rec, log):
    """CUDA kernel self time of the 4K denoise loop by class: Linear (INT8 Triton / bf16 cuBLAS) / attention / other (elementwise, norm, copy, ...)."""
    cats = {"linear_int8": 0.0, "linear_bf16_gemm": 0.0, "attention": 0.0, "other": 0.0}
    top = []
    for ev in prof.key_averages():
        t = float(getattr(ev, "self_device_time_total", getattr(ev, "self_cuda_time_total", 0.0)))
        if t <= 0 or getattr(ev, "device_type", None) is None:
            continue
        if str(ev.device_type) not in ("DeviceType.CUDA", "DeviceType.PrivateUse1") and "CUDA" not in str(ev.device_type):
            continue
        n = ev.key
        if "_int8_gemm_fused" in n or "_quant_rows_kernel" in n or "_dequant_epilogue" in n:
            c = "linear_int8"
        elif re.search(r"flash|fmha|attention|sdpa|attn", n, re.I):
            c = "attention"
        elif re.search(r"gemm|nvjet|cutlass|Cijk|xmma|matmul|mm_", n, re.I):
            c = "linear_bf16_gemm"
        else:
            c = "other"
        cats[c] += t / 1e3
        top.append((round(t / 1e3, 1), c, n[:110], int(ev.count)))
    top.sort(reverse=True)
    tot = sum(cats.values())
    rep = {"executor": ex, "cuda_kernel_ms_4k_denoise": {k: round(v, 1) for k, v in cats.items()}, "total_kernel_ms": round(tot, 1),
           "share": {k: round(v / max(tot, 1e-9), 4) for k, v in cats.items()}, "denoise_wall_s_4096": rec["denoise_s"]["4096"],
           "top_kernels": [{"ms": a, "cat": b, "name": c, "count": d} for a, b, c, d in top[:25]]}
    json.dump(rep, open(out_dir / f"profile4k_{ex}.json", "w"), indent=1)
    log(f"4K denoise CUDA kernel time {tot/1e3:.1f}s (wall {rec['denoise_s']['4096']:.1f}s): " + ", ".join(f"{k} {v/1e3:.1f}s ({rep['share'][k]*100:.0f}%)" for k, v in cats.items()))
    for a, b, c, d in top[:12]:
        log(f"   {a:8.1f} ms  {b:16s} x{d:<5d} {c}")


# --------------------------------------------------------------------------- per-forward numerics probe
def forward_error(args):
    """Numerics sanity (independent of latency): relative L2 error / cosine between one forward of the bf16
    transformer and of the INT8 transformer on the same input.  Input = real T5/CLIP embeddings + 1K packed
    latent: step 0 (t=1, pure noise) and step 2 (t=0.5, x_t obtained by two bf16 sampling steps)."""
    from pipeline_flux import FluxPipeline                              # noqa: E402
    from transformer_flux import FluxTransformer2DModel                 # noqa: E402
    from pyraquant.quant_unet import _ActFakeQuant                      # noqa: E402  (per-token dynamic symmetric A8 QDQ, fp32 math)
    _check_release_modules(FluxPipeline, FluxTransformer2DModel)
    int8_exec.set_backend(args.int8_backend)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda *a: print("[fwd-err]", *a, flush=True)
    name, prompt, seed = read_prompts(Path(args.prompts_file), 1, args.offset)[0]
    pipe = FluxPipeline.from_pretrained(args.model_id, revision=args.model_revision, transformer=None, torch_dtype=torch.bfloat16,
                                        local_files_only=args.local_files_only)
    pipe.text_encoder.to("cuda"); pipe.text_encoder_2.to("cuda")
    with torch.no_grad():
        pe, pp, ids = pipe.encode_prompt(prompt=prompt, prompt_2=None, device="cuda", num_images_per_prompt=1, max_sequence_length=256)
    pe, pp, ids = pe.detach(), pp.detach(), ids.detach()
    sched = pipe.scheduler
    pipe.text_encoder.to("cpu"); pipe.text_encoder_2.to("cpu"); pipe.text_encoder = None; pipe.text_encoder_2 = None
    gc.collect(); torch.cuda.empty_cache()
    tf = FluxTransformer2DModel.from_pretrained(args.model_id, revision=args.model_revision, subfolder="transformer", torch_dtype=torch.bfloat16,
                                                local_files_only=args.local_files_only).to("cuda")
    tf.NPAttn = False; tf.query_random_jitter = False
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
    mu = calculate_shift(4096, sched.config.base_image_seq_len, sched.config.max_image_seq_len, sched.config.base_shift, sched.config.max_shift)
    timesteps, _ = retrieve_timesteps(sched, 4, "cuda", sigmas=np.linspace(1.0, 1 / 4, 4), mu=mu)
    g = torch.Generator(device="cuda").manual_seed(seed)
    lat = torch.randn((1, 16, 128, 128), generator=g, device="cuda", dtype=torch.bfloat16)
    img_ids = pipe._prepare_latent_image_ids(1, 64, 64, "cuda", torch.bfloat16)

    def fwd(model, x4d, t):
        packed = pipe._pack_latents(x4d, 1, 16, 128, 128)
        with torch.no_grad():
            out = model(hidden_states=packed, timestep=t.expand(1).to(torch.bfloat16) / 1000, guidance=None, pooled_projections=pp,
                        encoder_hidden_states=pe, txt_ids=ids, img_ids=img_ids, joint_attention_kwargs={}, return_dict=False)[0]
        return pipe._unpack_latents(out, 128, 128)
    # bf16 reference: inputs / outputs of step 0 and step 2
    inputs, ref = {}, {}
    x = lat.clone(); sched._step_index = None                          # retrieve_timesteps already called set_timesteps; step() initialises step_index from t
    for i, t in enumerate(timesteps):
        if i in (0, 2):
            inputs[i] = x.clone(); ref[i] = fwd(tf, x, t).float()
            v = ref[i].to(torch.bfloat16)
        else:
            v = fwd(tf, x, t)
        x = sched.step(v, t, x, return_dict=False)[0]
        if i == 2:
            break
    rep = {"prompt": name, "int8_backend": int8_exec.BACKEND, "steps": {},
           "note": "sim = QDQ simulation math of the quality tables (per-channel W8 absmax QDQ on weights + per-token A8 fake-quant hooks, bf16 GEMM); "
                   "int8 = real INT8 kernels (same codes/scales). int8-vs-sim gap = bf16 rounding of dequantised inputs + accumulation order only."}
    # QDQ simulation (same A8 math as the quality tables): per-channel weight QDQ in place + per-token A8 hooks
    hooks = []
    for n_, m_ in tf.named_modules():
        if isinstance(m_, nn.Linear) and not int8_exec.is_bf16_whitelisted(n_):
            w_ = m_.weight.data; s_ = w_.float().abs().amax(1, keepdim=True).clamp_min(1e-8) / 127.0
            m_.weight.data = (torch.round(w_.float() / s_).clamp_(-127, 127) * s_).to(w_.dtype)
            hooks.append(m_.register_forward_pre_hook(_ActFakeQuant(8)))
    sim = {i: fwd(tf, inputs[i], timesteps[i]).float() for i in (0, 2)}
    for h in hooks:
        h.remove()
    t0 = time.perf_counter(); int8_exec.convert_linears_to_int8(tf, device="cuda"); int8_exec.warm_kernels()   # re-quantising the QDQ weights gives the same codes/scales
    log(f"converted to INT8 in {time.perf_counter()-t0:.0f}s")

    def _cmp(y, r):
        d = y - r
        return {"rel_l2": float(d.norm() / r.norm()), "cosine": float((y.flatten() @ r.flatten()) / (y.norm() * r.norm())),
                "max_abs": float(d.abs().max()), "snr_db": float(20 * torch.log10(r.norm() / d.norm()))}
    for i in (0, 2):
        y = fwd(tf, inputs[i], timesteps[i]).float(); r = ref[i]
        k = f"step{i}_t{float(timesteps[i]):.0f}"
        rep["steps"][k] = {"int8_vs_bf16": _cmp(y, r), "sim_vs_bf16": _cmp(sim[i], r), "int8_vs_sim": _cmp(y, sim[i]),
                           "ref_rms": float(r.pow(2).mean().sqrt())}
        log(f"1K forward {k}: int8 vs bf16 rel L2 {rep['steps'][k]['int8_vs_bf16']['rel_l2']:.4f} (SNR {rep['steps'][k]['int8_vs_bf16']['snr_db']:.1f} dB) | "
            f"QDQ-sim vs bf16 {rep['steps'][k]['sim_vs_bf16']['rel_l2']:.4f} | int8 vs QDQ-sim {rep['steps'][k]['int8_vs_sim']['rel_l2']:.4f} (SNR {rep['steps'][k]['int8_vs_sim']['snr_db']:.1f} dB)")
    json.dump(rep, open(out_dir / "forward_error.json", "w"), indent=1)
    log(f"saved {out_dir / 'forward_error.json'}")


# --------------------------------------------------------------------------- summary
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


def summarize(args):
    out_dir = Path(args.out_dir)
    res = {"gpu": None, "executors": {}, "kernel_bench": None, "psnr_sanity": {}, "caveats": []}
    kb = out_dir / "kernel_bench.json"                                   # optional: output of `python deploy/int8_exec.py kernel_bench.json`
    if kb.exists():
        res["kernel_bench"] = json.load(open(kb))
    for ex in ("bf16", "int8", "ours"):
        p = out_dir / f"rec_{ex}.json"
        if not p.exists():
            continue
        d = json.load(open(p)); meta, recs = d["meta"], [r for r in d["records"] if not r["warmup"]]
        res["gpu"] = meta["gpu"]
        e = {"n": len(recs), "meta": {k: meta[k] for k in ("int8_backend", "int8_report", "plan", "leaf_latent", "tau", "route", "resident_alloc_gib", "transformer_bytes_gib", "notes") if k in meta},
             "total_s": _stats([r["total_s"] for r in recs]),
             "stage_wall_s": {px: _stats([r["stage_wall_s"][px] for r in recs]) for px in ("1024", "2048", "4096")},
             "denoise_s": {px: _stats([r["denoise_s"][px] for r in recs]) for px in ("1024", "2048", "4096")},
             "vae_decode_s": {px: _stats([r["vae_decode_s"][px] for r in recs]) for px in ("1024", "2048", "4096")},
             "vae_encode_s": {px: _stats([r["vae_encode_s"][px] for r in recs]) for px in ("2048", "4096")},
             "select_s": {px: _stats([r["select_s"][px] for r in recs]) for px in ("2048", "4096")},
             "peak_alloc_gib": {px: _stats([r["peak_alloc_gib"][px] for r in recs]) for px in ("1024", "2048", "4096")},
             "text_encode_s_excluded": _stats([r["text_encode_s_excluded"] for r in recs]),
             "executed_token_frac_4096_incl_halo": _stats([r["executed_token_frac_4096_incl_halo"] for r in recs]),
             "selected_leaf_frac_4096": _stats([(r["coverage"].get("512", {}).get("hi", 0) + r["coverage"].get("512", {}).get("lo", 0)) if r["coverage"] else 1.0 for r in recs]),
             "selected_leaf_frac_2048": _stats([(r["coverage"].get("256", {}).get("hi", 0) + r["coverage"].get("256", {}).get("lo", 0)) if r["coverage"] else 1.0 for r in recs]),
             "transformer_forward_calls_4096": _stats([r["transformer_forward_calls"].get("4096") for r in recs]),
             "per_prompt": [{"prompt_name": r["prompt_name"], "total_s": r["total_s"], "stage_wall_s": r["stage_wall_s"], "peak4k": r["peak_alloc_gib"]["4096"],
                             "exec4k": r["executed_token_frac_4096_incl_halo"]} for r in recs]}
        res["executors"][ex] = e
    E = res["executors"]
    if "bf16" in E:
        for ex in ("int8", "ours"):
            if ex in E:
                E[ex]["speedup_vs_bf16"] = {k: (E["bf16"][k]["median"] / E[ex][k]["median"]) for k in ("total_s",)}
                E[ex]["speedup_vs_bf16"].update({f"stage_{px}": E["bf16"]["stage_wall_s"][px]["median"] / E[ex]["stage_wall_s"][px]["median"] for px in ("1024", "2048", "4096")})
        if "int8" in E and "ours" in E:
            E["ours"]["speedup_vs_int8"] = {"total_s": E["int8"]["total_s"]["median"] / E["ours"]["total_s"]["median"],
                                            "stage_4096": E["int8"]["stage_wall_s"]["4096"]["median"] / E["ours"]["stage_wall_s"]["4096"]["median"]}
    warm = {ex: sorted(out_dir.glob(f"warmup_{ex}_*_4096.png")) for ex in ("bf16", "int8", "ours")}
    if warm["bf16"]:
        for ex in ("int8", "ours"):
            if warm[ex]:
                res["psnr_sanity"][f"{ex}_vs_bf16_4096_db"] = _psnr(warm[ex][0], warm["bf16"][0])
    # per-stage PSNR (a --profile-4k re-run in <out-dir>/profile keeps the 1K/2K/4K warm-up images) + determinism check (main run vs re-run, 4K hash)
    prof_dir = out_dir / "profile"
    stage = {}
    for px in (1024, 2048, 4096):
        im = {ex: sorted(prof_dir.glob(f"warmup_{ex}_*_{px}.png")) for ex in ("bf16", "int8", "ours")}
        if im["bf16"]:
            for a, b in (("int8", "bf16"), ("ours", "bf16"), ("ours", "int8")):
                if im[a] and im[b]:
                    stage[f"{a}_vs_{b}_{px}_db"] = _psnr(im[a][0], im[b][0])
    res["psnr_sanity"]["per_stage"] = stage
    det = {}
    for ex in ("bf16", "int8", "ours"):
        a = sorted(prof_dir.glob(f"warmup_{ex}_*_4096.png"))
        if warm[ex] and a:
            det[ex] = {"rerun_bitwise_identical_4096": _sha256_file(a[0]) == _sha256_file(warm[ex][0]),
                       "rerun_psnr_db": _psnr(a[0], warm[ex][0])}
    res["determinism_check"] = det
    fe = out_dir / "forward_error.json"
    res["forward_error_sanity"] = json.load(open(fe)) if fe.exists() else None
    res["profile_4k_denoise"] = {}
    for ex in ("bf16", "int8", "ours"):
        pf = prof_dir / f"profile4k_{ex}.json"
        if pf.exists():
            d = json.load(open(pf)); res["profile_4k_denoise"][ex] = {k: d[k] for k in ("cuda_kernel_ms_4k_denoise", "total_kernel_ms", "share", "denoise_wall_s_4096")}
            res["profile_4k_denoise"][ex]["top_kernels"] = d["top_kernels"][:8]
    res["caveats"] = [
        "Wall-clock of the ScaleDiff-FLUX.1-schnell 1K->2K->4K ladder (4/2/2 NFE, NPA windows) on one GPU with all weights resident (no CPU offload); text encoding (T5-XXL+CLIP) is identical across executors and excluded (reported separately).",
        "int8/ours use the same INT8 kernels (deploy/int8_exec.py: Triton per-token absmax quantisation + Triton fused INT8 GEMM with the per-row/per-column dequant epilogue in registers; the cuBLAS torch._int_mm + separate epilogue path is available with --int8-backend cublas but its int32 round trip makes it slower than bf16 at 1K); attention (SDPA/flash) is bf16 in all three; no torch.compile or CUDA graphs anywhere.",
        "The INT8 numerics are per-channel-W / per-token-A absmax and differ from the QDQ simulation used for the quality tables; quality is NOT measured here (the 4K PSNR of the warm-up prompt is a numerics sanity check only).",
        "ours collapses the hi/lo (W4/W3) dual pass into ONE INT8 pass: the W3-vs-W4 distinction is a storage/BOPs effect only and contributes nothing to wall-clock; the kernel speed-up is credited to the base quantiser, the routed-vs-full gap is the executor effect.",
        "ours 2K stage runs the full canvas (the recipe has no cache at 2K), so its 2K time equals int8 uniform; the executor saving is entirely at 4K, where only relgap-selected leaves (+halo) are computed and the rest comes from the spatial cache. The selected-leaf and executed-token fractions vary per prompt (selected_leaf_frac_4096 / executed_token_frac_4096_incl_halo); overlapping halos of adjacent leaf rows can exceed the full canvas when most leaves are selected, which is why the 4K stage has a per-prompt spread.",
        "Whitelist (x_embedder, context_embedder, time_text_embed.*, norm_out.linear, proj_out) and adaLN modulation Linears stay bf16 in int8/ours; peak memory reflects that mix (INT8 token-level weights + bf16 rest); text encoders are not resident (peak memory excludes T5-XXL).",
        "Why INT8-uniform gains little end-to-end: in the host's 4K full-canvas step the token-level GEMMs are only a minority of GPU time; most of it is memory-bound glue of the NPA implementation (per-tile K/V window gathers, fp32 RoPE/modulation elementwise) which INT8 does not touch (see --profile-4k; the INT8 Linear kernels themselves are faster than bf16 F.linear on every FLUX shape, see kernel_bench).",
        "Why the routed executor saves more than its token fraction: local patches (leaf rows + halo, absolute RoPE ids, dense attention within the patch) bypass the NPA window-gather glue, so part of the 4K gain is executor overhead avoided, not only tokens skipped. Report both the token fraction and the wall-clock.",
        "PSNR between executors is trajectory divergence, not degradation: per-token absmax A8 has a per-forward relative error of the same magnitude as the A8 QDQ simulation whose rows carry the quality claims (see --forward-error), and the 4-step distilled trajectory turns that into composition-level differences. INT8-kernel and QDQ-sim outputs are not bit-equivalent to each other (dynamic rounding noise decorrelates through the 57 blocks), so quality must be read from the QDQ tables; here the numerics sanity is forward_error_sanity + visual inspection of the warm-up images.",
        "The 4K VAE encode + decode and the 2K VAE are host-inherent and identical across executors; they are inside the stage wall-clock and bound the attainable end-to-end speed-up.",
    ]
    json.dump(res, open(args.json, "w"), indent=1)
    # ---- markdown table ----
    lines = [f"## E2E latency, ScaleDiff-FLUX.1-schnell 1K->2K->4K, 1x {res['gpu']}, weights resident (median [IQR] over n timed prompts)", "",
             "| executor | n | 1K wall (s) | 2K wall (s) | 4K wall (s) | total (s) | speed-up vs bf16 | 4K exec. tokens | 4K peak (GiB) |", "|---|---|---|---|---|---|---|---|---|"]
    for ex, e in E.items():
        f = lambda s: (f"{s['median']:.1f} [{s['iqr']:.1f}]" if s else "-")
        sp = e.get("speedup_vs_bf16", {}).get("total_s")
        ef = e["executed_token_frac_4096_incl_halo"]
        ef_str = (f"{ef['median']:.2f} [{ef['iqr']:.2f}]" if ex == "ours" else "1.00")
        lines.append(f"| {ex} | {e['n']} | {f(e['stage_wall_s']['1024'])} | {f(e['stage_wall_s']['2048'])} | {f(e['stage_wall_s']['4096'])} | {f(e['total_s'])} | "
                     f"{(f'{sp:.2f}x' if sp else '1.00x')} | {ef_str} | {e['peak_alloc_gib']['4096']['median']:.1f} (max {e['peak_alloc_gib']['4096']['max']:.1f}) |")
    lines += ["", "denoise-loop only (s, median): " + "; ".join(f"{ex}: " + "/".join(f"{e['denoise_s'][px]['median']:.1f}" for px in ("1024", "2048", "4096")) for ex, e in E.items()),
              "VAE per stage (s, median, identical across executors): " + "; ".join(f"{ex}: dec " + "/".join(f"{e['vae_decode_s'][px]['median']:.1f}" for px in ("1024", "2048", "4096")) + ", enc " + "/".join(f"{e['vae_encode_s'][px]['median']:.1f}" for px in ("2048", "4096")) for ex, e in E.items()),
              "PSNR sanity (warm-up prompt, 4K): " + ", ".join(f"{k}={v:.1f} dB" for k, v in res["psnr_sanity"].items() if k != "per_stage")
              + (" | per stage: " + ", ".join(f"{k}={v:.1f}" for k, v in stage.items()) if stage else "")]
    if res.get("forward_error_sanity"):
        lines.append("per-forward error at 1K (same input): " + "; ".join(
            f"{k}: int8 vs bf16 rel L2 {v['int8_vs_bf16']['rel_l2']:.4f} (SNR {v['int8_vs_bf16']['snr_db']:.1f} dB, cos {v['int8_vs_bf16']['cosine']:.4f}), "
            f"QDQ-sim vs bf16 {v['sim_vs_bf16']['rel_l2']:.4f}, int8 vs QDQ-sim {v['int8_vs_sim']['rel_l2']:.4f} (SNR {v['int8_vs_sim']['snr_db']:.1f} dB)"
            for k, v in res["forward_error_sanity"]["steps"].items()))
    if res.get("profile_4k_denoise"):
        lines.append("4K denoise-loop CUDA time by kernel class (profiler, 1 prompt): " + "; ".join(
            f"{ex}: " + ", ".join(f"{k} {v/1e3:.1f}s ({d['share'][k]*100:.0f}%)" for k, v in d["cuda_kernel_ms_4k_denoise"].items() if v > 0) for ex, d in res["profile_4k_denoise"].items()))
    if res.get("determinism_check"):
        parts = []
        for ex, v in res["determinism_check"].items():
            parts.append(f"{ex}: bitwise" if v["rerun_bitwise_identical_4096"] else f"{ex}: {v['rerun_psnr_db']:.1f} dB")
        lines.append("determinism (main run vs re-run, 4K PNG): " + ", ".join(parts))
    if "int8" in E and "ours" in E:
        lines.append(f"ours vs int8 (same numerics): total x{E['ours']['speedup_vs_int8']['total_s']:.2f}, 4K stage x{E['ours']['speedup_vs_int8']['stage_4096']:.2f}")
    md = "\n".join(lines)
    print(md)
    (out_dir / "summary.md").write_text(md + "\n\nCaveats:\n" + "\n".join("- " + c for c in res["caveats"]) + "\n")
    print(f"[e2e] saved {args.json} and {out_dir / 'summary.md'}")


# --------------------------------------------------------------------------- speed-up breakdown ladder
BREAKDOWN_LADDER = (   # (rung, description, executor whose rec_<executor>.json holds the rung; default file = <out-dir>/<rung>/rec_<executor>.json)
    ("bf16_full", "bf16 Linears, full canvas (host NPA)", "bf16"),
    ("int8_full", "INT8 Linears, full canvas (host NPA)", "int8"),
    ("int8_routed_nocache", "INT8, routed patch executor, ALL 4K leaves computed (no cache reuse)", "ours"),
    ("int8_routed_cache", "INT8, routed + spatial cache, relgap selection (final recipe)", "ours"),
    ("bf16_routed_cache", "bf16 Linears, routed + spatial cache (strategy only, no kernel)", "ours"),
)


def _arm_stats(path: Path):
    d = json.load(open(path)); meta, recs = d["meta"], [r for r in d["records"] if not r["warmup"]]
    sel4k = lambda r: ((r["coverage"].get("512", {}).get("hi", 0) + r["coverage"].get("512", {}).get("lo", 0)) if r["coverage"] else 1.0)
    e = {"rec_path": str(path), "n": len(recs), "executor": meta["executor"], "arm_recorded": meta.get("arm"), "linear": meta.get("linear", "bf16" if meta["executor"] == "bf16" else "int8"),
         "no_cache": meta.get("no_cache", False), "tau": meta.get("tau"), "plan": meta.get("plan"), "route": meta.get("route"), "int8_backend": meta.get("int8_backend"),
         "weight_source": meta.get("weight_source"), "gpu": meta["gpu"], "resident_alloc_gib": meta.get("resident_alloc_gib"),
         "total_s": _stats([r["total_s"] for r in recs]),
         "stage_wall_s": {px: _stats([r["stage_wall_s"][px] for r in recs]) for px in ("1024", "2048", "4096")},
         "denoise_s": {px: _stats([r["denoise_s"][px] for r in recs]) for px in ("1024", "2048", "4096")},
         "vae_decode_s": {px: _stats([r["vae_decode_s"][px] for r in recs]) for px in ("1024", "2048", "4096")},
         "vae_encode_s": {px: _stats([r["vae_encode_s"][px] for r in recs]) for px in ("2048", "4096")},
         "select_s": {px: _stats([r["select_s"][px] for r in recs]) for px in ("2048", "4096")},
         "peak_alloc_gib": {px: _stats([r["peak_alloc_gib"][px] for r in recs]) for px in ("1024", "2048", "4096")},
         "executed_token_frac_4096_incl_halo": _stats([r["executed_token_frac_4096_incl_halo"] for r in recs]),
         "selected_leaf_frac_4096": _stats([sel4k(r) for r in recs]),
         "transformer_forward_calls_4096": _stats([r["transformer_forward_calls"].get("4096") for r in recs]),
         "per_prompt": [{"prompt_name": r["prompt_name"], "total_s": r["total_s"], "stage_wall_s": r["stage_wall_s"], "denoise_4096": r["denoise_s"]["4096"],
                         "peak4k": r["peak_alloc_gib"]["4096"], "exec4k": r["executed_token_frac_4096_incl_halo"], "sel4k": sel4k(r)} for r in recs]}
    return e


def breakdown(args):
    """Five-rung ladder summary: per-stage median [IQR] / 4K peak memory / 4K executed tokens of each rung, and the
    decomposition ratios kernel=t1/t2, routing=t2/t3, cache=t3/t4, total=t1/t4, strategy-only=t1/t5."""
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    over = dict(kv.split("=", 1) for kv in (args.ladder or []))
    arms, missing = {}, []
    for arm, desc, executor in BREAKDOWN_LADDER:
        p = Path(over[arm]) if arm in over else out_dir / arm / f"rec_{executor}.json"
        if not p.exists():
            missing.append((arm, str(p))); continue
        arms[arm] = _arm_stats(p); arms[arm]["description"] = desc
    if missing:
        print("[breakdown] missing arms: " + ", ".join(f"{a} ({p})" for a, p in missing))
    med = lambda a, k="total_s": arms[a][k]["median"]
    st4 = lambda a: arms[a]["stage_wall_s"]["4096"]["median"]
    dn4 = lambda a: arms[a]["denoise_s"]["4096"]["median"]

    def ratio(num, den, f):
        return (f(num) / f(den)) if (num in arms and den in arms) else None
    pairs = {"kernel_gain_t1_over_t2": ("bf16_full", "int8_full"), "routing_effect_t2_over_t3": ("int8_full", "int8_routed_nocache"),
             "cache_gain_t3_over_t4": ("int8_routed_nocache", "int8_routed_cache"), "total_t1_over_t4": ("bf16_full", "int8_routed_cache"),
             "strategy_only_bf16_t1_over_t5": ("bf16_full", "bf16_routed_cache"), "kernel_on_top_of_strategy_t5_over_t4": ("bf16_routed_cache", "int8_routed_cache"),
             "strategy_on_top_of_kernel_t2_over_t4": ("int8_full", "int8_routed_cache")}
    decomp = {k: {"total": ratio(a, b, med), "stage_4096": ratio(a, b, st4), "denoise_4096": ratio(a, b, dn4)} for k, (a, b) in pairs.items()}
    prof = {}
    for ex in ("bf16", "int8", "ours"):
        pf = out_dir / "profile" / f"profile4k_{ex}.json"                # optional: --profile-4k runs stored in <out-dir>/profile
        if pf.exists():
            d = json.load(open(pf)); prof[ex] = {k: d[k] for k in ("cuda_kernel_ms_4k_denoise", "total_kernel_ms", "share", "denoise_wall_s_4096")}
    tau = arms["int8_routed_cache"]["tau"] if "int8_routed_cache" in arms else None
    tau_txt = f"tau={tau}" if tau is not None else "tau as recorded"
    halo = ((arms["int8_routed_nocache"].get("route") or {}).get("halo") if "int8_routed_nocache" in arms else None)
    res = {"gpu": next(iter(arms.values()))["gpu"] if arms else None, "ladder": [a for a, _, _ in BREAKDOWN_LADDER], "arms": arms, "decomposition": decomp,
           "profile_4k_denoise_from_e2e_run": prof, "missing": missing,
           "definitions": {"t1": "bf16_full", "t2": "int8_full", "t3": "int8_routed_nocache", "t4": f"int8_routed_cache ({tau_txt})", "t5": f"bf16_routed_cache ({tau_txt})",
                           "ratios": "median total wall-clock (1K+2K+4K, text encoding excluded) of the numerator arm over the denominator arm; stage_4096 / denoise_4096 = same ratio on the 4K stage / 4K denoise loop only",
                           "executed_token_frac_4096_incl_halo": "sum over the two 4K steps of routed patch tokens (leaf rows merged per row + halo, clamped at the canvas) / (2 x 65536); full-canvas arms = 1.00 by definition"}}
    res["caveats"] = [
        "All five arms: ScaleDiff-FLUX.1-schnell 1K->2K->4K (4/2/2 NFE), one GPU, weights resident, the same timed prompts + warm-up, text encoding excluded; every rung is a rec_*.json written by this script (see --ladder / rec_path for the file behind each rung).",
        f"int8_routed_nocache = every one of the 64 4K leaves selected (threshold 0 on the median rule, no 2K inheritance) so the driver builds 8 full-row route boxes and a cache mask that is identically zero: every leaf goes through the routed patch executor with the {halo if halo is not None else 'configured'}-token halo and nothing is reused. It is NOT cache_bg=False (the driver would fall back to the host's NPA full canvas). Its 1K/2K stages are the same work as int8_full.",
        "hi/lo (W4/W3) are collapsed into ONE INT8 pass in every routed arm: W3 vs W4 has zero wall-clock effect here (storage/BOPs only); the kernel gain is credited entirely to the per-channel-W / per-token-A INT8 Triton GEMM.",
        "The INT8 numerics (per-channel weight requantisation + dynamic per-token absmax activations, fused epilogue) differ from the QDQ simulation that carries the quality tables; no quality claim is made from these runs.",
        "Attention (SDPA/flash) stays bf16 in every arm; no torch.compile / CUDA graphs anywhere; the whitelist (x_embedder, context_embedder, time_text_embed.*, norm_out.linear, proj_out) and adaLN modulation Linears stay bf16 in the INT8 arms.",
        "The routed executor bypasses the NPA window gathers (dense attention inside each patch), so its per-token cost is lower than the host's full-canvas step: part of the routing gain is executor overhead avoided, not tokens skipped; the nocache arm isolates exactly that (the per-kernel-class split is available from --profile-4k runs stored in <out-dir>/profile).",
        "The 2K stage of the routed arms runs the full canvas (no cache at 2K in the recipe) and the 4K VAE encode/decode is host-inherent, so the end-to-end ratios are bounded by those fixed costs; the 4K-stage and 4K-denoise columns show the executor effect without them.",
        "Per-prompt variance of the cache arms comes from the adaptive relgap selection (executed-token fraction incl. halo varies per prompt and can exceed 1.0 when most leaves are selected); the full-canvas and nocache arms are content-independent.",
    ]
    json.dump(res, open(args.json, "w"), indent=1)
    # ---- markdown ----
    f = lambda s, d=1: (f"{s['median']:.{d}f} [{s['iqr']:.{d}f}]" if s else "-")
    lines = [f"## Speed-up breakdown, ScaleDiff-FLUX.1-schnell 1K->2K->4K, 1x {res['gpu']}, weights resident, relgap {tau_txt} (median [IQR] over n timed prompts)", "",
             "| # | arm | Linear | 4K executor | n | 1K (s) | 2K (s) | 4K (s) | total (s) | 4K denoise (s) | 4K exec. tokens (incl. halo) | 4K leaves computed | 4K peak (GiB) | vs bf16_full |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    execs = {"bf16_full": "full canvas (NPA)", "int8_full": "full canvas (NPA)", "int8_routed_nocache": "routed patches, all leaves, no cache",
             "int8_routed_cache": "routed patches + spatial cache", "bf16_routed_cache": "routed patches + spatial cache"}
    for i, (arm, desc, _) in enumerate(BREAKDOWN_LADDER, 1):
        if arm not in arms:
            lines.append(f"| {i} | {arm} | - | - | - | (missing) | | | | | | | | |"); continue
        e = arms[arm]; full = e["executor"] != "ours"
        ef = "1.00" if full else f(e["executed_token_frac_4096_incl_halo"], 2)
        sl = "1.00" if full else f(e["selected_leaf_frac_4096"], 2)
        sp = med("bf16_full") / med(arm) if "bf16_full" in arms else float("nan")
        lines.append(f"| {i} | {arm} | {e['linear']} | {execs[arm]} | {e['n']} | {f(e['stage_wall_s']['1024'])} | {f(e['stage_wall_s']['2048'])} | {f(e['stage_wall_s']['4096'])} | "
                     f"{f(e['total_s'])} | {f(e['denoise_s']['4096'])} | {ef} | {sl} | {e['peak_alloc_gib']['4096']['median']:.1f} (max {e['peak_alloc_gib']['4096']['max']:.1f}) | {sp:.2f}x |")
    lines += ["", "Decomposition (ratio of median wall-clock; total / 4K stage / 4K denoise loop):", "", "| factor | ratio | total | 4K stage | 4K denoise |", "|---|---|---|---|---|"]
    names = {"kernel_gain_t1_over_t2": "INT8 kernel (quantisation)", "routing_effect_t2_over_t3": "block routing / patch execution (no cache)", "cache_gain_t3_over_t4": "spatial cache reuse",
             "total_t1_over_t4": "total (final recipe)", "strategy_only_bf16_t1_over_t5": "strategy only at bf16 (no kernel)", "kernel_on_top_of_strategy_t5_over_t4": "kernel on top of strategy",
             "strategy_on_top_of_kernel_t2_over_t4": "strategy on top of kernel"}
    for k, (a, b) in pairs.items():
        d = decomp[k]
        g = lambda v: (f"{v:.2f}x" if v else "-")
        lines.append(f"| {names[k]} | {a} / {b} | {g(d['total'])} | {g(d['stage_4096'])} | {g(d['denoise_4096'])} |")
    if all(a in arms for a in ("int8_full", "int8_routed_nocache")):
        r = decomp["routing_effect_t2_over_t3"]
        lines += ["", (f"Routing without cache is {'SLOWER' if r['total'] < 1 else 'faster'} than the INT8 full canvas end-to-end ({r['total']:.2f}x; 4K stage {r['stage_4096']:.2f}x, 4K denoise {r['denoise_4096']:.2f}x) "
                       f"while executing {arms['int8_routed_nocache']['executed_token_frac_4096_incl_halo']['median']:.2f}x the canvas tokens (halo overhead); "
                       f"per executed token the patch executor is {r['denoise_4096'] * arms['int8_routed_nocache']['executed_token_frac_4096_incl_halo']['median']:.2f}x cheaper than the host's NPA full-canvas step.")]
    if prof:
        lines.append("4K denoise-loop CUDA time by kernel class (profiler, e2e run, 1 prompt): " + "; ".join(
            f"{ex}: " + ", ".join(f"{k} {v/1e3:.1f}s ({d['share'][k]*100:.0f}%)" for k, v in d["cuda_kernel_ms_4k_denoise"].items() if v > 0) for ex, d in prof.items()))
    md = "\n".join(lines)
    print(md)
    (out_dir / "summary.md").write_text(md + "\n\nCaveats:\n" + "\n".join("- " + c for c in res["caveats"]) + "\n")
    print(f"[breakdown] saved {args.json} and {out_dir / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--executor", choices=("bf16", "int8", "ours"), default="bf16")
    ap.add_argument("--weight-source", choices=("host", "method"), default="host",
                    help="host: derive the INT8 codes from the stock bf16 weights. "
                         "method: put the SVDQuant W4 package in first (--svdquant-dir) and derive the INT8 codes from those weights (the paper's PyraQuant row)")
    ap.add_argument("--svdquant-dir", type=Path, default=None,
                    help="directory of the SVDQuant external-quant package (the same one flux/run_flux.py reads); required with --weight-source method")
    ap.add_argument("--n", type=int, default=20, help="number of timed prompts (the paper: 20)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1, help="warm-up runs of the first prompt, excluded from the statistics (the paper: 1)")
    ap.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS,
                    help="JSONL with name/prompt (optional seed, default 42); default prompts/eval_ultrahr_2000.jsonl, of which the paper timed the first 20")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/latency_flux"), help="per-prompt records rec_<executor>.json, warm-up images, summary.md")
    ap.add_argument("--json", type=Path, default=None, help="summary / breakdown JSON (default: <out-dir>.json)")
    ap.add_argument("--tau", type=float, default=0.10, help="relative-gap threshold of the 2K/4K selection (the paper: 0.10)")
    ap.add_argument("--route-halo", type=int, default=2, help="halo in tokens around each routed 4K patch (the paper's deployment row: 2)")
    ap.add_argument("--route-batch", type=int, default=8, help="maximum number of patches per routed transformer call")
    ap.add_argument("--int8-backend", choices=("triton", "cublas"), default="triton",
                    help="INT8 GEMM backend of deploy/int8_exec: triton = fused INT8 GEMM + epilogue (the paper); cublas = torch._int_mm + Triton epilogue")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--model-revision", default=MODEL_REVISION)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--summarize", action="store_true", help="summarise rec_bf16/int8/ours.json of --out-dir into --json and summary.md")
    ap.add_argument("--forward-error", action="store_true", help="numerics sanity: one 1K forward, INT8 vs bf16 vs A8 QDQ simulation (not part of the latency table)")
    ap.add_argument("--profile-4k", action="store_true", help="1 warm-up + 1 profiled prompt: torch.profiler over the 4K denoise loop only, CUDA time by kernel class (not part of the latency table)")
    ap.add_argument("--linear", choices=("int8", "bf16"), default=None,
                    help="Linear numerics (default: bf16 for --executor bf16, INT8 otherwise); ours --linear bf16 = strategy-only rung (no kernel)")
    ap.add_argument("--no-cache", action="store_true", help="(ours) select all 64 4K leaves -> every leaf through the routed patch executor, cache weight 0 (ladder rung 3)")
    ap.add_argument("--breakdown", action="store_true", help="summarise the five-rung speed-up ladder (BREAKDOWN_LADDER) from <out-dir>/<rung>/rec_*.json into --json and summary.md")
    ap.add_argument("--ladder", action="append", default=None, help="override the record file of a rung: rung=path (repeatable)")
    args = ap.parse_args()
    if args.json is None:
        out_dir = Path(args.out_dir)
        args.json = out_dir.parent / (out_dir.name + ".json")
    if args.breakdown:
        breakdown(args)
    elif args.summarize:
        summarize(args)
    elif args.forward_error:
        forward_error(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
