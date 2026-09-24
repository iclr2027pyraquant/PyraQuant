# PyraQuant — anonymous code release (ICLR 2027 submission)

Training-free, spatially adaptive mixed-precision and cached inference for progressive
high-resolution diffusion (1K → 2K → 4K).  This repository contains the inference code for the
two **main configurations** reported in the paper:

| Pipeline | Backbone | Base quantizer | Low-precision state | Cached regions |
|---|---|---|---|---|
| FLUX (DiT) | ScaleDiff-FLUX.1-schnell, 4/2/2 steps at 1K/2K/4K | released SVDQuant W4A4 | nested W3 (derived from the stored W4 codes) | yes (4K stage) |
| SDXL (UNet) | ScaleDiff-SDXL, 50/20/20 steps at 1K/2K/4K | round-to-nearest W8A8 (group 32) | nested W4 (re-rounded from the W8 codes) | yes (4K stage) |

Each script generates 4096 × 4096 images for a few example prompts.  Ablations, hyper-parameter
sweeps, competitor baselines and the INT8 deployment executor are **not** included.

Precision is simulated with quantize–dequantize (QDQ) weights and activations, exactly as in the
paper's quality experiments; no low-bit kernels are required.

## 1. Environment

Python 3.11, CUDA 12.x.  The pinned versions in `requirements.txt` are the ones used for the paper.

```bash
conda create -n pyraquant python=3.11 -y
conda activate pyraquant
pip install -r requirements.txt
```

A single 24 GB GPU is sufficient.  FLUX runs with sequential CPU offload and keeps the transformer
and its nested-W3 copy in host memory (≥ 96 GB of host RAM recommended).

## 2. Models

Both scripts download from the Hugging Face Hub unless `--local-files-only` is given.

* **FLUX.1-schnell** (`black-forest-labs/FLUX.1-schnell`, revision `741f7c3ce8b383c54771c7003378a50191e9efe9`)
* **SDXL base 1.0** (`stabilityai/stable-diffusion-xl-base-1.0`, revision `462165984030d82259a11f4367a4eed129e94a7b`, fp16 variant)
* **SVDQuant W4A4 package for FLUX.1-schnell** (required by `flux/run_flux.py --svdquant-dir`).
  The FLUX base quantizer is the released SVDQuant checkpoint `mit-han-lab/svdq-int4-flux.1-schnell`.
  Its kernel-packed tensors are converted once into a framework-neutral "external-quant" package
  (dequantized bf16 weights + activation scales + low-rank branches, ≈ 24 GB on disk) that the
  QDQ simulation reads:
  ```bash
  # deepcompressor provides the nunchaku tile layout; install it without its (unpinned) dependencies
  pip install --no-deps git+https://github.com/mit-han-lab/deepcompressor@69f3473f5e1c1504bae35cc50c7858ef900a9b17
  hf download mit-han-lab/svdq-int4-flux.1-schnell --local-dir ckpt/svdq-int4-flux.1-schnell
  python tools/svdquant_export.py --ckpt ckpt/svdq-int4-flux.1-schnell --out ckpt/svdquant_flux_w4a4
  ```
  The conversion runs on the CPU.  The same checkpoint is also published as a single file in
  `nunchaku-tech/nunchaku-flux.1-schnell` (`svdq-int4_r32-flux.1-schnell.safetensors`); that file
  can be passed directly to `--ckpt`.

## 3. Run

```bash
# FLUX (DiT), main configuration: SVDQuant W4A4 + nested W3 + cache, routed executor
python flux/run_flux.py --svdquant-dir ckpt/svdquant_flux_w4a4 --output-dir outputs/flux --n 4

# SDXL (UNet), main configuration: RTN W8A8 g32 + nested W4 + cache, routed executor
python sdxl/run_sdxl.py --output-dir outputs/sdxl --n 4 --offset 4
```

Common options: `--prompt "..."` (single prompt), `--prompts-file` (JSONL with `name`/`prompt`,
default `prompts/examples.jsonl`; the first four prompts are the FLUX examples, the next four the
SDXL examples), `--n`/`--offset`, `--seed` (default 42), `--config` (default
`configs/<pipeline>_pyraquant.json`), `--save-masks` (also writes the per-stage active / high-precision
leaf maps as PNG), `--local-files-only`.

The evaluation prompts of the paper are in `prompts/eval_ultrahr_2000.jsonl` (2,000 UltraHR-100K
captions); all generations use seed 42 (`--seed`), for every compared method.

Outputs: `<name>_4096.png` per prompt and a `records.json` with the per-stage coverage
(high / low / cached fraction of the canvas).

### Threshold calibration

The relative-gap threshold τ is selected on 32 COCO captions (`prompts/calibration_coco32.jsonl`)
that are disjoint from the evaluation prompts:

```bash
python tools/threshold_sweep.py --pipeline sdxl --taus 0.03 0.05 0.10 0.15 --output-dir outputs/sweep_sdxl
python tools/threshold_sweep.py --pipeline flux --taus 0.05 0.10 0.20 --output-dir outputs/sweep_flux --svdquant-dir ckpt/svdquant_flux_w4a4
```

Each τ runs the main configuration with that threshold on the calibration captions and
`sweep_summary.json` reports the mean allocation per stage (high / low / cached fractions) and the
computed fraction of the 4K canvas.

## 4. Where the method lives

| Paper component | Code |
|---|---|
| Residual score e(B) = RMS(L_FM − L_U) per leaf | `leaf_scores_residual` in `flux/run_flux.py`, `sdxl/run_sdxl.py` |
| Relative-gap sibling rule (τ = 0.10) | `_relgap_select`, `build_leaf_hi` |
| 4K rule, FLUX: cascaded inheritance + relative-gap test inside the inherited sibling groups + median split | `make_spatial_hook` in `flux/run_flux.py` (`inherit`, `thresh`/`norm`, `inner_thresh`) |
| 4K rule, SDXL: all children of the 2K high-precision leaves stay active + median split | `make_spatial_hook` in `sdxl/run_sdxl.py` (`inherit_hi`, `inner_thresh`) |
| Cached reuse of the stage-entry prediction outside the active regions | `_cache_mask` / `_x0_cache` handling in `flux/pipeline_flux.py`, `sdxl/pipeline_sdxl.py` |
| Routed executor (only active leaves + halo are computed) | `flux/routing.py`, `get_routed_noise_pred` in `flux/pipeline_flux.py`; `routed_noise_pred` in `sdxl/pipeline_sdxl.py` |
| Hi/lo dual pass blended with the per-leaf mask | `SpatialMaskTransformer`, `SpatialMaskUNet` |
| Nested W3 from SVDQuant W4 codes | `build_extq_lo_variant` in `pyraquant/external_quant.py` |
| RTN W8A8 (g32) and nested W4 for SDXL | `build_recipe_variant` in `pyraquant/sim_recipes.py` (`rtn`, `nested4`) |
| Activation fake-quantization hooks | `_ActFakeQuant` / `_ActFakeQuantConv` in `pyraquant/quant_unet.py` (SDXL), `ExtActQuant` in `pyraquant/external_quant.py` (FLUX) |

The stage plans (which precision state is used at 1K / 2K / 4K, the thresholds, the executor and
the halo) are in `configs/flux_pyraquant.json` and `configs/sdxl_pyraquant.json`.

## 5. Acknowledgements and licenses

The progressive 1K → 2K → 4K pipelines (`flux/pipeline_flux.py`, `flux/transformer_flux.py`,
`sdxl/pipeline_sdxl.py`, `sdxl/attention_scalediff.py`) are derived from ScaleDiff
(Apache-2.0) and from the Hugging Face diffusers library (Apache-2.0); the quantized FLUX base
is the released SVDQuant checkpoint (MIT-HAN-Lab).  Our additions are released under the MIT
license (see `LICENSE`).
