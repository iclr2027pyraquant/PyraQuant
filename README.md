# PyraQuant — anonymous code release (ICLR 2027 submission)

Training-free, spatially adaptive mixed-precision and cached inference for progressive
high-resolution diffusion (1K → 2K → 4K).  Inference code for the two main configurations of the paper:

| Pipeline | Backbone | Base quantizer | Low-precision state | Cached regions |
|---|---|---|---|---|
| FLUX (DiT) | ScaleDiff-FLUX.1-schnell, 4/2/2 steps at 1K/2K/4K | released SVDQuant W4A4 | nested W3 (derived from the stored W4 codes) | yes (4K stage) |
| SDXL (UNet) | ScaleDiff-SDXL, 50/20/20 steps at 1K/2K/4K | round-to-nearest W8A8 (group 32) | nested W4 (re-rounded from the W8 codes) | yes (4K stage) |

Precision is simulated with quantize–dequantize (QDQ) weights and activations, as in the paper's
quality experiments; the quality scripts need no low-bit kernels.  The INT8 latency executor of the
deployment table is in `deploy/` (section 5).

## Contents

| Path | What it is |
|---|---|
| `flux/run_flux.py`, `sdxl/run_sdxl.py` | drivers for the two pipelines (selection, inheritance, cache, routed executor) |
| `flux/`, `sdxl/` | progressive pipelines (from ScaleDiff) with the stage-entry hook, cached reuse and routed executor |
| `pyraquant/` | quantizers: RTN, nested W3/W4, activation fake-quantization, SVDQuant package loader |
| `configs/flux_pyraquant.json`, `configs/sdxl_pyraquant.json` | stage plans of the two main configurations (states, threshold τ = 0.10, executor, halo) |
| `prompts/examples.jsonl` | 8 example prompts (first four FLUX, next four SDXL) |
| `prompts/eval_ultrahr_2000.jsonl` | the 2,000 UltraHR-eval4K evaluation prompts, in the order used by the paper |
| `prompts/calibration_coco32.jsonl` | the 32 COCO captions used to select τ |
| `tools/threshold_sweep.py` | threshold sweep on the calibration captions |
| `tools/svdquant_export.py`, `tools/svdquant_unpack.py` | conversion of the released SVDQuant checkpoint into the package read by `run_flux.py` |
| `deploy/int8_exec.py` | INT8 kernels (Triton): per-token activation quantization, fused INT8 GEMM with dequant epilogue and low-rank up-projection, `Int8Linear` |
| `deploy/run_flux_latency.py`, `deploy/run_sdxl_latency.py` | latency drivers of the deployment table and the decomposition ladder |

All generations use seed 42 (`--seed`).

## 1. Environment

Python 3.11, CUDA 12.x, one 24 GB GPU.  FLUX runs with sequential CPU offload and keeps the
transformer and its nested-W3 copy in host memory (≥ 96 GB of host RAM recommended).  The latency
executor in `deploy/` needs one 48 GB GPU with all weights resident.

```bash
conda create -n pyraquant python=3.11 -y
conda activate pyraquant
pip install -r requirements.txt
```

## 2. Models

Both scripts download from the Hugging Face Hub unless `--local-files-only` is given.

* **FLUX.1-schnell** (`black-forest-labs/FLUX.1-schnell`, revision `741f7c3ce8b383c54771c7003378a50191e9efe9`)
* **SDXL base 1.0** (`stabilityai/stable-diffusion-xl-base-1.0`, revision `462165984030d82259a11f4367a4eed129e94a7b`, fp16 variant)
* **SVDQuant W4A4 package for FLUX.1-schnell** (required by `flux/run_flux.py --svdquant-dir`), built once
  from the released checkpoint `mit-han-lab/svdq-int4-flux.1-schnell` (CPU only, ≈ 24 GB on disk):
  ```bash
  pip install --no-deps git+https://github.com/mit-han-lab/deepcompressor@69f3473f5e1c1504bae35cc50c7858ef900a9b17
  hf download mit-han-lab/svdq-int4-flux.1-schnell --local-dir ckpt/svdq-int4-flux.1-schnell
  python tools/svdquant_export.py --ckpt ckpt/svdq-int4-flux.1-schnell --out ckpt/svdquant_flux_w4a4
  ```
  The same checkpoint is also published as a single file in `nunchaku-tech/nunchaku-flux.1-schnell`
  (`svdq-int4_r32-flux.1-schnell.safetensors`); that file can be passed directly to `--ckpt`.

## 3. Run

```bash
# FLUX (DiT): SVDQuant W4A4 + nested W3 + cache, routed executor
python flux/run_flux.py --svdquant-dir ckpt/svdquant_flux_w4a4 --output-dir outputs/flux --n 4

# SDXL (UNet): RTN W8A8 g32 + nested W4 + cache, routed executor
python sdxl/run_sdxl.py --output-dir outputs/sdxl --n 4 --offset 4
```

Options: `--prompt "..."` (single prompt), `--prompts-file` (JSONL with `name`/`prompt`, default
`prompts/examples.jsonl`; use `prompts/eval_ultrahr_2000.jsonl` for the paper's evaluation set),
`--n`/`--offset`, `--seed` (default 42), `--config` (default `configs/<pipeline>_pyraquant.json`),
`--save-masks` (also writes the per-stage active / high-precision leaf maps as PNG), `--local-files-only`.

Outputs: `<name>_4096.png` per prompt and `records.json` with the per-stage coverage
(high / low / cached fraction of the canvas).

### Threshold sweep

```bash
python tools/threshold_sweep.py --pipeline sdxl --taus 0.03 0.05 0.10 0.15 --output-dir outputs/sweep_sdxl
python tools/threshold_sweep.py --pipeline flux --taus 0.05 0.10 0.20 --output-dir outputs/sweep_flux --svdquant-dir ckpt/svdquant_flux_w4a4
```

Runs the calibration captions with each τ (sibling test at 2K and 4K) and writes `sweep_summary.json`
with the mean allocation per stage and the computed fraction of the 4K canvas.  `--keep-4k-rule`
keeps the 4K rule of the config instead (for SDXL: the final recipe).

## 4. Where the method lives

| Paper component | Code |
|---|---|
| Residual score e(B) = RMS(L_FM − L_U) per leaf | `leaf_scores_residual` in `flux/run_flux.py`, `sdxl/run_sdxl.py` |
| Relative-gap sibling rule (τ = 0.10) | `_relgap_select`, `build_leaf_hi` |
| 4K rule, FLUX: inheritance + relative-gap test + median split | `make_spatial_hook` in `flux/run_flux.py` (`inherit`, `thresh`/`norm`, `inner_thresh`) |
| 4K rule, SDXL: children of the 2K high-precision leaves + median split | `make_spatial_hook` in `sdxl/run_sdxl.py` (`inherit_hi`, `inner_thresh`) |
| Cached reuse outside the active regions | `_cache_mask` / `_x0_cache` in `flux/pipeline_flux.py`, `sdxl/pipeline_sdxl.py` |
| Routed executor (active leaves + halo only) | `flux/routing.py`, `get_routed_noise_pred` in `flux/pipeline_flux.py`; `routed_noise_pred` in `sdxl/pipeline_sdxl.py` |
| Hi/lo dual pass blended with the per-leaf mask | `SpatialMaskTransformer`, `SpatialMaskUNet` |
| Nested W3 from SVDQuant W4 codes | `build_extq_lo_variant` in `pyraquant/external_quant.py` |
| RTN W8A8 (g32) and nested W4 for SDXL | `build_recipe_variant` in `pyraquant/sim_recipes.py` (`rtn`, `nested4`) |
| Activation fake-quantization hooks | `_ActFakeQuant` / `_ActFakeQuantConv` in `pyraquant/quant_unet.py` (SDXL), `ExtActQuant` in `pyraquant/external_quant.py` (FLUX) |

## 5. Latency executor

`deploy/` is the INT8 executor used for the deployment table and the latency decomposition ladder
of the appendix (packed INT8 weights, Triton kernels, the hi/lo dual pass collapsed into one INT8
pass; convolutions and attention in 16 bit).  Triton 3.2 comes with `torch==2.6.0`.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M="--weight-source method --svdquant-dir ckpt/svdquant_flux_w4a4"

# deployment table, FLUX (20 timed prompts = the first 20 of prompts/eval_ultrahr_2000.jsonl, after 1 warm-up)
python deploy/run_flux_latency.py --executor bf16 --n 20 --warmup 1 --out-dir outputs/latency_flux/bf16_full
python deploy/run_flux_latency.py --executor ours $M --tau 0.10 --route-halo 2 --n 20 --warmup 1 --out-dir outputs/latency_flux/int8_routed_cache

# deployment table, SDXL
python deploy/run_sdxl_latency.py --arm fp16_full --n 20 --warmup 1 --out-dir outputs/latency_sdxl/fp16_full
python deploy/run_sdxl_latency.py --arm int8_routed_cache --weight-source method --cascade --tau 0.10 --route-halo 8 --n 20 --warmup 1 --out-dir outputs/latency_sdxl/int8_routed_cache

# decomposition ladder (appendix): remaining rungs, then the summary
python deploy/run_flux_latency.py --executor int8 $M --n 20 --warmup 1 --out-dir outputs/latency_flux/int8_full
python deploy/run_flux_latency.py --executor ours $M --no-cache --tau 0.10 --route-halo 2 --n 20 --warmup 1 --out-dir outputs/latency_flux/int8_routed_nocache
python deploy/run_flux_latency.py --executor ours --linear bf16 --tau 0.10 --route-halo 2 --n 20 --warmup 1 --out-dir outputs/latency_flux/bf16_routed_cache
python deploy/run_flux_latency.py --breakdown --out-dir outputs/latency_flux
python deploy/run_sdxl_latency.py --arm int8_full --weight-source method --n 20 --warmup 1 --out-dir outputs/latency_sdxl/int8_full
python deploy/run_sdxl_latency.py --arm int8_routed_nocache --weight-source method --cascade --tau 0.10 --route-halo 8 --n 20 --warmup 1 --out-dir outputs/latency_sdxl/int8_routed_nocache
python deploy/run_sdxl_latency.py --arm fp16_routed_cache --cascade --tau 0.10 --route-halo 8 --n 20 --warmup 1 --out-dir outputs/latency_sdxl/fp16_routed_cache
python deploy/run_sdxl_latency.py --breakdown --out-dir outputs/latency_sdxl

# kernel self-test / micro-benchmark
python deploy/int8_exec.py kernel_bench.json
```

Each run writes per-prompt records and the warm-up image into its `--out-dir`; `--breakdown` writes
`summary.md` and a JSON with the median over the timed prompts.

## 6. Licenses

`flux/pipeline_flux.py`, `flux/transformer_flux.py`, `sdxl/pipeline_sdxl.py` and
`sdxl/attention_scalediff.py` are derived from ScaleDiff and the Hugging Face diffusers library
(Apache-2.0); the FLUX base is the released SVDQuant checkpoint.  Our additions are released under
the MIT license (see `LICENSE`).
