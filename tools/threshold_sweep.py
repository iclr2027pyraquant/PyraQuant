"""Threshold calibration: sweep the relative-gap threshold tau on the 32 COCO calibration captions.

For every tau the main configuration is run with that threshold for the relative-gap rule at the
2K and 4K stages (see below for the SDXL 4K rule), and the resulting allocation is summarised:
the mean fraction of the canvas in the high / low precision states and in the cached state
at each stage, and the mean fraction of the 4K canvas that is computed (not cached).

Example:
    python tools/threshold_sweep.py --pipeline sdxl --taus 0.03 0.05 0.10 0.15 --output-dir outputs/sweep_sdxl
    python tools/threshold_sweep.py --pipeline flux --taus 0.05 0.10 0.20 --output-dir outputs/sweep_flux \\
        --svdquant-dir ckpt/svdquant_flux_w4a4

By default the sweep re-applies the sibling test at 4K for both pipelines (the FLUX recipe);
``--keep-4k-rule`` keeps the 4K rule of the given config instead (for SDXL: all children of the
2K high-precision leaves stay active).
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=["flux", "sdxl"], required=True)
    ap.add_argument("--taus", type=float, nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None, help="base config (default configs/<pipeline>_pyraquant.json)")
    ap.add_argument("--prompts-file", type=Path, default=ROOT / "prompts" / "calibration_coco32.jsonl")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--svdquant-dir", type=Path, default=None, help="FLUX only")
    ap.add_argument("--keep-4k-rule", action="store_true", help="do not replace the 4K rule of the config by the sibling test")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()
    if args.pipeline == "flux" and args.svdquant_dir is None:
        ap.error("--svdquant-dir is required for the FLUX pipeline")
    tags = [f"{tau:g}" for tau in args.taus]
    if len(set(tags)) != len(tags):
        ap.error("duplicate tau values")

    base_cfg = json.loads((args.config or ROOT / "configs" / f"{args.pipeline}_pyraquant.json").read_text())
    driver = ROOT / args.pipeline / f"run_{args.pipeline}.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for tau, tag in zip(args.taus, tags):
        cfg = copy.deepcopy(base_cfg)
        for stage in ("256", "512"):
            st = cfg["plan"][stage]
            if stage == "512" and not args.keep_4k_rule:
                st.pop("inherit_hi", None)
                st.update({"inherit": True, "norm": "relgap"})
            if st.get("norm") == "relgap" or "thresh" in st:
                st["thresh"] = float(tau)
        run_dir = args.output_dir / f"tau_{tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "records.json").unlink(missing_ok=True)      # the driver appends to an existing file
        cfg_path = run_dir / "config.json"
        cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")
        cmd = [sys.executable, str(driver), "--config", str(cfg_path), "--output-dir", str(run_dir),
               "--prompts-file", str(args.prompts_file), "--n", str(args.n), "--seed", str(args.seed)]
        if args.pipeline == "flux":
            cmd += ["--svdquant-dir", str(args.svdquant_dir)]
        if args.local_files_only:
            cmd.append("--local-files-only")
        print(f"[sweep] tau={tau}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        records = json.loads((run_dir / "records.json").read_text())
        stages = sorted({s for r in records for s in r["coverage_per_stage"]}, key=int)
        cov = {s: {k: sum(r["coverage_per_stage"][s][k] for r in records) / len(records) for k in ("hi", "lo", "cache")}
               for s in stages}
        summary[tag] = {
            "num_prompts": len(records),
            "mean_coverage_per_stage": cov,
            "computed_fraction_4096": 1.0 - cov.get("4096", {"cache": 0.0})["cache"],
        }
        (args.output_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(f"[sweep] tau={tau}: 4K computed fraction {summary[tag]['computed_fraction_4096']:.3f}, "
              f"coverage {json.dumps(cov)}", flush=True)
    print(f"[sweep] done -> {args.output_dir / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
