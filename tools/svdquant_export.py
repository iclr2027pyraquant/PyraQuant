"""Convert a released SVDQuant checkpoint (nunchaku format, e.g. Hugging Face
mit-han-lab/svdq-int4-flux.1-schnell) into the external-quant package format read by
this repository (flux/run_flux.py --svdquant-dir <dir>).

Input
  --ckpt      directory of the released checkpoint (its transformer_blocks.safetensors
              is used) or a path to that safetensors file
  --flux-dir  optional: directory of the original bf16 FLUX.1-schnell transformer
              (diffusers layout with diffusion_pytorch_model.safetensors.index.json);
              when given, a per-layer identity check is run on a few blocks
Output (--out)
  weights.safetensors   dequantized bf16 weights (+ bias) of the quantized linears
  aux.safetensors       smoothing vectors, low-rank A/B, W4 group scales
  act_spec.json         per-module activation quantization spec
  weight_spec.json      per-module weight bit-width / group / scale spec
  meta.json             package metadata
  export_checks.json    self-check results (empty without --flux-dir)

Requires deepcompressor (https://github.com/mit-han-lab/deepcompressor), whose packer
layout is inverted by svdquant_unpack.py (see README for the pinned install command).  Runs on CPU.

Semantics of the nunchaku W4A4 kernel, which act_spec encodes:
    h  = x_raw @ lora_down                              (low-rank branch takes the raw input; lora_down is already divided by smooth)
    x' = (x_raw + shift) / smooth  -> INT4 per-token g64 (post-GELU layers: unsigned, shift=0.171875; others: signed, shift=0)
    y  = x'_q @ W_dq^T + bias + h @ lora_up^T           (bias already contains the -W*shift and low-rank shift compensation)
Self-check: W_dq + lora_up @ (lora_down * smooth_orig)^T ~= W_orig * diag(smooth_orig), 4-bit level relative error.
Every tensor in the checkpoint (qweight, wscales, lora_up/down, bias, smooth) is in kernel layout.  The released weight
codes come from SVDQuant+GPTQ (not round-to-nearest), so only the dequantized weights can be compared with the original.

Usage: python tools/svdquant_export.py --ckpt <ckpt_dir> --out <package_dir> [--flux-dir <transformer_dir>]
"""
from __future__ import annotations
import argparse, json, os, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from svdquant_unpack import unpack_weight_w4, unpack_scale, unpack_lowrank, unpack_w4x16_adanorm, unpack_vec

CKPT_FILE = "transformer_blocks.safetensors"
SHIFT_GELU = 0.171875
G = 64

# nunchaku name -> (diffusers target modules in output-row order, input is post-GELU (unsigned + shift))
JOINT = {
    "qkv_proj":         (["attn.to_q", "attn.to_k", "attn.to_v"], False),
    "qkv_proj_context": (["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"], False),
    "out_proj":         (["attn.to_out.0"], False),
    "out_proj_context": (["attn.to_add_out"], False),
    "mlp_fc1":          (["ff.net.0.proj"], False),
    "mlp_fc2":          (["ff.net.2"], True),
    "mlp_context_fc1":  (["ff_context.net.0.proj"], False),
    "mlp_context_fc2":  (["ff_context.net.2"], True),
}
SINGLE = {
    "qkv_proj": (["attn.to_q", "attn.to_k", "attn.to_v"], False),
    "out_proj": (["proj_out.linears.0"], False),
    "mlp_fc1":  (["proj_mlp"], False),
    "mlp_fc2":  (["proj_out.linears.1"], True),
}
ADANORM = {"transformer_blocks": [("norm1.linear", 6), ("norm1_context.linear", 6)], "single_transformer_blocks": [("norm.linear", 3)]}


class Orig:
    """Original bf16 transformer weights, read lazily from the sharded safetensors (self-check only)."""
    def __init__(self, root: str):
        self.root = root
        self.idx = json.load(open(os.path.join(root, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
        self.h = {}
    def get(self, k):
        f = self.idx[k]
        if f not in self.h: self.h[f] = safe_open(os.path.join(self.root, f), "pt")
        return self.h[f].get_tensor(k)


def main():
    ap = argparse.ArgumentParser(description="Unpack a released SVDQuant (nunchaku) FLUX checkpoint into an external-quant package.")
    ap.add_argument("--ckpt", required=True, help=f"released checkpoint directory (containing {CKPT_FILE}) or the safetensors file itself")
    ap.add_argument("--flux-dir", default=None, help="original bf16 FLUX.1-schnell transformer directory; enables the per-layer self-check")
    ap.add_argument("--out", required=True, help="output package directory")
    args = ap.parse_args()
    ckpt = os.path.join(args.ckpt, CKPT_FILE) if os.path.isdir(args.ckpt) else args.ckpt
    os.makedirs(args.out, exist_ok=True)
    sf = safe_open(ckpt, "pt")
    orig = Orig(args.flux_dir) if args.flux_dir else None
    W, AUX, ACT, WSPEC = {}, {}, {}, {}
    fp16_modules = ["x_embedder", "context_embedder", "time_text_embed.timestep_embedder.linear_1", "time_text_embed.timestep_embedder.linear_2",
                    "time_text_embed.text_embedder.linear_1", "time_text_embed.text_embedder.linear_2", "norm_out.linear", "proj_out"]
    split_modules = {}
    checks = []; t0 = time.time()
    n_joint, n_single = 19, 38
    for kind, nblk, table in (("transformer_blocks", n_joint, JOINT), ("single_transformer_blocks", n_single, SINGLE)):
        for b in range(nblk):
            blk = f"{kind}.{b}"
            if kind == "single_transformer_blocks":
                split_modules[f"{blk}.proj_out"] = 3072
            for nk, (targets, gelu_side) in table.items():
                p = f"{blk}.{nk}"
                qw = sf.get_tensor(f"{p}.qweight"); ws = sf.get_tensor(f"{p}.wscales")
                n = qw.shape[0]; k = qw.shape[1] * 2
                q = unpack_weight_w4(qw, n, k).float()                       # [n, k] codes in [-8, 7]
                sc = unpack_scale(ws, n, G, k).float()                        # [n, k/64]
                wdq = (q.reshape(n, k // G, G) * sc[:, :, None]).reshape(n, k).to(torch.bfloat16)
                bias = unpack_vec(sf.get_tensor(f"{p}.bias"), n)                                   # bias/smooth share the wscales lane layout
                smooth = unpack_vec(sf.get_tensor(f"{p}.smooth"), k).float(); smooth_orig = unpack_vec(sf.get_tensor(f"{p}.smooth_orig"), k).float()
                A = unpack_lowrank(sf.get_tensor(f"{p}.lora_down"), down=True).t().contiguous()   # unpacked [r, k] -> [k, r]; already divided by smooth (or fused)
                Bup = unpack_lowrank(sf.get_tensor(f"{p}.lora_up"), down=False)        # [n, r]
                r = A.shape[1]
                fused = bool(torch.all(smooth == 1))
                akey = f"{blk}.{targets[0]}.lowrank_A"; AUX[akey] = A.contiguous()
                if not fused:
                    skey = f"{blk}.{targets[0]}.smooth"; AUX[skey] = smooth.to(torch.bfloat16).contiguous()
                rows = n // len(targets)
                for i, tname in enumerate(targets):
                    full = f"{blk}.{tname}"
                    W[f"{full}.weight"] = wdq[i * rows:(i + 1) * rows].contiguous()
                    bi = bias[i * rows:(i + 1) * rows].contiguous()
                    if not (tname == "proj_out.linears.0" and bool(torch.all(bi == 0))):     # attention half has no bias (all-zero placeholder in the ckpt)
                        W[f"{full}.bias"] = bi
                    bkey = f"{full}.lowrank_B"; AUX[bkey] = Bup[i * rows:(i + 1) * rows].t().contiguous()      # [r, rows]
                    wskey = f"{full}.wscale"; AUX[wskey] = sc[i * rows:(i + 1) * rows].to(torch.float16).contiguous()   # [rows, k/64] W4 group scales (needed to derive nested W3 from the codes)
                    spec = {"bits": 4, "mode": "dyn_group", "group": G, "lowrank_A": akey, "lowrank_B": bkey, "lowrank_raw": True}
                    if not fused: spec["smooth"] = skey
                    if gelu_side: spec.update({"unsigned": True, "shift": SHIFT_GELU})
                    ACT[full] = spec
                    WSPEC[full] = {"bits": 4, "group": G, "scale_bits": 16, "wscale": wskey, "code_range": [-8, 7]}
                # ---- self-check: dequantized weight + low-rank ~= original weight * smooth_orig ----
                if orig is not None and (b in (0, n_joint - 1) or b == 5):
                    # original weight: concatenated (qkv) or one half of proj_out
                    if targets[0].startswith("proj_out.linears"):
                        w_o = orig.get(f"{blk}.proj_out.weight"); w_o = (w_o[:, :3072] if targets[0].endswith(".0") else w_o[:, 3072:])
                    else:
                        w_o = torch.cat([orig.get(f"{blk}.{t}.weight") for t in targets], 0)
                    w_o = w_o.float()
                    lhs = wdq.float() + Bup.float() @ (A.float() * smooth_orig[:, None]).t() if not fused else wdq.float() + Bup.float() @ A.float().t()
                    rhs = w_o * smooth_orig[None, :]
                    rel = ((lhs - rhs).norm() / rhs.norm()).item()
                    # reference: without the low-rank branch (or with smooth applied in the wrong direction) the error is O(1)
                    rel_nolr = ((wdq.float() - rhs).norm() / rhs.norm()).item()
                    checks.append({"module": p, "rel_err": rel, "rel_err_without_lowrank": rel_nolr, "fused_smooth": fused, "rank": r})
            for nname, splits in ADANORM[kind]:
                p = f"{blk}.{nname}"
                qw = sf.get_tensor(f"{p}.qweight"); ws = sf.get_tensor(f"{p}.wscales"); wz = sf.get_tensor(f"{p}.wzeros"); bias = sf.get_tensor(f"{p}.bias")
                oc = ws.shape[1]
                ic = qw.numel() * 8 // oc                                    # 8 nibbles per int32
                wdq, brec = unpack_w4x16_adanorm(qw, ws, wz, bias, oc, ic, splits)
                W[f"{p}.weight"] = wdq.to(torch.bfloat16).contiguous(); W[f"{p}.bias"] = brec.to(torch.bfloat16).contiguous()
                WSPEC[p] = {"bits": 4, "group": G, "scale_bits": 32, "note": "W4A16 asymmetric (scale+zero) g64, no activation quantization"}
                if orig is not None and b == 0:
                    w_o = orig.get(f"{p}.weight").float(); b_o = orig.get(f"{p}.bias").float()
                    checks.append({"module": p, "rel_err": ((wdq - w_o).norm() / w_o.norm()).item(), "bias_max_abs_err": (brec - b_o).abs().max().item(), "adanorm_splits": splits})
            if b % 5 == 0: print(f"[svdquant_export] {blk} done, {time.time()-t0:.0f}s", flush=True)
    print(f"[svdquant_export] weights={len(W)} aux={len(AUX)} act={len(ACT)}; saving...", flush=True)
    save_file(W, f"{args.out}/weights.safetensors"); save_file(AUX, f"{args.out}/aux.safetensors")
    json.dump(ACT, open(f"{args.out}/act_spec.json", "w"), indent=1); json.dump(WSPEC, open(f"{args.out}/weight_spec.json", "w"), indent=1)
    meta = {"method": "SVDQuant", "host": "FLUX.1-schnell", "w_bits": 4, "a_bits": 4, "group": G, "scale_dtype_bits": 16, "lowrank_rank": 32,
            "fp16_modules": fp16_modules, "split_modules": split_modules, "lowrank_raw": True,
            "calibration": "released checkpoint mit-han-lab/svdq-int4-flux.1-schnell (deepcompressor calibration: 128 prompts, 1024x1024, 4 steps); "
                           "not re-calibrated for 2K/4K",
            "notes": "unpacked from the released nunchaku checkpoint; INT4 W4A4 with per-token x g64 dynamic activation quantization; "
                     "post-GELU layers unsigned with shift 0.171875; adaLN W4A16 g64 asymmetric; rank-32 bf16 low-rank branch "
                     "(A shared per fused group, B per module); fp16 modules listed in fp16_modules",
            "paper_label": "SVDQuant (released ckpt)"}
    json.dump(meta, open(f"{args.out}/meta.json", "w"), indent=1)
    json.dump(checks, open(f"{args.out}/export_checks.json", "w"), indent=1)
    for c in checks: print("  check", c)
    bad = [c for c in checks if c.get("rel_err", 0) > 0.25]
    print(f"[svdquant_export] done in {time.time()-t0:.0f}s -> {args.out}; {len(bad)} self-checks above the 0.25 threshold")


if __name__ == "__main__":
    main()
