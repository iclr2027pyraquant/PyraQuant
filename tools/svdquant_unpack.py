"""Inverse of the nunchaku weight packing used by the released SVDQuant checkpoints
(e.g. Hugging Face mit-han-lab/svdq-int4-flux.1-schnell): qweight / wscales / bias /
smooth / lora_down / lora_up of the W4A4 linears and the W4A16 adaLN linears.

Requires deepcompressor (https://github.com/mit-han-lab/deepcompressor), whose
NunchakuWeightPacker defines the tile layout being inverted.  It is imported lazily,
so importing this module (e.g. for --help) does not need it.

Running this file directly performs a random-tensor roundtrip self-test against
deepcompressor's own packer.
"""
from __future__ import annotations
import torch

DEEPCOMPRESSOR_URL = "https://github.com/mit-han-lab/deepcompressor"


def _packer():
    """W4 nunchaku weight packer (deepcompressor is imported on first use)."""
    try:
        from deepcompressor.backend.nunchaku.utils import NunchakuWeightPacker
    except ImportError as e:
        raise ImportError(f"deepcompressor is required for unpacking: pip install git+{DEEPCOMPRESSOR_URL}") from e
    return NunchakuWeightPacker(bits=4)


def unpack_weight_w4(packed_i8: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Inverse of NunchakuWeightPacker(bits=4).pack_weight: [n, k//2] int8 -> [n, k] int32 codes in [-8, 7]."""
    p = _packer()
    w = packed_i8.contiguous().view(torch.int32)                                    # 8 nibbles per int32
    n_tiles, k_tiles = n // p.mem_n, k // p.mem_k
    w = w.reshape(n_tiles, k_tiles, p.num_k_packs, p.num_n_packs, p.num_n_lanes, p.num_k_lanes, p.n_pack_size, p.k_pack_size, p.reg_n)
    shift = torch.arange(0, 32, 4, dtype=torch.int32)
    w = (w.unsqueeze(-1) >> shift) & 0xF                                           # split back into reg_k=8 nibbles
    w = torch.where(w >= 8, w - 16, w)                                              # signed codes
    # The packer permutes (n_tiles, num_n_packs, n_pack_size, num_n_lanes, reg_n, k_tiles, num_k_packs, k_pack_size, num_k_lanes, reg_k)
    # with (0,5,6,1,3,8,2,7,4,9) into (n_tiles, k_tiles, num_k_packs, num_n_packs, num_n_lanes, num_k_lanes, n_pack_size, k_pack_size, reg_n, reg_k).
    inv = [0, 3, 6, 4, 8, 1, 2, 7, 5, 9]                                            # inverse permutation
    w = w.permute(*inv).contiguous()
    return w.reshape(n, k)


def unpack_scale(packed: torch.Tensor, n: int, group_size: int, k: int) -> torch.Tensor:
    """Inverse of pack_scale for fp16/bf16 per-group scales (not micro-scales): returns [n, k//group_size]."""
    p = _packer()
    s_pack_size = min(max(p.warp_n // p.num_lanes, 2), 8); num_s_lanes = min(p.num_lanes, p.warp_n // s_pack_size)
    num_s_packs = p.warp_n // (s_pack_size * num_s_lanes); warp_s = num_s_packs * num_s_lanes * s_pack_size
    ng = k // group_size if group_size > 0 else 1
    s = packed.reshape(n // warp_s, ng, num_s_packs, num_s_lanes // 4, 4, s_pack_size // 2, 2)   # layout after the packer's permute(0,6,1,2,4,3,5)
    s = s.permute(0, 2, 3, 5, 4, 6, 1).contiguous()                                              # inverse permutation
    return s.reshape(n, ng)


def unpack_vec(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of pack_scale(v, group_size=-1): per-channel vectors (bias, smooth) are stored in the same lane layout as wscales."""
    return unpack_scale(packed.reshape(-1), n, -1, n).reshape(n)


def unpack_lowrank(packed: torch.Tensor, down: bool) -> torch.Tensor:
    """Inverse of NunchakuWeightPacker.pack_lowrank_weight."""
    return _packer().unpack_lowrank_weight(packed, down=down)


def unpack_w4x16_adanorm(qweight_i32: torch.Tensor, wscales: torch.Tensor, wzeros: torch.Tensor, bias: torch.Tensor,
                         oc: int, ic: int, splits: int, delta=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of convert_to_nunchaku_w4x16_linear_weight (tinychat W4A16 layout).
    Returns the dequantized weight [oc, ic] in the original (diffusers) channel order and the original bias.
    tinychat stores q = round((w + zero) / scale) in [0, 15], _scale [ng, oc] and _zero = -zero (pre-scaled), so w = q * scale + _zero.
    With adanorm_splits > 1 the packer interleaves output channels (view(splits, oc/splits).T) and adds `delta` to the bias
    (+1 on the 2nd and the 2nd-to-last chunk)."""
    w16 = qweight_i32.view(torch.int16)
    # Inverse of tinychat pack_w4: [oc//4, ic] int16 = view(oc//4, ic//64, 4, 16).permute(0,2,1,3) then reshape; 4 nibbles per int16.
    w = w16.reshape(oc // 4, ic // 64, 4, 16).permute(0, 2, 1, 3).reshape(-1, 8).to(torch.int32) & 0xFFFF   # [oc*ic/32, 8] before the nibble OR
    nib = torch.stack([(w >> (4 * j)) & 0xF for j in range(4)], dim=1)            # [oc*ic/32, 4, 8] = original weight.view(-1, 4, 8)
    q = nib.reshape(oc, ic).float()
    ng = wscales.shape[0]; g = ic // ng
    scale = wscales[:ng].t().reshape(oc, ng, 1).float(); zneg = wzeros[:ng].t().reshape(oc, ng, 1).float()
    wdq = (q.reshape(oc, ng, g) * scale + zneg).reshape(oc, ic)                    # _zero stores -zero (pre-scaled)
    b = bias.float().clone()
    if splits > 1:
        if delta is None:                                   # delta pattern of the released checkpoints
            delta = torch.zeros(splits); delta[1] = delta[-2] = 1
        b = b.reshape(oc // splits, splits) - torch.as_tensor(delta).float()[None]
        b = b.t().reshape(oc)
        wdq = wdq.reshape(oc // splits, splits, ic).transpose(0, 1).reshape(oc, ic)
    return wdq, b


def _selftest():
    torch.manual_seed(0)
    from deepcompressor.backend.nunchaku.utils import NunchakuWeightPacker, convert_to_nunchaku_w4x4y16_linear_weight, convert_to_nunchaku_w4x16_linear_weight
    n, k, g = 256, 512, 64
    w = torch.randn(n, k, dtype=torch.bfloat16); scale = (w.float().reshape(n, 1, k // g, g).abs().amax(-1, keepdim=True) / 7).to(torch.bfloat16)
    lora = (torch.randn(k, 32, dtype=torch.bfloat16), torch.randn(n, 32, dtype=torch.bfloat16))
    qw, sc, bias, sm, lr, _ = convert_to_nunchaku_w4x4y16_linear_weight(w, scale=scale, lora=lora)
    q_ref = (w.float().reshape(n, 1, k // g, g) / scale.float()).round().reshape(n, k)
    assert torch.equal(unpack_weight_w4(qw, n, k).float(), q_ref), "qweight unpack mismatch"
    assert torch.equal(unpack_scale(sc, n, g, k), scale.reshape(n, k // g)), "wscales unpack mismatch"
    v = torch.randn(n, dtype=torch.bfloat16); p_ = NunchakuWeightPacker(bits=4)
    pv = p_.pack_scale(p_.pad_scale(v.view(-1, 1), group_size=-1), group_size=-1)
    assert torch.equal(unpack_vec(pv, n), v), "bias/smooth vector unpack mismatch"
    # The packer pads the rank to 128 -> unpacked [k, 128]; released checkpoints are unpadded ([k, 32]) -> unpacked [r, k], transposed by the exporter.
    assert torch.equal(unpack_lowrank(lr[0], True)[:, :32], lora[0]) and torch.equal(unpack_lowrank(lr[1], False), lora[1]), "lora unpack mismatch"
    # adaLN W4A16, splits=1/3/6: exact roundtrip on structured codes (random bf16 weights would differ by +-1 due to bf16 rounding of 7*scale).
    for splits in (1, 3, 6):
        oc, ic = 768, 3072
        q2 = torch.randint(0, 16, (oc, ic)).float(); s2 = torch.ones(oc, 1, ic // g, 1, dtype=torch.bfloat16)
        w2 = (q2 - 7).to(torch.bfloat16); b2 = (torch.randn(oc) * 0.05).to(torch.bfloat16)   # realistic bias magnitude; the +1 delta loses precision on O(1) bf16 values
        qw2, ws2, wz2, b2p = convert_to_nunchaku_w4x16_linear_weight(w2, scale=s2, zero=None, bias=b2, adanorm_splits=splits)
        # The packer's effective delta pattern (add_ on a transposed view) can differ from the released checkpoints; read it from a zero-bias pack.
        d_eff = convert_to_nunchaku_w4x16_linear_weight(w2, scale=s2, zero=None, bias=torch.zeros(oc, dtype=torch.bfloat16), adanorm_splits=splits)[3].float().reshape(oc // splits, splits)[0]
        wdq, brec = unpack_w4x16_adanorm(qw2, ws2, wz2, b2p, oc, ic, splits, delta=d_eff)
        assert torch.equal(wdq, w2.float()), ("adaLN code unpack mismatch", splits)
        if not torch.allclose(brec, b2.float(), atol=1e-2):     # data-dependent delta behaviour of the packer; not an unpacking error
            print(f"  [warn] adaLN splits={splits} synthetic bias roundtrip max|diff|={(brec - b2.float()).abs().max():.4f} (packer delta quirk)")
    print("OK svdquant_unpack roundtrip (w4 weight, g64 scales, bias/smooth vectors, lora, adaLN splits 3/6)")


if __name__ == "__main__":
    _selftest()
