"""Patch packing and fusion utilities for the routed FLUX executor.

Leaf boxes are expressed in unpacked VAE-latent coordinates.  This module maps
them onto the FLUX packed token sequence used by the pipeline, extracts the
selected local patches together with their exact absolute positional IDs, and
fuses the per-patch predictions back into the full-canvas prediction with an
order-independent, feathered continuous-beta blend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch

Tensor = torch.Tensor
Box = Tuple[int, int, int, int]  # y0, y1, x0, x1


def _smooth_ramp(values: Tensor, *, mode: str, power: float) -> Tensor:
    out = values.clamp(0.0, 1.0)
    if mode in {"cosine", "hann", "raised_cosine"}:
        out = 0.5 - 0.5 * torch.cos(torch.pi * out)
    elif mode in {"smoothstep", "cubic"}:
        out = out * out * (3.0 - 2.0 * out)
    elif mode not in {"linear", "lin"}:
        raise ValueError(f"Unsupported feather ramp mode: {mode}")
    if float(power) != 1.0:
        out = out.pow(max(float(power), 1e-6))
    return out.clamp(0.0, 1.0)


def halo_feather_mask(
    patch_hw: Tuple[int, int],
    inner_box_in_patch: Box,
    *,
    device: torch.device,
    dtype: torch.dtype,
    min_alpha: float = 0.0,
    mode: str = "cosine",
    power: float = 1.0,
) -> Tensor:
    """Return an ``H,W`` blend weight that is one inside the core box and ramps to zero across the halo."""

    height, width = int(patch_hw[0]), int(patch_hw[1])
    iy0, iy1, ix0, ix1 = [int(value) for value in inner_box_in_patch]
    iy0, iy1 = max(0, min(iy0, height)), max(iy0, min(iy1, height))
    ix0, ix1 = max(0, min(ix0, width)), max(ix0, min(ix1, width))
    y_weight = torch.ones(height, device=device, dtype=torch.float32)
    x_weight = torch.ones(width, device=device, dtype=torch.float32)
    if iy0 > 0:
        y_weight[:iy0] = _smooth_ramp(
            torch.linspace(0.0, 1.0, iy0, device=device), mode=mode, power=power
        )
    if iy1 < height:
        y_weight[iy1:] = _smooth_ramp(
            torch.linspace(1.0, 0.0, height - iy1, device=device),
            mode=mode,
            power=power,
        )
    if ix0 > 0:
        x_weight[:ix0] = _smooth_ramp(
            torch.linspace(0.0, 1.0, ix0, device=device), mode=mode, power=power
        )
    if ix1 < width:
        x_weight[ix1:] = _smooth_ramp(
            torch.linspace(1.0, 0.0, width - ix1, device=device),
            mode=mode,
            power=power,
        )
    mask = torch.minimum(y_weight[:, None], x_weight[None, :]).clamp(0.0, 1.0)
    if float(min_alpha) > 0.0:
        mask = float(min_alpha) + (1.0 - float(min_alpha)) * mask
    return mask.clamp(0.0, 1.0).to(dtype)


@dataclass(frozen=True)
class PatchGeometry:
    """Geometry for one selected leaf in packed-token coordinates."""

    core_box: Box
    halo_box: Box
    inner_box_in_patch: Box


@dataclass
class PatchBucket:
    """Same-shaped local patches that can share one transformer invocation."""

    patch_hw: Tuple[int, int]
    hidden_states: Tensor
    img_ids: Tensor
    geometries: List[PatchGeometry]


def token_hw_for_image(height: int, width: int) -> Tuple[int, int]:
    """Return FLUX packed-token grid size for an image resolution."""

    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError(f"FLUX image size must be positive and divisible by 16: {(height, width)}")
    return height // 16, width // 16


def packed_to_grid(packed: Tensor, token_hw: Tuple[int, int]) -> Tensor:
    """Convert ``B,N,C`` packed tokens to ``B,H,W,C`` without changing values."""

    if packed.ndim != 3:
        raise ValueError(f"Expected B,N,C packed tensor, got {tuple(packed.shape)}")
    height, width = (int(token_hw[0]), int(token_hw[1]))
    if int(packed.shape[1]) != height * width:
        raise ValueError(
            f"Packed token count {packed.shape[1]} does not match token grid {height}x{width}"
        )
    return packed.reshape(int(packed.shape[0]), height, width, int(packed.shape[2]))


def grid_to_packed(grid: Tensor) -> Tensor:
    """Convert ``B,H,W,C`` token grid back to ``B,N,C``."""

    if grid.ndim != 4:
        raise ValueError(f"Expected B,H,W,C token grid, got {tuple(grid.shape)}")
    return grid.reshape(int(grid.shape[0]), int(grid.shape[1]) * int(grid.shape[2]), int(grid.shape[3]))


def unpack_flux_tokens(packed: Tensor, token_hw: Tuple[int, int]) -> Tensor:
    """Invert FLUX 2x2 latent packing and return ``B,C,2H,2W``.

    Same permutation as Diffusers ``_unpack_latents`` but takes the token grid
    directly.
    """

    if packed.ndim != 3:
        raise ValueError(f"Expected B,N,C packed tensor, got {tuple(packed.shape)}")
    token_height, token_width = (int(token_hw[0]), int(token_hw[1]))
    batch, tokens, channels = (int(value) for value in packed.shape)
    if tokens != token_height * token_width:
        raise ValueError(
            f"Packed token count {tokens} does not match token grid "
            f"{token_height}x{token_width}"
        )
    if channels % 4:
        raise ValueError(f"FLUX packed channels must be divisible by four: {channels}")
    unpacked_channels = channels // 4
    unpacked = packed.reshape(
        batch,
        token_height,
        token_width,
        unpacked_channels,
        2,
        2,
    )
    unpacked = unpacked.permute(0, 3, 1, 4, 2, 5)
    return unpacked.reshape(
        batch,
        unpacked_channels,
        2 * token_height,
        2 * token_width,
    )


def pack_flux_latents(unpacked: Tensor) -> Tensor:
    """Apply FLUX 2x2 latent packing to a ``B,C,H,W`` tensor."""

    if unpacked.ndim != 4:
        raise ValueError(f"Expected B,C,H,W tensor, got {tuple(unpacked.shape)}")
    batch, channels, height, width = (int(value) for value in unpacked.shape)
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError(
            "FLUX unpacked latent dimensions must be positive and even: "
            f"{(height, width)}"
        )
    packed = unpacked.reshape(
        batch,
        channels,
        height // 2,
        2,
        width // 2,
        2,
    )
    packed = packed.permute(0, 2, 4, 1, 3, 5)
    return packed.reshape(
        batch,
        (height // 2) * (width // 2),
        channels * 4,
    )


def ids_to_grid(img_ids: Tensor, token_hw: Tuple[int, int]) -> Tensor:
    """Reshape the exact full-image FLUX IDs to ``H,W,3``."""

    if img_ids.ndim != 2 or int(img_ids.shape[-1]) != 3:
        raise ValueError(f"Expected N,3 image IDs, got {tuple(img_ids.shape)}")
    height, width = (int(token_hw[0]), int(token_hw[1]))
    if int(img_ids.shape[0]) != height * width:
        raise ValueError(
            f"Image-ID count {img_ids.shape[0]} does not match token grid {height}x{width}"
        )
    return img_ids.reshape(height, width, 3)


def unpacked_box_to_token_box(
    box: Box,
    *,
    packing_offset_latent: int = 0,
) -> Box:
    """Map a VAE-latent box into packed-token coordinates.

    ``packing_offset_latent`` is the latent-cell shift of the current 2x2
    packing phase (zero for the standard pipeline).  With offset one, a leaf
    boundary falls halfway through a packed token; the floor/ceil mapping
    covers both straddling boundary tokens so no selected latent cell is
    dropped.
    """

    values = tuple(int(v) for v in box)
    if any(v < 0 for v in values):
        raise ValueError(f"Negative route coordinate: {values}")
    if any(v % 2 for v in values):
        raise ValueError(f"Route coordinates must align to FLUX 2x2 packing: {values}")
    offset = int(packing_offset_latent)
    if offset not in (0, 1):
        raise ValueError(f"Packing offset must be 0 or 1, got {offset}")
    y0, y1, x0, x1 = values
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"Empty route box: {values}")
    return (
        (y0 + offset) // 2,
        (y1 + offset + 1) // 2,
        (x0 + offset) // 2,
        (x1 + offset + 1) // 2,
    )


def _expand_box(box: Box, halo_tokens: int, token_hw: Tuple[int, int]) -> Box:
    y0, y1, x0, x1 = (int(v) for v in box)
    height, width = (int(token_hw[0]), int(token_hw[1]))
    halo = max(0, int(halo_tokens))
    return (
        max(0, y0 - halo),
        min(height, y1 + halo),
        max(0, x0 - halo),
        min(width, x1 + halo),
    )


def _inner_box(outer: Box, inner: Box) -> Box:
    oy0, _oy1, ox0, _ox1 = (int(v) for v in outer)
    iy0, iy1, ix0, ix1 = (int(v) for v in inner)
    return iy0 - oy0, iy1 - oy0, ix0 - ox0, ix1 - ox0


def build_patch_buckets(
    hidden_states: Tensor,
    img_ids: Tensor,
    *,
    token_hw: Tuple[int, int],
    route_boxes_unpacked: Sequence[Box],
    halo_tokens: int,
    packing_offset_latent: int = 0,
) -> List[PatchBucket]:
    """Extract selected local tokens and their exact absolute FLUX IDs.

    Edge/corner patches have smaller clamped halos.  They are grouped by shape,
    so no fabricated or reset positional IDs are needed.
    """

    if int(hidden_states.shape[0]) != 1:
        raise ValueError("The routed executor supports one generated image per run")
    hidden_grid = packed_to_grid(hidden_states, token_hw)
    id_grid = ids_to_grid(img_ids, token_hw)
    by_shape: dict[Tuple[int, int], List[Tuple[Tensor, Tensor, PatchGeometry]]] = {}
    seen = set()
    for route_box in route_boxes_unpacked:
        core = unpacked_box_to_token_box(
            tuple(route_box),
            packing_offset_latent=packing_offset_latent,
        )
        if core in seen:
            continue
        seen.add(core)
        y0, y1, x0, x1 = core
        height, width = (int(token_hw[0]), int(token_hw[1]))
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError(f"Route box {core} lies outside token grid {token_hw}")
        outer = _expand_box(core, halo_tokens, token_hw)
        oy0, oy1, ox0, ox1 = outer
        patch = hidden_grid[:, oy0:oy1, ox0:ox1, :]
        patch_ids = id_grid[oy0:oy1, ox0:ox1, :]
        patch_hw = (int(oy1 - oy0), int(ox1 - ox0))
        geometry = PatchGeometry(
            core_box=core,
            halo_box=outer,
            inner_box_in_patch=_inner_box(outer, core),
        )
        by_shape.setdefault(patch_hw, []).append(
            (
                patch.reshape(1, patch_hw[0] * patch_hw[1], int(patch.shape[-1])),
                patch_ids.reshape(patch_hw[0] * patch_hw[1], 3),
                geometry,
            )
        )

    buckets: List[PatchBucket] = []
    for patch_hw in sorted(by_shape):
        items = by_shape[patch_hw]
        buckets.append(
            PatchBucket(
                patch_hw=patch_hw,
                hidden_states=torch.cat([item[0] for item in items], dim=0),
                img_ids=torch.stack([item[1] for item in items], dim=0),
                geometries=[item[2] for item in items],
            )
        )
    return buckets


def repeat_conditioning(value: Tensor, count: int) -> Tensor:
    """Repeat batch-one conditioning for a bucket without changing values."""

    if int(value.shape[0]) == count:
        return value
    if int(value.shape[0]) != 1:
        raise ValueError(f"Cannot repeat conditioning batch {value.shape[0]} to {count}")
    return value.expand(count, *value.shape[1:])


def fuse_patch_predictions(
    background: Tensor,
    bucket_outputs: Iterable[Tuple[PatchBucket, Tensor]],
    *,
    fusion_beta: float,
    feather_mode: str = "cosine",
    feather_power: float = 1.0,
) -> Tensor:
    """Fuse local raw flow predictions into the background with continuous beta fields."""

    if not 0.0 <= float(fusion_beta) <= 1.0:
        raise ValueError(f"fusion_beta must be in [0,1], got {fusion_beta}")
    batch, height, width, channels = background.shape
    if int(batch) != 1:
        raise ValueError("Only batch-one full-image fusion is supported")
    local_accum = torch.zeros_like(background)
    beta_accum = torch.zeros(
        (1, int(height), int(width), 1),
        device=background.device,
        dtype=background.dtype,
    )
    for bucket, output in bucket_outputs:
        patch_h, patch_w = bucket.patch_hw
        if output.ndim != 3:
            raise ValueError(f"Expected bucket output K,N,C, got {tuple(output.shape)}")
        if int(output.shape[0]) != len(bucket.geometries):
            raise ValueError("Bucket output batch does not match geometry count")
        if int(output.shape[1]) != patch_h * patch_w or int(output.shape[2]) != int(channels):
            raise ValueError("Bucket output shape does not match patch geometry/background channels")
        output_grid = output.reshape(int(output.shape[0]), patch_h, patch_w, int(channels))
        for index, geometry in enumerate(bucket.geometries):
            y0, y1, x0, x1 = geometry.halo_box
            beta = halo_feather_mask(
                bucket.patch_hw,
                geometry.inner_box_in_patch,
                device=background.device,
                dtype=background.dtype,
                min_alpha=0.0,
                mode=feather_mode,
                power=feather_power,
            )
            beta = (float(fusion_beta) * beta).clamp(0.0, 1.0)[None, :, :, None]
            local_accum[:, y0:y1, x0:x1, :] += beta * output_grid[index : index + 1]
            beta_accum[:, y0:y1, x0:x1, :] += beta

    beta_map = beta_accum.clamp(0.0, 1.0)
    combined = torch.where(
        beta_accum > 0.0,
        local_accum / beta_accum.clamp_min(1e-8),
        torch.zeros_like(local_accum),
    )
    return (1.0 - beta_map) * background + beta_map * combined


def reference_flow_prediction(sample: Tensor, pred_x0_ref: Tensor, sigma: Tensor) -> Tensor:
    """Project an x0 reference into FLUX/FlowMatch model-output space."""

    if sample.shape != pred_x0_ref.shape:
        raise ValueError(f"sample/reference shape mismatch: {sample.shape} vs {pred_x0_ref.shape}")
    return (sample - pred_x0_ref) / (sigma.to(sample.device, torch.float32) + 1e-6)
