"""ScaleDiff-SDXL window attention (NPA) with arbitrary crop sizes.

The host window attention (AttnControl / AttnProcessor2_0_local) requires the
token grid on every side to be a multiple of window//2 and >= window; in latent
units each crop side must be a multiple of 64 latent and >= 128 latent. Under
that constraint the routed executor could only use a halo of 0 or 64 latent.

Here the token grid is zero-padded to a valid size inside the attention module,
the padded keys are masked out via attn_mask, and the output is cropped back.
Convolutions and linear layers still run only on the real crop, so the halo can
be any multiple of 8 latent. On the full canvas (already a valid size) no padding
is triggered and the result is bit-identical to the host implementation.
"""
import math
from typing import Optional
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from attention_scalediff import AttnControl, AttnProcessor2_0_local


def _ceil_to(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


class PadAttnControl(AttnControl):
    """height_scale/width_scale always describe the padded (valid) grid; true_scale holds the real grid (None = no padding)."""
    def __init__(self):
        super().__init__(); self.true_scale = None

    def initialize(self, height_scale, width_scale):
        self.true_scale = None
        super().initialize(height_scale, width_scale)

    def initialize_crop(self, crop_h_lat: int, crop_w_lat: int, base: int = 128):
        """Crop size (latent px) -> pad to a multiple of 64 and >= 128, build the window views, and remember the true size."""
        ph, pw = max(128, _ceil_to(crop_h_lat, 64)), max(128, _ceil_to(crop_w_lat, 64))
        super().initialize(ph / base, pw / base)
        self.true_scale = None if (ph, pw) == (crop_h_lat, crop_w_lat) else (crop_h_lat / base, crop_w_lat / base)

    def grids(self, window_size: int):
        """-> (h, w, hp, wp): real and padded token grids for this layer."""
        hp, wp = int(self.height_scale * window_size), int(self.width_scale * window_size)
        if self.true_scale is None:
            return hp, wp, hp, wp
        return int(self.true_scale[0] * window_size), int(self.true_scale[1] * window_size), hp, wp


class PadAttnProcessor(AttnProcessor2_0_local):
    def __call__(self, attn: Attention, hidden_states: torch.Tensor,
                 encoder_hidden_states: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None, temb: Optional[torch.Tensor] = None,
                 *args, **kwargs) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states); value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]; head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None: query = attn.norm_q(query)
        if attn.norm_k is not None: key = attn.norm_k(key)

        if self.controller.active:
            c = self.controller; win = self.window_size
            h, w, hp, wp = c.grids(win)
            padded = (hp, wp) != (h, w)
            if padded:
                if attention_mask is not None:
                    raise NotImplementedError("padded crop path does not support an external attention_mask")
                def pad2d(t):                                   # [B,H,h*w,C] -> [B,H,hp*wp,C], zero-pad right/bottom
                    B, H, _, C = t.shape
                    t = t.view(B, H, h, w, C)
                    t = F.pad(t, (0, 0, 0, wp - w, 0, hp - h))
                    return t.reshape(B, H, hp * wp, C)
                query, key, value = pad2d(query), pad2d(key), pad2d(value)
                valid = torch.zeros(hp, wp, dtype=torch.bool, device=key.device); valid[:h, :w] = True
                views = c.kv_views[win].to(key.device)          # [num_views, L]
                kmask = valid.flatten()[views]                  # [num_views, L]  True = real key
                nv = views.shape[0]
                attention_mask = kmask[None, :, None, None, :].expand(batch_size, nv, 1, 1, -1).reshape(batch_size * nv, 1, 1, -1)
            q_ = c.patchify_q(query, win); k_ = c.patchify_kv(key, win); v_ = c.patchify_kv(value, win)
            hidden_states = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
            hidden_states = c.unpatchify(hidden_states, win)
            if padded:                                          # crop back to the real grid
                B, H, _, C = hidden_states.shape
                hidden_states = hidden_states.view(B, H, hp, wp, C)[:, :, :h, :w, :].reshape(B, H, h * w, C)
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states); hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def register_attention_control_pad(pipe):
    """Same structure as the host register_attention_control, using the padding-aware controller/processor."""
    controller = PadAttnControl(); attn_procs = {}; window_size_lst = []
    for name in pipe.unet.attn_processors.keys():
        if name.startswith("down_blocks.1"): window_size = 64
        elif name.startswith("down_blocks.2"): window_size = 32
        elif name.startswith("mid_block"): window_size = 32
        elif name.startswith("up_blocks.0"): window_size = 32
        elif name.startswith("up_blocks.1"): window_size = 64
        else: raise ValueError("unexpected attention name")
        if window_size not in window_size_lst: window_size_lst.append(window_size)
        attn_procs[name] = PadAttnProcessor(controller, window_size) if name.endswith("attn1.processor") else pipe.unet.attn_processors[name]
    pipe.unet.set_attn_processor(attn_procs)
    controller.set_window_sizes(window_size_lst)
    return controller
