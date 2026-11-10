"""Widen MOSS-Image video_embedder for a ComfyUI tile control adapter.

ComfyUI concatenates the noisy latent with the control latent before
video_embedder. The built-in MOSS-Image Color/Depth node sends two latents,
color then depth. This tile adapter uses the color slot for the degraded
image and leaves depth at zero, so the saved weight is 128 + 128 + 128 = 384
input channels. Load it with the normal LoRA loader, then connect the
structure image to color_map and leave depth_map empty.
"""

from __future__ import annotations

import torch
from torch import nn


LATENT_CHANNELS = 128
# color slot + zero depth slot, matching MOSSImageControl
CONTROL_SLOTS = 2


class TileVideoEmbedder(nn.Module):
    """Base columns stay frozen. Only the new control columns train."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        if base.in_features != LATENT_CHANNELS:
            raise ValueError(f"expected video_embedder in_features {LATENT_CHANNELS}, got {base.in_features}")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.control = nn.Linear(LATENT_CHANNELS * CONTROL_SLOTS, base.out_features, bias=False)
        nn.init.zeros_(self.control.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        base_channels = self.base.in_features
        if tokens.shape[-1] == base_channels:
            return self.base(tokens)
        expected = base_channels * (1 + CONTROL_SLOTS)
        if tokens.shape[-1] != expected:
            raise RuntimeError(f"tile embedder expected {base_channels} or {expected} channels, got {tokens.shape[-1]}")
        return self.base(tokens[..., :base_channels]) + self.control(tokens[..., base_channels:])

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return torch.cat([self.base.weight.detach(), self.control.weight.detach()], dim=1).contiguous()

    @torch.no_grad()
    def merged_bias(self) -> torch.Tensor:
        return self.base.bias.detach().contiguous()


def split_merged_embedder(weight: torch.Tensor, bias: torch.Tensor | None = None):
    """Split a ComfyUI 384-column video_embedder back into base and control columns."""
    control_columns = LATENT_CHANNELS * CONTROL_SLOTS
    expected = LATENT_CHANNELS + control_columns
    if weight.ndim != 2 or weight.shape[1] != expected:
        raise ValueError(f"expected video_embedder weight [dim, {expected}], got {tuple(weight.shape)}")
    base_weight = weight[:, :LATENT_CHANNELS].contiguous()
    control_weight = weight[:, LATENT_CHANNELS:].contiguous()
    return base_weight, control_weight, bias


def pack_tile_tokens(noisy_tokens: torch.Tensor, control_latents: torch.Tensor) -> torch.Tensor:
    """noisy_tokens is [B, HW, 128]. control_latents is [B, 128, H, W]."""
    control_tokens = control_latents.flatten(2).transpose(1, 2)
    if control_tokens.shape != noisy_tokens.shape:
        raise RuntimeError(
            f"control tokens {tuple(control_tokens.shape)} != noisy tokens {tuple(noisy_tokens.shape)}"
        )
    depth = torch.zeros_like(control_tokens)
    return torch.cat((noisy_tokens, control_tokens, depth), dim=-1)
