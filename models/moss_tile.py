"""Mixin that teaches diffusion-pipe's MossImagePipeline to train a tile adapter.

Copy this file to diffusion-pipe/models/moss_tile.py and run apply_moss_tile.py.
The saved safetensors keeps the usual diffusion-pipe LoRA keys and adds:

    diffusion_model.video_embedder.set_weight
    diffusion_model.video_embedder.bias.set_weight

ComfyUI's LoRA loader replaces video_embedder with that 384-channel weight.
Put the structure image on MOSS-Image Color/Depth Control -> color_map.
Leave depth_map empty. Depth was trained as zeros.
"""

from __future__ import annotations

import torch
from torch import nn

from models.tile_embedder import CONTROL_SLOTS, LATENT_CHANNELS, TileVideoEmbedder, pack_tile_tokens

__all__ = ["MossTileMixin", "TileVideoEmbedder", "pack_tile_tokens"]


class MossTileMixin:
    def load_diffusion_model(self):
        super().load_diffusion_model()
        if self.model_config.get("control_adapter") != "tile":
            self.tile_embedder = None
            return
        current = self.diffusion_model.video_embedder
        if not isinstance(current, nn.Linear) or current.in_features != LATENT_CHANNELS:
            raise RuntimeError(
                f"tile control expected video_embedder Linear({LATENT_CHANNELS}, dim), got {type(current)} "
                f"in_features={getattr(current, 'in_features', None)}"
            )
        self.tile_embedder = TileVideoEmbedder(current)
        self.diffusion_model.video_embedder = self.tile_embedder
        print(
            f"MOSS tile control: video_embedder {LATENT_CHANNELS} -> "
            f"{LATENT_CHANNELS * (1 + CONTROL_SLOTS)} (color=degraded image, depth=zeros)"
        )

    def get_call_vae_fn(self, vae):
        parent = super().get_call_vae_fn(vae)

        def fn(*args):
            if len(args) == 1:
                return parent(args[0])
            if len(args) != 2:
                raise RuntimeError(f"unexpected VAE encode args: {len(args)}")
            tensor, control_tensor = args
            result = parent(tensor)
            result["control_latents"] = parent(control_tensor)["latents"]
            return result

        return fn

    def get_target_modules(self, target_model):
        modules = super().get_target_modules(target_model)
        if self.model_config.get("control_adapter") == "tile":
            modules = [name for name in modules if "text_embedder" not in name]
        return modules

    def configure_adapter(self, adapter_config):
        super().configure_adapter(adapter_config)
        embedder = getattr(self, "tile_embedder", None)
        if embedder is None:
            return
        weight = embedder.control.weight
        weight.requires_grad_(True)
        weight.data = weight.data.to(adapter_config["dtype"])
        weight.original_name = "video_embedder.control.weight"

    def get_param_groups(self, parameters):
        embedder = getattr(self, "tile_embedder", None)
        if embedder is None:
            return [{"params": parameters}]
        control = [parameter for parameter in parameters if parameter is embedder.control.weight]
        rest = [parameter for parameter in parameters if parameter is not embedder.control.weight]
        groups = []
        if rest:
            groups.append({"params": rest})
        if control:
            multiplier = float(self.model_config.get("control_lr_multiplier", 10.0))
            groups.append({"params": control, "lr": float(self.config["optimizer"]["lr"]) * multiplier})
        return groups

    def save_adapter(self, save_dir, sd):
        sd = {key: value for key, value in sd.items() if not str(key).endswith("video_embedder.control.weight")}
        super().save_adapter(save_dir, sd)
        embedder = getattr(self, "tile_embedder", None)
        if embedder is None:
            return
        import safetensors.torch

        path = save_dir / "adapter_model.safetensors"
        state = safetensors.torch.load_file(path)
        weight = embedder.merged_weight()
        bias = embedder.merged_bias()
        save_dtype = self.config.get("save_dtype")
        if isinstance(save_dtype, str):
            weight = weight.to(getattr(torch, save_dtype))
            bias = bias.to(weight.dtype)
        state["diffusion_model.video_embedder.set_weight"] = weight
        state["diffusion_model.video_embedder.bias.set_weight"] = bias
        safetensors.torch.save_file(state, path, metadata={"format": "pt"})
        print(f"Wrote ComfyUI tile embedder {tuple(weight.shape)} into {path}")
