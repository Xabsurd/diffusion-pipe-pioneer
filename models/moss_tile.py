"""Mixin that teaches diffusion-pipe's MossImagePipeline to train a tile adapter.

The condition images are an offline degradation already stored beside the
targets. Training caches those pixels with the VAE. It does not build a new
degradation each step.

ComfyUI loads the merged projection:

    diffusion_model.video_embedder.set_weight          [dim, 384]
    diffusion_model.video_embedder.bias.set_weight

init_from_existing also reads diffusion_model.video_embedder.control.weight,
the trainable 256 control columns. DeepSpeed checkpoints store that same
parameter because it requires grad. The frozen 128 base columns stay in the
base checkpoint.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn

from models.tile_embedder import (
    CONTROL_SLOTS,
    LATENT_CHANNELS,
    TileVideoEmbedder,
    pack_tile_tokens,
    split_merged_embedder,
)

__all__ = ["MossTileMixin", "TileVideoEmbedder", "pack_tile_tokens"]

def _prefer_non_cudnn_attention():
    """cuDNN SDPA can report no execution plan and then not fall through.

    ComfyUI pins that backend with priority. Only tile runs replace it.
    """
    import comfy.ops
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def scaled_dot_product_attention(q, k, v, *args, **kwargs):
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)

    comfy.ops.scaled_dot_product_attention = scaled_dot_product_attention


SET_WEIGHT_KEY = "diffusion_model.video_embedder.set_weight"
SET_BIAS_KEY = "diffusion_model.video_embedder.bias.set_weight"
CONTROL_WEIGHT_KEY = "diffusion_model.video_embedder.control.weight"


class MossTileMixin:
    def load_diffusion_model(self):
        super().load_diffusion_model()
        if self.model_config.get("control_adapter") != "tile":
            self.tile_embedder = None
            return
        current = self.diffusion_model.video_embedder
        if isinstance(current, TileVideoEmbedder):
            self.tile_embedder = current
            return
        if not isinstance(current, nn.Linear) or current.in_features != LATENT_CHANNELS:
            raise RuntimeError(
                f"tile control expected video_embedder Linear({LATENT_CHANNELS}, dim), got {type(current)} "
                f"in_features={getattr(current, 'in_features', None)}"
            )
        self.tile_embedder = TileVideoEmbedder(current)
        self.diffusion_model.video_embedder = self.tile_embedder
        _prefer_non_cudnn_attention()
        print(
            f"MOSS tile control: video_embedder {LATENT_CHANNELS} -> "
            f"{LATENT_CHANNELS * (1 + CONTROL_SLOTS)} (color=degraded image, depth=zeros). "
            "Condition latents are the offline degraded images, not a new degradation each step."
        )

    def get_call_vae_fn(self, vae):
        """Encode the dataset control image. That file is the offline condition."""
        parent = super().get_call_vae_fn(vae)

        def fn(*args):
            if len(args) == 1:
                return parent(args[0])
            if len(args) != 2:
                raise RuntimeError(f"unexpected VAE encode args: {len(args)}")
            tensor, control_tensor = args
            if tuple(tensor.shape[-2:]) != tuple(control_tensor.shape[-2:]):
                raise RuntimeError(
                    f"target pixels {tuple(tensor.shape)} and condition pixels {tuple(control_tensor.shape)} differ"
                )
            result = parent(tensor)
            result["control_latents"] = parent(control_tensor)["latents"]
            target_latents = result["latents"]
            if tuple(result["control_latents"].shape) != tuple(target_latents.shape):
                raise RuntimeError(
                    f"control latents {tuple(result['control_latents'].shape)} != "
                    f"target latents {tuple(target_latents.shape)}"
                )
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
        """Return a control group with lr_scale. The optimizer applies the base lr."""
        embedder = getattr(self, "tile_embedder", None)
        if embedder is None:
            return [{"params": parameters}]
        control = []
        rest = []
        for parameter in parameters:
            name = getattr(parameter, "original_name", "")
            if parameter is embedder.control.weight or name == "video_embedder.control.weight":
                control.append(parameter)
            else:
                rest.append(parameter)
        groups = []
        if rest:
            groups.append({"params": rest})
        if control:
            multiplier = float(self.model_config.get("control_lr_multiplier", 10.0))
            groups.append({"params": control, "lr_scale": multiplier})
        return groups

    def _cast_save_tensor(self, tensor):
        save_dtype = self.config.get("save_dtype")
        if isinstance(save_dtype, str):
            save_dtype = getattr(torch, save_dtype)
        if isinstance(save_dtype, torch.dtype):
            return tensor.to(dtype=save_dtype)
        return tensor

    def save_adapter(self, save_dir, sd):
        control_keys = [key for key in sd if str(key).endswith("video_embedder.control.weight")]
        sd = {key: value for key, value in sd.items() if key not in control_keys}
        super().save_adapter(save_dir, sd)
        embedder = getattr(self, "tile_embedder", None)
        if embedder is None:
            return
        import safetensors.torch

        path = Path(save_dir) / "adapter_model.safetensors"
        state = safetensors.torch.load_file(path)
        control_weight = self._cast_save_tensor(embedder.control.weight.detach().cpu().contiguous())
        merged_weight = self._cast_save_tensor(embedder.merged_weight().cpu())
        merged_bias = self._cast_save_tensor(embedder.merged_bias().cpu())
        state[CONTROL_WEIGHT_KEY] = control_weight
        state[SET_WEIGHT_KEY] = merged_weight
        state[SET_BIAS_KEY] = merged_bias
        safetensors.torch.save_file(state, path, metadata={"format": "pt"})
        print(f"Wrote ComfyUI tile embedder {tuple(merged_weight.shape)} into {path}")

    def load_adapter_weights(self, adapter_path):
        embedder = getattr(self, "tile_embedder", None)
        if embedder is None:
            return super().load_adapter_weights(adapter_path)
        adapter_path = Path(adapter_path)
        safetensors_files = list(adapter_path.glob("*.safetensors"))
        if len(safetensors_files) != 1:
            raise RuntimeError(f"expected one safetensors file in {adapter_path}, found {len(safetensors_files)}")
        adapter_state_dict = __import__("safetensors.torch", fromlist=["load_file"]).load_file(safetensors_files[0])
        self._load_control_projection(adapter_state_dict)
        lora_state = {
            key: value
            for key, value in adapter_state_dict.items()
            if key not in (SET_WEIGHT_KEY, SET_BIAS_KEY, CONTROL_WEIGHT_KEY)
        }
        print(f"Loading adapter weights from path {adapter_path}")
        modified_state_dict = {}
        model_parameters = {name for name, _parameter in self.diffusion_model.named_parameters()}
        for key, value in lora_state.items():
            renamed = re.sub(r"^(transformer|diffusion_model)\.", "", key)
            renamed = re.sub(r"\.weight$", ".default.weight", renamed)
            if renamed not in model_parameters:
                raise RuntimeError(f"modified_state_dict key {renamed} is not in the model parameters")
            modified_state_dict[renamed] = value
        self.diffusion_model.load_state_dict(modified_state_dict, strict=False)
        embedder.control.weight.requires_grad_(True)

    def _load_control_projection(self, state):
        embedder = self.tile_embedder
        if CONTROL_WEIGHT_KEY in state:
            control_weight = state[CONTROL_WEIGHT_KEY]
        elif SET_WEIGHT_KEY in state:
            _base_weight, control_weight, _bias = split_merged_embedder(state[SET_WEIGHT_KEY])
        else:
            raise RuntimeError("tile adapter is missing video_embedder control columns")
        if tuple(control_weight.shape) != tuple(embedder.control.weight.shape):
            raise RuntimeError(
                f"control projection {tuple(control_weight.shape)} != {tuple(embedder.control.weight.shape)}"
            )
        with torch.no_grad():
            embedder.control.weight.copy_(control_weight.to(device=embedder.control.weight.device, dtype=embedder.control.weight.dtype))
        embedder.control.weight.requires_grad_(True)
