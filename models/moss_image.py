import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
from torch import nn
import torch.nn.functional as F

from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.moss_image.model import modulated_rms_norm

from models.base import ComfyPipeline, make_contiguous
from models.flux3 import _sample_t
from utils.common import AUTOCAST_DTYPE
from utils.offloading import ModelOffloader


class MossImagePipeline(ComfyPipeline):
    """MOSS-Image text-to-image. The released model has no reference-image path."""

    name = 'moss_image'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['Block', 'TextEmbedder']
    keep_in_high_precision = [
        'video_embedder', 'text_embedder', 'time_embedding', 'shared_block_modulation',
        'final_norm_video', 'final_linear_video', 'video_head_modulation',
    ]
    spatial_compression = 16
    channels = 128
    default_shift = 5.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_format = comfy.latent_formats.Flux2()
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

    def model_specific_dataset_config_validation(self, dataset_config):
        for directory_config in dataset_config['directory']:
            if 'control_path' in directory_config or 'control_paths' in directory_config:
                raise RuntimeError('MOSS-Image training is text-to-image only. Reference-image editing is not in this release.')

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.layers):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(diffusion_model))
        return layers

    def get_conds(self, inputs):
        text_embeds = inputs['text_embeds_0']
        if torch.is_tensor(text_embeds):
            return (text_embeds,)
        max_len = max(embed.shape[0] for embed in text_embeds)
        text_embeds = torch.stack([
            torch.cat([embed, embed.new_zeros(max_len - embed.shape[0], embed.shape[1])])
            if embed.shape[0] < max_len else embed
            for embed in text_embeds
        ])
        return (text_embeds,)

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']
        text_embeds, = self.get_conds(inputs)
        bs, _, h, w = latents.shape
        device = latents.device

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')

        t = _sample_t(self.model_config, bs, device, h, w, timestep_quantile, self.default_shift)
        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents
        return (noisy_latents, t, text_embeds), (target, mask)

    def enable_block_swap(self, blocks_to_swap):
        diffusion_model = self.diffusion_model
        blocks = diffusion_model.layers
        num_blocks = len(blocks)
        assert blocks_to_swap <= num_blocks - 2, (
            f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        )
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        diffusion_model.layers = None
        diffusion_model.to('cuda')
        diffusion_model.layers = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.video_embedder = model.video_embedder
        self.text_embedder = model.text_embedder
        self.time_embedding = model.time_embedding
        self.shared_block_modulation = model.shared_block_modulation
        self.video_head_modulation = model.video_head_modulation
        self.pe_embedder = model.pe_embedder
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    def _rope(self, height, width, text_len, device):
        # pe_embedder lives on this layer. MOSSImage.rope looks for it on the DiT.
        ids = torch.zeros(height * width + text_len, 4, device=device, dtype=torch.float32)
        ids[:height * width, 1] = text_len
        ids[:height * width, 2] = (torch.arange(height, device=device, dtype=torch.float32) - (height - 1) / 2).repeat_interleave(width)
        ids[:height * width, 3] = (torch.arange(width, device=device, dtype=torch.float32) - (width - 1) / 2).repeat(height)
        ids[height * width:, 1] = torch.arange(text_len, device=device, dtype=torch.float32)
        return self.pe_embedder(ids.unsqueeze(0)).transpose(1, 2).contiguous()

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        x, timesteps, context = inputs
        batch, _, height, width = x.shape
        device = x.device
        dtype = x.dtype
        text_len = context.shape[1]
        pe = self._rope(height, width, text_len, device)
        vec = self.time_embedding(timestep_embedding(timesteps, 256).to(dtype))
        mod = self.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
        head_shift, head_scale = self.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
        txt = self.text_embedder(context, pe[:, height * width:], {})
        hidden = torch.cat((self.video_embedder(x.flatten(2).transpose(1, 2)), txt), dim=1)
        sizes = torch.tensor([height * width, height, width], dtype=torch.int64, device=device)
        pe.requires_grad_(True)
        return make_contiguous(hidden, pe, sizes, head_shift, head_scale, *mod)


class TransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden, pe, sizes, head_shift, head_scale, *mod = inputs
        self.offloader.wait_for_block(self.block_idx)
        hidden = self.layer(hidden, mod, pe, {})
        self.offloader.submit_move_blocks_forward(self.block_idx)
        return make_contiguous(hidden, pe, sizes, head_shift, head_scale, *mod)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_norm_video = model.final_norm_video
        self.final_linear_video = model.final_linear_video
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden, pe, sizes, head_shift, head_scale, *mod = inputs
        img_len, height, width = (int(v) for v in sizes.tolist())
        out = self.final_linear_video(modulated_rms_norm(hidden[:, :img_len], self.final_norm_video, head_scale, head_shift))
        return out.transpose(1, 2).reshape(hidden.shape[0], self.latent_channels, height, width)
