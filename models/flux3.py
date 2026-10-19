import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
from torch import nn
import torch.nn.functional as F

import comfy.ops
from comfy.ldm.flux.layers import ModulationOut, modulated_norm, timestep_embedding
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

from models.base import ComfyPipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift
from utils.offloading import ModelOffloader


def _batched_text_rope(llama, position_ids):
    """One 1D RoPE row per caption.

    Qwen3-VL interleaved MRoPE reads position_ids.shape[0] as the three T/H/W
    axes. Flux3 text positions are (batch, sequence), so that read is only
    valid for a single caption.
    """
    pieces = [llama.compute_freqs_cis(row.unsqueeze(0), row.device) for row in position_ids]
    first = pieces[0]
    if isinstance(first, list):
        return [
            tuple(torch.cat([piece[axis][part] for piece in pieces], dim=0) for part in range(3))
            for axis in range(len(first))
        ]
    return tuple(torch.cat([piece[part] for piece in pieces], dim=0) for part in range(3))


def _install_flux3_text_rope_batch_fix():
    import comfy.text_encoders.flux as flux_text

    cls = flux_text.Flux3Qwen3VL
    if getattr(cls.forward, '_flux3_batched_text_rope', False):
        return
    original = cls.forward

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is not None and kwargs.get('freqs_cis') is None:
            # Same positions as Flux3Qwen3VL: real tokens follow the mask
            # cumsum, and padding stays at rope position 1.
            position_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 1)
            if position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0)
            if position_ids.shape[0] != 1:
                kwargs['freqs_cis'] = _batched_text_rope(self.model, position_ids)
        return original(self, input_ids, attention_mask=attention_mask, **kwargs)

    forward._flux3_batched_text_rope = True
    cls.forward = forward


_install_flux3_text_rope_batch_fix()


def training_block_forward(block, x, spans, pe, transformer_options=None):
    """Flux3 Block.forward without in-place residuals, so autograd can see every row."""
    if transformer_options is None:
        transformer_options = {}
    batch, length, _ = x.shape
    base_mod = spans[0][2]
    x_mod = modulated_norm(x, block.pre_norm, base_mod.scale, base_mod.shift)
    for start, end, mod in spans[1:]:
        piece = modulated_norm(x[:, start:end], block.pre_norm, mod.scale, mod.shift)
        x_mod = torch.cat((x_mod[:, :start], piece, x_mod[:, end:]), dim=1)

    if block.fused_qkv:
        q, k, v = block.qkv(x_mod).chunk(3, dim=-1)
    else:
        q, k, v = block.q_proj(x_mod), block.k_proj(x_mod), block.v_proj(x_mod)
    mlp = block.mlp_in(x_mod)
    q, k = block.norm(q.view(batch, length, block.num_heads, -1), k.view(batch, length, block.num_heads, -1), pe)
    q = AttentionTensorContainer(q.flatten(2))
    k = AttentionTensorContainer(k.flatten(2))
    v = AttentionTensorContainer(v)
    attn = block.attn_out(optimized_attention(
        q, k, v, block.num_heads, transformer_options=transformer_options, preferred_attention=block.comfy_attention,
    ))
    out = comfy.ops.linear_input_act(block.mlp_out, mlp, 'swiglu', residual=attn, residual_scale=attn.new_ones(attn.shape[-1]))

    pieces = []
    cursor = 0
    for start, end, mod in spans:
        if start > cursor:
            pieces.append(x[:, cursor:start])
        pieces.append(torch.addcmul(x[:, start:end], out[:, start:end], mod.gate))
        cursor = end
    if cursor < length:
        pieces.append(x[:, cursor:])
    return torch.cat(pieces, dim=1)


class Flux3Pipeline(ComfyPipeline):
    name = 'flux3'
    checkpointable_layers = ['SingleTransformerLayer']
    adapter_target_modules = ['Block']
    keep_in_high_precision = ['emb_in', 'txt_in', 'time_in', 'ref_time_in', 'early_stream_modulations', 'single_stream_modulations', 'final_layer']
    spatial_compression = 16
    channels = 128
    # ComfyUI sampling_settings shift is mu = ln(6.93). This pipeline's shift is the multiplier.
    default_shift = 6.93

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_format = comfy.latent_formats.Flux2()
        self.offloader_early = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.offloader_single = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

    def load_diffusion_model(self):
        super().load_diffusion_model()
        if getattr(self.diffusion_model, 'flow_map', False):
            raise RuntimeError(
                'This is a step-distilled FLUX 3 Klein checkpoint (it has ref_time_in). '
                'Train the Klein Base weights. LoRAs trained on Base can be loaded on the distilled model afterwards.'
            )

    def get_call_vae_fn(self, vae):
        parent_vae_fn = super().get_call_vae_fn(vae)

        def fn(*args, **kwargs):
            if len(args) == 1:
                return parent_vae_fn(args[0], **kwargs)
            if len(args) != 2:
                raise RuntimeError(f'Unexpected number of args: {len(args)}')
            tensor, control_tensor = args
            result = parent_vae_fn(tensor, **kwargs)
            if control_tensor.ndim == tensor.ndim + 1:
                bs, num_controls = control_tensor.shape[:2]
                encoded = parent_vae_fn(control_tensor.flatten(0, 1), **kwargs)['latents']
                result['control_latents'] = encoded.unflatten(0, (bs, num_controls))
            else:
                result['control_latents'] = parent_vae_fn(control_tensor, **kwargs)['latents']
            return result
        return fn

    def model_specific_dataset_config_validation(self, dataset_config):
        ref_counts = [_reference_count(directory) for directory in dataset_config['directory']]
        if any(count != ref_counts[0] for count in ref_counts):
            raise RuntimeError(
                'FLUX 3 Klein needs the same number of reference images in every dataset directory, including zero for text-to-image. '
                'Mixing counts changes the pipeline input.'
            )

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [InitialLayer(diffusion_model, self.offloader_early)]
        for i, block in enumerate(diffusion_model.single_blocks):
            layers.append(SingleTransformerLayer(block, i, self.offloader_single))
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

        extra_inputs = ()
        if 'control_latents' in inputs:
            control_latents = inputs['control_latents'].float()
            if control_latents.ndim == latents.ndim + 1:
                assert control_latents.shape[0] == bs and control_latents.shape[2:] == latents.shape[1:], (
                    control_latents.shape, latents.shape
                )
            else:
                assert control_latents.shape == latents.shape, (control_latents.shape, latents.shape)
            extra_inputs = (control_latents,)
        return (noisy_latents, t, text_embeds) + extra_inputs, (target, mask)

    def enable_block_swap(self, blocks_to_swap):
        model = self.diffusion_model
        early_blocks = (
            list(model.txt_mode_blocks)
            + list(model.content_mode_blocks['image_cond'])
            + list(model.content_mode_blocks['image'])
        )
        single_blocks = list(model.single_blocks)
        single_to_swap = min(blocks_to_swap, len(single_blocks) - 2)
        early_to_swap = blocks_to_swap - single_to_swap
        if early_to_swap > len(early_blocks) - 2:
            raise RuntimeError(
                f'Cannot swap {blocks_to_swap} blocks. Early blocks can swap at most {len(early_blocks) - 2} '
                f'and single blocks at most {len(single_blocks) - 2}.'
            )
        self.offloader_early = ModelOffloader(
            'EarlyBlock', early_blocks, len(early_blocks), early_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        self.offloader_single = ModelOffloader(
            'SingleBlock', single_blocks, len(single_blocks), single_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        txt_blocks = model.txt_mode_blocks
        content_blocks = model.content_mode_blocks
        single_block_list = model.single_blocks
        model.txt_mode_blocks = None
        model.content_mode_blocks = None
        model.single_blocks = None
        model.to('cuda')
        model.txt_mode_blocks = txt_blocks
        model.content_mode_blocks = content_blocks
        model.single_blocks = single_block_list
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {early_to_swap} early blocks and {single_to_swap} single blocks.')

    def prepare_block_swap_training(self):
        for offloader in (self.offloader_early, self.offloader_single):
            offloader.enable_block_swap()
            offloader.set_forward_only(False)
            offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        for offloader in (self.offloader_early, self.offloader_single):
            if disable_block_swap:
                offloader.disable_block_swap()
            offloader.set_forward_only(True)
            offloader.prepare_block_devices_before_forward()


class InitialLayer(nn.Module):
    def __init__(self, model, offloader):
        super().__init__()
        self.emb_in = model.emb_in
        self.txt_in = model.txt_in
        self.time_in = model.time_in
        self.early_stream_modulations = model.early_stream_modulations
        self.single_stream_modulations = model.single_stream_modulations
        self.content_mode_blocks = model.content_mode_blocks
        self.txt_mode_blocks = model.txt_mode_blocks
        self.pe_embedder = model.pe_embedder
        self.offloader = offloader
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    def _rope(self, ids):
        # pe_embedder lives on this layer. The original Flux3.rope method would look for it on the DiT.
        return self.pe_embedder(ids.unsqueeze(0)).transpose(1, 2).contiguous()

    def _run_early(self, blocks, tokens, mod, pe, index):
        span = None if tokens is None else [(0, tokens.shape[1], mod)]
        for block in blocks:
            self.offloader.wait_for_block(index)
            if tokens is not None:
                tokens = training_block_forward(block, tokens, span, pe, {})
            self.offloader.submit_move_blocks_forward(index)
            index += 1
        return tokens, index

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        x, timesteps, context, *extra = inputs
        ref_latents = []
        if extra:
            control_latents = extra[0]
            if control_latents.ndim == x.ndim:
                ref_latents = [control_latents]
            else:
                ref_latents = [control_latents[:, i] for i in range(control_latents.shape[1])]

        batch, _, height, width = x.shape
        device = x.device
        dtype = x.dtype
        vec_zero_t = context.new_zeros(1)
        # Text and references are modulated at t = 0. The image uses the sampled t.
        vec = self.time_in(timestep_embedding(torch.cat([timesteps, vec_zero_t]).to(dtype), 256))
        vec_img, vec_zero = vec[:-1], vec[-1:]

        txt = self.txt_in(context)
        txt_mod = self.early_stream_modulations['txt'](vec_zero)[0]
        txt, index = self._run_early(self.txt_mode_blocks, txt, txt_mod, self._rope(self.text_ids(txt.shape[1], device)), 0)

        if ref_latents:
            refs = self.emb_in['image_cond'](torch.cat([ref.flatten(2).transpose(1, 2) for ref in ref_latents], dim=1))
            ref_shapes = [ref.shape[-2:] for ref in ref_latents]
            ref_mod = self.early_stream_modulations['image_cond'](vec_zero)[0]
            refs, index = self._run_early(
                self.content_mode_blocks['image_cond'], refs, ref_mod, self._rope(self.ref_ids(ref_shapes, device)), index,
            )
        else:
            refs = None
            ref_shapes = []
            _, index = self._run_early(self.content_mode_blocks['image_cond'], None, None, None, index)

        img = self.emb_in['image'](x.flatten(2).transpose(1, 2))
        img_len = img.shape[1]
        ref_len = 0 if refs is None else refs.shape[1]
        txt_len = txt.shape[1]
        img_ids = self.image_ids(height, width, 0, device)
        ref_ids = self.ref_ids(ref_shapes, device) if ref_shapes else img_ids.new_zeros((0, 4))
        txt_ids = self.text_ids(txt_len, device)
        pe = self._rope(torch.cat([img_ids, ref_ids, txt_ids], dim=0))
        img_mod = self.early_stream_modulations['image'](vec_img)[0]
        img, index = self._run_early(self.content_mode_blocks['image'], img, img_mod, pe[:, :img_len], index)

        parts = [img]
        if refs is not None:
            parts.append(refs)
        parts.append(txt)
        hidden = torch.cat(parts, dim=1)

        def modulation_rows(mode, vec_mode):
            mod = self.single_stream_modulations[mode](vec_mode)[0]
            return mod.shift, mod.scale, mod.gate

        img_shift, img_scale, img_gate = modulation_rows('image', vec_img)
        cond_shift, cond_scale, cond_gate = modulation_rows('image_cond', vec_zero)
        txt_shift, txt_scale, txt_gate = modulation_rows('txt', vec_zero)
        lengths = torch.tensor([img_len, ref_len, txt_len, height, width], dtype=torch.int64, device=device)
        pe.requires_grad_(True)
        return make_contiguous(
            hidden, pe, vec_img, lengths,
            img_shift, img_scale, img_gate,
            cond_shift, cond_scale, cond_gate,
            txt_shift, txt_scale, txt_gate,
        )


class SingleTransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden, pe, vec_img, lengths, img_shift, img_scale, img_gate, cond_shift, cond_scale, cond_gate, txt_shift, txt_scale, txt_gate = inputs
        img_len, ref_len, txt_len, _, _ = (int(v) for v in lengths.tolist())
        spans = [(0, img_len, ModulationOut(img_shift, img_scale, img_gate))]
        cursor = img_len
        if ref_len:
            spans.append((cursor, cursor + ref_len, ModulationOut(cond_shift, cond_scale, cond_gate)))
            cursor += ref_len
        spans.append((cursor, cursor + txt_len, ModulationOut(txt_shift, txt_scale, txt_gate)))

        self.offloader.wait_for_block(self.block_idx)
        hidden = training_block_forward(self.layer, hidden, spans, pe, {})
        self.offloader.submit_move_blocks_forward(self.block_idx)
        return make_contiguous(
            hidden, pe, vec_img, lengths,
            img_shift, img_scale, img_gate,
            cond_shift, cond_scale, cond_gate,
            txt_shift, txt_scale, txt_gate,
        )


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden, pe, vec_img, lengths, *_ = inputs
        img_len, _, _, height, width = (int(v) for v in lengths.tolist())
        out = self.final_layer['image'](hidden[:, :img_len], vec_img)
        return out.transpose(1, 2).reshape(hidden.shape[0], self.in_channels, height, width)


def _reference_count(directory_config):
    if 'control_path' in directory_config and 'control_paths' in directory_config:
        raise RuntimeError('control_path and control_paths are mutually exclusive')
    if 'control_paths' in directory_config:
        return len(directory_config['control_paths'])
    if 'control_path' in directory_config:
        return 1
    return 0


def _sample_t(model_config, batch, device, height, width, timestep_quantile, default_shift):
    timestep_sample_method = model_config.get('timestep_sample_method', 'logit_normal')
    if timestep_sample_method == 'logit_normal':
        dist = torch.distributions.normal.Normal(0, 1)
    elif timestep_sample_method == 'uniform':
        dist = torch.distributions.uniform.Uniform(0, 1)
    else:
        raise NotImplementedError()

    if timestep_quantile is not None:
        t = dist.icdf(torch.full((batch,), timestep_quantile, device=device))
    else:
        t = dist.sample((batch,)).to(device)

    if timestep_sample_method == 'logit_normal':
        t = t * model_config.get('sigmoid_scale', 1.0)
        t = torch.sigmoid(t)

    if model_config.get('flux_shift', False):
        mu = get_lin_function(y1=0.5, y2=1.15)((height // 2) * (width // 2))
        t = time_shift(mu, 1.0, t)
    else:
        shift = model_config.get('shift', default_shift)
        t = (t * shift) / (1 + (shift - 1) * t)
    return t
