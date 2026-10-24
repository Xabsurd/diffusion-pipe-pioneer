import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.base import ComfyPipeline, make_contiguous, maybe_pad
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift
from utils.offloading import ModelOffloader
import comfy.latent_formats
from comfy.ldm.qwen_image21.model import _modulated_norm, block_causal_attention


def _training_gated_residual(x, y, gate, prefix_len):
    # The ComfyUI block adds the residual with addcmul_ on a view of x. Autograd rejects that.
    g_prefix, g_target = gate
    if prefix_len:
        return torch.cat([
            torch.addcmul(x[:, :prefix_len], y[:, :prefix_len], g_prefix),
            torch.addcmul(x[:, prefix_len:], y[:, prefix_len:], g_target),
        ], dim=1)
    return torch.addcmul(x, y, g_target)


def training_qwen21_block(block, x, mod, pe, attn_fn, prefix_len, transformer_options):
    scale1, gate1, scale2, gate2, zero = mod
    # Autocast promotes LayerNorm to FP32. Cast after modulation, just as the
    # following BF16 linear would, so Comfy does not dequantize INT8 weights.
    attn_input = _modulated_norm(block.img_norm1, x, scale1, prefix_len, zero).to(x.dtype)
    attn_out = block.attn(attn_input, pe, attn_fn, prefix_len, transformer_options)
    x = _training_gated_residual(x, attn_out, gate1, prefix_len)
    mlp_input = _modulated_norm(block.img_norm2, x, scale2, prefix_len, zero).to(x.dtype)
    mlp_out = block.img_mlp(mlp_input)
    x = _training_gated_residual(x, mlp_out, gate2, prefix_len)
    if x.dtype == torch.float16:
        x = x.clamp(-65504, 65504)
    return x


def _gated_residual(x, y, gate, prefix_len):
    g_prefix, g_target = gate
    out = torch.addcmul(x, y, g_target)
    if prefix_len:
        out[:, :prefix_len] = torch.addcmul(x[:, :prefix_len], y[:, :prefix_len], g_prefix)
    return out

# remove in-place operation which breaks backward()
comfy.ldm.qwen_image21.model._gated_residual = _gated_residual


def _split_rows(p):
    # shared modulation rows: (t = 0 row for text and references, sampled-t rows for the target)
    return p[-1:].unsqueeze(1), p[:-1].unsqueeze(1)


def pack_segments(segments):
    endpoints = torch.tensor([row[:2] for row in segments])
    masks = [row[2] for row in segments]
    masks = [m if m is not None else torch.tensor([], dtype=bool) for m in masks]
    return endpoints, masks


def unpack_segments(endpoints, masks):
    endpoints = endpoints.tolist()
    segments = []
    for ((start, end), mask) in zip(endpoints, masks):
        if mask.numel() == 0:
            mask = None
        segments.append((start, end, mask))
    return segments


class QwenImage21Pipeline(ComfyPipeline):
    name = 'qwen_image21'
    spatial_compression = 16
    channels = 64
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['QwenImage21TransformerBlock']
    keep_in_high_precision = ['img_in', 'txt_in', 'time_text_embed', 'modulation', 'norm_out', 'proj_out']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_format = comfy.latent_formats.QwenImage21()
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

    def to_layers(self):
        diffusion_model = self.diffusion_model
        transformer_options = {
            'total_blocks': len(diffusion_model.transformer_blocks),
            'block_type': 'single',
        }
        layers = [InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.transformer_blocks):
            layers.append(TransformerLayer(block, i, transformer_options, self.offloader))
        layers.append(FinalLayer(diffusion_model))
        return layers

    def get_conds(self, inputs):
        text_embeds = inputs['text_embeds_0']
        attention_mask = inputs['attention_mask_0']
        # text embeds are variable length
        max_seq_len = max([e.size(0) for e in text_embeds])
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in text_embeds]
        )
        attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attention_mask]
        )
        assert text_embeds.shape[:2] == attention_mask.shape[:2]
        attention_mask = attention_mask.to(torch.bool)
        return text_embeds, attention_mask

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']

        bs, c, h, w = latents.shape
        device = latents.device

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

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
            if 'image_slots' not in inputs:
                raise RuntimeError(
                    'Qwen-Image-2.1 edit latents have no image_slots. Delete the dataset cache and rerun so the text encoder sees the reference images.'
                )
            image_slots = inputs['image_slots']
            if not torch.is_tensor(image_slots):
                raise RuntimeError(
                    'image_slots were not batched into a tensor. Keep uncond_fraction at 0, and use the same number of reference images for every sample.'
                )
            extra_inputs = (control_latents, image_slots.to(device=device, dtype=torch.long))

        return (noisy_latents, t, *self.get_conds(inputs)) + extra_inputs, (target, mask)

    def enable_block_swap(self, blocks_to_swap):
        diffusion_model = self.diffusion_model
        blocks = diffusion_model.transformer_blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        diffusion_model.transformer_blocks = None
        diffusion_model.to('cuda')
        diffusion_model.transformer_blocks = blocks
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

    def model_specific_dataset_config_validation(self, dataset_config):
        ref_counts = []
        for directory_config in dataset_config['directory']:
            if 'control_path' in directory_config and 'control_paths' in directory_config:
                raise RuntimeError('control_path and control_paths are mutually exclusive')
            if 'control_paths' in directory_config:
                ref_counts.append(len(directory_config['control_paths']))
            elif 'control_path' in directory_config:
                ref_counts.append(1)
            else:
                ref_counts.append(0)
        if any(count > 0 for count in ref_counts) and any(count != ref_counts[0] for count in ref_counts):
            raise RuntimeError(
                'Qwen-Image-2.1 edit needs the same number of reference images in every dataset directory. '
                'Mixing text-to-image samples with edit samples changes how many tensors the pipeline passes between stages.'
            )
        if any(count > 0 for count in ref_counts) and self.config.get('uncond_fraction', 0) > 0:
            raise RuntimeError(
                'Qwen-Image-2.1 edit does not support uncond_fraction > 0. The shared empty-caption embedding has no reference images. Set uncond_fraction = 0.'
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

    def _load_vlm_image(self, path):
        # Same resize the ComfyUI Qwen-Image-2.1 encoder uses: about N^2 pixels, multiples of 32.
        resolution = int(self.model_config.get('edit_vlm_resolution', 1024))
        pil_img = Image.open(path).convert('RGB')
        width, height = pil_img.size
        if resolution > 0:
            ratio = width / height
            width = max(32, round(math.sqrt(resolution * resolution * ratio) / 32) * 32)
            height = max(32, round(math.sqrt(resolution * resolution / ratio) / 32) * 32)
        else:
            width = max(32, round(width / 32) * 32)
            height = max(32, round(height / 32) * 32)
        if (width, height) != pil_img.size:
            pil_img = pil_img.resize((width, height), Image.Resampling.LANCZOS)
        return transforms.ToTensor()(pil_img).movedim(0, -1).unsqueeze(0)

    def get_call_text_encoder_fn(self, text_encoder):
        # Text-to-image keeps the shared encoder, so existing caches stay valid.
        parent_fn = super().get_call_text_encoder_fn(text_encoder)
        te_idx = None
        for i, te in enumerate(self.text_encoders):
            if text_encoder == te:
                te_idx = i
                break
        if te_idx is None:
            raise RuntimeError('Unknown text encoder')

        @torch.inference_mode()
        def fn(captions: list[str], is_video: list[bool], control_files=None):
            # One sample at a time. The Qwen-Image-2.1 encoder records image slots from a single token sequence.
            assert not any(is_video)
            files_per_sample = _control_files_for_batch(control_files, len(captions))
            if all(len(files) == 0 for files in files_per_sample):
                return parent_fn(captions, is_video)
            tokenizer = getattr(text_encoder.tokenizer, text_encoder.tokenizer.clip)
            original_min_length = tokenizer.min_length
            encoded = []
            try:
                for caption, files in zip(captions, files_per_sample):
                    images = [self._load_vlm_image(file) for file in files]
                    tokens = text_encoder.tokenize(caption, images=images, keep_vision=len(images) == 0)
                    token_lists = {k: v for k, v in tokens.items() if k != 'keep_vision'}
                    max_length = max(len(v[0]) for v in token_lists.values())
                    tokenizer.min_length = max_length
                    tokens_dict = {}
                    for key, value in token_lists.items():
                        token_list = value[0]
                        maybe_pad(token_list, max_length, tokenizer)
                        tokens_dict[key] = [token_list]
                    tokens_dict['keep_vision'] = len(images) == 0
                    encoded_tokens = text_encoder.encode_from_tokens_scheduled(tokens_dict)
                    embed = encoded_tokens[0][0]
                    if embed.ndim == 3:
                        embed = embed[0]
                    extra = encoded_tokens[0][1]
                    if 'attention_mask' in extra:
                        mask = extra['attention_mask']
                        if mask.ndim == 2:
                            mask = mask[0]
                    else:
                        mask = torch.ones(embed.shape[0], dtype=torch.int64)
                    encoded.append((embed.cpu(), mask.cpu().to(torch.int64), list(extra.get('image_slots', []))))
            finally:
                tokenizer.min_length = original_min_length

            slot_counts = {len(slots) for _, _, slots in encoded}
            if len(slot_counts) != 1:
                raise RuntimeError(f'Reference image slot counts differ inside one text-encoder batch: {sorted(slot_counts)}')
            max_len = max(embed.shape[0] for embed, _, _ in encoded)
            embeds, masks = [], []
            for embed, mask, _ in encoded:
                if embed.shape[0] < max_len:
                    embed = torch.cat([embed, embed.new_zeros(max_len - embed.shape[0], embed.shape[1])])
                    mask = torch.cat([mask, mask.new_zeros(max_len - mask.shape[0])])
                embeds.append(embed)
                masks.append(mask)
            result = {
                f'text_embeds_{te_idx}': torch.stack(embeds),
                f'attention_mask_{te_idx}': torch.stack(masks),
            }
            num_slots = slot_counts.pop()
            if num_slots > 0:
                result['image_slots'] = torch.tensor([slots for _, _, slots in encoded], dtype=torch.long)
            return result

        return fn


def _control_files_for_batch(control_files, batch_size):
    if control_files is None:
        return [[] for _ in range(batch_size)]
    if isinstance(control_files, str):
        return [[control_files]]
    if len(control_files) == 0:
        return [[] for _ in range(batch_size)]
    first = control_files[0]
    if isinstance(first, str):
        return [[file] for file in control_files]
    return [list(files) for files in control_files]


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.img_in = model.img_in
        self.txt_in = model.txt_in
        self.time_text_embed = model.time_text_embed
        self.modulation = model.modulation
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    def build_sequence(self, x, context, context_mask, ref_latents, image_slots):
        # text with each reference image spliced in at its slot, target image last
        bs = x.shape[0]
        txt = self.txt_in(context)
        txt_len = txt.shape[1]
        slots = [int(slot) for slot in image_slots]
        if any(slot < 0 or slot > txt_len for slot in slots):
            raise RuntimeError(f'Reference image slot {slots} is outside the text length {txt_len}')
        if len(slots) != len(ref_latents):
            raise RuntimeError(f'{len(ref_latents)} reference latents but {len(slots)} image slots')
        bounds = [0] + slots + [txt_len]

        parts, ids, segments = [], [], []
        pos, length = 0, 0
        prefix_valid = context_mask.new_zeros((bs, 0), dtype=torch.bool)
        for (start, end), img in zip(zip(bounds[:-1], bounds[1:]), list(ref_latents) + [x]):
            n = end - start
            if n > 0:
                parts.append(txt[:, start:end])
                ids.append(torch.arange(pos, pos + n, device=x.device, dtype=torch.float32).unsqueeze(1).expand(n, 3))
                # Causal over the sequence so far, and padding text tokens stay masked.
                text_valid = context_mask[:, start:end].to(torch.bool)
                key_valid = torch.cat([prefix_valid, text_valid], dim=1)
                causal = torch.ones((n, length + n), dtype=torch.bool, device=x.device).tril(length)
                mask = key_valid[:, None, None, :] & causal[None, None, :, :]
                segments.append((length, length + n, mask))
                prefix_valid = key_valid
                pos += n
                length += n
            h, w = img.shape[-2:]
            parts.append(self.img_in(img.flatten(2).transpose(1, 2)))
            # half a token where a reference grid has the other parity, so it centres on the target
            hh = torch.arange(h, device=x.device, dtype=torch.float32) - (h - h // 2) + 0.5 * (h % 2 - x.shape[-2] % 2)
            ww = torch.arange(w, device=x.device, dtype=torch.float32) - (w - w // 2) + 0.5 * (w % 2 - x.shape[-1] % 2)
            ids.append(torch.stack([torch.full((h, w), pos, device=x.device, dtype=torch.float32), hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1).flatten(0, 1))
            image_valid = torch.ones((bs, h * w), dtype=torch.bool, device=x.device)
            key_valid = torch.cat([prefix_valid, image_valid], dim=1)
            segments.append((length, length + h * w, key_valid[:, None, None, :]))
            prefix_valid = key_valid
            pos += max(h, w)
            length += h * w

        # (1, N, 1, ...): the layout the fused rms_rope wants for (B, N, H, D) queries
        pe = self.pe_embedder(torch.cat(ids, dim=0).unsqueeze(0)).transpose(1, 2).contiguous()
        return torch.cat(parts, dim=1), pe, segments

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        x, timesteps, context, context_mask, *extra = inputs

        B, C, H, W = x.shape
        dtype = x.dtype
        ref_latents = []
        image_slots = []
        if len(extra) == 2:
            control_latents, image_slots = extra
            if control_latents.ndim == x.ndim:
                ref_latents = [control_latents]
            elif control_latents.ndim == x.ndim + 1:
                ref_latents = [control_latents[:, i] for i in range(control_latents.shape[1])]
            else:
                raise RuntimeError(f'Unexpected control latent shape {tuple(control_latents.shape)}')
            if image_slots.ndim == 2:
                if image_slots.shape[0] > 1 and not torch.equal(image_slots, image_slots[:1].expand_as(image_slots)):
                    raise RuntimeError(
                        'Reference image slots differ inside one batch. Qwen-Image-2.1 edit needs uncond_fraction = 0, '
                        'and every sample in the run needs the same number of reference images.'
                    )
                image_slots = image_slots[0]
            image_slots = image_slots.tolist()
        elif len(extra) != 0:
            raise RuntimeError(f'Expected target latents only, or control latents plus image slots, got {len(extra)} extra tensors')

        hidden_states, pe, segments = self.build_sequence(x, context, context_mask, ref_latents, image_slots)
        sizes = torch.tensor([hidden_states.shape[1], H, W])

        # pipeline rounds t*1000 and t to the compute dtype; text and reference tokens modulate from t = 0
        t = ((timesteps * 1000).to(dtype) / 1000).to(dtype)
        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)

        mod_scale1_a, mod_scale1_b = _split_rows(scale1)
        mod_gate1_a, mod_gate1_b = _split_rows(gate1.tanh())
        mod_scale2_a, mod_scale2_b = _split_rows(scale2)
        mod_gate2_a, mod_gate2_b = _split_rows(gate2.tanh())
        mod_zeros = torch.zeros_like(scale1[:1, None])

        endpoints, masks = pack_segments(segments)

        return make_contiguous(hidden_states, temb, pe, mod_scale1_a, mod_scale1_b, mod_gate1_a, mod_gate1_b, mod_scale2_a, mod_scale2_b, mod_gate2_a, mod_gate2_b, mod_zeros, sizes, endpoints, *masks)


class TransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, transformer_options, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        transformer_options['block_index'] = block_idx
        self.transformer_options = transformer_options
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, temb, pe, mod_scale1_a, mod_scale1_b, mod_gate1_a, mod_gate1_b, mod_scale2_a, mod_scale2_b, mod_gate2_a, mod_gate2_b, mod_zeros, sizes, endpoints, *masks = inputs
        mod = ((mod_scale1_a, mod_scale1_b), (mod_gate1_a, mod_gate1_b), (mod_scale2_a, mod_scale2_b), (mod_gate2_a, mod_gate2_b), mod_zeros)
        seq_len, H, W = sizes
        prefix_len = seq_len - H*W
        segments = unpack_segments(endpoints, masks)

        attn_fn = block_causal_attention(segments, self.transformer_options, None, self.block_idx, prefix_len)

        self.offloader.wait_for_block(self.block_idx)
        hidden_states = training_qwen21_block(self.layer, hidden_states, mod, pe, attn_fn, prefix_len, self.transformer_options)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(hidden_states, temb, pe, mod_scale1_a, mod_scale1_b, mod_gate1_a, mod_gate1_b, mod_scale2_a, mod_scale2_b, mod_gate2_a, mod_gate2_b, mod_zeros, sizes, endpoints, *masks)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        hidden_states, temb, pe, mod_scale1_a, mod_scale1_b, mod_gate1_a, mod_gate1_b, mod_scale2_a, mod_scale2_b, mod_gate2_a, mod_gate2_b, mod_zeros, sizes, endpoints, *masks = inputs
        seq_len, H, W = sizes
        prefix_len = seq_len - H*W
        B = hidden_states.shape[0]
        hidden_states = self.norm_out(hidden_states[:, prefix_len:], temb[:-1])
        hidden_states = self.proj_out(hidden_states)
        return hidden_states.transpose(1, 2).reshape(B, self.out_channels, H, W)
