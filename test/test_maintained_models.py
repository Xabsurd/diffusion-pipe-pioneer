import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

# ComfyUI also has a top-level utils package. Import this repo's utils before any model module prepends ComfyUI to sys.path.
import utils.common  # noqa: F401

import models.flux3 as flux3
import models.moss_image as moss_image
import models.qwen_image21 as qwen_image21
from comfy.ldm.flux.layers import EmbedND, ModulationOut
from comfy.ldm.flux3.model import Block
import comfy.ops


class _Pe(nn.Module):
    def forward(self, ids):
        return torch.zeros(ids.shape[0], 1, ids.shape[1], 2, 2)


class _QwenStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt_in = nn.Linear(8, 8, bias=False)
        self.img_in = nn.Linear(4, 8, bias=False)
        self.time_text_embed = nn.Identity()
        self.modulation = nn.Identity()
        self.pe_embedder = _Pe()


class _MossStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_embedder = nn.Identity()
        self.text_embedder = nn.Identity()
        self.time_embedding = nn.Identity()
        self.shared_block_modulation = nn.Identity()
        self.video_head_modulation = nn.Identity()
        self.pe_embedder = EmbedND(dim=8, theta=10000, axes_dim=[2, 2, 2, 2])


def _qwen_layer():
    return qwen_image21.InitialLayer(_QwenStub())


class QwenImage21EditTests(unittest.TestCase):
    def test_t2i_masks_match_the_single_text_segment(self):
        layer = _qwen_layer()
        x = torch.randn(2, 4, 2, 3)
        context = torch.randn(2, 5, 8)
        context_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        hidden, _, segments = layer.build_sequence(x, context, context_mask, [], [])
        self.assertEqual(hidden.shape[1], 5 + 2 * 3)
        text_start, text_end, text_mask = segments[0]
        image_start, image_end, image_mask = segments[1]
        self.assertEqual((text_start, text_end, image_start, image_end), (0, 5, 5, 11))
        expected_text = context_mask[:, None, None, :].to(torch.bool).expand(2, 1, 5, 5).tril()
        self.assertTrue(torch.equal(text_mask, expected_text))
        expected_image = torch.ones(2, 1, 1, 11, dtype=torch.bool)
        expected_image[:, 0, 0, :5] = context_mask.to(torch.bool)
        self.assertTrue(torch.equal(image_mask, expected_image))

    def test_two_references_extend_the_sequence_and_keep_padding_masked(self):
        layer = _qwen_layer()
        x = torch.randn(1, 4, 2, 2)
        refs = [torch.randn(1, 4, 2, 2), torch.randn(1, 4, 2, 2)]
        context = torch.randn(1, 5, 8)
        context_mask = torch.tensor([[1, 1, 1, 0, 0]])
        hidden, _, segments = layer.build_sequence(x, context, context_mask, refs, [1, 3])
        # text 1 + ref 4 + text 2 + ref 4 + text 2 + target 4
        self.assertEqual(hidden.shape[1], 17)
        self.assertEqual([end - start for start, end, _ in segments], [1, 4, 2, 4, 2, 4])
        # The last text chunk is the two padding tokens. Nothing may attend to them.
        pad_keys = segments[4][2]
        self.assertFalse(bool(pad_keys[..., -2:].any()))
        target_keys = segments[5][2]
        self.assertFalse(bool(target_keys[..., 11:13].any()))
        self.assertTrue(bool(target_keys[..., :11].all()))
        self.assertTrue(bool(target_keys[..., 13:].all()))

    def test_slot_at_the_ends_drops_the_empty_text_chunk(self):
        layer = _qwen_layer()
        x = torch.randn(1, 4, 2, 2)
        ref = [torch.randn(1, 4, 2, 2)]
        context = torch.randn(1, 5, 8)
        mask = torch.ones(1, 5)
        _, _, at_start = layer.build_sequence(x, context, mask, ref, [0])
        _, _, at_end = layer.build_sequence(x, context, mask, ref, [5])
        _, _, middle = layer.build_sequence(x, context, mask, ref, [2])
        self.assertEqual(len(at_start), 3)
        self.assertEqual(len(at_end), 3)
        self.assertEqual(len(middle), 4)

    def test_mismatched_slots_in_one_batch_fail(self):
        layer = _qwen_layer()
        inputs = (
            torch.randn(2, 4, 2, 2),
            torch.rand(2),
            torch.randn(2, 4, 8),
            torch.ones(2, 4),
            torch.randn(2, 1, 4, 2, 2),
            torch.tensor([[1], [0]]),
        )
        with self.assertRaises(RuntimeError):
            layer.forward(inputs)

    def test_gated_residual_is_out_of_place(self):
        x = torch.randn(1, 6, 4, requires_grad=True)
        y = torch.randn(1, 6, 4, requires_grad=True)
        gate = (
            torch.randn(1, 1, 4, requires_grad=True),
            torch.randn(1, 1, 4, requires_grad=True),
        )
        out = qwen_image21._training_gated_residual(x, y, gate, 2)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)
        self.assertIsNotNone(gate[0].grad)
        self.assertIsNotNone(gate[1].grad)


class Flux3Tests(unittest.TestCase):
    def test_training_block_backward(self):
        block = Block(
            hidden_size=32, num_heads=4, mlp_ratio=1.0, fused_qkv=False,
            dtype=torch.float32, device='cpu', operations=comfy.ops.disable_weight_init,
        )
        for parameter in block.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
        pe_embedder = EmbedND(dim=8, theta=10000, axes_dim=[2, 2, 2, 2])
        ids = torch.zeros(6, 4)
        ids[:, 3] = torch.arange(6)
        pe = pe_embedder(ids.unsqueeze(0)).transpose(1, 2).contiguous()
        x = torch.randn(2, 6, 32, requires_grad=True)

        def modulation():
            return ModulationOut(
                torch.randn(2, 1, 32, requires_grad=True),
                torch.randn(2, 1, 32, requires_grad=True),
                torch.randn(2, 1, 32, requires_grad=True),
            )

        image_mod = modulation()
        text_mod = ModulationOut(
            torch.randn(1, 1, 32, requires_grad=True),
            torch.randn(1, 1, 32, requires_grad=True),
            torch.randn(1, 1, 32, requires_grad=True),
        )
        out = flux3.training_block_forward(block, x, [(0, 4, image_mod), (4, 6, text_mod)], pe, {})
        self.assertEqual(tuple(out.shape), (2, 6, 32))
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(float(block.q_proj.weight.grad.abs().sum()), 0)
        self.assertGreater(float(image_mod.gate.grad.abs().sum()), 0)
        self.assertGreater(float(text_mod.gate.grad.abs().sum()), 0)


class MossTests(unittest.TestCase):
    def test_rope_length_matches_image_plus_text(self):
        layer = moss_image.InitialLayer(_MossStub())
        pe = layer._rope(2, 3, 4, torch.device('cpu'))
        self.assertEqual(pe.shape[1], 2 * 3 + 4)


class DatasetRuleTests(unittest.TestCase):
    def test_qwen21_rejects_mixed_reference_counts_and_uncond(self):
        pipe = object.__new__(qwen_image21.QwenImage21Pipeline)
        pipe.config = {'uncond_fraction': 0}
        pipe.model_specific_dataset_config_validation({'directory': [{}, {}]})
        pipe.model_specific_dataset_config_validation({'directory': [{'control_paths': ['a', 'b']}]})
        with self.assertRaises(RuntimeError):
            pipe.model_specific_dataset_config_validation({'directory': [{'control_path': 'a'}, {}]})
        pipe.config = {'uncond_fraction': 0.1}
        with self.assertRaises(RuntimeError):
            pipe.model_specific_dataset_config_validation({'directory': [{'control_path': 'a'}]})

    def test_flux3_reference_count_must_be_uniform(self):
        pipe = object.__new__(flux3.Flux3Pipeline)
        pipe.model_specific_dataset_config_validation({'directory': [{}, {}]})
        pipe.model_specific_dataset_config_validation({'directory': [{'control_paths': ['a', 'b']}, {'control_paths': ['c', 'd']}]})
        with self.assertRaises(RuntimeError):
            pipe.model_specific_dataset_config_validation({'directory': [{'control_path': 'a'}, {'control_paths': ['b', 'c']}]})
        with self.assertRaises(RuntimeError):
            flux3._reference_count({'control_path': 'a', 'control_paths': ['b']})

    def test_moss_is_text_to_image_only(self):
        pipe = object.__new__(moss_image.MossImagePipeline)
        pipe.model_specific_dataset_config_validation({'directory': [{}, {}]})
        with self.assertRaises(RuntimeError):
            pipe.model_specific_dataset_config_validation({'directory': [{'control_path': 'a'}]})
        with self.assertRaises(RuntimeError):
            pipe.model_specific_dataset_config_validation({'directory': [{'control_paths': ['a']}]})

    def test_control_file_batch_shapes(self):
        files = qwen_image21._control_files_for_batch(None, 3)
        self.assertEqual(files, [[], [], []])
        self.assertEqual(qwen_image21._control_files_for_batch(['a.jpg', 'b.jpg'], 2), [['a.jpg'], ['b.jpg']])
        nested = [['a.jpg', 'b.jpg'], ['c.jpg', 'd.jpg']]
        self.assertEqual(qwen_image21._control_files_for_batch(nested, 2), nested)


class ImportTests(unittest.TestCase):
    def test_comfy_models_import(self):
        from comfy.ldm.flux3.model import Flux3
        from comfy.ldm.moss_image.model import MOSSImage
        ops = comfy.ops.disable_weight_init
        base = Flux3(hidden_size=32, num_heads=4, depth=1, depth_single_blocks=1, dtype=torch.float32, device='cpu', operations=ops)
        distilled = Flux3(hidden_size=32, num_heads=4, depth=1, depth_single_blocks=1, flow_map=True, dtype=torch.float32, device='cpu', operations=ops)
        moss = MOSSImage(dim=32, ffn_dim=64, num_heads=4, num_layers=1, text_dim=16, latent_channels=4, dtype=torch.float32, device='cpu', operations=ops)
        self.assertFalse(base.flow_map)
        self.assertTrue(distilled.flow_map)
        self.assertTrue(hasattr(moss, 'text_embedder'))
        self.assertEqual(len(moss.layers), 1)


if __name__ == '__main__':
    unittest.main()
