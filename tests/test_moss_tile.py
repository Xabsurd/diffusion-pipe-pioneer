"""Tile adapter save/load, learning-rate scale, and linear schedule."""

import torch
from torch import nn

from models.tile_embedder import TileVideoEmbedder, pack_tile_tokens, split_merged_embedder
from utils.common import apply_param_group_lr_scales, linear_scheduler_steps


def test_pack_requires_matching_spatial_size():
    noisy = torch.zeros(2, 12, 128)
    control = torch.zeros(2, 128, 3, 4)
    packed = pack_tile_tokens(noisy, control)
    assert packed.shape == (2, 12, 384)
    assert torch.count_nonzero(packed[..., 256:]) == 0
    try:
        pack_tile_tokens(torch.zeros(2, 8, 128), control)
    except RuntimeError:
        return
    raise AssertionError("mismatched target and condition token lengths were accepted")


def test_merged_weight_roundtrip():
    base = nn.Linear(128, 16)
    embedder = TileVideoEmbedder(base)
    with torch.no_grad():
        embedder.control.weight.normal_()
    merged = embedder.merged_weight()
    base_weight, control_weight, bias = split_merged_embedder(merged, embedder.merged_bias())
    assert torch.equal(base_weight, base.weight)
    assert torch.equal(control_weight, embedder.control.weight)
    assert torch.equal(bias, base.bias)
    assert not base.weight.requires_grad
    assert embedder.control.weight.requires_grad


def test_control_group_uses_lr_scale_not_optimizer_lr():
    base = nn.Linear(128, 4)
    embedder = TileVideoEmbedder(base)
    embedder.control.weight.original_name = "video_embedder.control.weight"
    other = nn.Parameter(torch.zeros(2))
    other.original_name = "layers.0.lora"

    class Model:
        tile_embedder = embedder
        model_config = {"control_lr_multiplier": 10}

        def get_param_groups(self, parameters):
            from models.moss_tile import MossTileMixin
            return MossTileMixin.get_param_groups(self, parameters)

    groups = Model().get_param_groups([other, embedder.control.weight])
    assert "lr" not in groups[0]
    assert groups[1]["lr_scale"] == 10.0
    resolved = apply_param_group_lr_scales(groups, 2e-4)
    assert resolved[1]["lr"] == 2e-3
    assert "lr_scale" not in resolved[1]
    assert "lr" not in resolved[0]


def test_linear_schedule_respects_max_steps_and_end_lr():
    config = {
        "epochs": 10,
        "max_steps": 30,
        "warmup_steps": 5,
        "optimizer": {"lr": 2e-4},
        "lr_scheduler_end_lr": 2e-5,
    }
    end_factor, decay_steps = linear_scheduler_steps(config, steps_per_epoch=100)
    assert decay_steps == 25
    assert abs(end_factor - 0.1) < 1e-8

    untouched = {"epochs": 10, "warmup_steps": 5, "optimizer": {"lr": 2e-4}}
    end_factor, decay_steps = linear_scheduler_steps(untouched, steps_per_epoch=100)
    assert end_factor == 0.0
    assert decay_steps == 1000


def test_init_from_existing_restores_control_columns(tmp_path=None):
    from pathlib import Path
    import tempfile
    from safetensors.torch import load_file, save_file
    from models.moss_tile import CONTROL_WEIGHT_KEY, SET_WEIGHT_KEY, MossTileMixin

    base = nn.Linear(128, 8)
    embedder = TileVideoEmbedder(base)
    with torch.no_grad():
        embedder.control.weight.normal_()
    control = embedder.control.weight.detach().clone()
    merged = embedder.merged_weight()
    with torch.no_grad():
        embedder.control.weight.zero_()

    class Diffusion:
        def named_parameters(self):
            return [("layers.0.proj.q.lora_A.default.weight", nn.Parameter(torch.zeros(1)))]

        def load_state_dict(self, state, strict=False):
            self.loaded = state

    class Holder(MossTileMixin):
        pass

    holder = Holder()
    holder.tile_embedder = embedder
    holder.diffusion_model = Diffusion()

    directory = Path(tempfile.mkdtemp())
    lora_key = "diffusion_model.layers.0.proj.q.lora_A.weight"
    save_file(
        {
            lora_key: torch.ones(1),
            SET_WEIGHT_KEY: merged,
            CONTROL_WEIGHT_KEY: control,
        },
        directory / "adapter_model.safetensors",
    )
    holder.load_adapter_weights(directory)
    assert torch.allclose(embedder.control.weight, control)
    assert "layers.0.proj.q.lora_A.default.weight" in holder.diffusion_model.loaded
    assert not any("set_weight" in key for key in holder.diffusion_model.loaded)


def test_gradient_release_groups_without_explicit_lr():
    groups = apply_param_group_lr_scales([{"params": [nn.Parameter(torch.zeros(1))]}], 1e-4)
    assert "lr" not in groups[0]


if __name__ == "__main__":
    test_pack_requires_matching_spatial_size()
    test_merged_weight_roundtrip()
    test_control_group_uses_lr_scale_not_optimizer_lr()
    test_linear_schedule_respects_max_steps_and_end_lr()
    test_init_from_existing_restores_control_columns()
    test_gradient_release_groups_without_explicit_lr()
    print("ok")
