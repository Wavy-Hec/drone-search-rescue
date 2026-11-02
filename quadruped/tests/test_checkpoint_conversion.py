# Copyright (c) 2026 Hector Lugo
# SPDX-License-Identifier: MIT

"""rsl-rl 3.x -> 5.x checkpoint conversion (``scripts/convert_rslrl_checkpoint.py``), on the CPU."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl", reason="needs rsl-rl-lib 5.x")
pytest.importorskip("tensordict")

from rsl_rl.models import MLPModel
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


@pytest.fixture(scope="module")
def conv(load_script: Callable[[str], ModuleType]) -> ModuleType:
    """The converter script as a module."""
    return load_script("convert_rslrl_checkpoint")


def _synthetic_mlp(net_dims: list[int], num_actions: int) -> dict[str, torch.Tensor]:
    """Actor and critic weights of an rsl-rl 3.x ``ActorCritic`` (Linear + ELU) with the given layer sizes."""
    sd = {}
    for net, out in (("actor", num_actions), ("critic", 1)):
        dims = [*net_dims, out]
        for i in range(len(dims) - 1):
            sd[f"{net}.{2 * i}.weight"] = torch.randn(dims[i + 1], dims[i]) * 0.05
            sd[f"{net}.{2 * i}.bias"] = torch.randn(dims[i + 1]) * 0.05
    return sd


def test_convert_synthetic_checkpoint(conv: ModuleType) -> None:
    """Round trip of a synthetic rsl-rl 3.x checkpoint, including an observation normalizer."""
    torch.manual_seed(0)
    sd = {"std": torch.rand(12) + 0.1, **_synthetic_mlp([235, 512, 256, 128], 12)}
    for net in ("actor", "critic"):
        sd[f"{net}_obs_normalizer._mean"] = torch.randn(1, 235)
        sd[f"{net}_obs_normalizer._var"] = torch.rand(1, 235) + 0.5
        sd[f"{net}_obs_normalizer._std"] = sd[f"{net}_obs_normalizer._var"].sqrt()
        sd[f"{net}_obs_normalizer.count"] = torch.tensor(1000)
    new, info = conv.convert_checkpoint(
        {"model_state_dict": sd, "optimizer_state_dict": None, "iter": 7, "infos": None}
    )
    assert info["actor"]["obs_normalizer"] and info["num_actions"] == 12
    assert set(new["actor_state_dict"]) == {"distribution.std_param"} | {
        f"mlp.{i}.{k}" for i in (0, 2, 4, 6) for k in ("weight", "bias")
    } | {f"obs_normalizer.{b}" for b in ("_mean", "_var", "_std", "count")}

    obs = TensorDict({"policy": torch.randn(64, 235)}, batch_size=[64])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    dist = {"class_name": "GaussianDistribution", "init_std": 1.0}
    actor = MLPModel(obs, groups, "actor", 12, [512, 256, 128], "elu", obs_normalization=True, distribution_cfg=dist)
    critic = MLPModel(obs, groups, "critic", 1, [512, 256, 128], "elu", obs_normalization=True)
    actor.load_state_dict(new["actor_state_dict"], strict=True)
    critic.load_state_dict(new["critic_state_dict"], strict=True)
    actor.eval()
    critic.eval()
    with torch.inference_mode():
        assert torch.allclose(actor(obs), conv.old_forward(sd, "actor", obs["policy"]), atol=1e-6)
        assert torch.allclose(critic(obs), conv.old_forward(sd, "critic", obs["policy"]), atol=1e-6)


def test_convert_log_std_checkpoint(conv: ModuleType, tmp_path: Path) -> None:
    """A log-std checkpoint converts to ``distribution.log_std_param`` and says which agent cfg it needs."""
    torch.manual_seed(1)
    sd = {"log_std": torch.log(torch.rand(12) * 0.5 + 0.1), **_synthetic_mlp([48, 64, 64], 12)}
    old = {"model_state_dict": sd, "optimizer_state_dict": None, "iter": 3, "infos": None}
    new, info = conv.convert_checkpoint(old)
    assert info["std_type"] == "log" and "std_type='log'" in info["agent_cfg_hint"]
    assert "distribution.log_std_param" in new["actor_state_dict"]

    obs = TensorDict({"policy": torch.randn(16, 48)}, batch_size=[16])
    dist = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"}
    actor = MLPModel(obs, {"actor": ["policy"]}, "actor", 12, [64, 64], "elu", distribution_cfg=dist)
    actor.load_state_dict(new["actor_state_dict"], strict=True)
    actor.eval()
    with torch.inference_mode():
        assert torch.allclose(actor(obs), conv.old_forward(sd, "actor", obs["policy"]), atol=1e-6)
        actor.distribution.update(actor(obs))
        assert torch.allclose(actor.distribution.std[0], sd["log_std"].exp(), atol=1e-6)

    # the default (scalar-std) agent cfg is refused up front, with the hint instead of a strict-load key error
    pytest.importorskip("isaaclab_rl", reason="needs Isaac Lab 3.0 (the verification builds the play runner)")
    old_path, new_path = tmp_path / "old.pt", tmp_path / "new.pt"
    torch.save(old, old_path)
    conv.save_checkpoint(new, new_path, force=False)
    with pytest.raises(ValueError, match="std_type='log'"):
        conv.verify(old_path, new_path, conv.DEFAULT_AGENT_CFG, 8, 0)


def test_converted_checkpoint_matches_original(
    conv: ModuleType, project_dir: Path, require_file: Callable[[Path], Path]
) -> None:
    """The committed rsl-rl 5.x checkpoint loads through the play runner and reproduces the 2.x policy exactly."""
    pytest.importorskip("isaaclab_rl", reason="needs Isaac Lab 3.0 (the verification builds the play runner)")
    old_path = require_file(project_dir / "checkpoints_2x" / "height_scanner_model.pt")
    new_path = require_file(project_dir / "checkpoints_2x" / "height_scanner_model_rslrl5.pt")
    report = conv.verify(old_path, new_path, conv.DEFAULT_AGENT_CFG, 1024, 0)
    assert report["obs_dim"] == 235 and report["num_actions"] == 12
    assert report["loaded_iteration"] == report["checkpoint_iteration"] == 13900
    assert max(v for k, v in report.items() if k.startswith("max_abs_diff")) == 0.0
