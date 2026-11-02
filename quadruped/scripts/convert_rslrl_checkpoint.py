# Copyright (c) 2026 Hector Lugo
# SPDX-License-Identifier: MIT

"""Convert an rsl-rl <= 3.x ``ActorCritic`` checkpoint (Isaac Lab 2.x) to the rsl-rl 5.x format (Isaac Lab 3.0).

rsl-rl 3.x saves one ``model_state_dict`` for an ``ActorCritic`` module::

    {"model_state_dict": {"std", "actor.N.*", "critic.N.*", ["actor_obs_normalizer.*", "critic_obs_normalizer.*"]},
     "optimizer_state_dict", "iter", "infos"}

rsl-rl 5.x (``OnPolicyRunner.load`` -> ``PPO.load``) expects separate ``MLPModel`` state dicts::

    {"actor_state_dict": {"distribution.std_param", "mlp.N.*", ["obs_normalizer.*"]},
     "critic_state_dict": {"mlp.N.*", ["obs_normalizer.*"]},
     "optimizer_state_dict", "iter", "infos"}

The Adam state is copied unchanged: rsl-rl 3.x optimized ``ActorCritic.parameters()`` (std, actor layers, critic
layers) and rsl-rl 5.x optimizes ``chain(actor.parameters(), critic.parameters())`` (std_param, actor mlp, critic mlp),
i.e. the same tensors in the same order. The input file is only read, never modified.

Examples (from the project root, conda env ``env_isaaclab`` active)::

    python scripts/convert_rslrl_checkpoint.py checkpoints_2x/height_scanner_model.pt \\
        checkpoints_2x/height_scanner_model_rslrl5.pt --verify

    # verify an existing converted file only
    python scripts/convert_rslrl_checkpoint.py checkpoints_2x/height_scanner_model.pt \\
        checkpoints_2x/height_scanner_model_rslrl5.pt --verify-only

``--verify`` runs on CPU without Isaac Sim: it builds the rsl-rl 5.x ``OnPolicyRunner`` from the project's agent
config (exactly what ``isaaclab play`` builds), loads the converted file through ``runner.load`` (strict state-dict
loading plus optimizer and iteration), and compares the new actor/critic with a hand-written forward pass of the old
weights on random observations. The project must be importable (installed, or ``PYTHONPATH=<project>/src``).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensordict import TensorDict

    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

DEFAULT_AGENT_CFG = (
    "quadruped_lab.tasks.locomotion.config.custom_quad.agents.rsl_rl_ppo_cfg:CustomQuadRoughPPORunnerCfg"
)
"""Agent config whose actor/critic must match the checkpoint (the rough height-scan runner)."""

_MLP_KEY = re.compile(r"^(actor|critic)\.(\d+)\.(weight|bias)$")
_NORMALIZER_KEY = re.compile(r"^(actor|critic)_obs_normalizer\.(_mean|_var|_std|count)$")

STD_TYPE_HINTS = {
    "scalar": None,
    "log": (
        "the checkpoint stores log_std, converted to 'distribution.log_std_param': the agent cfg must use "
        "actor=RslRlMLPModelCfg(..., distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, "
        "std_type='log')); with the default std_type='scalar' rsl-rl expects 'distribution.std_param' and strict "
        "loading fails with an unexpected-key error (rsl_rl/modules/distribution.py:165-168)"
    ),
}
"""What the agent config must look like for each std parameterization of the old checkpoint."""


class CheckpointFormatError(RuntimeError):
    """Raised when the input does not look like an rsl-rl <= 3.x ActorCritic checkpoint."""


def load_checkpoint(path: str | Path) -> dict:
    """Load a checkpoint on the CPU with the safe (weights-only) unpickler."""
    return torch.load(path, map_location="cpu", weights_only=True)


def _mlp_layers(state_dict: dict[str, torch.Tensor], prefix: str) -> list[int]:
    """Return the sorted ``nn.Sequential`` indices of the Linear layers stored under ``prefix``."""
    return sorted({int(m.group(2)) for k in state_dict if (m := _MLP_KEY.match(k)) and m.group(1) == prefix})


def describe_old_model(model_state_dict: dict[str, torch.Tensor]) -> dict:
    """Infer the architecture of an rsl-rl 3.x ActorCritic from its weights."""
    info: dict = {}
    for net in ("actor", "critic"):
        layers = _mlp_layers(model_state_dict, net)
        if not layers:
            raise CheckpointFormatError(f"no '{net}.N.weight' tensors found")
        weights = [model_state_dict[f"{net}.{i}.weight"] for i in layers]
        info[net] = {
            "layer_indices": layers,
            "input_dim": int(weights[0].shape[1]),
            "hidden_dims": [int(w.shape[0]) for w in weights[:-1]],
            "output_dim": int(weights[-1].shape[0]),
            "obs_normalizer": f"{net}_obs_normalizer._mean" in model_state_dict,
        }
    if "std" in model_state_dict:
        info["std_type"] = "scalar"
        info["std"] = model_state_dict["std"]
    elif "log_std" in model_state_dict:
        info["std_type"] = "log"
        info["std"] = model_state_dict["log_std"].exp()
    else:
        raise CheckpointFormatError(
            "no 'std' / 'log_std' parameter: state-dependent std (heteroscedastic) checkpoints are not supported"
        )
    info["num_actions"] = int(info["std"].numel())
    if info["actor"]["output_dim"] != info["num_actions"]:
        raise CheckpointFormatError("actor output size does not match the std size")
    info["agent_cfg_hint"] = STD_TYPE_HINTS[info["std_type"]]
    return info


def convert_checkpoint(old: dict) -> tuple[dict, dict]:
    """Convert a loaded rsl-rl 3.x checkpoint dictionary to the rsl-rl 5.x layout.

    Returns:
        The converted checkpoint dictionary and a description of the model.
    """
    if "actor_state_dict" in old:
        raise CheckpointFormatError("the input already uses the rsl-rl >= 4 layout ('actor_state_dict')")
    if "model_state_dict" not in old:
        raise CheckpointFormatError(f"expected key 'model_state_dict', found {sorted(old)}")
    sd: dict[str, torch.Tensor] = old["model_state_dict"]
    info = describe_old_model(sd)

    actor: dict[str, torch.Tensor] = {}
    critic: dict[str, torch.Tensor] = {}
    # rsl-rl 5.x MLPModel registers obs_normalizer -> distribution -> mlp; keep that key order
    for key, value in sd.items():
        if m := _NORMALIZER_KEY.match(key):
            (actor if m.group(1) == "actor" else critic)[f"obs_normalizer.{m.group(2)}"] = value.clone()
    if info["std_type"] == "scalar":
        actor["distribution.std_param"] = sd["std"].clone()
    else:
        actor["distribution.log_std_param"] = sd["log_std"].clone()
    used = {"std", "log_std"} | {k for k in sd if _NORMALIZER_KEY.match(k)}
    for net, target in (("actor", actor), ("critic", critic)):
        for idx in info[net]["layer_indices"]:
            for kind in ("weight", "bias"):
                key = f"{net}.{idx}.{kind}"
                target[f"mlp.{idx}.{kind}"] = sd[key].clone()
                used.add(key)
    unknown = sorted(set(sd) - used)
    if unknown:
        raise CheckpointFormatError(f"unexpected tensors in model_state_dict: {unknown}")

    # the optimizer state is index-based; check that the parameter order/shape matches the new models
    new_param_order = [v for k, v in actor.items() if not k.startswith("obs_normalizer.")]
    new_param_order += [v for k, v in critic.items() if not k.startswith("obs_normalizer.")]
    optim = old.get("optimizer_state_dict")
    if optim is not None:
        groups = optim["param_groups"]
        indices = [i for g in groups for i in g["params"]]
        if len(indices) != len(new_param_order):
            raise CheckpointFormatError(
                f"optimizer holds {len(indices)} parameters but the converted models have {len(new_param_order)}"
            )
        for position, (param_id, tensor) in enumerate(zip(indices, new_param_order, strict=True)):
            state = optim["state"].get(param_id, {})
            if "exp_avg" in state and tuple(state["exp_avg"].shape) != tuple(tensor.shape):
                raise CheckpointFormatError(
                    f"optimizer state #{position} has shape {tuple(state['exp_avg'].shape)}, "
                    f"expected {tuple(tensor.shape)}"
                )

    new = {
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "optimizer_state_dict": optim,
        "iter": old.get("iter", 0),
        "infos": old.get("infos"),
    }
    return new, info


def save_checkpoint(checkpoint: dict, output: Path, force: bool) -> None:
    """Write the checkpoint atomically next to ``output``."""
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; pass --force to overwrite it")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(checkpoint, tmp_path)
        # NamedTemporaryFile creates 0600 files; give the result the usual umask-based permissions
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_path, 0o666 & ~umask)
        os.replace(tmp_path, output)
    finally:
        tmp_path.unlink(missing_ok=True)


def old_forward(model_state_dict: dict[str, torch.Tensor], net: str, obs: torch.Tensor) -> torch.Tensor:
    """Hand-written forward pass of an rsl-rl 3.x MLP (Linear + ELU, linear output), independent of rsl-rl."""
    x = obs
    if f"{net}_obs_normalizer._mean" in model_state_dict:
        mean = model_state_dict[f"{net}_obs_normalizer._mean"]
        std = model_state_dict[f"{net}_obs_normalizer._std"]
        x = (x - mean) / (std + 1e-2)  # EmpiricalNormalization(eps=1e-2).forward
    layers = _mlp_layers(model_state_dict, net)
    for n, idx in enumerate(layers):
        x = torch.nn.functional.linear(
            x, model_state_dict[f"{net}.{idx}.weight"], model_state_dict[f"{net}.{idx}.bias"]
        )
        if n < len(layers) - 1:
            x = torch.nn.functional.elu(x)
    return x


class _FakeVecEnv:
    """Minimal stand-in for ``RslRlVecEnvWrapper``: enough for ``OnPolicyRunner`` construction on the CPU."""

    def __init__(self, obs: dict[str, torch.Tensor], num_actions: int) -> None:
        from tensordict import TensorDict

        self._obs = TensorDict(obs, batch_size=[next(iter(obs.values())).shape[0]])
        self.num_envs = self._obs.batch_size[0]
        self.num_actions = num_actions
        self.max_episode_length = 1000
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        """The fixed observations, as the wrapper returns them."""
        return self._obs


def _load_agent_cfg(spec: str) -> RslRlOnPolicyRunnerCfg:
    """Instantiate an agent config given as ``module:Class``."""
    module_name, class_name = spec.split(":")
    return getattr(importlib.import_module(module_name), class_name)()


def verify(old_path: Path, new_path: Path, agent_cfg_spec: str, num_samples: int, seed: int) -> dict:
    """Load the converted checkpoint through the rsl-rl 5.x runner and compare against the old weights."""
    from isaaclab_rl.rsl_rl import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    old = load_checkpoint(old_path)
    old_sd = old["model_state_dict"]
    info = describe_old_model(old_sd)
    obs_dim, num_actions = info["actor"]["input_dim"], info["num_actions"]

    # build the runner exactly like `isaaclab play` (play_rsl_rl.py): deprecated-cfg handling, then the runner
    agent_cfg = _load_agent_cfg(agent_cfg_spec)
    agent_cfg.device = "cpu"
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    groups = sorted({g for gs in agent_cfg.obs_groups.values() for g in gs})
    if groups != ["policy"]:
        raise ValueError(f"verification expects an MLP agent on the 'policy' group only, got obs_groups={groups}")
    cfg_std_type = getattr(agent_cfg.actor.distribution_cfg, "std_type", "scalar")
    if cfg_std_type != info["std_type"]:
        raise ValueError(
            f"agent cfg {agent_cfg_spec} uses std_type={cfg_std_type!r} but the checkpoint has std_type="
            f"{info['std_type']!r}: {info['agent_cfg_hint'] or 'use the default scalar GaussianDistributionCfg'}"
        )

    generator = torch.Generator().manual_seed(seed)
    obs = torch.randn(num_samples, obs_dim, generator=generator)
    obs[: num_samples // 2] *= 3.0  # exercise the ELU saturation region as well
    env = _FakeVecEnv({"policy": obs}, num_actions)
    runner = create_rsl_rl_runner(env, agent_cfg, log_dir=None)
    runner.load(str(new_path))  # strict=True; loads actor, critic, optimizer and iteration
    policy = runner.get_inference_policy(device="cpu")

    with torch.inference_mode():
        new_actions = policy(env.get_observations())
        new_values = runner.alg.critic(env.get_observations())
        old_actions = old_forward(old_sd, "actor", obs)
        old_values = old_forward(old_sd, "critic", obs)
        actor = runner.alg.get_policy()
        actor.distribution.update(new_actions)
        new_std = actor.distribution.std[0]

    report = {
        "old_checkpoint": str(old_path),
        "new_checkpoint": str(new_path),
        "agent_cfg": agent_cfg_spec,
        "actor_model": type(runner.alg.actor).__name__,
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "actor_hidden_dims": info["actor"]["hidden_dims"],
        "critic_hidden_dims": info["critic"]["hidden_dims"],
        "obs_normalizer_in_old_checkpoint": info["actor"]["obs_normalizer"] or info["critic"]["obs_normalizer"],
        "loaded_iteration": runner.current_learning_iteration,
        "checkpoint_iteration": old["iter"],
        "optimizer_lr": runner.alg.optimizer.param_groups[0]["lr"],
        "num_samples": num_samples,
        "max_abs_diff_actions": float((new_actions - old_actions).abs().max()),
        "max_abs_diff_values": float((new_values - old_values).abs().max()),
        "max_abs_diff_std": float((new_std - info["std"]).abs().max()),
        "max_abs_action": float(old_actions.abs().max()),
    }

    # the export path used by `isaaclab play` (exported/policy.pt) must reproduce the same actions
    with tempfile.TemporaryDirectory() as tmp:
        runner.export_policy_to_jit(path=tmp, filename="policy.pt")
        jit_policy = torch.jit.load(os.path.join(tmp, "policy.pt"), map_location="cpu")
        with torch.inference_mode():
            report["max_abs_diff_jit_export"] = float((jit_policy(obs) - old_actions).abs().max())
    return report


def main(argv: list[str] | None = None) -> int:
    """Convert (and/or verify) a checkpoint; exit code 1 if the verification exceeds the tolerance."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="rsl-rl <= 3.x checkpoint (read only).")
    parser.add_argument("output", type=Path, help="Destination of the rsl-rl 5.x checkpoint.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    parser.add_argument("--verify", action="store_true", help="Verify the converted file on the CPU after writing.")
    parser.add_argument("--verify-only", action="store_true", help="Skip conversion and only verify OUTPUT.")
    parser.add_argument("--agent-cfg", default=DEFAULT_AGENT_CFG, help="module:Class of the matching agent cfg.")
    parser.add_argument("--num-samples", type=int, default=4096, help="Random observations used for verification.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the random verification observations.")
    parser.add_argument("--atol", type=float, default=1e-5, help="Maximum accepted absolute difference.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON file for the verification report.")
    args = parser.parse_args(argv)

    if args.input.resolve() == args.output.resolve():
        parser.error("OUTPUT must differ from INPUT (checkpoints are never converted in place)")

    if not args.verify_only:
        old = load_checkpoint(args.input)
        try:
            new, info = convert_checkpoint(old)
            save_checkpoint(new, args.output, force=args.force)
        except (CheckpointFormatError, FileExistsError) as err:
            parser.error(str(err))
        print(f"[convert] {args.input} -> {args.output}")
        print(
            f"[convert] actor {info['actor']['input_dim']} -> {info['actor']['hidden_dims']} -> {info['num_actions']}, "
            f"critic {info['critic']['input_dim']} -> {info['critic']['hidden_dims']} -> 1, "
            f"std_type={info['std_type']}, obs normalizer={info['actor']['obs_normalizer']}, iter={new['iter']}"
        )
        print(f"[convert] actor keys:  {list(new['actor_state_dict'])}")
        print(f"[convert] critic keys: {list(new['critic_state_dict'])}")
        if info["agent_cfg_hint"]:
            print(f"[convert] NOTE: {info['agent_cfg_hint']}")

    if args.verify or args.verify_only:
        report = verify(args.input, args.output, args.agent_cfg, args.num_samples, args.seed)
        print(json.dumps(report, indent=2))
        if args.report is not None:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        diffs = [report[k] for k in report if k.startswith("max_abs_diff")]
        if max(diffs) > args.atol or report["loaded_iteration"] != report["checkpoint_iteration"]:
            print(f"[verify] FAILED (tolerance {args.atol})", file=sys.stderr)
            return 1
        print(f"[verify] OK: max abs diff {max(diffs):.3e} <= {args.atol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
