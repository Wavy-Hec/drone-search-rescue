"""Rescue finale with the Unitree Go2 and Isaac Lab's rough-terrain velocity policy.

The base is ``isaaclab_tasks.core.velocity.config.go2.rough_env_cfg:UnitreeGo2RoughEnvCfg`` (task
``Isaac-Velocity-Rough-UnitreeGo2``): same robot, actuators (``use_newton_actuators``), actions (joint position,
scale 0.25), policy observations (base velocities, projected gravity, velocity command, joint state, last action,
187-ray height scan), decimation 4 x 5 ms and physics presets (default ``newton_mjwarp``, the backend today's
checkpoint was trained on; ``physics=isaacsim_physx`` also resolves). :func:`apply_rescue_task` swaps in the
survey terrain, the path-following command and deterministic events, and puts a person at the goal.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

from drone_lab.tasks.rescue.rescue_env_cfg import DEFAULT_GOAL_XY, apply_rescue_task, person_asset_cfg


@configclass
class RescueGo2EnvCfg(UnitreeGo2RoughEnvCfg):
    """Go2 walks the planned route to the person; see the module docstring."""

    def __post_init__(self):
        super().__post_init__()
        # the default person stands on the flat ground of the default goal (terrain height 0 there);
        # scripts/rescue_finale.py replaces it with the planned goal and its terrain height
        apply_rescue_task(self, num_envs=1, episode_length_s=300.0, person=person_asset_cfg(DEFAULT_GOAL_XY, 0.0))

    def play_mode(self):
        """``isaaclab play``: only the shared defaults (no corruption, <= 50 envs).

        The velocity base's ``play_mode`` would shrink the terrain grid to 5 x 5 tiles; the survey area must stay
        exactly as the drone mapped it, so it is skipped on purpose.
        """
        from isaaclab.envs import ManagerBasedRLEnvCfg

        ManagerBasedRLEnvCfg.play_mode(self)
