# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026 Hector Lugo
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL (rsl-rl-lib 5.5) PPO runner configurations for the custom quadruped.

Ported from the 2.3.2 ``agents/rsl_rl_ppo_cfg.py``: the deprecated ``policy=RslRlPpoActorCriticCfg(...)`` and
``empirical_normalization`` (unsupported from Isaac Lab 3.1) become separate ``actor`` / ``critic`` model configs with
explicit ``obs_groups``. The rough/flat hyperparameters are unchanged. The rough runner builds exactly the network
stored in the converted checkpoint ``checkpoints_2x/height_scanner_model_rslrl5.pt``: MLP 235 -> [512, 256, 128] -> 12,
ELU, scalar Gaussian std (init 1.0), no observation normalization. The experimental depth runner is new (CNN actor).
"""

from isaaclab.utils import configclass, replace

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

ROUGH_PPO_ALGORITHM_CFG = RslRlPpoAlgorithmCfg(
    value_loss_coef=0.5,  # A1 template: 1.0
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.0025,  # A1 template: 0.01
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)
"""PPO settings of the 2.x rough/flat runners (unchanged)."""

DEPTH_PPO_ALGORITHM_CFG = replace(ROUGH_PPO_ALGORITHM_CFG, schedule="fixed", learning_rate=7.0e-5, num_mini_batches=8)
"""Starting point for the CNN actor: the rough settings with the 3.0 camera-actor schedule.

Isaac Lab 3.0 trains its camera actors with a fixed learning rate: under the adaptive KL schedule the encoder features
change faster than the policy head can track and training does not converge
(``core/lift/config/kuka_allegro/agents/rsl_rl_ppo_cfg.py:80-82``, which uses exactly these three values). Eight
mini-batches also halve the CNN activation memory of each PPO update. Untuned: the other 3.0 CNN-actor configs use a
fixed 1.5e-4 or 3e-4, or an adaptive schedule starting at 1e-4.
"""


@configclass
class CustomQuadRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner of the rough task: MLP actor and critic on the 235-value policy group (the 2.x network)."""

    num_steps_per_env = 24
    max_iterations = 80000  # A1 template: 1500
    save_interval = 50
    experiment_name = "custom_quad_rough"
    # 2.x: no obs_groups -> rsl-rl used the "policy" group for both networks
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    # 2.x: policy=RslRlPpoActorCriticCfg(init_noise_std=1.0, actor_hidden_dims=[512, 256, 128], activation="elu")
    #      with empirical_normalization=False
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = ROUGH_PPO_ALGORITHM_CFG


@configclass
class CustomQuadFlatPPORunnerCfg(CustomQuadRoughPPORunnerCfg):
    """PPO runner of the flat task: the rough settings on the 48-value proprioceptive policy group."""

    def __post_init__(self) -> None:
        """Name the experiment and keep the rough network size."""
        super().__post_init__()

        self.max_iterations = 80000
        self.experiment_name = "custom_quad_flat"
        self.actor.hidden_dims = [512, 256, 128]  # A1 template: [128, 128, 128]
        self.critic.hidden_dims = [512, 256, 128]  # A1 template: [128, 128, 128]


@configclass
class CustomQuadRoughDepthPPORunnerCfg(CustomQuadRoughPPORunnerCfg):
    """EXPERIMENTAL: CNN actor on proprioception + depth image, MLP critic on proprioception + privileged height scan.

    The 2.x depth task reused the rough MLP runner, which cannot consume a 2-D image; this runner is new in the port.
    PPO uses :data:`DEPTH_PPO_ALGORITHM_CFG` (fixed learning rate 7e-5, 8 mini-batches).
    """

    experiment_name = "custom_quad_rough_depth"
    obs_groups = {"actor": ["policy", "depth"], "critic": ["policy", "privileged"]}
    actor = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=1.0),
        # (1, 48, 64) -> (16, 22, 30) -> (32, 10, 14) -> (32, 4, 6) -> 768 features, concatenated with the 48 proprio
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[5, 3, 3],
            stride=[2, 2, 2],
            activation="elu",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = DEPTH_PPO_ALGORITHM_CFG
    # The per-step NaN scan reads every observation group (including the image) and syncs with the host. The 3.0 camera
    # runners turn it off; keep it on until the first depth smoke run is NaN-free, then pass agent.check_for_nan=false.
    check_for_nan = True
