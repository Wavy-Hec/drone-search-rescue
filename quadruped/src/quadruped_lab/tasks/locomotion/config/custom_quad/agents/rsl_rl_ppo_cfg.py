# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL (rsl-rl-lib 5.5) PPO runner configurations for the custom quadruped.

Ported from the 2.3.2 ``agents/rsl_rl_ppo_cfg.py``: the deprecated ``policy=RslRlPpoActorCriticCfg(...)`` and
``empirical_normalization`` (unsupported from Isaac Lab 3.1) become separate ``actor`` / ``critic`` model configs with
explicit ``obs_groups``. Hyperparameters are unchanged. The rough runner builds exactly the network stored in the
converted checkpoint ``checkpoints_2x/height_scanner_model_rslrl5.pt``: MLP 235 -> [512, 256, 128] -> 12, ELU,
scalar Gaussian std (init 1.0), no observation normalization.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class CustomQuadRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 80000  # 1500
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
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,  # 1.0
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,  # 0.01
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class CustomQuadFlatPPORunnerCfg(CustomQuadRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 80000
        self.experiment_name = "custom_quad_flat"
        self.actor.hidden_dims = [512, 256, 128]  # [128, 128, 128]
        self.critic.hidden_dims = [512, 256, 128]  # [128, 128, 128]


@configclass
class CustomQuadRoughDepthPPORunnerCfg(CustomQuadRoughPPORunnerCfg):
    """EXPERIMENTAL: CNN actor on proprioception + depth image, MLP critic on proprioception + privileged height scan.

    The 2.x depth task reused the rough MLP runner, which cannot consume a 2-D image; this runner is new in the port.
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
