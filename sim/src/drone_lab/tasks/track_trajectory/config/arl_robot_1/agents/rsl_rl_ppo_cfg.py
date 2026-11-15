"""RSL-RL PPO configuration of the trajectory-tracking task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class TrackTrajectoryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO with the network size of the built-in drone task (Isaac Lab 3.0.0,
    ``TrackPositionNoObstaclesEnvPPORunnerCfg``) and most of its hyper-parameters. Four values differ:

    * observation normalization is on (the 26-D observation mixes metres, metres per second and quaternions);
    * ``entropy_coef`` is 0.002 instead of 0.001;
    * ``max_iterations`` is 2000 instead of 1500;
    * ``save_interval`` is 100 instead of 50.
    """

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "drone_lab_track_trajectory"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=4.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
