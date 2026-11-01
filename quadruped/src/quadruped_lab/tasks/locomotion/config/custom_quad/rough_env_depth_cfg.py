# Copyright (c) 2026, Hector.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EXPERIMENTAL: rough-terrain velocity tracking with a body-mounted depth camera (Isaac Lab 3.0 redesign).

This is not a 1:1 port. The 2.x depth task relied on a local patch of the 2.3.2 velocity base config (a 320x240
``TiledCameraCfg`` plus ``mdp.image`` concatenated into the 1-D policy group). That combination could not be built
(a (240, 320, 1) image cannot be concatenated with 1-D terms), and ``depth_camera_model.pt`` (240 inputs = 48 + 192)
matches no recorded camera config, so it is not reusable and this task must be retrained.

Design (3.0 APIs):

* Camera: :class:`~isaaclab.sensors.CameraCfg` (``TiledCameraCfg`` is now a deprecated alias) on the trunk with the
  2.x mount pose. The offset quaternion is converted from 2.x (w, x, y, z) = (0.66446, 0.24184, -0.24184, -0.66446)
  to the 3.0 (x, y, z, w) order. The camera sits ~1.42 m above flat ground, 0.25 m behind the trunk origin, pitched
  50 deg down, 60 x 47 deg field of view.
* Far clip raised from 1.0 m to :data:`DEPTH_MAX_RANGE` (4 m): from 1.42 m height the closest ground ray (bottom image
  edge, 73.4 deg down) is ~1.48 m away and the top edge ~3.2 m, so the 2.x far clip saw no ground at all.
* Image: ``mdp.image_depth`` (``mdp.image`` was removed) in its own ``depth`` observation group, channel-first
  ``(N, 1, 48, 64)``, normalized to [0, 1] by the far clip; pixels without a hit read as far (1.0).
* Policy: rsl-rl ``CNNModel`` actor on ``policy`` (48 proprio) + ``depth``; MLP critic on ``policy`` + a noise-free
  ``privileged`` height scan (asymmetric actor-critic). The actor never sees the height scan.
* Renderer: Newton Warp by default (kit-less Warp ray tracer, documented pairing with Isaac Sim PhysX, much lighter
  than RTX for depth-only images). ``renderer=isaacsim_rtx`` switches to the legacy Isaac RTX renderer inside Kit.
  OVRTX is not offered: it cannot share a process with Kit (Isaac Sim PhysX) and its runtime is not installed.
* Size: 64x48 pixels and 1024 envs by default (see PORT_NOTES.md for the 16 GB memory budget).
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import MySceneCfg, ObservationsCfg
from isaaclab_tasks.utils import PresetCfg

from .rough_env_cfg import CustomQuadRoughEnvCfg

DEPTH_CAMERA_WIDTH = 64
"""Depth image width [px] (2.x patch: 320)."""

DEPTH_CAMERA_HEIGHT = 48
"""Depth image height [px] (2.x patch: 240). Same 4:3 aspect, so the field of view is unchanged."""

DEPTH_MAX_RANGE = 4.0
"""Far clipping plane and normalization range of the depth image [m] (2.x patch: 1.0)."""

DEPTH_NUM_ENVS = 1024
"""Default number of environments for the depth task (the 2.x commands also trained with 1024)."""


@configclass
class DepthCameraRendererCfg(PresetCfg):
    """Renderer presets for the depth camera, selected with ``renderer=<name>`` on the command line.

    Both renderers report rays without a hit as the far clip (``depth_clipping_behavior="max"``), so the normalized
    image reads 1.0 for "nothing within range" instead of Newton's 0.0 / RTX's ``inf`` sentinels.
    """

    newton_renderer: NewtonWarpRendererCfg = NewtonWarpRendererCfg(depth_clipping_behavior="max")
    isaacsim_rtx: IsaacRtxRendererCfg = IsaacRtxRendererCfg(depth_clipping_behavior="max")
    default: NewtonWarpRendererCfg = NewtonWarpRendererCfg(depth_clipping_behavior="max")


@configclass
class CustomQuadDepthSceneCfg(MySceneCfg):
    """Velocity-task scene plus the trunk-mounted depth camera."""

    depth_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk/depth_camera",
        # the camera rides on the trunk, so non-Kit renderers need its pose refreshed on every capture
        update_latest_camera_pose=True,
        offset=CameraCfg.OffsetCfg(
            pos=(-0.25, 0.0, 0.7),
            # 2.x (w, x, y, z) = (0.66446, 0.24184, -0.24184, -0.66446) -> 3.0 (x, y, z, w)
            rot=(0.24184, -0.24184, -0.66446, 0.66446),
            convention="opengl",
        ),
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.14756,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, DEPTH_MAX_RANGE),
        ),
        width=DEPTH_CAMERA_WIDTH,
        height=DEPTH_CAMERA_HEIGHT,
        renderer_cfg=DepthCameraRendererCfg(),
    )


@configclass
class CustomQuadDepthObservationsCfg(ObservationsCfg):
    """Velocity observations with a separate depth-image group and a privileged critic group."""

    @configclass
    class DepthCfg(ObsGroup):
        """Depth image for the CNN actor, shape ``(N, 1, H, W)``."""

        depth_image = ObsTerm(
            func=mdp.image_depth,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_camera",
                "normalize": True,
                "max_depth": DEPTH_MAX_RANGE,
                "invalid_value": DEPTH_MAX_RANGE,
                "channel_first": True,
            },
            # 2.x: +-0.02 on metric depth (= +-2 cm); the image is now normalized by 4 m, so +-0.005 keeps 2 cm
            noise=Unoise(n_min=-0.005, n_max=0.005),
            clip=(0.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Noise-free height scan for the critic only (asymmetric actor-critic)."""

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    depth: DepthCfg = DepthCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class CustomQuadRoughDepthEnvCfg(CustomQuadRoughEnvCfg):
    """EXPERIMENTAL depth-camera variant of :class:`CustomQuadRoughEnvCfg` (retrain from scratch)."""

    scene: CustomQuadDepthSceneCfg = CustomQuadDepthSceneCfg(num_envs=DEPTH_NUM_ENVS, env_spacing=2.5)
    observations: CustomQuadDepthObservationsCfg = CustomQuadDepthObservationsCfg()

    def __post_init__(self):
        # post init of parent (robot, terrain scaling, events, rewards, physics default)
        super().__post_init__()

        # the actor sees proprioception + depth only; the height scan feeds the privileged critic group
        self.observations.policy.height_scan = None
        # render once per policy step (the 2.x patch used update_period=0.0, i.e. every 5 ms physics step)
        self.scene.depth_camera.update_period = self.decimation * self.sim.dt
