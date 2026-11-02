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

* Camera: :class:`~isaaclab.sensors.CameraCfg` (``TiledCameraCfg`` is now a deprecated alias) on the front face of
  the trunk (:data:`DEPTH_CAMERA_POS`) with the 2.x orientation: looking forward, pitched 50 deg down, 60 x 46.8 deg
  field of view. The 2.x mount (0.7 m above and 0.25 m behind the trunk origin) is not kept: from there the robot's
  own visual meshes filled a third of the image, and the nearest visible ground was 1.26 m ahead of the trunk origin
  on the centre line and 0.89 m in the foot tracks. At the default standing pose (trunk origin 0.452 m above flat
  ground) the front-face camera is 0.53 m above the ground, no pixel sees the robot, and flat ground is visible from
  0.63 m to ~1.5 m ahead of the trunk origin (``scripts/depth_camera_study.py``; PORT_NOTES.md, depth-camera task).
* The offset quaternion is the 2.x (w, x, y, z) = (0.66446, 0.24184, -0.24184, -0.66446) in the 3.0 (x, y, z, w) order.
* Far clip raised from 1.0 m to :data:`DEPTH_MAX_RANGE` (4 m). Flat ground is 0.56-1.32 m away along the pixel rays of
  this mount, so the 2.x 1.0 m clip would blank the top sixth of the image; 4 m leaves headroom for downhill terrain
  and a pitched-up trunk. (With the 2.x mount every ground pixel was beyond 1.0 m.)
* Image: ``mdp.image_depth`` (``mdp.image`` was removed) in its own ``depth`` observation group, channel-first
  ``(N, 1, 48, 64)``, normalized to [0, 1] by the far clip; pixels without a hit read as far (1.0).
* Update rate: ``update_period=0.0``. Sensors update lazily (``InteractiveSceneCfg.lazy_sensor_update=True``), so the
  camera renders when the observation reads it, i.e. once per policy step, whatever the period. A positive period
  would only add a device-to-host sync per read and, with float32 sensor timestamps, skip every other pose refresh
  late in long episodes.
* Camera pose: the observation term :class:`body_mounted_image_depth` writes the trunk pose composed with the mount
  offset into the camera right before each read. Without it, the Newton Warp renderer on Isaac Sim PhysX renders
  from the camera's spawn pose for the whole run (GPU check 2026-11-01): ``update_latest_camera_pose`` re-reads the
  camera prim's Fabric world matrix, and only the Kit RTX path keeps that matrix in step with the trunk.
* Policy: rsl-rl ``CNNModel`` actor on ``policy`` (48 proprio) + ``depth``; MLP critic on ``policy`` + a noise-free
  ``privileged`` height scan (asymmetric actor-critic). The actor never sees the height scan.
* Renderer: Newton Warp by default (kit-less Warp ray tracer, documented pairing with Isaac Sim PhysX, much lighter
  than RTX for depth-only images). ``renderer=isaacsim_rtx`` switches to the legacy Isaac RTX renderer inside Kit.
  OVRTX is not offered: it cannot share a process with Kit, which Isaac Sim PhysX needs.
* Size: 64x48 pixels and 1024 envs by default. Measured on the RTX 5080 (2026-11-01, 1024 envs, training): 5.5 GB
  of GPU memory with Newton Warp, 7.3 GB with RTX; the camera render takes 4-5 ms (Warp) vs 17.5 ms (RTX) of a
  70-110 ms env step, most of which is PhysX.
"""

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

DEPTH_CAMERA_WIDTH = 64
"""Depth image width [px] (2.x patch: 320)."""

DEPTH_CAMERA_HEIGHT = 48
"""Depth image height [px] (2.x patch: 240). Same 4:3 aspect, so the field of view is unchanged."""

DEPTH_MAX_RANGE = 4.0
"""Far clipping plane and normalization range of the depth image [m] (2.x patch: 1.0)."""

DEPTH_NUM_ENVS = 1024
"""Default number of environments for the depth task (the 2.x commands also trained with 1024)."""

DEPTH_CAMERA_POS = (0.47, 0.0, 0.08)
"""Camera position in the trunk frame [m]: centre of the trunk's front face, 1.1 cm ahead of its visual mesh, which
ends at x = 0.459 m (2.x patch: (-0.25, 0.0, 0.7))."""

DEPTH_CAMERA_ROT = (0.24184, -0.24184, -0.66446, 0.66446)
"""Camera orientation (x, y, z, w) w.r.t. the trunk, OpenGL convention: forward, pitched 50 deg down. Same as the 2.x
patch, whose (w, x, y, z) value was (0.66446, 0.24184, -0.24184, -0.66446)."""


@configclass
class DepthCameraRendererCfg(PresetCfg):
    """Renderer presets for the depth camera, selected with ``renderer=<name>`` on the command line.

    Both renderers report rays without a hit as the far clip (``depth_clipping_behavior="max"``), so the normalized
    image reads 1.0 for "nothing within range" instead of Newton's 0.0 / RTX's ``inf`` sentinels.
    """

    newton_renderer: NewtonWarpRendererCfg = NewtonWarpRendererCfg(depth_clipping_behavior="max")
    isaacsim_rtx: IsaacRtxRendererCfg = IsaacRtxRendererCfg(depth_clipping_behavior="max")
    default: NewtonWarpRendererCfg = NewtonWarpRendererCfg(depth_clipping_behavior="max")


_DEFAULT_CAMERA_CFG = SceneEntityCfg("tiled_camera")
"""Default ``sensor_cfg`` of :class:`body_mounted_image_depth`, as in ``mdp.image_depth`` (the task sets its own)."""

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
"""Default ``asset_cfg`` of :class:`body_mounted_image_depth` (the task sets its own)."""


class body_mounted_image_depth(mdp.image_depth):
    """:class:`mdp.image_depth` for a camera rigidly mounted on a robot link; refreshes the camera pose first.

    Isaac Lab 3.0 with Isaac Sim PhysX and a non-Kit renderer (Newton Warp): :class:`~isaaclab.sensors.Camera` reads
    its pose from the camera prim's Fabric world matrix (``FabricFrameView``), and ``update_latest_camera_pose`` only
    re-reads that matrix. Only the Kit RTX path keeps it in step with the parent link: it enables
    ``/physics/fabricUpdateTransformations`` (``isaaclab/sensors/camera/camera.py``) and runs the Fabric hierarchy
    update before rendering (``isaaclab_physx/renderers/fabric.py``). In a headless Newton Warp run nothing moves
    the child camera prim, so the camera renders from its spawn pose for the whole run. GPU probe, 2026-11-01:
    camera displacement exactly 0 over 150 policy steps, while the trunk walked 3.4 m away and turned 136 deg;
    the robot then walked into its own camera's view.

    Before every read this term composes the link pose (``body_link_pose_w``, which brings the articulation
    kinematics up to date after resets) with the camera's configured offset and writes it with
    :meth:`~isaaclab.sensors.Camera.set_world_poses`. The renderer then draws from the current pose (measured: camera
    pose error at most 2.4e-7 m; >= 99.998 % of the pixels match a terrain-only ray cast within 3 cm or lie beyond
    the far clip in both, also right after resets; the rest are the robot's own legs).
    The term is active only for Isaac Sim PhysX with a non-Kit renderer. The RTX images already follow the trunk
    (measured), and in a control test a pose written this way did not reach the RTX image. Newton physics computes
    the camera pose from the body state (``NewtonSiteFrameView``; measured error 1e-7 m).

    Extra parameter (besides those of :class:`mdp.image_depth`):

    * ``asset_cfg``: the articulation and the single body the camera prim is attached to.
    """

    def __init__(self, cfg: ObsTerm, env: "ManagerBasedEnv") -> None:
        """Resolve the camera and its parent body and decide whether this backend needs the pose refresh."""
        super().__init__(cfg, env)
        from isaaclab_physx.physics import PhysxCfg  # runtime import: keeps the cfg module light

        self._camera = env.scene.sensors[cfg.params["sensor_cfg"].name]
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset = env.scene[asset_cfg.name]
        body_ids = asset_cfg.body_ids
        if isinstance(body_ids, slice) or len(body_ids) != 1:
            raise ValueError(f"body_mounted_image_depth needs exactly one body, got {asset_cfg.body_names!r}.")
        self._body_id = int(body_ids[0])
        renderer_type = getattr(self._camera.cfg.renderer_cfg, "renderer_type", None)
        # Kit RTX ("default" / "isaac_rtx", see isaaclab/app/sim_launcher.py `_is_kit_camera`) keeps the pose current
        self.refresh_pose = isinstance(env.cfg.sim.physics, PhysxCfg) and renderer_type not in (
            None,
            "default",
            "isaac_rtx",
        )
        offset = self._camera.cfg.offset
        rot = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([offset.rot], dtype=torch.float32), origin=offset.convention, target="opengl"
        )
        self._offset_rot = math_utils.normalize(rot).to(env.device).expand(env.num_envs, 4).contiguous()
        self._offset_pos = torch.tensor([offset.pos], dtype=torch.float32, device=env.device)
        body_name = self._asset.body_names[self._body_id]
        print(
            f"[INFO] {self._camera.cfg.prim_path}: pose refresh from '{body_name}' "
            f"{'on' if self.refresh_pose else 'off'} (physics {type(env.cfg.sim.physics).__name__}, "
            f"renderer {renderer_type})"
        )

    def __call__(
        self,
        env: "ManagerBasedEnv",
        sensor_cfg: SceneEntityCfg = _DEFAULT_CAMERA_CFG,
        data_type: str = "distance_to_image_plane",
        convert_perspective_to_orthogonal: bool = False,
        normalize: bool = True,
        invalid_value: float = 0.0,
        max_depth: float | None = None,
        tanh_scale: float | None = None,
        channel_first: bool = False,
        frame_stack: int = 1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ) -> torch.Tensor:
        """Write the camera pose from its parent body (if needed), then return :class:`mdp.image_depth`'s image."""
        if self.refresh_pose:
            link_pose = self._asset.data.body_link_pose_w.torch[:, self._body_id]
            link_quat = link_pose[:, 3:7]
            pos = link_pose[:, :3] + math_utils.quat_apply(link_quat, self._offset_pos)
            self._camera.set_world_poses(pos, math_utils.quat_mul(link_quat, self._offset_rot), convention="opengl")
        return super().__call__(
            env,
            sensor_cfg,
            data_type,
            convert_perspective_to_orthogonal,
            normalize,
            invalid_value,
            max_depth,
            tanh_scale,
            channel_first,
            frame_stack,
        )


@configclass
class CustomQuadDepthSceneCfg(MySceneCfg):
    """Velocity-task scene plus the trunk-mounted depth camera."""

    depth_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk/depth_camera",
        # lazy sensor updates render the camera when the observation reads it (once per policy step); a positive
        # period would add a host sync per read (camera.py `_env_mask_has_any`) and skip pose refreshes once the
        # float32 sensor timestamps grow past ~16 s
        update_period=0.0,
        # the camera rides on the trunk: re-read its pose on every capture. On Isaac Sim PhysX + Newton Warp that
        # read alone returns the spawn pose; `body_mounted_image_depth` writes the current pose first
        update_latest_camera_pose=True,
        offset=CameraCfg.OffsetCfg(pos=DEPTH_CAMERA_POS, rot=DEPTH_CAMERA_ROT, convention="opengl"),
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
            # mdp.image_depth plus a camera pose refresh from the trunk (frozen spawn pose otherwise, see the class)
            func=body_mounted_image_depth,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_camera",
                "normalize": True,
                "max_depth": DEPTH_MAX_RANGE,
                "invalid_value": DEPTH_MAX_RANGE,
                "channel_first": True,
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            },
            # 2.x: +-0.02 on metric depth (= +-2 cm); the image is now normalized by 4 m, so +-0.005 keeps 2 cm
            noise=Unoise(n_min=-0.005, n_max=0.005),
            clip=(0.0, 1.0),
        )

        def __post_init__(self) -> None:
            """Image noise during training; the group is the single ``(N, 1, H, W)`` image tensor."""
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

        def __post_init__(self) -> None:
            """Noise-free height scan as one flat tensor."""
            self.enable_corruption = False
            self.concatenate_terms = True

    depth: DepthCfg = DepthCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class CustomQuadRoughDepthEnvCfg(CustomQuadRoughEnvCfg):
    """EXPERIMENTAL depth-camera variant of :class:`CustomQuadRoughEnvCfg` (retrain from scratch)."""

    scene: CustomQuadDepthSceneCfg = CustomQuadDepthSceneCfg(num_envs=DEPTH_NUM_ENVS, env_spacing=2.5)
    observations: CustomQuadDepthObservationsCfg = CustomQuadDepthObservationsCfg()

    def __post_init__(self) -> None:
        """Rough-task settings (robot, terrain, events, rewards, physics), without the height scan for the actor."""
        super().__post_init__()

        # the actor sees proprioception + depth only; the height scan feeds the privileged critic group
        self.observations.policy.height_scan = None
