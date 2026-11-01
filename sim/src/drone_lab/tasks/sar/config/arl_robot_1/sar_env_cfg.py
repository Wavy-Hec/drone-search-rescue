"""Search-and-rescue (SAR) mission for ARL-Robot-1: the survey task over a dressed disaster site, with a camera.

``SarEnvCfg`` subclasses :class:`~drone_lab.tasks.survey.config.arl_robot_1.survey_env_cfg.SurveyEnvCfg` and only
adds to the scene; observations, actions, commands, events, rewards and terminations are the survey task's, so
the policy sees exactly the same 17 inputs and the built-in tracking policy flies it unchanged.

Scene additions:

* ``sar_site`` -- the whole dressed site as one entity (:class:`drone_lab.sar.scene.SarSiteCfg`): 6-10 people
  standing or lying (some next to blocks), a forklift, crates, barrels, pallets, traffic cones, a few trees, a
  landing pad at the start, all with ``class`` semantic labels (trees unlabelled), plus MDL ground materials.
  ``env.scene.sar_site.spawn.seed=<n>`` gives another site.
* ``sar_camera`` -- downward 640 x 480 RGB-D camera under the drone (f = 480 px, 67 x 53 deg; the pixel budget is
  derived in :mod:`drone_lab.sar.camera`) with ``rgb``, ``distance_to_image_plane`` and raw
  ``instance_segmentation`` (one id per labelled object; ``camera.data.info["instance_segmentation"]`` maps ids
  to prim paths and ``{"class": ...}`` labels, see :mod:`drone_lab.sar.labels`). It is not part of the policy
  observations. Anti-aliasing is FXAA (sharper than the kit's DLSS and free of its smearing; DLSS is ~20 %
  faster: ``env.scene.sar_camera.renderer_cfg.global_settings.antialiasing_mode=DLSS``).
* ``sky_light`` (clear-sky HDRI dome) and ``sun_light`` (distant light aligned with the HDRI's own sun,
  43 deg elevation) for the RTX renderer.

The camera needs the RTX renderer (Kit): run with ``physics=isaacsim_physx`` (the only preset of this task);
``launch_simulation`` enables cameras automatically.
"""

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from drone_lab.sar.camera import SAR_CAMERA
from drone_lab.sar.scene import SarSiteCfg, sar_camera_cfg, sky_light_cfg, sun_light_cfg
from drone_lab.tasks.survey.config.arl_robot_1.survey_env_cfg import SurveyEnvCfg, SurveySceneCfg

SITE_PRIM_PATH = "/World/sar"
"""Root prim of the dressed site; objects are at ``/World/sar/objects/<name>``."""


@configclass
class SarSceneCfg(SurveySceneCfg):
    """Survey scene + dressed site, camera and sun."""

    sky_light = sky_light_cfg(intensity=800.0)
    # aligned with the sun inside the HDRI (measured from tree shadows in the GPU preview: azimuth ~136 deg,
    # elevation ~43-45 deg; the file name says 43), otherwise every object casts two shadows
    sun_light = sun_light_cfg(intensity=2000.0, elevation_deg=43.0, azimuth_deg=136.0)

    sar_site = AssetBaseCfg(prim_path=SITE_PRIM_PATH, spawn=SarSiteCfg(seed=0))

    sar_camera: CameraCfg = sar_camera_cfg(SAR_CAMERA)

    # optional world-fixed cameras (None = not spawned); scripts/sar_preview.py sets them for its renders
    overview_camera: CameraCfg | None = None
    topdown_camera: CameraCfg | None = None


@configclass
class SarEnvCfg(SurveyEnvCfg):
    """Survey mission over the dressed SAR site; see the module docstring."""

    scene: SarSceneCfg = SarSceneCfg(num_envs=1, env_spacing=2.0)

    def __post_init__(self):
        super().__post_init__()
        site = self.scene.sar_site.spawn
        # place the objects on exactly the terrain the scene imports, and keep trees below the survey height
        site.terrain_generator = self.scene.terrain.terrain_generator
        site.terrain_prim_path = self.scene.terrain.prim_path
        site.survey_clearance = self.commands.survey_target.altitude
        site.survey_spacing = self.commands.survey_target.spacing
        # one camera frame per policy step (the env renders once per step: render_interval = decimation)
        self.scene.sar_camera.update_period = 0.0
