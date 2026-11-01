"""Search-and-rescue (SAR) dressing of the survey site: a seeded disaster scene plus the drone's downward camera.

Modules (the first four need no simulator):

* :mod:`.assets`   -- verified asset URLs, unit fixes, measured sizes, semantic classes, ground materials.
* :mod:`.camera`   -- intrinsics / mount of the downward RGB-D camera and the pixel-budget math.
* :mod:`.scenario` -- seeded placement on the terrain (height queries, flatness, spacing) and the ground-truth JSON.
* :mod:`.labels`   -- instance-segmentation ids -> objects / classes / 2D boxes (numpy only).
* :mod:`.topdown`  -- top-down object map figure (matplotlib).
* :mod:`.scene`    -- Isaac Lab spawners: the whole site as one scene entity, camera and light configs.

Ground truth (``Scenario.ground_truth()``, ``format: drone_lab.sar.ground_truth/v1``): one record per object
with ``class`` (None for trees), ``position`` (asset origin, e.g. the feet of a person), ``center`` (centre of the
object's box: use it for pins and pin errors), ``size``, ``yaw_deg``, ``aabb_min/max``, ``footprint_xy``,
``in_area`` and the stage ``prim_path``; plus the landing pad (``home``), the survey flight height and the camera.
"""

from .assets import ASSETS, CLASS_IDS, CLASSES, UNLABELED  # noqa: F401
from .camera import SAR_CAMERA, CameraSpec  # noqa: F401
