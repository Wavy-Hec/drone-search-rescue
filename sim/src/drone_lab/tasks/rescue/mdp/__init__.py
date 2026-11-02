"""MDP terms of the rescue task.

The command implementation (:mod:`.commands`) is referenced lazily by its config's ``class_type`` string, so
importing this package (and the env config) stays cheap and works before the simulator is launched.
"""

from .commands_cfg import GOAL_MARKER_CFG, ROUTE_MARKER_CFG, TARGET_MARKER_CFG, PathFollowCommandCfg
from .rescue_terms import reset_root_on_route, route_complete
