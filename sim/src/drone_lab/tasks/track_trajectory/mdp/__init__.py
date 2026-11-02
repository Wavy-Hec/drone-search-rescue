"""MDP terms of the trajectory-tracking task (the command class is loaded lazily via its config)."""

from .commands_cfg import REFERENCE_MARKER_CFG, TrajectoryReferenceCommandCfg
from .track_terms import (
    reference_lookahead,
    reference_position_error,
    reference_too_far,
    reference_tracking_exp,
    reference_velocity_error,
    reference_velocity_exp,
    reset_near_reference,
)
