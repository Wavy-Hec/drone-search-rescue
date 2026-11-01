"""Rescue finale: a quadruped walks the drone's planned route to a person (air-ground teaming).

The task reuses a velocity-tracking locomotion policy unchanged: its ``base_velocity`` command is replaced by a
path-following command (pure pursuit on a route planned with :mod:`drone_lab.teaming`), and the robot walks on
the same seeded survey terrain the drone mapped. See ``rescue_env_cfg.py`` and ``scripts/rescue_finale.py``.
"""
