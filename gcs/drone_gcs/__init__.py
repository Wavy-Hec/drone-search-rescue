"""drone_gcs: ground-control-station side of the simulated drone survey.

The simulator (Isaac Sim / Isaac Lab, or ``scripts/fake_drone.py``) streams MISSION / TELEMETRY / SCAN /
CAMERA / GT_OBJECTS / END messages over a local socket (``drone_gcs.link``). The GCS builds the map
incrementally (``drone_gcs.mapping``), runs detection (``drone_gcs.detector``), shows a live dashboard
(``drone_gcs.dashboard``) and writes a mission report (``drone_gcs.report``).

This package ``__init__`` imports nothing on purpose: ``from drone_gcs.link import Sender`` must work inside
Isaac Sim's Python with only the standard library and numpy.
"""

__version__ = "0.1.0"
