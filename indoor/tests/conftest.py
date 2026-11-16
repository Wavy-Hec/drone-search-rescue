"""Put ``indoor_sar`` and, when this folder sits in the drone-search-rescue repository, ``drone_lab`` on the path."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT.parent / "sim" / "src"):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
