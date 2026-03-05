from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SDK_SRC = ROOT / "sdk" / "src"

if SDK_SRC.exists():
    sys.path.insert(0, str(SDK_SRC))

# Ensure tests run against deterministic mock backend by default.
os.environ.setdefault("SOARMMOCE_TRANSPORT", "mock")
