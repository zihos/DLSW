"""Entry point that boots the refactored dl_software package."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dl_software.app import run_app


if __name__ == "__main__":
    sys.exit(run_app())
