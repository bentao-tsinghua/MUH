"""Ensure `src/` is on sys.path when scripts are executed directly."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
