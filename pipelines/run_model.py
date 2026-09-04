"""Run MUH pipeline for one model: metrics → indicators → XGBoost."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cli import add_model_arg  # noqa: E402

STEPS = [
    SRC / "methods" / "ours" / "get_metrics.py",
    SRC / "methods" / "ours" / "get_indicators.py",
    SRC / "methods" / "xgboost_train.py",
]


def subprocess_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MUH for one model, reading from final_data."
    )
    add_model_arg(parser)
    args = parser.parse_args()

    for i, script in enumerate(STEPS, 1):
        cmd = [sys.executable, str(script), "--model", args.model]
        print("=" * 72)
        print(f"[{i}/{len(STEPS)}] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=subprocess_env())
    print("MUH pipeline finished.")


if __name__ == "__main__":
    main()
