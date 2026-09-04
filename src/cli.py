"""Shared CLI helpers."""

from __future__ import annotations

import argparse

from paths import list_models


def add_model_arg(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument(
        "--model",
        required=required,
        choices=list_models(),
        help="Model data folder name under data/ (e.g. 1_llama)",
    )
