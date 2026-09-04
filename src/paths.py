"""Resolve project and per-model data paths (portable, no hardcoded drive letters)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

MUH_DIR = "1_ours"


def project_root() -> Path:
    """Project root = parent of `data/` (npj_digital_medicine repo root)."""
    env = os.environ.get("NPJ_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # 00_code_to_github/src/paths.py -> repo root is three levels up from src/
    return Path(__file__).resolve().parents[2].parent


def config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "config"


def load_config() -> dict[str, Any]:
    cfg_path = config_dir() / "models.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_data_root(model: str) -> Path:
    cfg = load_config()
    if model not in cfg["models"]:
        raise ValueError(f"Unknown model '{model}'. See config/models.yaml")
    return project_root() / "data" / model


def final_data_dir(model: str) -> Path:
    cfg = load_config()
    rel = cfg["models"][model]["final_data_dir"]
    return model_data_root(model) / "1_final_data" / Path(rel).relative_to("1_final_data")


def muh_root(model: str) -> Path:
    return model_data_root(model) / "2_method" / MUH_DIR


def metrics_dir(model: str) -> Path:
    path = muh_root(model) / "1_get_metrics" / "1_get_metrics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def indicators_file(model: str) -> Path:
    path = muh_root(model) / "2_get_indicators" / "indicators.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def xgboost_dir(model: str) -> Path:
    path = muh_root(model) / "3_xgboost"
    path.mkdir(parents=True, exist_ok=True)
    return path


def method_result_path(model: str) -> Path:
    return xgboost_dir(model) / "ours_result.jsonl"


def method_model_path(model: str) -> Path:
    return xgboost_dir(model) / "hallucination_xgb.pkl"


def list_models() -> list[str]:
    return list(load_config()["models"].keys())
