"""Resolve project and per-model data paths (portable, no hardcoded drive letters)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

MUH_DIR = "1_ours"
CODE_ROOT = Path(__file__).resolve().parents[1]


def load_local_env() -> None:
    """Load KEY=VALUE pairs from `.env` without overriding the shell."""
    env_path = CODE_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"Missing {name}. Copy .env.example to .env and fill it in, "
            "or export the variable in the shell."
        )
    return value


def project_root() -> Path:
    """Directory that contains `data/`. Override with NPJ_PROJECT_ROOT."""
    env = os.environ.get("NPJ_PROJECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # 00_code_to_github/src/paths.py -> repository root is three levels above src/
    return Path(__file__).resolve().parents[2].parent


def config_dir() -> Path:
    return CODE_ROOT / "config"


def prompt_file(name: str) -> Path:
    path = config_dir() / "prompts" / name
    if not path.is_file():
        raise SystemExit(f"Prompt file not found: {path}")
    return path


def load_config() -> dict[str, Any]:
    cfg_path = config_dir() / "models.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_spec(model: str) -> dict[str, Any]:
    cfg = load_config()
    models = cfg.get("models", {})
    if model not in models:
        raise ValueError(f"Unknown model '{model}'. See config/models.yaml")
    return models[model]


def model_data_root(model: str) -> Path:
    model_spec(model)
    return project_root() / "data" / model


def questions_dir(model: str) -> Path:
    return model_data_root(model) / "0_questions"


def model_answers_dir(model: str) -> Path:
    return model_data_root(model) / "1_model_answers"


def chatglm_judge_dir(model: str) -> Path:
    return model_data_root(model) / "2_chatglm_judge"


def deepseek_judge_dir(model: str) -> Path:
    return model_data_root(model) / "3_deepseek_judge"


def final_data_dir(model: str) -> Path:
    rel = model_spec(model)["final_data_dir"]
    return model_data_root(model) / "1_final_data" / Path(rel).relative_to("1_final_data")


def muh_root(model: str) -> Path:
    return model_data_root(model) / "2_method" / MUH_DIR


def entities_dir(model: str) -> Path:
    return muh_root(model) / "entities"


def token_prob_dir(model: str) -> Path:
    return muh_root(model) / "token_prob"


def token_sim_dir(model: str) -> Path:
    return muh_root(model) / "token_sim"


def token_position_dir(model: str) -> Path:
    return muh_root(model) / "token_position"


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
