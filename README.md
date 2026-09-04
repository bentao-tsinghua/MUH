# MUH: Meaningful Uncertainty Hierarchy for Medical LLM Hallucination Detection

Large language models are increasingly used in clinical decision support, yet fluent answers can hide medically harmful hallucinations. Standard uncertainty scores often mix two different things: harmless variation in wording, and genuine disagreement about medical meaning. A model may look uncertain when it is only choosing among equivalent phrases, or look confident while stating an incorrect fact. **MUH** (Meaningful Uncertainty Hierarchy) is designed to separate these cases.

MUH estimates uncertainty at three linked levels. Token-level entropy locates unstable generation steps. Semantic consistency among candidate tokens asks whether alternatives still mean the same thing. Entity-level aggregation, supported by biomedical knowledge, asks whether remaining uncertainty involves clinically important concepts such as diseases, drugs, or tests. The resulting features—U, S, SAU, and EAU—are used to detect medical hallucinations and to rank outputs that most need human review.

This repository provides portable experiment code for MUH. Choose a source model with `--model` (`1_llama`, `2_qwen3_30b`, `3_llama32_3b`, `4_qwen25_7b`). Paths are resolved from `NPJ_PROJECT_ROOT` (or an inferred project root), so the scripts are not bound to a local machine. The pipeline reads `final_data` JSONL, extracts hierarchical features, trains an XGBoost detector, and reports Accuracy, F1, AUC-ROC, AUC-PR, TPR@5%FPR, and SHAP summaries.

## Layout

```
├── config/models.yaml
├── requirements.txt
├── src/
│   ├── paths.py
│   ├── methods/ours/       # get_metrics, get_indicators
│   └── methods/xgboost_train.py
└── pipelines/
    ├── run_model.py        # one model
    └── run_all_models.py   # all models
```

## Setup

```bash
pip install -r requirements.txt
```

Optional project root (otherwise inferred as the parent of this folder’s grandparent):

```bash
# Linux / macOS
export NPJ_PROJECT_ROOT=/path/to/your/project

# Windows PowerShell
$env:NPJ_PROJECT_ROOT = "D:\path\to\your\project"
```

Data are read from `{NPJ_PROJECT_ROOT}/data/{model}/1_final_data/...`. Outputs go to `{NPJ_PROJECT_ROOT}/data/{model}/2_method/1_ours/`.

## Run

```bash
# One model: metrics → indicators → XGBoost
python pipelines/run_model.py --model 1_llama

# All configured models
python pipelines/run_all_models.py
```

Individual steps:

```bash
export PYTHONPATH=src   # PowerShell: $env:PYTHONPATH="src"

python src/methods/ours/get_metrics.py --model 1_llama
python src/methods/ours/get_indicators.py --model 1_llama
python src/methods/xgboost_train.py --model 1_llama
```

Add a new model in `config/models.yaml` and place JSONL under `data/{model}/`.
