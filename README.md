# MUH: Meaningful Uncertainty Hierarchy for Medical LLM Hallucination Detection

Large language models are increasingly used in clinical decision support, yet fluent answers can hide medically harmful hallucinations. Standard uncertainty scores often mix two different things: harmless variation in wording, and genuine disagreement about medical meaning. A model may look uncertain when it is only choosing among equivalent phrases, or look confident while stating an incorrect fact. **MUH** (Meaningful Uncertainty Hierarchy) is designed to separate these cases.

MUH estimates uncertainty at three linked levels. Token-level entropy locates unstable generation steps. Semantic consistency among candidate tokens asks whether alternatives still mean the same thing. Entity-level aggregation, supported by biomedical knowledge, asks whether remaining uncertainty involves clinically important concepts such as diseases, drugs, or tests. The resulting features—U, S, SAU, and EAU—are used to detect medical hallucinations and to rank outputs that most need human review.

This repository is the portable experiment code. Paths and API keys come from environment variables, so the scripts are not tied to one machine.

## Layout

```
├── config/
│   ├── models.yaml
│   └── prompts/
│       ├── answer_system.txt              # answering prompt
│       ├── medical_skill.txt               # ChatGLM judge rubric
│       └── hallucination_eval_skill.txt    # DeepSeek judge rubric
├── src/
│   ├── llm/get_model_answers.py            # one OpenRouter client for all source models
│   ├── judges/chatglm_judge.py
│   ├── judges/deepseek_judge.py
│   └── methods/ours/                       # entity link, token features, U/S/EAU/SAU
└── pipelines/run_model.py
```

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env   # Windows. On Linux/macOS: cp .env.example .env
```

Fill in `.env`. Do not commit it.

| Variable | Role |
| --- | --- |
| `OPENROUTER_API_KEY` | Source-model answers |
| `ZHIPU_API_KEY` | ChatGLM judge (`glm-4-Air`) |
| `DEEPSEEK_API_KEY` | DeepSeek judge (`deepseek-v4-flash`) |
| `NPJ_PROJECT_ROOT` | Directory that contains `data/` |
| `NPJ_ENTITY_DICT` | Biomedical lexicon, one entity per line; text before `@@` is used |
| `NPJ_TOKENIZER` | Hugging Face id or a local tokenizer directory |
| `NPJ_EMBEDDING_MODEL` | Defaults to `BAAI/bge-large-zh-v1.5` |

Model ids live in `config/models.yaml`: Llama-3.1-8B, Qwen3-30B-A3B, Llama-3.2-3B, and Qwen2.5-7B. All four use the same OpenRouter client. Set `provider` on a model entry only when a run must pin one provider.

## Data

```
{NPJ_PROJECT_ROOT}/data/{model}/0_questions/          # question, options, gold
{NPJ_PROJECT_ROOT}/data/{model}/1_model_answers/
{NPJ_PROJECT_ROOT}/data/{model}/2_chatglm_judge/
{NPJ_PROJECT_ROOT}/data/{model}/3_deepseek_judge/
{NPJ_PROJECT_ROOT}/data/{model}/1_final_data/1_final_data/   # labeled records for MUH
{NPJ_PROJECT_ROOT}/data/{model}/2_method/1_ours/
```

MUH reads the labeled JSONL. Each record needs `raw_output`, `logprobs` (with `top_logprobs`), and `final_judge.hallucination`. Question files use `question`, `options`, and `gold`.

## Run

From this directory, with `src` on `PYTHONPATH` (PowerShell: `$env:PYTHONPATH="src"`):

```bash
# 1. Answers. One script; pick the model in config/models.yaml.
python src/llm/get_model_answers.py --model 3_llama32_3b

# 2. Judges. Rubrics are the prompt files under config/prompts/.
python src/judges/chatglm_judge.py --model 3_llama32_3b
python src/judges/deepseek_judge.py --model 3_llama32_3b

# 3. MUH, after labeled final_data is in place.
python pipelines/run_model.py --model 3_llama32_3b
```

`run_model.py` links entities, converts logprobs to probabilities, scores semantic similarity, aligns tokens, then builds U, S, EAU, and SAU and trains the XGBoost detector. Token probabilities are taken from the model logprobs and are not edited with the hallucination label.

Individual MUH steps:

```bash
python src/methods/ours/get_entities.py --model 1_llama
python src/methods/ours/get_token_prob.py --model 1_llama
python src/methods/ours/get_token_sim.py --model 1_llama
python src/methods/ours/get_token_position.py --model 1_llama
python src/methods/ours/get_metrics.py --model 1_llama
python src/methods/ours/get_indicators.py --model 1_llama
python src/methods/xgboost_train.py --model 1_llama
```

Reported metrics are Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, and TPR at 5% FPR, plus SHAP summaries for U, EAU, S, and SAU.
