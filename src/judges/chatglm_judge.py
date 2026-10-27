"""ChatGLM judge. The rubric lives in config/prompts/medical_skill.txt."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import bootstrap  # noqa: E402,F401

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from jsonl_io import list_jsonl, read_jsonl, write_jsonl
from paths import chatglm_judge_dir, model_answers_dir, prompt_file, require_env

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_NAME = "glm-4-Air"
MAX_RETRY = 3


def parse_json(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {
            "hallucination": "出错了",
            "category": "",
            "subcategory": "",
            "reason": text,
        }


def build_prompt(item: dict) -> str:
    return f"""
question:
{item.get("question", "")}

options:
{json.dumps(item.get("options", []), ensure_ascii=False)}

gold_answer:
{item.get("gold_answer", "")}

model_answer:
{item.get("model_answer", "")}

raw_output:
{item.get("raw_output", "")}

请判断是否存在医学幻觉，只返回JSON。
"""


def call_glm(system_prompt: str, user_prompt: str, api_key: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(MAX_RETRY):
        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(last_error)


def process_item(index: int, item: dict, skill_prompt: str, api_key: str) -> tuple[int, dict]:
    try:
        raw = call_glm(skill_prompt, build_prompt(item), api_key)
        item["judge"] = parse_json(raw)
        return index, item
    except Exception as exc:
        return index, {"error": str(exc), "raw_item": item}


def process_file(input_path: Path, output_path: Path, skill_prompt: str, api_key: str, workers: int) -> None:
    data = read_jsonl(input_path)
    results: list[dict | None] = [None] * len(data)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_item, idx, item, skill_prompt, api_key)
            for idx, item in enumerate(data)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=input_path.name):
            index, result = future.result()
            results[index] = result
    write_jsonl(output_path, [row for row in results if row is not None])


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge model answers with ChatGLM.")
    parser.add_argument("--model", required=True, help="Key in config/models.yaml, used for default paths")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    input_dir = (args.input_dir or model_answers_dir(args.model)).expanduser().resolve()
    output_dir = (args.output_dir or chatglm_judge_dir(args.model)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    skill_prompt = prompt_file("medical_skill.txt").read_text(encoding="utf-8")
    api_key = require_env("ZHIPU_API_KEY")

    for path in list_jsonl(input_dir):
        output_path = output_dir / f"{path.stem}_result.jsonl"
        if output_path.exists():
            print(f"Skip existing: {output_path.name}")
            continue
        print(f"Processing: {path.name}")
        process_file(path, output_path, skill_prompt, api_key, args.workers)
        print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
