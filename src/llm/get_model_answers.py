"""Collect multiple-choice answers from one OpenRouter model.

The four source models share this client. Select one with --model.
API keys and data roots come from the environment or .env.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import bootstrap  # noqa: E402,F401

import argparse
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from jsonl_io import list_jsonl, read_jsonl, write_jsonl
from paths import (
    model_answers_dir,
    model_spec,
    prompt_file,
    questions_dir,
    require_env,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRY = 5


def extract_answer(text: str) -> tuple[str, str]:
    analysis = ""
    answer = ""
    analysis_match = re.search(r"解析[:：]\s*(.*?)(?=答案[:：]|$)", text, re.S)
    if analysis_match:
        analysis = analysis_match.group(1).strip()
    answer_match = re.search(r"答案[:：]\s*([A-E])", text, re.I)
    if answer_match:
        answer = answer_match.group(1).upper()
    return analysis, answer


def build_user_prompt(question: str, options: list) -> str:
    option_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
    return f"题目：\n\n{question}\n\n选项：\n\n{option_text}\n\n请回答。\n"


def serialize_logprobs(choice) -> list | None:
    if not choice.logprobs or not choice.logprobs.content:
        return None
    rows = []
    for token_info in choice.logprobs.content:
        item = {
            "token": token_info.token,
            "logprob": token_info.logprob,
            "top_logprobs": [],
        }
        if token_info.top_logprobs:
            for top in token_info.top_logprobs:
                item["top_logprobs"].append(
                    {"token": top.token, "logprob": top.logprob}
                )
        rows.append(item)
    return rows


def call_model(client: OpenAI, model_name: str, system_prompt: str, question: str, options: list, provider: str | None) -> dict:
    user_prompt = build_user_prompt(question, options)
    request = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 512,
        "logprobs": True,
        "top_logprobs": 5,
    }
    if provider:
        request["extra_body"] = {
            "provider": {"order": [provider], "allow_fallbacks": False}
        }

    last_error = "请求失败"
    for attempt in range(MAX_RETRY):
        try:
            response = client.chat.completions.create(**request)
            choice = response.choices[0]
            raw_answer = (choice.message.content or "").strip()
            analysis, answer = extract_answer(raw_answer)
            usage = None
            if response.usage is not None:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return {
                "raw_output": raw_answer,
                "analysis": analysis,
                "model_answer": answer,
                "logprobs": serialize_logprobs(choice),
                "usage": usage,
                "error": None,
            }
        except Exception as exc:
            last_error = str(exc)
            print(f"请求失败，第 {attempt + 1}/{MAX_RETRY} 次: {last_error}")
            time.sleep(2 ** attempt)

    return {
        "raw_output": None,
        "analysis": None,
        "model_answer": None,
        "logprobs": None,
        "usage": None,
        "error": last_error,
    }


def process_one(index: int, item: dict, client: OpenAI, model_name: str, system_prompt: str, provider: str | None) -> tuple[int, dict]:
    try:
        result = call_model(
            client,
            model_name,
            system_prompt,
            item.get("question", ""),
            item.get("options", []),
            provider,
        )
        output = {
            "question": item.get("question"),
            "options": item.get("options"),
            "raw_output": result["raw_output"],
            "analysis": result["analysis"],
            "model_answer": result["model_answer"],
            "gold_answer": item.get("gold", item.get("gold_answer")),
            "meta_info": item.get("meta_info"),
            "logprobs": result["logprobs"],
            "usage": result["usage"],
            "error": result["error"],
        }
    except Exception as exc:
        output = {
            "question": item.get("question"),
            "options": item.get("options"),
            "raw_output": None,
            "analysis": None,
            "model_answer": None,
            "gold_answer": item.get("gold", item.get("gold_answer")),
            "meta_info": item.get("meta_info"),
            "logprobs": None,
            "usage": None,
            "error": str(exc),
        }
    time.sleep(random.uniform(0.1, 0.5))
    return index, output


def process_file(input_path: Path, output_path: Path, client: OpenAI, model_name: str, system_prompt: str, provider: str | None, workers: int) -> None:
    data = read_jsonl(input_path)
    results: list[dict | None] = [None] * len(data)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_one, idx, item, client, model_name, system_prompt, provider
            )
            for idx, item in enumerate(data)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=input_path.name):
            index, output = future.result()
            results[index] = output
    write_jsonl(output_path, [row for row in results if row is not None])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query one OpenRouter model for medical multiple-choice answers."
    )
    parser.add_argument("--model", required=True, help="Key in config/models.yaml")
    parser.add_argument("--model-name", default=None, help="Override the OpenRouter model id")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    spec = model_spec(args.model)
    model_name = args.model_name or spec["openrouter_id"]
    provider = spec.get("provider") or None
    input_dir = (args.input_dir or questions_dir(args.model)).expanduser().resolve()
    output_dir = (args.output_dir or model_answers_dir(args.model)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = prompt_file("answer_system.txt").read_text(encoding="utf-8")

    client = OpenAI(api_key=require_env("OPENROUTER_API_KEY"), base_url=OPENROUTER_BASE_URL)
    files = list_jsonl(input_dir)
    print(f"Model: {model_name}")
    print(f"Found {len(files)} jsonl files")

    for path in files:
        output_path = output_dir / path.name
        if output_path.exists():
            print(f"Skip existing: {output_path.name}")
            continue
        print(f"Processing: {path.name}")
        process_file(path, output_path, client, model_name, system_prompt, provider, args.workers)
        print(f"Done: {output_path}")

    print("All files finished.")


if __name__ == "__main__":
    main()
