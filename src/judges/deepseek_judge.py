"""DeepSeek judge. The rubric lives in config/prompts/hallucination_eval_skill.txt."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import bootstrap  # noqa: E402,F401

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from jsonl_io import list_jsonl, read_jsonl, write_jsonl
from paths import deepseek_judge_dir, model_answers_dir, prompt_file, require_env

BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.last_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            delay = self.interval - (now - self.last_time)
            if delay > 0:
                time.sleep(delay)
            self.last_time = time.time()


def parse_json(text: str) -> dict:
    try:
        if "```" in text:
            text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "hallucination": "",
            "category": "",
            "subcategory": "",
            "reason": text,
        }


def build_prompt(skill: str, item: dict) -> str:
    return f"""
{skill}


下面是待检测样本：

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


请严格按照要求输出JSON。
""".strip()


def call_deepseek(client: OpenAI, limiter: RateLimiter, prompt: str) -> str:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            limiter.wait()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(last_error)


def process_one(index: int, item: dict, client: OpenAI, limiter: RateLimiter, skill: str) -> tuple[int, dict, str | None]:
    try:
        parsed = parse_json(call_deepseek(client, limiter, build_prompt(skill, item)))
        item["deepseek_judge"] = {
            "hallucination": parsed.get("hallucination", ""),
            "category": parsed.get("category", ""),
            "subcategory": parsed.get("subcategory", ""),
            "reason": parsed.get("reason", ""),
        }
        return index, item, None
    except Exception as exc:
        return index, item, str(exc)


def process_file(path: Path, output_path: Path, client: OpenAI, limiter: RateLimiter, skill: str, workers: int) -> None:
    data = read_jsonl(path)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_one, idx, item, client, limiter, skill)
            for idx, item in enumerate(data)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=path.name):
            index, item, error = future.result()
            if error:
                print(f"Item {index} failed: {error}")
            data[index] = item
    write_jsonl(output_path, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge model answers with DeepSeek.")
    parser.add_argument("--model", required=True, help="Key in config/models.yaml, used for default paths")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--requests-per-second", type=float, default=2)
    args = parser.parse_args()

    input_dir = (args.input_dir or model_answers_dir(args.model)).expanduser().resolve()
    output_dir = (args.output_dir or deepseek_judge_dir(args.model)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    skill = prompt_file("hallucination_eval_skill.txt").read_text(encoding="utf-8")
    client = OpenAI(api_key=require_env("DEEPSEEK_API_KEY"), base_url=BASE_URL)
    limiter = RateLimiter(args.requests_per_second)

    for path in list_jsonl(input_dir):
        output_path = output_dir / f"{path.stem}_deepseek_judge.jsonl"
        if output_path.exists():
            print(f"Skip existing: {output_path.name}")
            continue
        print(f"Processing: {path.name}")
        process_file(path, output_path, client, limiter, skill, args.workers)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
