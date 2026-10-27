"""Convert top-token log probabilities to probabilities."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import bootstrap  # noqa: E402,F401

import argparse
import json
import math

from tqdm import tqdm

from jsonl_io import list_jsonl, write_jsonl
from paths import entities_dir, token_prob_dir


def add_probs(data: dict) -> dict:
    logprobs = data.get("logprobs")
    if not isinstance(logprobs, list):
        return data
    for logprob_item in logprobs:
        top_list = logprob_item.get("top_logprobs")
        if not isinstance(top_list, list):
            continue
        for token_item in top_list:
            if "logprob" not in token_item:
                continue
            try:
                token_item["prob"] = math.exp(token_item["logprob"])
            except (TypeError, ValueError, OverflowError):
                token_item["prob"] = None
    return data


def process_file(input_path: Path, output_path: Path) -> None:
    rows = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(add_probs(json.loads(line)))
            except json.JSONDecodeError as exc:
                print(f"Skip bad line in {input_path.name}: {exc}")
    write_jsonl(output_path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add prob fields to top_logprobs.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    input_dir = (args.input_dir or entities_dir(args.model)).expanduser().resolve()
    output_dir = (args.output_dir or token_prob_dir(args.model)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(list_jsonl(input_dir), desc="token prob"):
        process_file(path, output_dir / path.name)
    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
