"""Align generated tokens back onto the raw output string."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import bootstrap  # noqa: E402,F401

import argparse
import json

from tqdm import tqdm

from jsonl_io import list_jsonl
from paths import token_position_dir, token_sim_dir

REPLACEMENT = "�"


def next_token_fits(raw: str, start_pos: int, next_token: str) -> bool:
    if next_token in ("", REPLACEMENT):
        return True
    return raw.startswith(next_token, start_pos)


def align_tokens(raw_output: str, tokens: list[dict]) -> list[dict]:
    pos = 0
    for index, item in enumerate(tokens):
        token = item.get("token", "")
        if token == "":
            item["start_position"] = None
            item["end_position"] = None
            continue

        if token != REPLACEMENT:
            token_len = len(token)
            if raw_output.startswith(token, pos):
                item["start_position"] = pos
                item["end_position"] = pos + token_len - 1
                pos += token_len
                continue
            found = raw_output.find(token, pos)
            if found != -1:
                item["start_position"] = found
                item["end_position"] = found + token_len - 1
                pos = found + token_len
                continue
        else:
            start = pos
            occupy = 1
            next_token = ""
            if index + 1 < len(tokens):
                next_token = tokens[index + 1].get("token", "")
            while start + occupy <= len(raw_output):
                if next_token_fits(raw_output, start + occupy, next_token):
                    break
                occupy += 1
            item["start_position"] = start
            item["end_position"] = start + occupy - 1
            pos += occupy
            continue

        item["start_position"] = pos
        item["end_position"] = pos + len(token) - 1
        pos += len(token)
    return tokens


def process_file(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            data = json.loads(line)
            data["logprobs"] = align_tokens(
                data.get("raw_output", "") or "",
                data.get("logprobs", []) or [],
            )
            target.write(json.dumps(data, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align logprob tokens to raw_output.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    input_dir = (args.input_dir or token_sim_dir(args.model)).expanduser().resolve()
    output_dir = (args.output_dir or token_position_dir(args.model)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(list_jsonl(input_dir), desc="token position"):
        process_file(path, output_dir / path.name)
    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
