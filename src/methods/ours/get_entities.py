"""Link biomedical entities in model outputs with an Aho-Corasick lexicon."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import bootstrap  # noqa: E402,F401

import argparse
import json
import os

import ahocorasick
from tqdm import tqdm
from transformers import AutoTokenizer

from jsonl_io import list_jsonl
from paths import entities_dir, final_data_dir, require_env


def build_automaton(entity_path: Path) -> ahocorasick.Automaton:
    automaton = ahocorasick.Automaton()
    count = 0
    with entity_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            text = line.strip()
            if not text:
                continue
            entity = text.split("@@")[0]
            if len(entity) < 2:
                continue
            automaton.add_word(entity, (idx, entity))
            count += 1
    automaton.make_automaton()
    print(f"Entities loaded: {count}")
    return automaton


def longest_match(matches: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    if not matches:
        return []
    ordered = sorted(matches, key=lambda item: (item[0], -(item[1] - item[0])))
    kept: list[tuple[int, int, str]] = []
    for start, end, entity in ordered:
        overlap = False
        for index, (kept_start, kept_end, _) in enumerate(kept):
            if not (end < kept_start or start > kept_end):
                overlap = True
                if end - start > kept_end - kept_start:
                    kept[index] = (start, end, entity)
                break
        if not overlap:
            kept.append((start, end, entity))
    return kept


def token_span(offsets, start_char: int, end_char: int) -> tuple[int, int] | None:
    token_ids = [
        idx
        for idx, (start, end) in enumerate(offsets)
        if end > start_char and start <= end_char
    ]
    if not token_ids:
        return None
    return token_ids[0], token_ids[-1]


def process_item(item: dict, tokenizer, automaton) -> dict:
    text = item.get("raw_output") or ""
    if not text:
        item["entities"] = []
        return item

    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    matches = [
        (end - len(entity) + 1, end, entity)
        for end, (_, entity) in automaton.iter(text)
    ]
    entities = []
    for start, end, entity in longest_match(matches):
        if len(entity) < 2:
            continue
        entities.append(
            {
                "entity": entity,
                "start_char": start,
                "end_char": end,
                "token_span": token_span(offsets, start, end),
            }
        )
    item["entities"] = entities
    return item


def process_file(input_path: Path, output_path: Path, tokenizer, automaton) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            item = process_item(json.loads(line), tokenizer, automaton)
            target.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Match biomedical entities in raw_output.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--entity-dict", type=Path, default=None)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()

    input_dir = (args.input_dir or final_data_dir(args.model)).expanduser().resolve()
    output_dir = (args.output_dir or entities_dir(args.model)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entity_path = (args.entity_dict or Path(require_env("NPJ_ENTITY_DICT"))).expanduser().resolve()
    if not entity_path.is_file():
        raise SystemExit(f"Entity lexicon not found: {entity_path}")
    tokenizer_name = args.tokenizer or os.environ.get("NPJ_TOKENIZER", "").strip() or require_env("NPJ_TOKENIZER")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    automaton = build_automaton(entity_path)

    for path in tqdm(list_jsonl(input_dir), desc="entities"):
        process_file(path, output_dir / path.name, tokenizer, automaton)
    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
