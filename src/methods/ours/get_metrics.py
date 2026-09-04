"""MUH (ours): compute token entropy, semantic similarity, entity entropy."""

from __future__ import annotations

import bootstrap  # noqa: F401

import argparse
import json
import math
from pathlib import Path

from tqdm import tqdm

from cli import add_model_arg
from paths import final_data_dir, metrics_dir


def calc_entropy(top_logprobs: list) -> float:
    entropy = 0.0
    if not top_logprobs:
        return entropy
    for item in top_logprobs:
        p = item.get("prob", 0)
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def calc_token_similarity(sim_each_other: list) -> float | None:
    if not sim_each_other:
        return None
    score = 0.0
    weight = 0.0
    for item in sim_each_other:
        w = item.get("token1_prob", 0) * item.get("token2_prob", 0)
        score += item.get("sim", 0) * w
        weight += w
    if weight == 0:
        return None
    return score / weight


def calc_entity_entropy(logprobs: list, entropy_list: list, entities: list) -> list:
    entity_entropy_list = []
    for entity in entities:
        start = entity["start_char"]
        end = entity["end_char"]
        pingjie_token = ""
        values = []
        for idx, item in enumerate(logprobs):
            token_start = item.get("start_position")
            token_end = item.get("end_position")
            if token_start is None or token_end is None:
                continue
            if token_start <= end and token_end >= start:
                pingjie_token += item.get("token", "")
                if idx < len(entropy_list):
                    values.append(entropy_list[idx])
        if values:
            entity_entropy_list.append(
                {
                    "entity": entity.get("entity", ""),
                    "pingjie_token": pingjie_token,
                    "entropy_list": values,
                }
            )
    return entity_entropy_list


def run(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.jsonl"))
    for filename in tqdm(files, desc="ours get_metrics", unit="file"):
        out_path = output_dir / filename.name
        with filename.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                data = json.loads(line)
                logprobs = data.get("logprobs", [])
                entities = data.get("entities", [])
                entropy_list = [
                    calc_entropy(token.get("top_logprobs", [])) for token in logprobs
                ]
                token_sim_list = [
                    calc_token_similarity(token.get("sim_each_other", [])) for token in logprobs
                ]
                entity_entropy_list = calc_entity_entropy(logprobs, entropy_list, entities)
                new_item = {
                    "id": data.get("id"),
                    "meta_info": data.get("meta_info"),
                    "entropy_list": entropy_list,
                    "token_sim_list": token_sim_list,
                    "entity_entropy_list": entity_entropy_list,
                    "hallucination": data.get("final_judge", {}).get("hallucination"),
                }
                fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="MUH ours: extract metrics.")
    add_model_arg(parser)
    args = parser.parse_args()
    input_dir = final_data_dir(args.model)
    output_dir = metrics_dir(args.model)
    run(input_dir, output_dir)
    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
