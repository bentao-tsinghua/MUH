"""MUH (ours): aggregate U, S, EAU, SAU indicators."""

from __future__ import annotations

import bootstrap  # noqa: F401
import argparse
import json

from tqdm import tqdm

from cli import add_model_arg
from paths import indicators_file, metrics_dir


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def process_item(data: dict) -> dict:
    entropy_list = data.get("entropy_list", [])
    token_sim_list = data.get("token_sim_list", [])
    entity_entropy_list = data.get("entity_entropy_list", [])
    U = mean(entropy_list)
    S = mean(token_sim_list)
    entity_first_entropy = []
    for entity in entity_entropy_list:
        values = entity.get("entropy_list", [])
        if values:
            entity_first_entropy.append(values[0])
    EAU = mean(entity_first_entropy)
    multiply_values = []
    length = min(len(entropy_list), len(token_sim_list))
    for i in range(length):
        e, t = entropy_list[i], token_sim_list[i]
        if e is None or t is None:
            continue
        multiply_values.append(e * t)
    SAU = mean(multiply_values)
    return {
        "U": U,
        "S": S,
        "EAU": EAU,
        "SAU": SAU,
        "hallucination": data.get("hallucination"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MUH ours: build indicators.")
    add_model_arg(parser)
    args = parser.parse_args()
    input_dir = metrics_dir(args.model)
    output_file = indicators_file(args.model)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    with output_file.open("w", encoding="utf-8") as fout, tqdm(
        total=len(jsonl_files), desc="ours get_indicators", unit="file"
    ) as pbar:
        for file in jsonl_files:
            with file.open("r", encoding="utf-8", errors="replace") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        result = process_item(data)
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"Error in {file}: {e}")
            pbar.update(1)
    print(f"Done. Output: {output_file}")


if __name__ == "__main__":
    main()
