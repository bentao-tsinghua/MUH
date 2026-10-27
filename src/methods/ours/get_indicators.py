"""MUH (ours): aggregate U, S, SAU, and EAU indicators.

Response-level definitions follow the manuscript:

- U = mean_t U_t, where U_t is token entropy.
- S = mean_t S_t, where S_t is probability-weighted pairwise similarity.
- SAU_t = U_t * (1 - S_t); SAU = mean_t SAU_t.
- E_m = U_{m,1} (first token of entity m);
  EAU = sum_m L_m E_m / sum_m L_m, with L_m the entity token count.
"""

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

from cli import add_model_arg
from paths import indicators_file, metrics_dir


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def entity_aggregated_uncertainty(entity_entropy_list: list) -> float | None:
    """Length-weighted mean of each entity's first-token entropy."""
    weighted = 0.0
    total_length = 0.0
    for entity in entity_entropy_list:
        values = entity.get("entropy_list", [])
        if not values or values[0] is None:
            continue
        length = len(values)
        weighted += length * values[0]
        total_length += length
    if total_length == 0:
        return None
    return weighted / total_length


def process_item(data: dict) -> dict:
    """Aggregate token entropy (U), semantic consistency (S), semantic-adjusted uncertainty (SAU), and entity-aggregated uncertainty (EAU)."""
    entropy_list = data.get("entropy_list", [])
    token_sim_list = data.get("token_sim_list", [])
    entity_entropy_list = data.get("entity_entropy_list", [])
    U = mean(entropy_list)
    S = mean(token_sim_list)
    EAU = entity_aggregated_uncertainty(entity_entropy_list)
    sau_values = []
    length = min(len(entropy_list), len(token_sim_list))
    for i in range(length):
        entropy, similarity = entropy_list[i], token_sim_list[i]
        if entropy is None or similarity is None:
            continue
        sau_values.append(entropy * (1.0 - similarity))
    SAU = mean(sau_values)
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
