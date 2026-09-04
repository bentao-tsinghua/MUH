"""Add prob field from logprob in final judge JSONL files."""

from __future__ import annotations

import bootstrap  # noqa: F401
import argparse
import json
import math
from pathlib import Path

from tqdm import tqdm

from cli import add_model_arg
from paths import final_data_dir, project_root


def add_prob_from_logprob(data: dict) -> dict:
    logprobs = data.get("logprobs", [])
    if isinstance(logprobs, list):
        for item in logprobs:
            if (
                isinstance(item, dict)
                and "logprob" in item
                and item["logprob"] is not None
            ):
                try:
                    item["prob"] = math.exp(float(item["logprob"]))
                except (ValueError, OverflowError, TypeError):
                    item["prob"] = None
    return data


def run(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.jsonl"))
    for filename in tqdm(files, desc="prepare_final_data"):
        out_path = output_dir / filename.name
        with input_dir / filename.name.open("r", encoding="utf-8") as fin, out_path.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                data = add_prob_from_logprob(data)
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert logprob to prob and write final_data JSONL."
    )
    add_model_arg(parser, required=False)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory of raw judge JSONL (overrides --model default)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: model final_data dir)",
    )
    args = parser.parse_args()

    if args.input_dir:
        input_dir = args.input_dir.expanduser().resolve()
    elif args.model:
        input_dir = project_root() / "data" / "raw" / args.model / "final_judge"
    else:
        parser.error("Provide --input-dir or --model")

    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
    elif args.model:
        output_dir = final_data_dir(args.model)
    else:
        output_dir = input_dir.parent / "final_data_prepared"

    run(input_dir, output_dir)
    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
