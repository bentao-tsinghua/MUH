"""Semantic similarity among each token's top alternatives."""

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

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from jsonl_io import list_jsonl
from paths import token_prob_dir, token_sim_dir


def embedding_model_name() -> str:
    return os.environ.get("NPJ_EMBEDDING_MODEL", "").strip() or "BAAI/bge-large-zh-v1.5"


class SimilarityEncoder:
    def __init__(self, model_name: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model on {self.device}: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.cache: dict[str, torch.Tensor] = {}

    def embeddings(self, tokens: list[str]) -> torch.Tensor:
        unique = list(dict.fromkeys(tokens))
        missing = [token for token in unique if token not in self.cache]
        if missing:
            encoded = self.model.encode(
                missing,
                batch_size=64,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for token, vector in zip(missing, encoded):
                self.cache[token] = vector.cpu()
        stacked = torch.stack([self.cache[token] for token in tokens])
        return stacked.to(self.device)

    def pairwise(self, top_logprobs: list[dict]) -> list[dict]:
        if len(top_logprobs) <= 1:
            return []
        tokens = [item.get("token", "") for item in top_logprobs]
        probs = [item.get("prob", 0) for item in top_logprobs]
        vectors = self.embeddings(tokens)
        matrix = torch.matmul(vectors, vectors.T).cpu()
        rows = []
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                rows.append(
                    {
                        "token1": tokens[i],
                        "token2": tokens[j],
                        "sim": float(matrix[i, j]),
                        "token1_prob": probs[i],
                        "token2_prob": probs[j],
                    }
                )
        return rows


def process_file(input_path: Path, output_path: Path, encoder: SimilarityEncoder) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            data = json.loads(line)
            logprobs = data.get("logprobs")
            if isinstance(logprobs, list):
                for item in logprobs:
                    top_logprobs = item.get("top_logprobs")
                    if isinstance(top_logprobs, list):
                        item["sim_each_other"] = encoder.pairwise(top_logprobs)
            target.write(json.dumps(data, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score semantic similarity of top tokens.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args()

    input_dir = (args.input_dir or token_prob_dir(args.model)).expanduser().resolve()
    output_dir = (args.output_dir or token_sim_dir(args.model)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = SimilarityEncoder(args.embedding_model or embedding_model_name())

    for path in tqdm(list_jsonl(input_dir), desc="token similarity"):
        process_file(path, output_dir / path.name, encoder)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Cached tokens: {len(encoder.cache)}")
    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
