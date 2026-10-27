"""Small JSONL helpers shared by the public scripts."""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                print(f"JSON parse failed {path}:{line_no}: {exc}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def list_jsonl(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"Input directory not found: {directory}")
    files = sorted(path for path in directory.glob("*.jsonl") if path.is_file())
    if not files:
        raise SystemExit(f"No JSONL files in {directory}")
    return files
