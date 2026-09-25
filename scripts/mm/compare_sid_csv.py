#!/usr/bin/env python3
"""Verify that SID CSV tracks share identical interaction rows and only SID values differ."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "user_id",
    "history_item_id",
    "item_id",
    "history_item_sid",
    "item_sid",
}


def _load(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows and path.stat().st_size == 0:
        raise ValueError(f"Empty CSV: {path}")
    if rows and not REQUIRED_COLUMNS.issubset(rows[0]):
        raise ValueError(f"{path} is missing required columns")
    return rows


def compare(paths: dict[str, Path]) -> dict[str, Any]:
    loaded = {name: _load(path) for name, path in paths.items()}
    names = list(loaded)
    reference = loaded[names[0]]
    errors: list[str] = []
    for name in names[1:]:
        rows = loaded[name]
        if len(rows) != len(reference):
            errors.append(f"{name}: row count {len(rows)} != {len(reference)}")
            continue
        for idx, (left, right) in enumerate(zip(reference, rows)):
            for field in ("user_id", "history_item_id", "item_id"):
                if left[field] != right[field]:
                    errors.append(f"{name}: row {idx} differs in {field}")
                    break
    result: dict[str, Any] = {
        "tracks": {name: str(path) for name, path in paths.items()},
        "row_counts": {name: len(rows) for name, rows in loaded.items()},
        "identical_interactions": not errors,
        "errors": errors,
    }
    if reference:
        result["unique_users"] = len({row["user_id"] for row in reference})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", action="append", nargs=2, metavar=("NAME", "CSV"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare({name: Path(path) for name, path in args.track})
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not result["identical_interactions"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
