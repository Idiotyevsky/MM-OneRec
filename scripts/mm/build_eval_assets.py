#!/usr/bin/env python3
"""Build catalog, info, and training-frequency files for SID evaluation."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path


def build_assets(index_path: Path, item_path: Path, train_csv: Path, output_dir: Path) -> dict[str, int]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    items = json.loads(item_path.read_text(encoding="utf-8"))
    sid_map = {str(item_id): "".join(map(str, tokens)) for item_id, tokens in index.items()}
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = sorted(set(sid_map.values()))
    (output_dir / "catalog.txt").write_text("\n".join(catalog) + "\n", encoding="utf-8")
    with (output_dir / "info.tsv").open("w", encoding="utf-8") as handle:
        for item_id in sorted(sid_map, key=int):
            title = str(items.get(item_id, {}).get("title", item_id)).replace("\t", " ").replace("\n", " ")
            handle.write(f"{sid_map[item_id]}\t{title}\t{item_id}\n")

    train_items: list[str] = []
    with train_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            train_items.extend(ast.literal_eval(row["history_item_sid"]))
            train_items.append(str(row["item_sid"]))
    (output_dir / "train-items.txt").write_text("\n".join(map(str, train_items)) + "\n", encoding="utf-8")
    stats = {"catalog_items": len(catalog), "train_item_events": len(train_items)}
    (output_dir / "assets.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_assets(args.index, args.items, args.train_csv, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
