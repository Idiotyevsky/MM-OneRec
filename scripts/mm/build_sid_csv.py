#!/usr/bin/env python3
"""Convert atomic interaction splits into MiniOneRec-compatible SID CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

LOGGER = logging.getLogger("build_sid_csv")


def load_sid_map(path: Path) -> dict[str, str]:
    """Load an item-to-token-list index and concatenate every SID token."""
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    sid_map: dict[str, str] = {}
    for item_id, tokens in raw.items():
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"Invalid SID for item {item_id!r}: expected a non-empty token list")
        sid_map[str(item_id)] = "".join(map(str, tokens))
    return sid_map


def build_sid_csv(
    interactions_path: Path,
    index_path: Path,
    output_path: Path,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, int | str]:
    """Write SFT rows while preserving item order and complete deduplicated SIDs."""
    sid_map = load_sid_map(index_path)
    rows: list[dict[str, str]] = []
    missing_items: set[str] = set()
    with interactions_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required = {"user_id:token", "item_id_list:token_seq", "item_id:token"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected interaction header: {reader.fieldnames}")
        for row in tqdm(reader, desc=f"Building {output_path.name}"):
            history_ids = row["item_id_list:token_seq"].split()
            target_id = row["item_id:token"].strip()
            unknown = [item for item in [*history_ids, target_id] if item not in sid_map]
            if unknown:
                missing_items.update(unknown)
                continue
            history_sids = [sid_map[item] for item in history_ids]
            rows.append(
                {
                    "user_id": row["user_id:token"],
                    "history_item_id": json.dumps(history_ids),
                    "item_id": target_id,
                    "history_item_sid": json.dumps(history_sids),
                    "item_sid": sid_map[target_id],
                }
            )
    if max_samples is not None and max_samples > 0 and len(rows) > max_samples:
        rows = random.Random(seed).sample(rows, max_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["user_id", "history_item_id", "item_id", "history_item_sid", "item_sid"]
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    stats: dict[str, int | str] = {
        "rows": len(rows),
        "items_in_index": len(sid_map),
        "missing_item_count": len(missing_items),
        "output": str(output_path),
    }
    output_path.with_suffix(output_path.suffix + ".stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Wrote %d rows to %s", len(rows), output_path)
    if missing_items:
        LOGGER.warning("Skipped rows containing %d unknown items", len(missing_items))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    stats = build_sid_csv(args.interactions, args.index, args.output, args.max_samples, args.seed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
