"""Convert MM-OneRec interaction CSV files to verl-ready parquet."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
from pathlib import Path
from typing import Any

from .reward import parse_sid_parts


def parse_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    text = str(value).strip()
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text.split()
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else [str(value)]


def make_prompt(history_sid: list[str], template: str) -> list[dict[str, str]]:
    content = template.format(history_sid=" ".join(history_sid), history=" ".join(history_sid))
    return [{"role": "user", "content": content}]


def convert_csv_to_parquet(
    interactions: Path,
    output: Path,
    *,
    dataset: str = "Amazon23",
    category: str = "Industrial_and_Scientific",
    prompt_template: str = "User history semantic IDs:\n{history_sid}\nPredict the next item semantic ID.",
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Write prompt/target fields without copying any embedding into parquet."""
    rows: list[dict[str, Any]] = []
    with interactions.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in {interactions}")
        for row in reader:
            history_sid = parse_list(row.get("history_item_sid"))
            history_items = parse_list(row.get("history_item_id"))
            target_sid = str(row.get("item_sid", "")).strip()
            target_item = str(row.get("item_id", "")).strip()
            if not history_sid or not target_sid:
                continue
            rows.append(
                {
                    "prompt": make_prompt(history_sid, prompt_template),
                    "target_sid": target_sid,
                    "target_sid_parts": parse_sid_parts(target_sid),
                    "target_item_id": target_item,
                    "history_sid": history_sid,
                    "history_item_ids": history_items,
                    "dataset": dataset,
                    "category": category,
                    "data_source": dataset,
                    "reward_model": {"ground_truth": target_sid},
                    "extra_info": {"target_item_id": target_item, "category": category},
                }
            )
    if max_samples is not None and max_samples > 0 and len(rows) > max_samples:
        rows = random.Random(seed).sample(rows, max_samples)
    if not rows:
        raise ValueError("No valid recommendation rows found")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError("Parquet conversion requires pyarrow") from exc
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output)
    metadata = {
        "rows": len(rows),
        "columns": table.column_names,
        "source": str(interactions),
        "dataset": dataset,
        "category": category,
        "seed": seed,
        "prompt_template": prompt_template,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="Amazon23")
    parser.add_argument("--category", default="Industrial_and_Scientific")
    parser.add_argument("--prompt-template", default="User history semantic IDs:\n{history_sid}\nPredict the next item semantic ID.")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(convert_csv_to_parquet(args.interactions, args.output, dataset=args.dataset, category=args.category, prompt_template=args.prompt_template, max_samples=args.max_samples, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
