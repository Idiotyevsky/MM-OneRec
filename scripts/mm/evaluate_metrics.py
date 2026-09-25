"""Compute ranking, coverage, validity, and frequency-bucket metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data["predictions"]


def _buckets(train_items: list[str]) -> dict[str, str]:
    counts = Counter(train_items)
    ordered = sorted(counts, key=lambda item: (-counts[item], item))
    total = len(ordered)
    head_end, tail_start = math.ceil(total * 0.2), math.floor(total * 0.8)
    return {item: ("head" if rank < head_end else "tail" if rank >= tail_start else "mid") for rank, item in enumerate(ordered)}


def _raw_sid(sid: str) -> str:
    """Drop an optional collision-disambiguation suffix from a SID string."""
    return "".join(re.findall(r"<[abc]_\d+>", str(sid))[:3])


def evaluate(
    rows: list[dict[str, Any]],
    catalog: set[str],
    train_items: list[str],
    ks: tuple[int, ...] = (5, 10, 20),
    sid_index: dict[str, list[str]] | None = None,
) -> dict[str, float | int]:
    buckets = _buckets(train_items)
    collision_counts = Counter()
    if sid_index:
        collision_counts.update("".join(map(str, tokens[:3])) for tokens in sid_index.values())
    metrics: dict[str, float | int] = {"samples": len(rows)}
    recommended: set[str] = set(); invalid = total_predictions = 0
    for k in ks:
        hits = ndcgs = 0.0
        bucket_hits = Counter(); bucket_ndcgs = Counter(); bucket_totals = Counter()
        group_hits = Counter(); group_ndcgs = Counter(); group_totals = Counter()
        for row in rows:
            target = str(row["target"]); predictions = [str(value) for value in row["predictions"]][:k]
            bucket = buckets.get(target, "tail"); bucket_totals[bucket] += 1
            if collision_counts:
                group = "collision" if collision_counts.get(_raw_sid(target), 1) > 1 else "non_collision"
                group_totals[group] += 1
            if target in predictions:
                rank = predictions.index(target) + 1; gain = 1 / math.log2(rank + 1)
                hits += 1; ndcgs += gain
                bucket_hits[bucket] += 1; bucket_ndcgs[bucket] += gain
                if collision_counts:
                    group_hits[group] += 1; group_ndcgs[group] += gain
        metrics[f"hr@{k}"] = hits / len(rows) if rows else 0.0
        metrics[f"ndcg@{k}"] = ndcgs / len(rows) if rows else 0.0
        if collision_counts and k == 10:
            for group in ("collision", "non_collision"):
                denom = group_totals[group]
                metrics[f"hr@10_{group}"] = group_hits[group] / denom if denom else 0.0
                metrics[f"ndcg@10_{group}"] = group_ndcgs[group] / denom if denom else 0.0
                metrics[f"samples_{group}"] = denom
        if k == 10:
            for bucket in ("head", "mid", "tail"):
                denom = bucket_totals[bucket]
                metrics[f"hr@10_{bucket}"] = bucket_hits[bucket] / denom if denom else 0.0
                metrics[f"ndcg@10_{bucket}"] = bucket_ndcgs[bucket] / denom if denom else 0.0
                metrics[f"samples_{bucket}"] = denom
            # Keep concise aliases for the README/summary schema.
            metrics["tail_hr@10"] = metrics["hr@10_tail"]
            metrics["tail_ndcg@10"] = metrics["ndcg@10_tail"]
    for row in rows:
        predictions = [str(value) for value in row["predictions"]]
        recommended.update(value for value in predictions if value in catalog)
        invalid += sum(value not in catalog for value in predictions); total_predictions += len(predictions)
    metrics["coverage"] = len(recommended) / len(catalog) if catalog else 0.0
    metrics["invalid_sid_rate"] = invalid / total_predictions if total_predictions else 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="JSON/JSONL rows: target + predictions list")
    parser.add_argument("--catalog", type=Path, required=True, help="One valid item/SID per line")
    parser.add_argument("--train-items", type=Path, required=True, help="One interacted item/SID per line, duplicates retained")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--model-name", default="unnamed")
    parser.add_argument("--sid-index", type=Path, help="Optional item-to-SID JSON for collision-group metrics")
    args = parser.parse_args()
    sid_index = json.loads(args.sid_index.read_text(encoding="utf-8")) if args.sid_index else None
    metrics = evaluate(
        _load_rows(args.predictions),
        set(args.catalog.read_text().splitlines()),
        args.train_items.read_text().splitlines(),
        sid_index=sid_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        # Rewrite the small summary atomically in one stable schema.  The old
        # append-only implementation produced rows wider than the header when
        # a later run exposed additional long-tail fields.
        base_fields = [
            "model", "status", "hr@5", "hr@10", "hr@20", "ndcg@5", "ndcg@10", "ndcg@20",
            "coverage", "tail_hr@10", "tail_ndcg@10", "invalid_sid_rate",
        ]
        existing: list[dict[str, Any]] = []
        if args.summary_csv.exists():
            with args.summary_csv.open(newline="", encoding="utf-8") as handle:
                existing = list(csv.DictReader(handle))
        existing = [row for row in existing if row.get("model") != args.model_name]
        row = {"model": args.model_name, "status": "evaluated", **metrics}
        fields = list(base_fields)
        for key in metrics:
            if key not in fields:
                fields.append(key)
        with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing)
            writer.writerow(row)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
