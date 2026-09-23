"""Report Semantic-ID collision statistics for an index JSON file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def collision_stats(index_path: Path) -> dict[str, float | int]:
    indices = json.loads(index_path.read_text(encoding="utf-8"))
    codes = [tuple(value) for value in indices.values()]
    counts = Counter(codes)
    collisions = sum(count - 1 for count in counts.values())
    return {"total_items": len(codes), "unique_sid": len(counts), "collision_count": collisions, "collision_rate": collisions / len(codes) if codes else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stats = collision_stats(args.index)
    rendered = json.dumps(stats, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
