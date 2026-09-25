"""Report Semantic-ID length and raw collision-group statistics."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

def stats(path: Path) -> dict[str, int | float | dict[str, int]]:
    index = json.loads(path.read_text(encoding="utf-8"))
    lengths = Counter(len(tokens) for tokens in index.values())
    raw_groups = Counter(tuple(tokens[:3]) for tokens in index.values())
    return {
        "items": len(index),
        "three_token_items": lengths.get(3, 0),
        "four_token_items": lengths.get(4, 0),
        "other_length_items": sum(v for k, v in lengths.items() if k not in (3, 4)),
        "three_token_proportion": lengths.get(3, 0) / len(index) if index else 0.0,
        "four_token_proportion": lengths.get(4, 0) / len(index) if index else 0.0,
        "average_sid_length": sum(k * v for k, v in lengths.items()) / len(index) if index else 0.0,
        "max_collision_group_size": max(raw_groups.values(), default=0),
        "raw_unique_sid": len(raw_groups),
        "raw_collision_count": sum(v - 1 for v in raw_groups.values() if v > 1),
        "raw_collision_rate": sum(v - 1 for v in raw_groups.values() if v > 1) / len(index) if index else 0.0,
        "suffix_token_count": len({token for tokens in index.values() for token in tokens[3:]}),
        "length_distribution": {str(k): v for k, v in sorted(lengths.items())},
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = stats(args.index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
if __name__ == "__main__":
    main()
