"""Build a deterministic dense item subset from processed Amazon sequences."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def build_subset(
    interactions_path: Path,
    items_path: Path,
    output_dir: Path,
    dataset_name: str,
    max_items: int = 1000,
    min_user_interactions: int = 5,
) -> dict[str, int]:
    """Keep frequent items, filter short users, remap IDs, and leave-two-out split."""
    interactions = json.loads(interactions_path.read_text(encoding="utf-8"))
    items = json.loads(items_path.read_text(encoding="utf-8"))
    counts = Counter(str(item) for sequence in interactions.values() for item in sequence)
    selected = [item for item, _ in sorted(counts.items(), key=lambda pair: (-pair[1], int(pair[0])))[:max_items]]
    selected_set = set(selected)
    filtered = []
    for old_user, sequence in sorted(interactions.items(), key=lambda pair: int(pair[0])):
        kept = [str(item) for item in sequence if str(item) in selected_set]
        if len(kept) >= min_user_interactions:
            filtered.append((str(old_user), kept))
    used_items = {item for _, sequence in filtered for item in sequence}
    ordered_items = [item for item in selected if item in used_items]
    item_map = {old: new for new, old in enumerate(ordered_items)}
    user_map = {old: new for new, (old, _) in enumerate(filtered)}
    remapped = {
        str(user_map[user]): [item_map[item] for item in sequence]
        for user, sequence in filtered
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / dataset_name
    (prefix.with_suffix(".inter.json")).write_text(json.dumps(remapped, indent=2) + "\n", encoding="utf-8")
    new_items = {str(item_map[old]): items[old] for old in ordered_items}
    (prefix.with_suffix(".item.json")).write_text(json.dumps(new_items, indent=2) + "\n", encoding="utf-8")
    (prefix.with_suffix(".item-map.json")).write_text(json.dumps(item_map, indent=2) + "\n", encoding="utf-8")
    (prefix.with_suffix(".user-map.json")).write_text(json.dumps(user_map, indent=2) + "\n", encoding="utf-8")
    header = "user_id:token\titem_id_list:token_seq\titem_id:token\n"
    split_rows = {"train": [], "valid": [], "test": []}
    for user, sequence in remapped.items():
        split_rows["train"].append(f"{user}\t{' '.join(map(str, sequence[:-3]))}\t{sequence[-3]}\n")
        split_rows["valid"].append(f"{user}\t{' '.join(map(str, sequence[:-2]))}\t{sequence[-2]}\n")
        split_rows["test"].append(f"{user}\t{' '.join(map(str, sequence[:-1]))}\t{sequence[-1]}\n")
    for split, rows in split_rows.items():
        (output_dir / f"{dataset_name}.{split}.inter").write_text(header + "".join(rows), encoding="utf-8")
    stats = {
        "source_users": len(interactions),
        "source_items": len(items),
        "users": len(remapped),
        "items": len(new_items),
        "interactions": sum(map(len, remapped.values())),
        "min_user_interactions": min_user_interactions,
    }
    (output_dir / f"{dataset_name}.stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--max-items", type=int, default=1000)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    args = parser.parse_args()
    stats = build_subset(args.interactions, args.items, args.output_dir, args.dataset_name, args.max_items, args.min_user_interactions)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
