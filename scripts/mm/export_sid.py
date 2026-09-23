"""Export Semantic IDs from a trusted, locally trained RQ-VAE checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rq"))

from datasets import EmbDataset  # noqa: E402
from models.rqvae import RQVAE  # noqa: E402


def export_sid(data_path: Path, checkpoint_path: Path, output_path: Path, device: str, batch_size: int) -> dict:
    """Restore an RQ-VAE and export item-aligned hierarchical token strings."""
    dataset = EmbDataset(str(data_path))
    # The checkpoint is explicitly a locally trained artifact, not an untrusted download.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    model = RQVAE(
        in_dim=dataset.dim,
        num_emb_list=saved.num_emb_list,
        e_dim=saved.e_dim,
        layers=saved.layers,
        dropout_prob=saved.dropout_prob,
        bn=saved.bn,
        loss_type=saved.loss_type,
        quant_loss_weight=saved.quant_loss_weight,
        beta=saved.beta,
        kmeans_init=False,
        kmeans_iters=saved.kmeans_iters,
        sk_epsilons=saved.sk_epsilons,
        sk_iters=saved.sk_iters,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    prefixes = ("a", "b", "c", "d", "e")
    result: dict[str, list[str]] = {}
    offset = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="export SID"):
            codes = model.get_indices(batch.to(device), use_sk=False).reshape(len(batch), -1).cpu().numpy()
            for row in codes:
                result[str(offset)] = [f"<{prefixes[level]}_{int(code)}>" for level, code in enumerate(row)]
                offset += 1
    counter = Counter(tuple(value) for value in result.values())
    collisions = len(result) - len(counter)
    ordinal: Counter[tuple[str, ...]] = Counter()
    for item_id in sorted(result, key=int):
        raw = tuple(result[item_id])
        if counter[raw] > 1:
            ordinal[raw] += 1
            result[item_id].append(f"<d_{ordinal[raw]}>" )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output_unique = len({tuple(value) for value in result.values()})
    stats = {
        "total_items": len(result),
        "raw_unique_sid": len(counter),
        "raw_collision_count": collisions,
        "raw_collision_rate": collisions / len(result) if result else 0.0,
        "deduplicated_unique_sid": output_unique,
        "deduplicated_collision_rate": 1.0 - output_unique / len(result) if result else 0.0,
    }
    output_path.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(export_sid(args.data_path, args.checkpoint, args.output, args.device, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
