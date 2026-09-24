#!/usr/bin/env python3
"""Analyze a trained RQ-VAE with the final nearest-neighbor SID assignment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rq"))

from diagnostics import (  # noqa: E402
    apply_zscore,
    compute_codebook_diagnostics,
    embedding_statistics,
    load_zscore_stats,
    prefix_examples,
)
from models.rqvae import RQVAE  # noqa: E402


class EmbeddingDataset(Dataset[torch.Tensor]):
    """Small item-aligned dataset used for deterministic diagnostics."""

    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.embeddings[index]


def _load_meta(path: Path | None) -> tuple[list[str], dict[str, Any] | list[dict[str, Any]] | None]:
    if path is None:
        return [], None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return [str(key) for key in raw.keys()], raw
    if isinstance(raw, list):
        return [str(i) for i in range(len(raw))], raw
    raise ValueError(f"Unsupported item metadata format: {path}")


def _model_from_checkpoint(checkpoint: dict[str, Any], input_dim: int) -> RQVAE:
    saved = checkpoint["args"]
    model = RQVAE(
        in_dim=input_dim,
        num_emb_list=list(saved.num_emb_list),
        e_dim=int(saved.e_dim),
        layers=list(saved.layers),
        dropout_prob=float(saved.dropout_prob),
        bn=bool(saved.bn),
        loss_type=str(saved.loss_type),
        quant_loss_weight=float(saved.quant_loss_weight),
        beta=float(saved.beta),
        kmeans_init=False,
        kmeans_iters=int(saved.kmeans_iters),
        sk_epsilons=list(saved.sk_epsilons),
        sk_iters=int(saved.sk_iters),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _flatten_row(metrics: dict[str, Any], *, experiment: str, embedding: str, preprocessing: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment": experiment,
        "embedding": embedding,
        "preprocessing": preprocessing,
        "codebooks": "/".join(map(str, metrics["codebooks"])),
        "latent_dim": metrics["latent_dim"],
        "beta": metrics.get("beta", ""),
        "quant_loss_weight": metrics.get("quant_loss_weight", ""),
        "sk_epsilons": metrics.get("sk_epsilons", ""),
        "best_epoch": metrics.get("epoch", ""),
        "reconstruction_loss": metrics["reconstruction_loss"],
        "raw_unique_sid": metrics["prefix_diversity"]["raw_unique_sid"],
        "collision_count": metrics["prefix_diversity"]["collision_count"],
        "collision_rate": metrics["prefix_diversity"]["collision_rate"],
    }
    for layer in metrics["layers"]:
        prefix = f"L{layer['layer']}"
        for key in ("used_codes", "utilization_rate", "entropy", "normalized_entropy", "perplexity", "most_frequent_code_ratio", "top5_code_ratio"):
            row[f"{prefix}_{key}"] = layer[key]
    for depth in range(1, metrics["num_layers"] + 1):
        row[f"prefix{depth}_unique"] = metrics["prefix_diversity"].get(f"unique_prefix@{depth}", 0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--preprocess", choices=("none", "zscore"), default="none")
    parser.add_argument("--preprocess-stats", type=Path)
    parser.add_argument("--item-meta", type=Path)
    parser.add_argument("--item-ids", type=Path)
    parser.add_argument("--prefix-examples", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_embeddings = np.load(args.embedding).astype(np.float32, copy=False)
    raw_stats = embedding_statistics(raw_embeddings)
    embeddings = raw_embeddings
    transformed_stats = None
    if args.preprocess == "zscore":
        if args.preprocess_stats is None:
            raise ValueError("--preprocess-stats is required for --preprocess zscore")
        embeddings = apply_zscore(raw_embeddings, load_zscore_stats(args.preprocess_stats))
        transformed_stats = embedding_statistics(embeddings)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _model_from_checkpoint(checkpoint, embeddings.shape[1])
    model.to(args.device)
    metrics = compute_codebook_diagnostics(
        model,
        EmbeddingDataset(embeddings),
        device=args.device,
        batch_size=args.batch_size,
    )
    metrics.update({
        "checkpoint": str(args.checkpoint),
        "embedding": str(args.embedding),
        "preprocess": args.preprocess,
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_collision_rate_checkpoint": checkpoint.get("best_collision_rate"),
        "beta": float(getattr(checkpoint["args"], "beta", 0.25)),
        "quant_loss_weight": float(getattr(checkpoint["args"], "quant_loss_weight", 1.0)),
        "sk_epsilons": list(getattr(checkpoint["args"], "sk_epsilons", [])),
        "embedding_diagnostics": {
            "raw": raw_stats,
            "preprocessed": transformed_stats,
        },
    })

    item_ids, item_meta = _load_meta(args.item_meta)
    if args.item_ids is not None:
        item_ids = [str(x) for x in json.loads(args.item_ids.read_text(encoding="utf-8"))]
    if item_meta is not None and len(item_ids) == len(embeddings):
        # Recompute codes once for compact prefix examples; diagnostics itself
        # intentionally does not retain all item codes in memory.
        loader = EmbeddingDataset(embeddings)
        all_codes = []
        with torch.inference_mode():
            for start in range(0, len(loader), args.batch_size):
                batch = loader.embeddings[start : start + args.batch_size].to(args.device)
                all_codes.append(model.get_indices(batch, use_sk=False).reshape(len(batch), -1).cpu().numpy())
        codes = np.concatenate(all_codes, axis=0)
        metrics["prefix_examples"] = prefix_examples(
            codes, embeddings, item_ids, item_meta, max_prefixes=args.prefix_examples
        )

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "embedding_stats.json").write_text(json.dumps(metrics["embedding_diagnostics"], indent=2) + "\n", encoding="utf-8")
    if "prefix_examples" in metrics:
        (args.output_dir / "prefix_examples.json").write_text(json.dumps(metrics["prefix_examples"], indent=2) + "\n", encoding="utf-8")

    row = _flatten_row(metrics, experiment=args.output_dir.name, embedding=args.embedding.name, preprocessing=args.preprocess)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps({
        "checkpoint": str(args.checkpoint),
        "epoch": metrics["epoch"],
        "reconstruction_loss": metrics["reconstruction_loss"],
        "raw_unique_sid": metrics["prefix_diversity"]["raw_unique_sid"],
        "collision_count": metrics["prefix_diversity"]["collision_count"],
        "collision_rate": metrics["prefix_diversity"]["collision_rate"],
        "layers": metrics["layers"],
        "prefix_diversity": metrics["prefix_diversity"],
    }, indent=2))


if __name__ == "__main__":
    main()
