"""Diagnostics for RQ-VAE codebook utilization and item representations.

All SID statistics in this module use nearest-neighbor assignment
(`use_sk=False`) so they match the repository's final SID export path.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _finite_float(value: float) -> float:
    """Return a JSON-safe finite float."""
    value = float(value)
    return value if math.isfinite(value) else 0.0


def _distribution_stats(counts: np.ndarray) -> dict[str, float | int]:
    """Summarize one discrete code distribution."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    probs = counts[counts > 0].astype(np.float64) / max(total, 1)
    entropy = float(-(probs * np.log(probs)).sum()) if len(probs) else 0.0
    total_codes = int(len(counts))
    top = np.sort(counts)[::-1]
    return {
        "used_codes": int((counts > 0).sum()),
        "total_codes": total_codes,
        "utilization_rate": _finite_float((counts > 0).mean() if total_codes else 0.0),
        "entropy": _finite_float(entropy),
        "normalized_entropy": _finite_float(entropy / math.log(total_codes))
        if total_codes > 1
        else 0.0,
        "perplexity": _finite_float(math.exp(entropy)),
        "most_frequent_code_ratio": _finite_float(top[0] / max(total, 1)),
        "top5_code_ratio": _finite_float(top[:5].sum() / max(total, 1)),
    }


def embedding_statistics(
    embeddings: np.ndarray,
    *,
    pairwise_sample_size: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute finite-value, norm, per-dimension, and cosine diagnostics."""
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected [num_items, dim] embedding matrix, got {array.shape}")
    finite = np.isfinite(array)
    norms = np.linalg.norm(np.nan_to_num(array), axis=1)
    result: dict[str, Any] = {
        "num_items": int(array.shape[0]),
        "dim": int(array.shape[1]),
        "dtype": str(array.dtype),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "non_finite_count": int((~finite).sum()),
        "norm": {
            "mean": _finite_float(norms.mean()),
            "std": _finite_float(norms.std()),
            "min": _finite_float(norms.min()) if len(norms) else 0.0,
            "max": _finite_float(norms.max()) if len(norms) else 0.0,
            "zero_count": int((norms == 0).sum()),
        },
        "per_dimension_mean": np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        .mean(axis=0)
        .astype(float)
        .tolist(),
        "per_dimension_std": np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        .std(axis=0)
        .astype(float)
        .tolist(),
    }
    if len(array) < 2:
        result["pairwise_cosine"] = {"pairs": 0}
        return result
    rng = np.random.default_rng(seed)
    num_pairs = min(int(pairwise_sample_size), max(1, len(array) * (len(array) - 1) // 2))
    left = rng.integers(0, len(array), size=num_pairs)
    right = rng.integers(0, len(array), size=num_pairs)
    same = left == right
    right[same] = (right[same] + 1) % len(array)
    safe = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = safe / np.maximum(np.linalg.norm(safe, axis=1, keepdims=True), 1e-12)
    cosine = np.sum(normalized[left] * normalized[right], axis=1)
    result["pairwise_cosine"] = {
        "pairs": int(len(cosine)),
        "mean": _finite_float(cosine.mean()),
        "std": _finite_float(cosine.std()),
        "min": _finite_float(cosine.min()),
        "max": _finite_float(cosine.max()),
        "p05": _finite_float(np.quantile(cosine, 0.05)),
        "p50": _finite_float(np.quantile(cosine, 0.50)),
        "p95": _finite_float(np.quantile(cosine, 0.95)),
        "fraction_gt_0999": _finite_float((cosine > 0.999).mean()),
    }
    return result


def fit_zscore(embeddings: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit item-content-only z-score statistics and transform embeddings."""
    array = np.asarray(embeddings, dtype=np.float32)
    mean = array.mean(axis=0, dtype=np.float64)
    std = array.std(axis=0, dtype=np.float64)
    safe_std = np.maximum(std, float(eps))
    transformed = ((array.astype(np.float64) - mean) / safe_std).astype(np.float32)
    stats = {
        "mode": "zscore",
        "eps": float(eps),
        "num_items": int(array.shape[0]),
        "dim": int(array.shape[1]),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }
    return transformed, stats


def apply_zscore(embeddings: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    """Apply saved z-score statistics to an embedding matrix."""
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    eps = float(stats.get("eps", 1e-6))
    if embeddings.shape[-1] != len(mean):
        raise ValueError("Embedding dimension does not match saved z-score statistics")
    return ((np.asarray(embeddings, dtype=np.float64) - mean) / np.maximum(std, eps)).astype(np.float32)


def save_zscore_stats(stats: Mapping[str, Any], path: str | Path) -> None:
    """Save reproducible z-score statistics as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(stats), indent=2) + "\n", encoding="utf-8")


def load_zscore_stats(path: str | Path) -> dict[str, Any]:
    """Load z-score statistics saved by :func:`save_zscore_stats`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _prefix_diversity(codes: np.ndarray) -> dict[str, int]:
    """Count unique hierarchical prefixes and complete raw codes."""
    result: dict[str, int] = {}
    for depth in range(1, codes.shape[1] + 1):
        result[f"unique_prefix@{depth}"] = int(np.unique(codes[:, :depth], axis=0).shape[0])
    result["raw_unique_sid"] = int(np.unique(codes, axis=0).shape[0])
    result["collision_count"] = int(len(codes) - result["raw_unique_sid"])
    result["collision_rate"] = _finite_float(result["collision_count"] / max(len(codes), 1))
    return result


def compute_codebook_diagnostics(
    model: torch.nn.Module,
    dataset: Dataset[torch.Tensor],
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 1024,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Evaluate reconstruction and nearest-neighbor codebook usage.

    The assignment is always made with ``use_sk=False``.  Thus the returned
    collision and utilization values are the same assignment semantics used by
    SID export, even when training used Sinkhorn assignments.
    """
    model.eval()
    target_device = torch.device(device)
    # The legacy trainer passes its training DataLoader here. Reuse it when
    # available so diagnostics remain compatible with the existing entry point.
    loader = dataset if isinstance(dataset, DataLoader) else DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    all_codes: list[np.ndarray] = []
    recon_sum = 0.0
    element_count = 0
    quant_sum = 0.0
    item_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(target_device)
            output, quant_loss, indices = model(batch, use_sk=False)
            recon_sum += float(F.mse_loss(output, batch, reduction="sum").item())
            element_count += int(batch.numel())
            quant_sum += float(quant_loss.item()) * len(batch)
            item_count += len(batch)
            all_codes.append(indices.reshape(len(batch), -1).detach().cpu().numpy())
    if not all_codes:
        raise ValueError("Cannot diagnose an empty embedding dataset")
    codes = np.concatenate(all_codes, axis=0).astype(np.int64, copy=False)
    prefix = _prefix_diversity(codes)
    layers: list[dict[str, Any]] = []
    num_emb_list: Sequence[int] = getattr(model.rq, "n_e_list", [int(codes[:, i].max() + 1) for i in range(codes.shape[1])])
    for layer, total_codes in enumerate(num_emb_list):
        counts = np.bincount(codes[:, layer], minlength=int(total_codes))
        stats = _distribution_stats(counts)
        stats["layer"] = int(layer + 1)
        layers.append(stats)
    reconstruction = recon_sum / max(element_count, 1)
    quantization = quant_sum / max(item_count, 1)
    return {
        "assignment": "nearest_neighbor_use_sk_false",
        "num_items": int(item_count),
        "num_layers": int(codes.shape[1]),
        "codebooks": [int(x) for x in num_emb_list],
        "latent_dim": int(getattr(model, "e_dim", 0)),
        "reconstruction_loss": _finite_float(reconstruction),
        "quantization_loss": _finite_float(quantization),
        "total_loss": _finite_float(reconstruction + float(getattr(model, "quant_loss_weight", 1.0)) * quantization),
        "layers": layers,
        "prefix_diversity": prefix,
    }


def _item_text(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Extract compact, JSON-safe item metadata for prefix inspection."""
    def first(*keys: str) -> Any:
        for key in keys:
            value = meta.get(key)
            if value not in (None, "", []):
                return value
        return ""
    return {
        "title": first("title", "item_title", "name"),
        "category": first("category", "categories", "main_category"),
        "brand": first("brand", "Brand"),
    }


def prefix_examples(
    codes: np.ndarray,
    embeddings: np.ndarray,
    item_ids: Sequence[str],
    item_meta: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    prefixes: Sequence[int] = (1, 2),
    max_prefixes: int = 5,
    max_items_per_prefix: int = 8,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return representative items sharing selected hierarchical prefixes."""
    if item_meta is None:
        return []
    if isinstance(item_meta, Mapping):
        metadata = {str(k): v for k, v in item_meta.items()}
    else:
        metadata = {str(i): v for i, v in enumerate(item_meta)}
    rng = np.random.default_rng(seed)
    tokens = ("a", "b", "c", "d", "e")
    normalized = np.asarray(embeddings, dtype=np.float32)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    output: list[dict[str, Any]] = []
    for depth in prefixes:
        groups: dict[tuple[int, ...], list[int]] = {}
        for row, code in enumerate(codes[:, :depth]):
            groups.setdefault(tuple(int(x) for x in code), []).append(row)
        eligible = [key for key, rows in groups.items() if len(rows) > 1]
        if not eligible:
            continue
        selected = eligible if len(eligible) <= max_prefixes else [eligible[i] for i in rng.choice(len(eligible), max_prefixes, replace=False)]
        for key in selected:
            rows = groups[key]
            centroid = normalized[rows].mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            rows = sorted(rows, key=lambda row: float(np.dot(normalized[row], centroid)), reverse=True)[:max_items_per_prefix]
            prefix = "".join(f"<{tokens[i]}_{value}>" for i, value in enumerate(key))
            items = []
            for row in rows:
                item_id = str(item_ids[row])
                meta = metadata.get(item_id, {})
                items.append({
                    "item_id": item_id,
                    "sid_prefix": prefix,
                    "sid": "".join(f"<{tokens[i]}_{value}>" for i, value in enumerate(codes[row])),
                    "cosine_to_prefix_centroid": _finite_float(float(np.dot(normalized[row], centroid))),
                    **_item_text(meta if isinstance(meta, Mapping) else {}),
                })
            output.append({"depth": int(depth), "prefix": prefix, "items": items})
    return output
