"""Build text-only, image-only, or weighted multimodal item embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _pca(values: np.ndarray, dimension: int) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:dimension].T


def fuse_embeddings(
    text: np.ndarray,
    image: np.ndarray,
    mode: str = "weighted",
    alpha: float = 0.7,
    image_mask: np.ndarray | None = None,
    projection: str = "pca",
    target_dim: int | None = None,
) -> np.ndarray:
    """Fuse item-aligned embeddings, falling back to text for missing images."""
    if text.ndim != 2 or image.ndim != 2 or len(text) != len(image):
        raise ValueError(f"Expected aligned 2-D arrays, got text={text.shape}, image={image.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if mode == "text":
        return text.astype(np.float32)
    if mode == "image":
        return image.astype(np.float32)
    if text.shape[1] != image.shape[1]:
        if projection != "pca":
            raise ValueError("Embedding dimensions differ; use --projection pca")
        maximum = min(len(text) - 1, text.shape[1], image.shape[1])
        dimension = min(target_dim or maximum, maximum)
        if dimension < 1:
            raise ValueError("PCA projection needs at least two aligned items")
        text, image = _pca(text, dimension), _pca(image, dimension)
    text_norm, image_norm = _normalize(text), _normalize(image)
    fused = alpha * text_norm + (1.0 - alpha) * image_norm
    if image_mask is not None:
        mask = np.asarray(image_mask, dtype=bool)
        if mask.shape != (len(text),):
            raise ValueError(f"image mask must have shape {(len(text),)}, got {mask.shape}")
        fused[~mask] = text_norm[~mask]
    return _normalize(fused).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-embedding", type=Path, required=True)
    parser.add_argument("--image-embedding", type=Path, required=True)
    parser.add_argument("--image-mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fusion", choices=("text", "image", "weighted"), default="weighted")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--projection", choices=("pca", "none"), default="pca")
    parser.add_argument("--target-dim", type=int)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); np.random.seed(args.seed)
    text = np.load(args.text_embedding); image = np.load(args.image_embedding)
    mask = np.load(args.image_mask) if args.image_mask else None
    if args.max_items is not None:
        text, image = text[: args.max_items], image[: args.max_items]
        mask = mask[: args.max_items] if mask is not None else None
    fused = fuse_embeddings(text, image, args.fusion, args.alpha, mask, args.projection, args.target_dim)
    args.output.parent.mkdir(parents=True, exist_ok=True); np.save(args.output, fused)
    config = {"fusion": args.fusion, "alpha": args.alpha, "projection": args.projection, "target_dim": args.target_dim, "shape": list(fused.shape), "text_embedding": str(args.text_embedding), "image_embedding": str(args.image_embedding), "valid_images": int(mask.sum()) if mask is not None else None}
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
