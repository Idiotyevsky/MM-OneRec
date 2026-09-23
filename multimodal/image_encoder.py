"""Encode cached product images with a frozen SigLIP/CLIP vision encoder."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger("mm_onerec.encoder")


def _histogram_features(images: list[Image.Image]) -> np.ndarray:
    """Small deterministic offline backend used only by smoke tests."""
    features = []
    for image in images:
        array = np.asarray(image.convert("RGB").resize((64, 64)))
        channels = [np.histogram(array[..., idx], bins=16, range=(0, 256), density=True)[0] for idx in range(3)]
        features.append(np.concatenate(channels).astype(np.float32))
    return np.stack(features)


def _load_manifest(path: Path, max_items: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:max_items] if max_items is not None else rows


def encode_images(
    manifest_path: Path,
    output_path: Path,
    mask_path: Path,
    model_name: str = "google/siglip-base-patch16-224",
    backend: str = "transformers",
    batch_size: int = 32,
    device: str | None = None,
    max_items: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode images in manifest order; missing images receive zeros plus a false mask."""
    rows = _load_manifest(manifest_path, max_items)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = model = None
    if backend == "transformers":
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
    vectors: list[np.ndarray | None] = [None] * len(rows)
    valid_indices = [idx for idx, row in enumerate(rows) if row.get("image_path") and Path(row["image_path"]).is_file()]
    for start in tqdm(range(0, len(valid_indices), batch_size), desc="encode images"):
        indices = valid_indices[start : start + batch_size]
        images = [Image.open(rows[idx]["image_path"]).convert("RGB") for idx in indices]
        try:
            if backend == "histogram":
                encoded = _histogram_features(images)
            else:
                assert processor is not None and model is not None
                inputs = processor(images=images, return_tensors="pt").to(device)
                with torch.inference_mode():
                    if hasattr(model, "get_image_features"):
                        tensor = model.get_image_features(**inputs)
                    else:
                        outputs = model(**inputs)
                        tensor = outputs.pooler_output
                encoded = tensor.float().cpu().numpy()
            for idx, vector in zip(indices, encoded, strict=True):
                vectors[idx] = vector
        finally:
            for image in images:
                image.close()
    dimension = next((len(vector) for vector in vectors if vector is not None), 48 if backend == "histogram" else 0)
    if dimension == 0:
        raise ValueError("No valid images were encoded; cannot infer output dimension")
    mask = np.asarray([vector is not None for vector in vectors], dtype=bool)
    matrix = np.stack([vector if vector is not None else np.zeros(dimension, dtype=np.float32) for vector in vectors]).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix)
    np.save(mask_path, mask)
    metadata = {"model": model_name, "backend": backend, "shape": list(matrix.shape), "valid": int(mask.sum()), "missing": int((~mask).sum())}
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved image embeddings: %s", metadata)
    return matrix, mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-output", type=Path, required=True)
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--backend", choices=("transformers", "histogram"), default="transformers")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    encode_images(args.manifest, args.output, args.mask_output, args.model, args.backend, args.batch_size, args.device, args.max_items)


if __name__ == "__main__":
    main()
