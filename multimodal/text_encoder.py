"""Encode item title and description with a frozen text encoder."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

LOGGER = logging.getLogger("mm_onerec.text_encoder")


def item_text(record: dict, max_characters: int = 2000) -> str:
    """Build the same simple title + description content used for SID creation."""
    title = str(record.get("title") or "").strip()
    description = record.get("description") or ""
    if isinstance(description, list):
        description = " ".join(str(value) for value in description)
    description = str(description).strip()
    return f"Title: {title}. Description: {description}"[:max_characters]


def encode_texts(
    item_path: Path,
    output_path: Path,
    model_name: str = "google/siglip-base-patch16-224",
    batch_size: int = 64,
    device: str | None = None,
    max_items: int | None = None,
) -> np.ndarray:
    """Encode processed item JSON in insertion/key order and save a float32 matrix."""
    from transformers import AutoModel, AutoProcessor

    items = json.loads(item_path.read_text(encoding="utf-8"))
    rows = list(items.items())
    if max_items is not None:
        rows = rows[:max_items]
    texts = [item_text(record) for _, record in rows]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    vectors: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode text"):
        batch = texts[start : start + batch_size]
        inputs = processor(text=batch, padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            if not hasattr(model, "get_text_features"):
                raise TypeError(f"{model_name} does not expose get_text_features")
            encoded = model.get_text_features(**inputs).float().cpu().numpy()
        vectors.extend(encoded)
    matrix = np.asarray(vectors, dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix)
    metadata = {"model": model_name, "shape": list(matrix.shape), "item_file": str(item_path)}
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved text embeddings: %s", metadata)
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    encode_texts(args.items, args.output, args.model, args.batch_size, args.device, args.max_items)


if __name__ == "__main__":
    main()
