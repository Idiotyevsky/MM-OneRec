"""Frozen Qwen3-VL item representations.

This module deliberately keeps the VLM on the item side.  It does not alter
the Qwen generator used by the recommendation task and it never fine-tunes
the VLM.  The public pooling functions are dependency-light so they can be
tested with a tiny mock model on CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger("mm_onerec.qwen3_vl")
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_as_text(item) for item in value)
    return str(value).strip()


def build_item_prompt(record: dict[str, Any], include_optional: bool = False) -> str:
    """Build the deterministic text prompt used for every item representation."""
    title = _as_text(record.get("title"))
    description = _as_text(record.get("description"))
    lines = [
        "Represent this product for recommendation.",
        "Focus on category, function, appearance, material, style and product attributes.",
        f"Title: {title}",
        f"Description: {description}",
    ]
    if include_optional:
        for field in ("category", "categories", "brand", "attributes"):
            value = record.get(field)
            if value not in (None, "", [], {}):
                lines.append(f"{field.title()}: {_as_text(value)}")
    lines.append("Product representation:")
    return "\n".join(lines)


def pool_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last valid token for each sequence.

    ``attention_mask`` is used instead of sequence length assumptions, which
    keeps left/right padding and mixed-length batches aligned.
    """
    if hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden_states must be [batch, seq, dim] and attention_mask [batch, seq]")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("hidden_states and attention_mask sequence dimensions differ")
    positions = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
    masked_positions = torch.where(attention_mask.bool(), positions, torch.full_like(positions, -1))
    lengths = masked_positions.max(dim=1).values.clamp_min(0)
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, lengths]


def pool_text_mean(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    text_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked mean pooling, optionally excluding image/special positions."""
    if hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden_states must be [batch, seq, dim] and attention_mask [batch, seq]")
    mask = attention_mask.bool() if text_token_mask is None else text_token_mask.bool() & attention_mask.bool()
    if mask.shape != attention_mask.shape or hidden_states.shape[:2] != mask.shape:
        raise ValueError("pooling masks must match hidden_states sequence dimensions")
    weights = mask.to(hidden_states.dtype).unsqueeze(-1)
    return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["item_id"])] = row
    return rows


def _load_items(path: Path, max_items: int | None = None) -> list[tuple[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected item JSON object, got {type(payload).__name__}")
    rows = [(str(item_id), record) for item_id, record in payload.items() if isinstance(record, dict)]
    return rows[:max_items] if max_items is not None else rows


def _to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _to_device(value, device) for key, value in batch.items()}
    return batch


def _last_hidden(outputs: Any) -> torch.Tensor:
    states = getattr(outputs, "hidden_states", None)
    if states is not None:
        return states[-1]
    value = getattr(outputs, "last_hidden_state", None)
    if value is None:
        raise ValueError("Qwen3-VL output does not contain hidden_states or last_hidden_state")
    return value


def _image_token_ids(model: Any, processor: Any) -> set[int]:
    ids: set[int] = set()
    config = getattr(model, "config", None)
    for name in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
        value = getattr(config, name, None)
        if isinstance(value, int):
            ids.add(value)
    tokenizer = getattr(processor, "tokenizer", processor)
    for token in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
        try:
            value = tokenizer.convert_tokens_to_ids(token)
        except (AttributeError, TypeError):
            value = None
        if isinstance(value, int) and value >= 0:
            ids.add(value)
    return ids


@dataclass(frozen=True)
class Qwen3VLConfig:
    model_name: str = DEFAULT_MODEL
    pooling: str = "last_token"
    projection: str = "none"
    target_dim: int | None = None
    batch_size: int = 1
    device: str | None = None
    dtype: str = "bfloat16"
    include_optional: bool = False
    normalize: bool = True


class Qwen3VLItemEncoder:
    """Encode aligned item records with a frozen Qwen3-VL model."""

    def __init__(
        self,
        config: Qwen3VLConfig,
        *,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if config.pooling not in {"last_token", "text_mean"}:
            raise ValueError("pooling must be 'last_token' or 'text_mean'")
        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if processor is None or model is None:
            try:
                from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            except Exception as exc:  # pragma: no cover - depends on local transformers build
                raise RuntimeError(
                    "Qwen3-VL loading requires a transformers build with Qwen3-VL support "
                    "and a torch-compatible DTensor implementation"
                ) from exc
            processor = AutoProcessor.from_pretrained(config.model_name)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.model_name,
                torch_dtype=self._dtype(config.dtype),
            )
        self.processor = processor
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._image_ids = _image_token_ids(self.model, self.processor)

    @staticmethod
    def _dtype(name: str) -> torch.dtype | str:
        if name == "auto":
            return "auto"
        if name in {"float16", "fp16"}:
            return torch.float16
        if name in {"float32", "fp32"}:
            return torch.float32
        return torch.bfloat16

    def _encode_batch(self, texts: Sequence[str], image_paths: Sequence[str | None]) -> torch.Tensor:
        has_images = any(path is not None for path in image_paths)
        images: list[Image.Image] = []
        try:
            if has_images and not all(path is not None for path in image_paths):
                raise ValueError("A batch must contain either all images or no images")
            if has_images:
                images = [Image.open(str(path)).convert("RGB") for path in image_paths]
                batch = self.processor(text=list(texts), images=images, padding=True, return_tensors="pt")
            else:
                batch = self.processor(text=list(texts), padding=True, return_tensors="pt")
            batch = _to_device(batch, self.device)
            with torch.inference_mode():
                outputs = self.model(**batch, output_hidden_states=True, return_dict=True)
            hidden = _last_hidden(outputs)
            attention = batch.get("attention_mask") if isinstance(batch, dict) else getattr(batch, "attention_mask", None)
            if attention is None:
                attention = torch.ones(hidden.shape[:2], dtype=torch.long, device=hidden.device)
            if self.config.pooling == "last_token":
                pooled = pool_last_token(hidden, attention)
            else:
                text_mask = None
                input_ids = batch.get("input_ids") if isinstance(batch, dict) else getattr(batch, "input_ids", None)
                if input_ids is not None and self._image_ids:
                    text_mask = torch.ones_like(attention, dtype=torch.bool)
                    for image_id in self._image_ids:
                        text_mask &= input_ids != image_id
                pooled = pool_text_mean(hidden, attention, text_mask)
            return pooled.float().cpu()
        finally:
            for image in images:
                image.close()

    def encode_records(
        self,
        records: Sequence[dict[str, Any]],
        image_paths: Sequence[str | None],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(embeddings, has_image)`` without changing item order."""
        if len(records) != len(image_paths):
            raise ValueError("records and image_paths must have identical length")
        texts = [build_item_prompt(record, self.config.include_optional) for record in records]
        output: list[torch.Tensor | None] = [None] * len(records)
        groups = {
            True: [idx for idx, path in enumerate(image_paths) if path is not None],
            False: [idx for idx, path in enumerate(image_paths) if path is None],
        }
        for has_image, indices in groups.items():
            for start in range(0, len(indices), max(1, self.config.batch_size)):
                batch_indices = indices[start : start + self.config.batch_size]
                vectors = self._encode_batch(
                    [texts[idx] for idx in batch_indices],
                    [image_paths[idx] for idx in batch_indices] if has_image else [None] * len(batch_indices),
                )
                for idx, vector in zip(batch_indices, vectors, strict=True):
                    output[idx] = vector
        if any(vector is None for vector in output):
            raise RuntimeError("Qwen3-VL failed to produce an embedding for an item")
        matrix = torch.stack([vector for vector in output if vector is not None]).numpy().astype(np.float32)
        if self.config.normalize:
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        return matrix, np.asarray([path is not None for path in image_paths], dtype=bool)


def _project(matrix: np.ndarray, projection: str, target_dim: int | None, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    if projection == "none":
        if target_dim is not None and target_dim != matrix.shape[1]:
            raise ValueError("projection=none cannot change the hidden dimension; use projection=pca")
        return matrix, {"projection": "none", "raw_dim": int(matrix.shape[1]), "projected_dim": int(matrix.shape[1])}
    if projection != "pca":
        raise ValueError("projection must be 'none' or 'pca'")
    if target_dim is None:
        raise ValueError("projection=pca requires --target-dim")
    from sklearn.decomposition import PCA

    components = min(int(target_dim), matrix.shape[0], matrix.shape[1])
    if components < 1:
        raise ValueError("PCA requires a non-empty matrix")
    pca = PCA(n_components=components, random_state=seed)
    projected = pca.fit_transform(matrix).astype(np.float32)
    return projected, {
        "projection": "pca",
        "raw_dim": int(matrix.shape[1]),
        "projected_dim": int(projected.shape[1]),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }


def encode_item_json(
    item_path: Path,
    manifest_path: Path,
    output_path: Path,
    config: Qwen3VLConfig,
    seed: int = 42,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Encode item JSON in insertion order and save ``.npy`` plus metadata."""
    rows = _load_items(item_path, max_items)
    manifest = _load_manifest(manifest_path)
    item_ids = [item_id for item_id, _ in rows]
    records = [record for _, record in rows]
    paths = [
        (manifest.get(item_id, {}).get("image_path") if manifest.get(item_id, {}).get("image_path") and Path(manifest.get(item_id, {}).get("image_path")).is_file() else None)
        for item_id in item_ids
    ]
    encoder = Qwen3VLItemEncoder(config)
    raw, has_image = encoder.encode_records(records, paths)
    matrix, projection_info = _project(raw, config.projection, config.target_dim, seed)
    if config.normalize:
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix.astype(np.float32))
    np.save(output_path.with_name(output_path.stem + ".has_image.npy"), has_image)
    output_path.with_name(output_path.stem + ".item_ids.json").write_text(json.dumps(item_ids, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "model_name": config.model_name,
        "pooling": config.pooling,
        "normalize": config.normalize,
        "include_optional": config.include_optional,
        "shape": list(matrix.shape),
        "has_image_count": int(has_image.sum()),
        "missing_image_count": int((~has_image).sum()),
        "item_path": str(item_path),
        "manifest_path": str(manifest_path),
        **projection_info,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved Qwen3-VL item representations: %s", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pooling", choices=("last_token", "text_mean"), default="last_token")
    parser.add_argument("--projection", choices=("none", "pca"), default="none")
    parser.add_argument("--target-dim", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = Qwen3VLConfig(
        model_name=args.model,
        pooling=args.pooling,
        projection=args.projection,
        target_dim=args.target_dim,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        include_optional=args.include_optional,
        normalize=not args.no_normalize,
    )
    print(json.dumps(encode_item_json(args.items, args.manifest, args.output, config, args.seed, args.max_items), indent=2))


if __name__ == "__main__":
    main()
