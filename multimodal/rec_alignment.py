"""Lightweight recommendation-aware alignment for frozen VLM item vectors.

The VLM remains frozen.  Only a small projector is optimized with next-item
signals from the training interaction split.  Validation and test files are
never read by this module.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

LOGGER = logging.getLogger("mm_onerec.rec_alignment")


def parse_id_sequence(value: object) -> list[str]:
    """Parse the list-like interaction fields used by Amazon preprocessing."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text.split()
    return [str(item) for item in parsed] if isinstance(parsed, (list, tuple)) else [str(parsed)]


def load_next_item_pairs(train_csv: Path, max_history: int = 10) -> list[tuple[list[str], str]]:
    """Read only train interactions and return ``(history_ids, next_id)`` pairs."""
    pairs: list[tuple[list[str], str]] = []
    with train_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in {train_csv}")
        history_key = next((key for key in ("history_item_id", "item_id_list:token_seq") if key in reader.fieldnames), None)
        target_key = next((key for key in ("item_id", "item_id:token") if key in reader.fieldnames), None)
        if history_key is None or target_key is None:
            raise ValueError(f"Could not find history/target columns in {reader.fieldnames}")
        for row in reader:
            history = parse_id_sequence(row.get(history_key))[-max_history:]
            target = str(row.get(target_key, "")).strip()
            if history and target:
                pairs.append((history, target))
    return pairs


def recency_pool(
    item_embeddings: np.ndarray | torch.Tensor,
    history_ids: Sequence[str],
    item_to_index: dict[str, int],
    decay: float = 0.0,
) -> np.ndarray | torch.Tensor:
    """Recency-weighted mean of known history item vectors."""
    indices = [item_to_index[item_id] for item_id in history_ids if item_id in item_to_index]
    if not indices:
        raise ValueError("history contains no item known to the representation matrix")
    if isinstance(item_embeddings, np.ndarray):
        vectors = item_embeddings[np.asarray(indices)]
        distances = np.arange(len(indices) - 1, -1, -1, dtype=np.float32)
        weights = np.exp(-float(decay) * distances) if decay else np.ones(len(indices), dtype=np.float32)
        return (vectors * (weights / weights.sum())[:, None]).sum(axis=0)
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=item_embeddings.device)
    vectors = item_embeddings.index_select(0, index_tensor)
    distances = torch.arange(len(indices) - 1, -1, -1, dtype=vectors.dtype, device=vectors.device)
    weights = torch.exp(-float(decay) * distances) if decay else torch.ones_like(distances)
    return (vectors * (weights / weights.sum()).unsqueeze(-1)).sum(dim=0)


class AlignmentProjector(nn.Module):
    """Small trainable map from frozen VLM space to recommendation space."""

    def __init__(self, input_dim: int, output_dim: int = 256, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(output_dim, min(1024, input_dim))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.network(values), dim=-1)


def info_nce_loss(
    user: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """In-batch positive plus sampled-negative InfoNCE objective."""
    if user.ndim != 2 or positive.ndim != 2 or negatives.ndim != 3:
        raise ValueError("Expected user [B,D], positive [B,D], negatives [B,N,D]")
    positive_score = (user * positive).sum(dim=-1, keepdim=True)
    negative_score = torch.einsum("bd,bnd->bn", user, negatives)
    logits = torch.cat([positive_score, negative_score], dim=1) / max(float(temperature), 1e-6)
    return torch.nn.functional.cross_entropy(logits, torch.zeros(user.shape[0], dtype=torch.long, device=user.device))


@dataclass(frozen=True)
class AlignmentConfig:
    output_dim: int = 256
    hidden_dim: int | None = None
    epochs: int = 3
    batch_size: int = 128
    negatives: int = 32
    temperature: float = 0.07
    learning_rate: float = 1e-3
    history_decay: float = 0.0
    seed: int = 42
    device: str | None = None


def _item_ids_from_args(item_ids_path: Path | None, items_path: Path | None, count: int) -> list[str]:
    if item_ids_path is not None:
        values = json.loads(item_ids_path.read_text(encoding="utf-8"))
        ids = [str(value) for value in values]
    elif items_path is not None:
        values = json.loads(items_path.read_text(encoding="utf-8"))
        ids = [str(key) for key in values]
    else:
        ids = [str(index) for index in range(count)]
    if len(ids) != count:
        raise ValueError(f"Expected {count} item IDs, got {len(ids)}")
    return ids


def train_alignment(
    embeddings_path: Path,
    train_csv: Path,
    output_path: Path,
    config: AlignmentConfig,
    *,
    item_ids_path: Path | None = None,
    items_path: Path | None = None,
) -> dict[str, object]:
    """Train the projector using train-only next-item supervision."""
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    raw = np.load(embeddings_path).astype(np.float32)
    if raw.ndim != 2:
        raise ValueError("item embeddings must be a 2-D .npy matrix")
    item_ids = _item_ids_from_args(item_ids_path, items_path, len(raw))
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    pairs = load_next_item_pairs(train_csv)
    pairs = [(history, target) for history, target in pairs if target in item_to_index and any(item in item_to_index for item in history)]
    if not pairs:
        raise ValueError("No usable train-only next-item pairs found")
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base = torch.from_numpy(raw).to(device)
    projector = AlignmentProjector(raw.shape[1], config.output_dim, config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=config.learning_rate)
    generator = random.Random(config.seed)
    all_indices = list(range(len(raw)))
    loss_history: list[float] = []
    projector.train()
    for epoch in range(config.epochs):
        shuffled = list(pairs); generator.shuffle(shuffled)
        epoch_losses: list[float] = []
        for start in tqdm(range(0, len(shuffled), max(1, config.batch_size)), desc=f"alignment epoch {epoch + 1}"):
            batch_pairs = shuffled[start : start + config.batch_size]
            users: list[torch.Tensor] = []
            positives: list[torch.Tensor] = []
            negatives: list[torch.Tensor] = []
            for history, target in batch_pairs:
                users.append(recency_pool(base, history, item_to_index, config.history_decay))
                positives.append(base[item_to_index[target]])
                forbidden = {item_to_index[target], *(item_to_index[item] for item in history if item in item_to_index)}
                candidates = [index for index in all_indices if index not in forbidden]
                if not candidates:
                    candidates = all_indices
                sampled = [candidates[index % len(candidates)] for index in generator.sample(range(len(candidates)), min(config.negatives, len(candidates)))] if len(candidates) >= config.negatives else [candidates[index % len(candidates)] for index in range(config.negatives)]
                negatives.append(base[torch.as_tensor(sampled, dtype=torch.long, device=device)])
            user_z = projector(torch.stack(users))
            positive_z = projector(torch.stack(positives))
            negative_z = projector(torch.stack(negatives))
            loss = info_nce_loss(user_z, positive_z, negative_z, config.temperature)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        loss_history.append(mean_loss)
        LOGGER.info("alignment epoch=%d loss=%.6f", epoch + 1, mean_loss)
    projector.eval()
    with torch.inference_mode():
        aligned = projector(base).cpu().numpy().astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, aligned)
    torch.save(projector.state_dict(), output_path.with_suffix(".projector.pt"))
    metadata: dict[str, object] = {
        "track": "Qwen3VL-RecAlign-SID",
        "source_embeddings": str(embeddings_path),
        "behavior_source": str(train_csv),
        "behavior_split": "train_only",
        "input_dim": int(raw.shape[1]),
        "output_dim": int(aligned.shape[1]),
        "pairs": len(pairs),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "negatives": config.negatives,
        "temperature": config.temperature,
        "seed": config.seed,
        "loss_history": loss_history,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--item-ids", type=Path)
    parser.add_argument("--items", type=Path)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--negatives", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--history-decay", type=float, default=0.0)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AlignmentConfig(
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        negatives=args.negatives,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        history_decay=args.history_decay,
        seed=args.seed,
        device=args.device,
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(train_alignment(args.embeddings, args.train_csv, args.output, config, item_ids_path=args.item_ids, items_path=args.items), indent=2))


if __name__ == "__main__":
    main()
