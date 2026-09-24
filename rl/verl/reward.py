"""Recommendation rewards for the native verl GRPO reward hook.

The module has no dependency on verl itself so reward ordering and SID parsing
can be tested on CPU.  ``compute_reward`` follows the common custom reward
function signature used by recent verl releases; extra keyword arguments are
accepted for compatibility across versions.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SID_TOKEN_RE = re.compile(r"(?:\[[^\[\]]+\]|<[^<>]+>)")


def parse_sid_parts(value: object) -> list[str]:
    """Parse angle/square SID tokens while retaining their order."""
    return SID_TOKEN_RE.findall(str(value).strip())


def semantic_sid_parts(value: object) -> list[str]:
    """Return only true RQ layers; collision suffix ``<d_n>`` is excluded."""
    parts = parse_sid_parts(value)
    return [part for part in parts if not re.match(r"^[\[<]d[_-]", part, flags=re.IGNORECASE)]


def normalize_sid(value: object) -> str:
    parts = parse_sid_parts(value)
    return "".join(parts) if parts else str(value).strip().splitlines()[0].strip()


def sid_hier_reward(predicted: object, target: object, weights: Sequence[float] = (0.5, 0.3, 0.2)) -> float:
    """Prefix-aware hierarchical SID reward, excluding disambiguation suffixes."""
    predicted_parts = semantic_sid_parts(predicted)
    target_parts = semantic_sid_parts(target)
    if not predicted_parts or not target_parts:
        return 0.0
    score = 0.0
    normalizer = float(sum(weights[: len(target_parts)])) or 1.0
    for index, weight in enumerate(weights[: len(target_parts)]):
        if index < len(predicted_parts) and predicted_parts[index] == target_parts[index]:
            score += float(weight)
        else:
            break
    return max(0.0, min(1.0, score / normalizer))


def cosine_reward(predicted_item: str, target_item: str, embeddings: np.ndarray, item_to_index: Mapping[str, int]) -> float:
    """Map cosine similarity to ``[0, 1]`` and return zero for invalid items."""
    if predicted_item not in item_to_index or target_item not in item_to_index:
        return 0.0
    predicted = embeddings[item_to_index[predicted_item]].astype(np.float32)
    target = embeddings[item_to_index[target_item]].astype(np.float32)
    denominator = max(float(np.linalg.norm(predicted) * np.linalg.norm(target)), 1e-12)
    cosine = float(np.dot(predicted, target) / denominator)
    return (max(-1.0, min(1.0, cosine)) + 1.0) / 2.0


def _sid_to_item(index: Mapping[str, Sequence[object]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item_id, tokens in index.items():
        sid = "".join(str(token) for token in tokens)
        mapping.setdefault(sid, str(item_id))
    return mapping


@dataclass
class RewardConfig:
    mode: str = "hybrid"
    lambda_sid: float = 0.6
    sid_weights: tuple[float, ...] = (0.5, 0.3, 0.2)
    index_path: str | None = None
    embedding_path: str | None = None
    item_ids_path: str | None = None


class RecommendationReward:
    """Callable reward object implementing exact/sid_hier/semantic/hybrid."""

    def __init__(self, config: RewardConfig, *, index: Mapping[str, Sequence[object]] | None = None, embeddings: np.ndarray | None = None, item_ids: Sequence[str] | None = None) -> None:
        if config.mode not in {"exact", "sid_hier", "semantic", "hybrid"}:
            raise ValueError("reward mode must be exact, sid_hier, semantic, or hybrid")
        if not 0.0 <= config.lambda_sid <= 1.0:
            raise ValueError("lambda_sid must be in [0, 1]")
        self.config = config
        self.index = dict(index or {})
        self.sid_to_item = _sid_to_item(self.index)
        self.embeddings = embeddings
        self.item_to_index = {str(item): index for index, item in enumerate(item_ids or [])}

    @classmethod
    def from_config(cls, config: RewardConfig) -> "RecommendationReward":
        index = json.loads(Path(config.index_path).read_text(encoding="utf-8")) if config.index_path else {}
        embeddings = np.load(config.embedding_path) if config.embedding_path else None
        item_ids = json.loads(Path(config.item_ids_path).read_text(encoding="utf-8")) if config.item_ids_path else list(index.keys())
        return cls(config, index=index, embeddings=embeddings, item_ids=item_ids)

    def components(self, predicted_sid: object, target_sid: object, target_item_id: str | None = None) -> dict[str, float]:
        """Return auditable exact/hierarchical/semantic/final reward pieces."""
        predicted = normalize_sid(predicted_sid)
        target = normalize_sid(target_sid)
        exact = float(bool(predicted) and predicted == target)
        hierarchical = 1.0 if exact else sid_hier_reward(predicted, target, self.config.sid_weights)
        predicted_item = self.sid_to_item.get(predicted)
        semantic = 0.0
        if self.embeddings is not None and target_item_id is not None and predicted_item is not None:
            semantic = cosine_reward(predicted_item, str(target_item_id), self.embeddings, self.item_to_index)
        if self.config.mode == "exact":
            final = exact
        elif self.config.mode == "sid_hier":
            final = hierarchical
        elif self.config.mode == "semantic":
            final = semantic
        elif exact:
            final = 1.0
        else:
            final = self.config.lambda_sid * hierarchical + (1.0 - self.config.lambda_sid) * semantic
        return {
            "exact": float(exact),
            "hierarchical": float(hierarchical),
            "semantic": float(semantic),
            "final": float(final),
            "predicted_item_id": predicted_item,
        }

    def score(self, predicted_sid: object, target_sid: object, target_item_id: str | None = None) -> float:
        return float(self.components(predicted_sid, target_sid, target_item_id)["final"])

    def __call__(self, completions: Sequence[object] | None = None, target_sid: Sequence[object] | object | None = None, target_item_id: Sequence[object] | object | None = None, solution_str: Sequence[object] | object | None = None, ground_truth: Sequence[object] | object | None = None, **_: Any) -> list[float]:
        predictions = completions if completions is not None else solution_str
        targets = target_sid if target_sid is not None else ground_truth
        if predictions is None or targets is None:
            raise ValueError("reward requires completions/solution_str and target_sid/ground_truth")
        if isinstance(predictions, (str, bytes)):
            predictions = [predictions]
        if isinstance(targets, (str, bytes)):
            targets = [targets] * len(predictions)
        target_items = target_item_id if isinstance(target_item_id, (list, tuple)) else [target_item_id] * len(predictions)
        return [self.score(prediction, target, None if item is None else str(item)) for prediction, target, item in zip(predictions, targets, target_items, strict=True)]


_ENV_REWARD: RecommendationReward | None = None


def _default_reward(config_path: str | None = None) -> RecommendationReward:
    global _ENV_REWARD
    if _ENV_REWARD is not None and config_path is None:
        return _ENV_REWARD
    path = config_path or os.environ.get("MM_ONEREC_REWARD_CONFIG")
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        _ENV_REWARD = RecommendationReward.from_config(RewardConfig(**payload))
    else:
        _ENV_REWARD = RecommendationReward(RewardConfig(mode="hybrid"))
    return _ENV_REWARD


def compute_reward(data_source: Any = None, solution_str: Any = None, ground_truth: Any = None, extra_info: Mapping[str, Any] | None = None, **kwargs: Any) -> float | list[float]:
    """Native verl custom reward hook; invalid/unmapped items receive zero semantic reward."""
    reward = _default_reward(kwargs.get("config_path"))
    extras = extra_info or {}
    target_item = extras.get("target_item_id") if isinstance(extras, Mapping) else None
    values = reward(solution_str=solution_str, ground_truth=ground_truth, target_item_id=target_item, **kwargs)
    return values[0] if isinstance(solution_str, (str, bytes)) else values


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Print auditable MM-OneRec reward components for real SID artifacts.")
    parser.add_argument("--config", type=Path, required=True, help="JSON RewardConfig with SID and embedding paths")
    parser.add_argument("--target-sid", required=True)
    parser.add_argument("--target-item-id", required=True)
    parser.add_argument("--prediction", action="append", required=True, help="Prediction SID; repeat for an ordered sanity list")
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    reward = RecommendationReward.from_config(RewardConfig(**payload))
    rows = []
    for prediction in args.prediction:
        rows.append({"prediction": prediction, **reward.components(prediction, args.target_sid, args.target_item_id)})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    _cli()
