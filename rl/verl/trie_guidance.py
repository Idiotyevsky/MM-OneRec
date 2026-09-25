"""vLLM V0 logits guidance for catalog-valid Semantic-ID rollouts.

The implementation intentionally uses only vLLM's public SamplingParams
logits_processors hook. It is used by the optional constrained training
rollout path; the default native verl path remains available unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


class CatalogTrie:
    """Prefix-to-children map over tokenized catalog SID sequences."""

    def __init__(self, sequences: Iterable[Sequence[int]]) -> None:
        children: dict[tuple[int, ...], set[int]] = {}
        terminals: set[tuple[int, ...]] = set()
        count = 0
        for sequence in sequences:
            values = tuple(int(token_id) for token_id in sequence)
            if not values:
                raise ValueError("catalog trie cannot contain an empty sequence")
            if values in terminals:
                raise ValueError(f"duplicate tokenized catalog sequence: {values}")
            terminals.add(values)
            count += 1
            for index, token_id in enumerate(values):
                children.setdefault(values[:index], set()).add(token_id)
        if count == 0:
            raise ValueError("catalog trie cannot be empty")
        self._children = {
            prefix: tuple(sorted(token_ids)) for prefix, token_ids in children.items()
        }
        self._terminals = terminals
        self.sequence_count = count

    def allowed(self, prefix: Sequence[int]) -> tuple[int, ...]:
        """Return valid next token ids for a generated response prefix."""
        return self._children.get(tuple(int(value) for value in prefix), ())

    def contains(self, sequence: Sequence[int]) -> bool:
        return tuple(int(value) for value in sequence) in self._terminals


def _sid_from_tokens(tokens: Any) -> str:
    if not isinstance(tokens, (list, tuple)) or not tokens:
        raise ValueError(f"invalid SID token list: {tokens!r}")
    return "".join(str(token) for token in tokens)


def _semantic_sid_from_tokens(tokens: Any) -> str:
    """Return the RQ hierarchy while excluding the <d_n> disambiguator."""
    values = tokens if isinstance(tokens, (list, tuple)) else []
    semantic = [
        str(token)
        for token in values
        if not (str(token).startswith("<d_") and str(token).endswith(">"))
    ]
    if not semantic:
        raise ValueError(f"SID has no semantic RQ levels: {tokens!r}")
    return "".join(semantic)


def build_catalog_trie(tokenizer: Any, index_path: str | Path) -> tuple[CatalogTrie, dict[str, int]]:
    """Build a trie using the exact response format used by SFT.

    Each catalog path is SID + newline + EOS. The newline is part of the
    supervised response in the existing SidSFTDataset, while <d_n> remains
    an ordinary collision-disambiguation token in the full path.
    """
    path = Path(index_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"SID index must be a non-empty JSON object: {path}")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    sequences: list[list[int]] = []
    for item_id, tokens in payload.items():
        sid = _sid_from_tokens(tokens)
        encoded = tokenizer.encode(f"{sid}\n", add_special_tokens=False)
        if not encoded:
            raise ValueError(f"SID encoded to an empty sequence for item {item_id!r}")
        sequence = [int(token_id) for token_id in encoded] + [int(eos_token_id)]
        sequences.append(sequence)
    trie = CatalogTrie(sequences)
    unique_raw_sid = len({_semantic_sid_from_tokens(tokens) for tokens in payload.values()})
    return trie, {
        "catalog_items": len(payload),
        "raw_unique_sid": unique_raw_sid,
        "raw_collision_count": len(payload) - unique_raw_sid,
        "tokenized_sequences": len(sequences),
        "trie_prefixes": len(trie._children),
    }


class TrieLogitsProcessor:
    """Stateless public-vLLM logits processor for a catalog trie.

    vLLM V0 calls a two-argument processor with generated output token ids and
    the next-token logits. Keeping the processor stateless is important:
    SamplingParams may be cloned and reused for batched rollouts.
    """

    def __init__(self, trie: CatalogTrie, eos_token_id: int | None = None) -> None:
        self.trie = trie
        self.eos_token_id = None if eos_token_id is None else int(eos_token_id)

    def __call__(self, past_token_ids: Sequence[int], logits: torch.Tensor) -> torch.Tensor:
        prefix = tuple(int(value) for value in past_token_ids)
        if self.eos_token_id is not None and prefix and prefix[-1] == self.eos_token_id:
            allowed = (self.eos_token_id,)
        else:
            allowed = self.trie.allowed(prefix)
        if not allowed:
            raise RuntimeError(
                "Trie rollout reached a prefix with no valid continuation; "
                "check tokenizer, SID index, and response newline format. "
                f"prefix={prefix[-12:]}"
            )
        mask = torch.full_like(logits, float("-inf"))
        allowed_ids = torch.as_tensor(allowed, device=logits.device, dtype=torch.long)
        mask.index_fill_(0, allowed_ids, 0.0)
        return logits + mask

    def __repr__(self) -> str:
        return f"TrieLogitsProcessor(sequences={self.trie.sequence_count})"
