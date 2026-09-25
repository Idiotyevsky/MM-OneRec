"""verl vLLM rollout subclass with catalog-valid training-time decoding."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from verl.workers.rollout.vllm_rollout import vLLMRollout

from .trie_guidance import TrieLogitsProcessor, build_catalog_trie

logger = logging.getLogger(__name__)


class TrieVLLMRollout(vLLMRollout):
    """Use vLLM V0's public per-request logits processor hook.

    run_grpo.py sets VLLM_USE_V1=0 before importing verl. vLLM V1 currently
    rejects user-provided per-request logits processors, whereas V0 supports
    the public SamplingParams.logits_processors API.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        index_value = os.environ.get("MM_ONEREC_TRIE_INDEX_PATH")
        if not index_value:
            raise RuntimeError(
                "MM_ONEREC_TRIE_INDEX_PATH is required for constrained rollout"
            )
        trie, stats = build_catalog_trie(self.model_config.tokenizer, Path(index_value))
        self._mm_onerec_trie_stats = stats
        self.sampling_params.logits_processors = [
            TrieLogitsProcessor(trie, self.model_config.tokenizer.eos_token_id)
        ]
        logger.info(
            "MM-OneRec Trie rollout enabled: index=%s stats=%s processor=%r",
            index_value,
            stats,
            self.sampling_params.logits_processors[0],
        )
