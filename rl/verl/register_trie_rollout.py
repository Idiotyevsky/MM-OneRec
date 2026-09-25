"""Register the MM-OneRec Trie rollout with verl's runtime registry.

This module is imported through verl's official model.external_lib hook inside
each actor/rollout worker, so Ray workers receive the registration as well as
the launcher process.
"""

from verl.workers.rollout import base

base._ROLLOUT_REGISTRY[("vllm", "sync")] = "rl.verl.trie_rollout.TrieVLLMRollout"
