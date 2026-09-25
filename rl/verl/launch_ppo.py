"""Bootstrap native verl PPO with the optional MM-OneRec Trie rollout."""

from __future__ import annotations

import runpy

from verl.workers.rollout import base


def main() -> None:
    # This is a runtime registry extension, not a modification of verl source.
    # Ray workers import the fully-qualified project class from the shared repo.
    base._ROLLOUT_REGISTRY[("vllm", "sync")] = "rl.verl.trie_rollout.TrieVLLMRollout"
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__")


if __name__ == "__main__":
    main()
