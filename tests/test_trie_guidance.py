import pytest
import torch

from rl.verl.trie_guidance import CatalogTrie, TrieLogitsProcessor


def test_catalog_trie_masks_only_valid_next_tokens():
    trie = CatalogTrie([[10, 20, 99], [10, 21, 99], [11, 22, 99]])
    processor = TrieLogitsProcessor(trie, eos_token_id=99)
    logits = torch.arange(128, dtype=torch.float32)

    root = processor([], logits)
    assert trie.allowed(()) == (10, 11)
    assert torch.isfinite(root[10])
    assert torch.isfinite(root[11])
    assert torch.isneginf(root[0])

    branch = processor([10], logits)
    assert torch.isfinite(branch[20])
    assert torch.isfinite(branch[21])
    assert torch.isneginf(branch[22])


def test_catalog_trie_keeps_eos_terminal_and_rejects_invalid_prefix():
    trie = CatalogTrie([[10, 99]])
    processor = TrieLogitsProcessor(trie, eos_token_id=99)
    logits = torch.zeros(128)

    terminal = processor([10, 99], logits)
    assert torch.isfinite(terminal[99])
    assert torch.isneginf(terminal[98])

    with pytest.raises(RuntimeError, match="no valid continuation"):
        processor([12], logits)
