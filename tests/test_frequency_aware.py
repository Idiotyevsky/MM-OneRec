import importlib.util

import torch
from transformers import GPT2Config, GPT2LMHeadModel


def _module():
    spec = importlib.util.spec_from_file_location("sft_frequency_test", "scripts/sft.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frequency_weights_downweight_popular_prefix():
    module = _module()

    class Tokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            mapping = {1: "<a_0>", 2: "<b_0>", 3: "<c_0>", 4: "<a_1>", 5: "<b_1>", 6: "<c_1>"}
            return "".join(mapping.get(int(token_id), "") for token_id in token_ids)

    rows = [
        {"labels": [-100, 1, 2, 3]},
        {"labels": [-100, 1, 2, 3]},
        {"labels": [-100, 4, 5, 6]},
    ]
    weights, payload = module._build_frequency_weights(rows, Tokenizer())
    assert weights["<a_0>"] < weights["<a_1>"]
    assert payload["stats"]["missing_prefix_samples"] == 0


def test_frequency_trainer_computes_weighted_token_loss_and_gradients():
    module = _module()
    config = GPT2Config(vocab_size=32, n_positions=16, n_ctx=16, n_embd=16, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(config)
    trainer = object.__new__(module.FrequencyAwareTrainer)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [1, 2, 5, 6]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, 2, 3, 4], [-100, 2, 5, 6]]),
        "sample_weight": torch.tensor([0.5, 2.0]),
        "task_type": ["recommendation", "auxiliary"],
    }
    loss = module.FrequencyAwareTrainer.compute_loss(trainer, model, inputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(param.grad is not None and torch.any(param.grad != 0) for param in model.parameters())
