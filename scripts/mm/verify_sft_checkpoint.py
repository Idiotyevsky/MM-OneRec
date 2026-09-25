"""Smoke-check a Qwen SFT checkpoint with an extended Semantic-ID vocabulary.

The check is intentionally small and deterministic.  It verifies that the
saved tokenizer/model pair can be reloaded, SID tokens remain atomic, one
recommendation example has a finite loss and non-zero gradients on target SID
rows, and greedy generation produces a decodable continuation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minionerec.data import SidSFTDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--sid-index", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    index = json.loads(args.sid_index.read_text(encoding="utf-8"))
    sid_tokens = sorted({token for sid in index.values() for token in sid})
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    non_atomic = {
        token: tokenizer.encode(token, add_special_tokens=False)
        for token in sid_tokens
        if len(tokenizer.encode(token, add_special_tokens=False)) != 1
    }
    if non_atomic:
        raise RuntimeError(f"non-atomic SID tokens after reload: {list(non_atomic.items())[:5]}")

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()
    if device.type == "cpu":
        model.to(device)
    model.config.use_cache = True

    dataset = SidSFTDataset(
        train_file=str(args.train_file),
        tokenizer=tokenizer,
        max_len=args.cutoff_len,
        sample=1,
        seed=42,
        category=args.category,
        test=False,
    )
    example = dataset[0]
    input_ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([example["attention_mask"]], dtype=torch.long, device=device)
    labels = torch.tensor([example["labels"]], dtype=torch.long, device=device)

    model.train()
    model.zero_grad(set_to_none=True)
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = output.loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"checkpoint loss is not finite: {loss.item()}")
    loss.backward()
    embedding = model.get_input_embeddings().weight
    grad = embedding.grad
    if grad is None:
        raise RuntimeError("input embedding gradient is missing")
    sid_token_ids = {
        int(tokenizer.convert_tokens_to_ids(token)) for token in sid_tokens
    }
    target_ids = sorted(
        {
            int(token_id)
            for token_id in labels[labels >= 0].detach().cpu().tolist()
            if int(token_id) in sid_token_ids
        }
    )
    target_norms = {
        tokenizer.convert_ids_to_tokens(token_id): float(grad[token_id].float().norm().item())
        for token_id in target_ids
    }
    nonzero_target = {token: value for token, value in target_norms.items() if value > 0.0}
    if not nonzero_target:
        raise RuntimeError(f"target SID embedding gradients are all zero: {target_norms}")

    model.eval()
    prompt_dataset = SidSFTDataset(
        train_file=str(args.train_file),
        tokenizer=tokenizer,
        max_len=args.cutoff_len,
        sample=1,
        seed=42,
        category=args.category,
        test=True,
    )
    prompt = prompt_dataset[0]
    prompt_ids = torch.tensor([prompt["input_ids"]], dtype=torch.long, device=device)
    prompt_mask = torch.tensor([prompt["attention_mask"]], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    continuation_ids = generated[0, prompt_ids.shape[1] :].detach().cpu().tolist()
    continuation = tokenizer.decode(continuation_ids, skip_special_tokens=False)

    # A tiny catalog Trie check exercises the same atomic SID contract as the
    # full evaluator without loading all 7,493 test prompts.
    sid_sequences = [
        [int(tokenizer.convert_tokens_to_ids(token)) for token in sid] + [tokenizer.eos_token_id]
        for sid in index.values()
    ]
    prompt_len = int(prompt_ids.shape[1])

    def allowed_tokens(_batch_id: int, sequence: torch.Tensor) -> list[int]:
        generated_prefix = sequence[prompt_len:].detach().cpu().tolist()
        if generated_prefix and generated_prefix[-1] == tokenizer.eos_token_id:
            return [tokenizer.eos_token_id]
        next_tokens = {
            candidate[len(generated_prefix)]
            for candidate in sid_sequences
            if candidate[: len(generated_prefix)] == generated_prefix
            and len(candidate) > len(generated_prefix)
        }
        return sorted(next_tokens) or [tokenizer.eos_token_id]

    with torch.no_grad():
        constrained = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=allowed_tokens,
        )
    constrained_ids = constrained[0, prompt_len:].detach().cpu().tolist()
    constrained_text = tokenizer.decode(constrained_ids, skip_special_tokens=False)
    constrained_sid = constrained_ids[:-1] if constrained_ids and constrained_ids[-1] == tokenizer.eos_token_id else constrained_ids
    catalog_sid_ids = {tuple(sequence[:-1]) for sequence in sid_sequences}

    result = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "vocab_size": len(tokenizer),
        "sid_token_count": len(sid_tokens),
        "non_atomic_count": len(non_atomic),
        "loss": float(loss.detach().cpu().item()),
        "target_sid_gradient_norms": target_norms,
        "nonzero_target_sid_gradient_count": len(nonzero_target),
        "generated_token_ids": continuation_ids,
        "generated_text": continuation,
        "constrained_generated_token_ids": constrained_ids,
        "constrained_generated_text": constrained_text,
        "constrained_catalog_sid": tuple(constrained_sid) in catalog_sid_ids,
        "target_text": tokenizer.decode(labels[labels >= 0].detach().cpu().tolist(), skip_special_tokens=False),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
