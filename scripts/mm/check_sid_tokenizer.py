#!/usr/bin/env python3
"""Validate atomic SID vocabulary extension for a causal-LM tokenizer."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


def check(model: str, index_path: Path) -> dict[str, object]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    tokens = sorted({str(token) for values in index.values() for token in values})
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    base_vocab_size = len(tokenizer)
    added = tokenizer.add_tokens(tokens)
    bad = {
        token: tokenizer.encode(token, add_special_tokens=False)
        for token in tokens
        if len(tokenizer.encode(token, add_special_tokens=False)) != 1
    }
    suffixes = [int(match.group(1)) for token in tokens if (match := re.fullmatch(r"<d_(\d+)>", token))]
    by_prefix = Counter(token[1] for token in tokens if re.fullmatch(r"<[a-z]_\d+>", token))
    return {
        "model": model,
        "index": str(index_path),
        "items": len(index),
        "base_vocab_size": base_vocab_size,
        "final_vocab_size": len(tokenizer),
        "added_tokens": int(added),
        "token_count": len(tokens),
        "tokens_by_prefix": dict(sorted(by_prefix.items())),
        "max_collision_suffix": max(suffixes, default=0),
        "collision_suffix_count": len(suffixes),
        "atomic": not bad,
        "non_atomic_examples": bad,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.model, args.index)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not result["atomic"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
