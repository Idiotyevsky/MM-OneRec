"""Full-catalog SID evaluation with vLLM and Trie-constrained beam search.

The output schema matches ``scripts/evaluate.py`` so the generated JSON can be
passed directly to ``scripts/mm/evaluate_metrics.py``. vLLM does not expose a
per-request prefix callback through its built-in beam-search helper, therefore
this module performs the small (normally five-token) beam loop explicitly and
uses vLLM's public ``allowed_token_ids`` parameter at every step.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CATEGORY_NAMES = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
    "Toys_and_Games": "toys and games",
    "Sports": "sports and outdoors",
    "Books": "books",
}


@dataclass(frozen=True)
class Beam:
    """One constrained beam; token_ids contains generated tokens only."""

    token_ids: tuple[int, ...]
    cumulative_logprob: float


class CatalogTrie:
    """Compact prefix-to-children map over tokenized catalog Semantic IDs."""

    def __init__(self, sequences: Iterable[Sequence[int]]) -> None:
        children: dict[tuple[int, ...], set[int]] = {}
        terminals: set[tuple[int, ...]] = set()
        sequence_count = 0
        for sequence in sequences:
            values = tuple(int(token_id) for token_id in sequence)
            if not values:
                raise ValueError("Catalog Trie cannot contain an empty sequence")
            if values in terminals:
                raise ValueError(f"Duplicate tokenized catalog SID: {values}")
            terminals.add(values)
            sequence_count += 1
            for index, token_id in enumerate(values):
                children.setdefault(values[:index], set()).add(token_id)
        if not sequence_count:
            raise ValueError("Catalog is empty")
        self._children = {
            prefix: tuple(sorted(values)) for prefix, values in children.items()
        }
        self._terminals = terminals
        self.sequence_count = sequence_count

    def allowed(self, prefix: Sequence[int]) -> tuple[int, ...]:
        return self._children.get(tuple(int(value) for value in prefix), ())

    def contains(self, sequence: Sequence[int]) -> bool:
        return tuple(int(value) for value in sequence) in self._terminals


def load_catalog(info_file: Path) -> list[str]:
    """Read the first tab-separated field from the existing catalog info file."""
    semantic_ids: list[str] = []
    for line_number, line in enumerate(
        info_file.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        sid = line.split("\t", 1)[0].strip()
        if not sid:
            raise ValueError(f"Empty Semantic ID at {info_file}:{line_number}")
        semantic_ids.append(sid)
    if len(set(semantic_ids)) != len(semantic_ids):
        raise ValueError(
            "Catalog contains duplicate final Semantic IDs; export collision suffixes first"
        )
    return semantic_ids


def build_catalog_trie(tokenizer: Any, semantic_ids: Sequence[str]) -> CatalogTrie:
    """Tokenize the exact response format and append EOS to every SID path."""
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    sequences = [
        tokenizer.encode(f"{sid}\n", add_special_tokens=False) + [int(eos_token_id)]
        for sid in semantic_ids
    ]
    return CatalogTrie(sequences)


def _beam_score(beam: Beam, eos_token_id: int, length_penalty: float) -> float:
    length = len(beam.token_ids)
    if beam.token_ids and beam.token_ids[-1] == eos_token_id:
        length -= 1
    return beam.cumulative_logprob / (max(length, 1) ** length_penalty)


def _logprob_value(value: Any) -> float:
    return float(value.logprob if hasattr(value, "logprob") else value)


def constrained_beam_search(
    engine: Any,
    prompt_token_ids: Sequence[Sequence[int]],
    trie: CatalogTrie,
    eos_token_id: int,
    beam_width: int,
    max_new_tokens: int,
    length_penalty: float,
    sampling_params_factory: Callable[[Sequence[int], int], Any],
) -> list[list[Beam]]:
    """Run deterministic beam search with a per-step catalog-token mask."""
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    prompts = [tuple(int(token_id) for token_id in row) for row in prompt_token_ids]
    active: list[list[Beam]] = [[Beam((), 0.0)] for _ in prompts]
    completed: list[list[Beam]] = [[] for _ in prompts]

    for _ in range(max_new_tokens):
        request_prompts: list[dict[str, list[int]]] = []
        request_params: list[Any] = []
        request_meta: list[tuple[int, Beam, frozenset[int]]] = []
        for prompt_index, beams in enumerate(active):
            for beam in beams:
                allowed = trie.allowed(beam.token_ids)
                if not allowed:
                    continue
                request_prompts.append(
                    {"prompt_token_ids": list(prompts[prompt_index] + beam.token_ids)}
                )
                request_params.append(sampling_params_factory(allowed, beam_width))
                request_meta.append((prompt_index, beam, frozenset(allowed)))

        if not request_prompts:
            break
        request_outputs = engine.generate(
            request_prompts,
            sampling_params=request_params,
            use_tqdm=False,
        )
        if len(request_outputs) != len(request_meta):
            raise RuntimeError(
                f"vLLM returned {len(request_outputs)} rows for "
                f"{len(request_meta)} requests"
            )

        candidates: list[list[Beam]] = [[] for _ in prompts]
        for output, (prompt_index, parent, allowed) in zip(
            request_outputs, request_meta
        ):
            if not output.outputs or not output.outputs[0].logprobs:
                raise RuntimeError(
                    "vLLM one-token request returned no log probabilities"
                )
            token_logprobs = output.outputs[0].logprobs[0]
            for token_id, logprob in token_logprobs.items():
                token_id = int(token_id)
                if token_id not in allowed:
                    continue
                child = Beam(
                    token_ids=parent.token_ids + (token_id,),
                    cumulative_logprob=(
                        parent.cumulative_logprob + _logprob_value(logprob)
                    ),
                )
                if token_id == eos_token_id:
                    if trie.contains(child.token_ids):
                        completed[prompt_index].append(child)
                else:
                    candidates[prompt_index].append(child)

        active = [
            sorted(
                rows,
                key=lambda beam: beam.cumulative_logprob,
                reverse=True,
            )[:beam_width]
            for rows in candidates
        ]
        for index in range(len(completed)):
            completed[index] = sorted(
                completed[index],
                key=lambda beam: _beam_score(
                    beam, eos_token_id, length_penalty
                ),
                reverse=True,
            )[:beam_width]
        if not any(active):
            break

    ranked: list[list[Beam]] = []
    for done, unfinished in zip(completed, active):
        rows = done if done else unfinished
        ranked.append(
            sorted(
                rows,
                key=lambda beam: _beam_score(
                    beam, eos_token_id, length_penalty
                ),
                reverse=True,
            )[:beam_width]
        )
    return ranked


def decode_beams(
    tokenizer: Any, rows: Sequence[Sequence[Beam]]
) -> list[list[str]]:
    """Decode generated suffixes into the evaluator's list-of-SID format."""
    eos_token_id = tokenizer.eos_token_id
    decoded: list[list[str]] = []
    for beams in rows:
        predictions: list[str] = []
        for beam in beams:
            token_ids = list(beam.token_ids)
            if token_ids and token_ids[-1] == eos_token_id:
                token_ids.pop()
            predictions.append(
                tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
            )
        decoded.append(predictions)
    return decoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        required=True,
        help="SFT checkpoint with its extended tokenizer",
    )
    parser.add_argument(
        "--info-file",
        type=Path,
        required=True,
        help="Catalog SID info TSV used by evaluate.py",
    )
    parser.add_argument(
        "--category", choices=sorted(CATEGORY_NAMES), required=True
    )
    parser.add_argument("--test-data-path", type=Path, required=True)
    parser.add_argument("--result-json-data", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of users per outer beam-search batch",
    )
    parser.add_argument("--num-beams", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from minionerec.data import EvalSidDataset

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    try:
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is required for this entry point; install "
            "requirements.verl.txt or the documented vLLM environment"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    semantic_ids = load_catalog(args.info_file)
    trie = build_catalog_trie(tokenizer, semantic_ids)

    dataset = EvalSidDataset(
        train_file=str(args.test_data_path),
        tokenizer=tokenizer,
        max_len=args.max_model_len,
        category=CATEGORY_NAMES[args.category],
        test=True,
        K=args.K,
        seed=args.seed,
        sample=args.sample,
    )
    encodings = [dataset[index] for index in range(len(dataset))]
    test_rows = dataset.get_all()
    prompt_ids = [row["input_ids"] for row in encodings]
    longest_prompt = max((len(values) for values in prompt_ids), default=0)
    if longest_prompt + args.max_new_tokens > args.max_model_len:
        raise ValueError(
            f"Longest prompt ({longest_prompt}) + max_new_tokens "
            f"({args.max_new_tokens}) exceeds max_model_len "
            f"({args.max_model_len})"
        )

    print(
        f"[vLLM Evaluate] model={args.base_model} samples={len(prompt_ids)} "
        f"catalog={len(semantic_ids)} beams={args.num_beams} "
        f"batch={args.batch_size}",
        flush=True,
    )
    engine = LLM(
        model=args.base_model,
        tokenizer=args.base_model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=not args.disable_prefix_caching,
        disable_custom_all_reduce=True,
        max_logprobs=max(2 * args.num_beams, 20),
    )

    def sampling_params_factory(
        allowed: Sequence[int], beam_width: int
    ) -> Any:
        return SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=2 * beam_width,
            allowed_token_ids=list(allowed),
            detokenize=False,
            skip_special_tokens=False,
            seed=args.seed,
        )

    predictions: list[list[str]] = []
    for start in tqdm(
        range(0, len(prompt_ids), args.batch_size),
        desc="vLLM constrained eval",
    ):
        batch_prompts = prompt_ids[start : start + args.batch_size]
        beams = constrained_beam_search(
            engine=engine,
            prompt_token_ids=batch_prompts,
            trie=trie,
            eos_token_id=int(tokenizer.eos_token_id),
            beam_width=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            length_penalty=args.length_penalty,
            sampling_params_factory=sampling_params_factory,
        )
        predictions.extend(decode_beams(tokenizer, beams))
        print(
            "[vLLM Evaluate Progress] "
            f"samples={min(start + args.batch_size, len(prompt_ids))}/"
            f"{len(prompt_ids)}",
            flush=True,
        )

    if len(predictions) != len(test_rows):
        raise RuntimeError(
            f"Prediction count mismatch: {len(predictions)} != "
            f"{len(test_rows)}"
        )
    metadata = {
        "backend": "vllm",
        "vllm_version": vllm.__version__,
        "constraint": "trie_allowed_token_ids",
        "search": "deterministic_beam",
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "length_penalty": args.length_penalty,
        "catalog_size": len(semantic_ids),
        "tensor_parallel_size": args.tensor_parallel_size,
        "seed": args.seed,
    }
    for row, output in zip(test_rows, predictions):
        row.pop("dedup", None)
        row["predict"] = output
        row["target"] = str(row.get("output", "")).strip()
        row["predictions"] = [str(value).strip() for value in output]
        row["evaluation_metadata"] = metadata

    args.result_json_data.parent.mkdir(parents=True, exist_ok=True)
    args.result_json_data.write_text(
        json.dumps(test_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[vLLM Evaluate] wrote {args.result_json_data}",
        flush=True,
    )


if __name__ == "__main__":
    main()
