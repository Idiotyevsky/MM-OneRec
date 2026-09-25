# Standard verl GRPO

This directory is the native RL entry point. It is intentionally separate from
`scripts/rl.py` and `minionerec/trainer.py`, which remain the historical
`legacy_group_relative_rl` baseline.

## Data conversion

```bash
python -m rl.verl.prepare_data \
  --interactions data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --output data/verl/industrial/train.parquet \
  --dataset Amazon23 --category Industrial_and_Scientific
```

Each parquet row contains `prompt`, `target_sid`, `target_item_id`,
`history_sid`, `history_item_ids`, `target_sid_parts`, `dataset`, and
`category`. Embeddings are loaded by the reward process and are not copied
into the parquet file.

## Native GRPO launch

```bash
python -m rl.verl.run_grpo \
  --config rl/verl/configs/grpo_small.yaml --dry-run

# After installing the verified verl release:
bash rl/verl/run_grpo.sh --config rl/verl/configs/grpo_formal.yaml
```

The launcher invokes native verl with
`algorithm.adv_estimator=grpo`, formal rollout group size `n=8`,
`actor_rollout_ref.actor.ppo_epochs`, clip ratio `0.2`, and an actor-side
reference KL coefficient `1e-3`. `pi_old` is the native verl rollout
policy, while `pi_ref` is the separate KL anchor.

## Training-time Trie guidance

The small configuration enables `training_rollout_constraint: trie`. In this
mode MM-OneRec registers a project rollout subclass at runtime and attaches a
catalog Trie through vLLM's public
`SamplingParams.logits_processors` API. The launcher sets
`VLLM_USE_V1=0` because the installed vLLM 0.8.5 V1 engine rejects
per-request user logits processors; no verl or vLLM source is modified.

Every generated response follows a valid catalog path:

```text
SID tokens -> optional <d_n> collision suffix -> newline -> EOS
```

The same SID index and checkpoint tokenizer are used to build the Trie. A
prefix with no valid continuation raises an explicit configuration error
instead of silently producing an all-masked distribution. Consequently,
constrained training rollouts cannot emit an invalid catalog SID. To retain
the previous ablation, set
`training_rollout_constraint: unconstrained_invalid_zero`; in that mode an
invalid completion is assigned zero reward.

For a real constrained run, use:

```bash
python -m rl.verl.run_grpo \
  --config rl/verl/configs/grpo_small.yaml \
  --constrained-rollout
```

The first constrained run should be treated as a vLLM V0 compatibility smoke
test before a longer multi-GPU job.

## Reward modes

`rl/verl/reward.py` supports `exact`, `sid_hier`, `semantic`, and
`hybrid`. The hybrid default is `0.6 * sid_hier + 0.4 * semantic` for
non-exact predictions. Semantic reward uses the representation track's item
matrix and maps cosine similarity from `[-1, 1]` into `[0, 1]`. `<d_n>`
collision suffixes are excluded from hierarchical semantic weights.
