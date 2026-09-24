# Standard verl GRPO

This directory is the new RL entry point.  It is intentionally separate from
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
`category`.  Embeddings are loaded by the reward process and are not copied
into the parquet file.

## Native GRPO launch

```bash
python -m rl.verl.run_grpo \
  --config rl/verl/configs/grpo_small.yaml --dry-run

# After installing a compatible verl release:
bash rl/verl/run_grpo.sh --config rl/verl/configs/grpo_formal.yaml
```

The launcher invokes `python -m verl.trainer.main_ppo` with
`algorithm.adv_estimator=grpo`, `actor_rollout_ref.rollout.n=8` for the formal
config, `actor_rollout_ref.actor.ppo_epochs`, clip ratio `0.2`, and an
actor-side reference KL coefficient `1e-3`.  `pi_old` is therefore the native
verl rollout policy, while `pi_ref` is the separate KL anchor.

Training rollouts are currently unconstrained: an invalid SID receives zero
reward.  The existing Trie processor remains enabled for evaluation.  The
launcher rejects a request for training-time Trie guidance instead of
monkey-patching vLLM.

## Reward modes

`rl/verl/reward.py` supports `exact`, `sid_hier`, `semantic`, and `hybrid`.
The hybrid default is `0.6 * sid_hier + 0.4 * semantic` for non-exact
predictions.  Semantic reward uses the representation track's item matrix and
maps cosine similarity from `[-1, 1]` into `[0, 1]`.  `<d_n>` collision
suffixes are excluded from hierarchical semantic weights.
