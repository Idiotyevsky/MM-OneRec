# Experiment Log

This file records provenance for the current MM-OneRec implementation.  The
previous log is preserved in
[`EXPERIMENT_LOG_LEGACY.md`](EXPERIMENT_LOG_LEGACY.md).

## 2026-09-24 — implementation audit and representation/RL refactor

### Base revision and data

- Base Git revision: `7de8e7e` (`Update comparable full-test evaluation results`).
- Dataset protocol: Amazon Reviews 2023, `Industrial_and_Scientific_1m` for
  the preserved formal artifacts; local smoke conversion also uses the
  repository's Amazon interaction CSV schema.
- Existing formal test split: 7,974 rows, 7,493 catalog items, 20-beam Trie
  evaluation, `max_new_tokens=8`.
- Seed for new smoke/config defaults: 42.

### Audit of the actual baseline chain

```text
Amazon interaction/item metadata
  -> SigLIP text/image (or existing text embedding)
  -> RQ-VAE / RQ-KMeans Semantic ID
  -> scripts/sft.py (Qwen generator)
  -> scripts/rl.py + minionerec/trainer.py (legacy group-relative RL)
  -> minionerec/logit_processor.py Trie constrained evaluation
  -> scripts/evaluate.py / scripts/mm/evaluate_metrics.py
```

The old trainer computes a detached self-ratio in its policy term.  It is
therefore retained and labelled `legacy_group_relative_rl`; it is not used as
the implementation of standard old-policy clipped GRPO.

### New code paths

- `multimodal/qwen3_vl_encoder.py`: frozen Qwen3-VL joint item encoding,
  `last_token`/`text_mean` pooling, missing-image text-only path, `none`/`pca`
  projection, item-order and `has_image` sidecars.
- `multimodal/rec_alignment.py`: frozen-VLM, train-only next-item InfoNCE
  projector for `Qwen3VL-RecAlign-SID`.
- `rl/verl/prepare_data.py`: interaction CSV to parquet with prompt,
  history/target SID and item IDs; embeddings are not duplicated.
- `rl/verl/reward.py`: `exact`, `sid_hier`, `semantic`, `hybrid` rewards;
  `<d_n>` is excluded from semantic RQ-layer scoring.
- `rl/verl/run_grpo.py`: native `verl.trainer.main_ppo` launcher with
  `algorithm.adv_estimator=grpo`, rollout group size, `ppo_epochs`, clip ratio,
  actor-side KL and custom reward hook.
- `rl/verl/grpo_math.py`: CPU diagnostic implementation of ratio/clipping, not
  a replacement for the native trainer.

### Runtime validation

| Check | Command | Result |
|---|---|---|
| New CLI help | `python -m multimodal.qwen3_vl_encoder --help` and corresponding alignment/verl commands | passed |
| Real-data parquet smoke | `python -m rl.verl.prepare_data ... --max-samples 4` | 4 rows; all required columns present |
| New CPU tests | `python -m pytest -q tests/test_rl_reward.py tests/test_qwen3vl_encoder.py tests/test_verl_pipeline.py tests/test_rec_alignment.py` | 10 passed |
| Full suite | `python -m pytest -q` | 56 passed, 3 skipped |
| verl launcher dry run | `python -m rl.verl.run_grpo --config rl/verl/configs/grpo_small.yaml --dry-run` | native command emitted |

### New-track execution status

The current environment reports `verl` unavailable.  In addition, the local
Transformers/Torch combination cannot import Qwen3-VL's model class because it
lacks the required `torch.distributed.tensor.DTensor` API.  The Qwen3-VL path
is tested with a CPU mock, but no real Qwen3-VL embedding, RQ model, SFT,
native-verl GRPO checkpoint or metric is claimed in this revision.  The
launcher exits instead of silently falling back to the legacy trainer.

| Track | SID | SFT | Native verl GRPO | Eval |
|---|:---:|:---:|:---:---:|:---:|
| Text baseline, preserved | ✓ | ✓ | — (legacy RL artifact) | ✓ |
| SigLIP-MM, preserved | ✓ | ✓ | — (legacy RL artifact) | ✓ |
| Text/SigLIP + native verl | data/launcher ready | — | not run | not run |
| Qwen3VL-MM | encoder/mock ready | not run | not run | not run |
| Qwen3VL-RecAlign | projector ready | not run | not run | not run |

### Preserved formal artifacts

The four comparable rows remain untouched under
`outputs/formal_amazon23_1m/` and `results/summary.csv`:

| Model | HR@10 | NDCG@10 | Coverage | Tail HR@10 | Invalid SID |
|---|---:|---:|---:|---:|---:|
| Text-SFT | 0.001129 | 0.000641 | 0.006806 | 0.000627 | 0.0 |
| Text-GRPO (legacy) | 0.001379 | 0.000554 | 0.004671 | 0.001046 | 0.0 |
| SigLIP-MM-SFT | 0.001630 | 0.000530 | 0.006940 | 0.000418 | 0.0 |
| SigLIP-MM-GRPO (legacy) | 0.002508 | 0.000830 | 0.004137 | 0.000418 | 0.0 |

### Preserved formal-run provenance

The following fields are available for the four existing formal checkpoints:

| Track | Dataset/split | Generator checkpoint | Prediction/metrics path | Seed | GPU/steps/wall-clock |
|---|---|---|---|---|---|
| Text-SFT | Amazon23 Industrial_and_Scientific_1m, 7,974 test rows | `outputs/formal_amazon23_1m/text_sft/final_checkpoint` | `outputs/formal_amazon23_1m/text_sft_predictions_k20.json`, `text_sft_metrics_k20.json` | 42 (run config) | not recorded in retained eval config |
| Text-GRPO legacy | same | `outputs/formal_amazon23_1m/text_grpo/final_checkpoint` | `outputs/formal_amazon23_1m/text_grpo_predictions_k20.json`, `text_grpo_metrics_k20.json` | 42 (run config) | not recorded in retained eval config |
| SigLIP-MM-SFT | same | `outputs/formal_amazon23_1m/mm_sft/final_checkpoint` | `outputs/formal_amazon23_1m/mm_sft_predictions_k20.json`, `mm_sft_metrics_k20.json` | 42 (run config) | not recorded in retained eval config |
| SigLIP-MM-GRPO legacy | same | `outputs/formal_amazon23_1m/mm_grpo/final_checkpoint` | `outputs/formal_amazon23_1m/mm_grpo_predictions_k20.json`, `mm_grpo_metrics_k20.json` | 42 (run config) | not recorded in retained eval config |

These are evaluation artifacts; no new native-verl training provenance is inferred from them.

### Formal-run provenance template

Every future Qwen3-VL or native-verl run must add a dated entry containing:

```text
git commit:
dataset/category and split:
representation model and pooling:
RQ config/checkpoint:
generator checkpoint:
verl version:
seed:
GPU(s):
training steps / ppo_epochs / rollout.n:
wall-clock:
checkpoint path:
metrics path:
```
