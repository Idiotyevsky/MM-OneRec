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

The checks below were run in the dedicated environment recorded in
`docs/ENVIRONMENT.md`.  The launcher invokes `verl.trainer.main_ppo` directly;
there is no fallback to the legacy trainer.  The formal Qwen3-VL encoder and
controlled RQ-VAE runs are recorded below; Qwen3VL SID export, SFT, native-verl
training, and recommendation evaluation are still separate downstream runs.

| Track | SID | SFT | Native verl GRPO | Eval |
|---|:---:|:---:|:---:---:|:---:|
| Text baseline, preserved | ✓ | ✓ | 4-step smoke ✓; formal NOT RUN | ✓ (preserved) |
| SigLIP-MM, preserved | ✓ | ✓ | NOT RUN (legacy artifact preserved) | ✓ (preserved) |
| Qwen3VL-MM | encoder ✓; RQ ✓; SID ✓ | NOT RUN | NOT RUN | NOT RUN |
| Qwen3VL-RecAlign | projector code ready | NOT RUN | NOT RUN | NOT RUN |

### Real Qwen3-VL encoder smoke (completed)

- Host: `4090-1` (`210.28.132.72`), `CUDA_VISIBLE_DEVICES=0`; model:
  `Qwen/Qwen3-VL-2B-Instruct`; the formal 4B run is documented below.
- Input: 21 real `Industrial_and_Scientific` items, 13 with a downloaded image
  and 8 without one. Image batches used the official message format,
  `apply_chat_template`, and `process_vision_info`; the recorded visual fields
  include `pixel_values` and `image_grid_thw`. Missing-image batches used the
  same Qwen3-VL model with title/description only.
- Pooling/projection: frozen model, `last_token`, L2 normalization. Output
  shape is `[21, 2048]`; no NaN/Inf was observed, and the cosine diagnostic had
  diagonal range `[0.99999976, 1.00000036]` with non-identical rows.
- A smoke PCA was also executed. Since the smoke set has only 21 items, its
  maximum output dimension is 21; the formal 7,493-item PCA-to-768 run is
  documented in the formal encoding section below.
- Artifacts: `outputs/smoke_qwen3vl/embedding.npy`, `metadata.json`,
  `has_image.npy`, `item_ids.json`, `similarity_debug.json`, and `run.log`.

### Formal Qwen3-VL item encoding (completed)

- Host/GPU: `4090-1`, `CUDA_VISIBLE_DEVICES=0`; model:
  `Qwen/Qwen3-VL-4B-Instruct`; batch size 4; dtype `bfloat16`; seed 42.
- Input: all 7,493 `Industrial_and_Scientific_1m` catalog items. 7,136 items
  have a valid cached image and 357 use the same Qwen3-VL text-only path for a
  missing image; no item is switched to the SigLIP space.
- Representation: `last_token`, L2-normalized, PCA fitted on the catalog
  content representations to 768 dimensions. Raw hidden size was 2,560 and
  PCA explained-variance ratio sum was `0.98346287`.
- Validation: output shape `[7493, 768]`, all values finite, mean row norm
  `1.0`, zero zero-norm rows, and item order exactly matched the item metadata.
- Artifacts: `outputs/qwen3vl/Industrial_and_Scientific_1m.qwen3vl-768.npy`
  and its `.npy.json`, `.has_image.npy`, `.item_ids.json`, plus the complete
  `outputs/qwen3vl/encode_4b.log`.

### Qwen3-VL RQ-VAE training (completed controlled runs)

Both runs use the same RQ configuration as the preserved Text/SigLIP formal
tracks (`32/32/32` codebooks, latent dimension 64, MLP `[512,256,128]`,
AdamW, learning rate `1e-3`, batch size 512, seed 2024). They are stored in
separate directories and do not overwrite the existing checkpoints.

| Run | Epochs | Best total train loss | Best collision rate | Collision at final eval | Best checkpoint |
|---|---:|---:|---:|---:|---|
| Qwen3VL RQ main | 100 | `0.0201108` (epoch 19) | `0.317763` (epoch 19) | `0.332977` (epoch 99) | `outputs/qwen3vl/rq_100/Sep-24-2026_12-46-42/best_collision_model.pth` |
| Qwen3VL RQ longer ablation | 300 | `0.0201108` (epoch 19) | `0.317763` (epoch 19) | `0.343254` (epoch 299) | `outputs/qwen3vl/rq_300/Sep-24-2026_12-49-48/best_collision_model.pth` |

The reconstruction component continued to fall (`0.0197` at epoch 0,
`0.0085` at epoch 99, `0.0077` at epoch 299), but the total objective rose
after its early minimum and collision did not improve with longer training.
Therefore the epoch-19 checkpoint is used for the first Qwen3VL SID export;
the 300-epoch run is retained as an ablation, not substituted into the main
comparison. The raw logs are `outputs/qwen3vl/rq_100.log` and
`outputs/qwen3vl/rq_300.log`.

### Qwen3-VL Semantic ID export (completed)

- The epoch-19 `best_collision_model.pth` from the 100-epoch controlled run was
  used; the 300-epoch final checkpoint was not substituted.
- Raw SID statistics: 7,493 items, 5,112 unique raw SIDs, 2,381 raw collisions,
  collision rate `0.317763`. Collision suffixes were appended only for duplicate
  codes, yielding 7,493 unique exported item SIDs.
- The existing interaction converter produced 63,788 train rows, 7,973 valid
  rows, and 7,974 test rows with zero missing catalog items. Artifacts are under
  `data/sid/amazon23_1m/Industrial_and_Scientific_1m.qwen3vl.*`.

### Native verl GRPO 4-step smoke (completed)

- Track: existing Text-SID SFT checkpoint
  `outputs/formal_amazon23_1m/text_sft/final_checkpoint`; 32 train and 8
  validation prompts from `data/verl/smoke_messages/`; seed 42.
- Configuration: native `verl==0.6.1`, `algorithm.adv_estimator=grpo`, `G=4`,
  `ppo_epochs=1`, `clip_ratio=0.2`, actor KL coefficient `1e-3`, SDPA,
  tensor-parallel size 1, four optimizer steps. Training rollout was
  unconstrained and invalid completions received zero reward; Trie decoding
  remains the evaluation path. Reward sanity on the real Text-SID artifacts
  gave exact `1.0`, same first two semantic levels `0.773066`, same first level
  `0.582510`, semantic-near valid item `0.366967`, unrelated valid item
  `0.291847`, and invalid SID `0.0`; the semantic component was non-zero for
  mapped valid items.
- Non-zero learning evidence in the retained console log:
  step 2 reward mean `0.009375`, advantage mean `-0.034798`, policy loss
  `0.0007567`, grad norm `44.5125`; step 3 reward mean `0.026864`, policy loss
  `-0.0000645`, grad norm `12.6117`; step 4 reward mean `0.009375`, policy loss
  `-0.0017756`, grad norm `20.9570`. Actor `pg_clipfrac` was `0.0` in this
  four-step smoke, while actor KL loss was non-zero at steps 3 and 4.
- Saved rollouts: 128 completions across `rollouts/1.jsonl`–`4.jsonl`;
  reward mean `0.0114035`, reward std `0.0566132`, 5 non-zero rewards, and
  14 syntactically parseable SID strings. Because the smoke rollout is
  intentionally unconstrained, this is not a final invalid-SID evaluation.
- Parameter verification: seven representative tensors were compared with
  the pre-run SFT checkpoint; all seven changed. Mean absolute delta was
  `1.15715e-6`, maximum delta `2.86102e-6`.
- Checkpoint/log artifacts: `outputs/verl_grpo/smoke_messages/global_step_4/`
  and `outputs/verl_grpo/smoke_messages/native_run.log`. Native verl did not
  emit ratio min/max statistics in its console logger; no such values are
  invented here.

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
