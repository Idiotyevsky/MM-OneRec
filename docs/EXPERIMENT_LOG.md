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


## 2026-09-24 — Qwen3-VL RQ-VAE quantization diagnostics and ablation

This round keeps the Qwen3-VL encoder, SFT/GRPO code, Trie evaluator, and the
existing SID artifacts unchanged.  The fixed embedding is
`outputs/qwen3vl/Industrial_and_Scientific_1m.qwen3vl-768.npy` (`7,493 x 768`).

### Embedding diagnostics

- finite values: `0` NaN/Inf; row norm mean `1.000000`, std `5.04e-8`, zero
  rows `0`;
- pairwise cosine sample (10,000 pairs): mean `0.002916`, std `0.206014`,
  p05 `-0.315778`, median `-0.008079`, p95 `0.353487`, fraction above `0.999`
  `0.0`;
- per-dimension standard deviation ranged from `0.006703` to `0.391615`
  (mean `0.0221887`, median `0.0137117`).  Thus the representation is not
  collapsed, but feature scales are strongly non-uniform; the full vectors are
  saved in `outputs/qwen3vl/embedding_diagnostics.json`.

### Assignment and diagnostics protocol

Every RQ evaluation uses `model.get_indices(..., use_sk=False)`, i.e. the same
nearest-neighbor assignment used by final SID export.  Sinkhorn is only a
training-time assignment when its configured epsilon is positive.  Each run
writes per-evaluation JSONL to `outputs/qwen3vl/rq_ablation_qwen3vl/<run>/diagnostics.jsonl`
and a consolidated CSV to `results/rq_ablation_qwen3vl.csv`.  The trainer now
also saves `best_collision_model.pth` and `best_reconstruction_model.pth`; the
main selector is minimum raw collision, not the last epoch or reconstruction
loss alone.

### Nine controlled runs (Q0–Q8)

All runs use seed `2024`, 100 epochs maximum, evaluation every 20 epochs,
`32/32/32` unless noted, beta `0.25`, quantization weight `1.0`, and the same
Qwen3-VL embedding.  The reported values below are from the best raw-collision
checkpoint.

| Run | Preprocess | Sinkhorn epsilon | Latent/codebooks | Best epoch | Recon. loss | Raw unique SID | Collision rate | Prefix@2 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Q0 | none | `0` | 64 / 32³ | 19 | `0.000712` | 5,124 | `0.316162` | 947 |
| Q1 | z-score | `0` | 64 / 32³ | 19 | `0.989669` | 3,341 | `0.554117` | 550 |
| Q2 | none | `0.001` | 64 / 32³ | 19 | `0.001299` | 46 | `0.993861` | 23 |
| Q3 | none | `0.003` | 64 / 32³ | 19 | `0.000716` | 5,290 | `0.294008` | 964 |
| Q4 | z-score | `0.003` | 64 / 32³ | 39 | `0.946734` | 3,892 | `0.480582` | 775 |
| Q5 | none (selected from Q3) | `0.003` | 128 / 32³ | 19 | `0.000692` | 5,168 | `0.310290` | 941 |
| Q6 | none (selected from Q3) | `0.003` | 128 / 64³ | 19 | `0.000658` | 6,515 | `0.130522` | 2,511 |
| Q7 | none (selected from Q3) | `0.003` | 128 / 128³ | 19 | `0.000628` | 7,075 | `0.055785` | 4,345 |
| Q8 | none (selected from Q3) | `0.003` | 128 / 256³ | 19 | `0.000611` | 7,261 | `0.030962` | 5,535 |

Codebook utilization for Q0/Q3/Q5/Q6 was `100%` at every layer.  Q7
kept `128/128/128` active codes, normalized entropy `0.9896/0.9922/0.9912`,
and perplexity `121.69/123.23/122.65` out of 128.  Q8 used `256/256/249`
codes, normalized entropy `0.9866/0.9884/0.9792`, and perplexity
`237.61/240.02/228.09` out of 256; its third layer is the first capacity run
with a small unused-code tail.  Q2 remains a clear assignment-collapse regime:
only `11/8/7` codes were used in the three layers.  Z-score preprocessing also
hurt this RQ setup: its reconstruction is measured in the transformed space
and its raw collision rate is substantially worse.

With the fixed Q3 training assignment (`epsilon=0.003`, no z-score), increasing
capacity gives a smooth RQ-only trade-off: Q6 reaches `13.0522%` collision, Q7
reaches `5.5785%`, and Q8 reaches `3.0962%`.  Q7 improves reconstruction from
`0.0006579` (Q6) to `0.0006279`; Q8 reaches `0.0006112`.  Collision is selected
at epoch 19 for all three capacity runs; later reconstruction improvements are
accompanied by collision worsening (Q7 final-eval `8.4746%`, Q8 `4.9780%`).

Q8 is the RQ-only minimum-collision candidate, while Q7 is the more
conservative semantic-granularity candidate: all three Q7 layers remain fully
utilized and its first-level prefix has 128 branches, whereas Q8 has 256 first-level
branches and 5,535 unique two-level prefixes.  Both candidates were subsequently
passed through the common Qwen3-4B SFT/evaluation protocol below.

### Semantic prefix inspection

Prefix examples for all candidates are stored under each analysis directory as
`prefix_examples.json`.  The sampled groups remain semantically interpretable
when capacity increases: Q6 `<a_40>` groups caster-wheel products (centroid
cosine around `0.905–0.918`), Q7 `<a_105>` groups tabletop epoxy products
(about `0.779–0.803`), and Q8 includes `<a_37>` 3D-printer filaments and
`<a_100>` 3D printers (about `0.784–0.939`).  These are sampled diagnostic
evidence, not a substitute for HR/NDCG evaluation.

The formal exporter independently reproduced the same raw statistics for all
selected capacity candidates.  The artifact paths are:

- `...qwen3vl_q3.index.stats.json`: 5,290 unique / `0.294008` collision;
- `...qwen3vl_q6.index.stats.json`: 6,515 unique / `0.130522` collision;
- `...qwen3vl_q7.index.stats.json`: 7,075 unique / `0.055785` collision;
- `...qwen3vl_q8.index.stats.json`: 7,261 unique / `0.030962` collision.

The Q7/Q8 capacity runs used commit `79b9b16` as their code base, host
`4090-1` / `CUDA_VISIBLE_DEVICES=0`, the fixed Qwen3-VL PCA-768 embedding,
seed `2024`, 100-epoch maximum, and no z-score preprocessing.

## 2026-09-25 — Qwen3-4B Q6/Q7/Q8 SFT comparison

This run tests the downstream effect of the three fixed Qwen3-VL RQ-VAE
capacity candidates.  The interaction rows were generated independently for
each index and compared with `scripts/mm/compare_sid_csv.py`; train/valid/test
contain exactly `63,788 / 7,973 / 7,974` rows in all tracks, with identical
user/history/target item columns and zero missing items.

| Field | Value |
|---|---|
| dataset | Amazon Reviews 2023, `Industrial_and_Scientific_1m` |
| splits | train 63,788; valid 7,973; test 7,974 |
| item representation | fixed Qwen3-VL PCA-768 embedding |
| generator | `Qwen/Qwen3-4B-Instruct-2507` |
| SFT | full-parameter BF16, seed 42, 1 epoch, lr `1e-5`, effective batch 32, micro batch 1, cutoff 256, rec repeat 2 |
| auxiliary mixture | recommendation fraction `0.6179` on full data; SID/title and fusion auxiliary tasks retained |
| checkpoint policy | model-only checkpoints, minimum validation loss rule, independent output directory per tokenizer |
| GPUs | Q6 CUDA 0, Q7 CUDA 1, Q8 CUDA 2 on host `4090-1` |
| code commit | `fa4d9501090f7d45cd7872092c236cb041956ff5` |

### Atomic vocabulary and SID length

| Track | Added SID tokens | `a/b/c` tokens | suffix tokens | 3-token items | 4-token items | max raw collision group |
|---|---:|---:|---:|---:|---:|---:|
| Q6 (`64^3`) | 222 | 64 / 64 / 64 | 30 | 5,826 | 1,667 | 30 |
| Q7 (`128^3`) | 394 | 128 / 128 / 128 | 10 | 6,755 | 738 | 10 |
| Q8 (`256^3`) | 784 | 256 / 256 / 256* | 23 | 7,092 | 401 | 23 |

`*` Q8 uses 249 distinct third-level codes in the exported catalog.  Every
added token was checked after `tokenizer.add_tokens` and encodes to exactly one
ID.  `<d_n>` remains a collision disambiguation token, not a fourth RQ layer.

### SFT artifacts

| Track | Train runtime | Train loss | Validation loss | Checkpoint | Status |
|---|---:|---:|---:|---|---|
| Q6 | 40,208 s | 2.1361 | 2.5633 | `outputs/qwen3_4b/q6_sft/final_checkpoint` | complete |
| Q7 | 39,929 s | 2.2415 | 2.9814 | `outputs/qwen3_4b/q7_sft/final_checkpoint` | complete |
| Q8 | 40,559 s | 2.3668 | 3.5308 | `outputs/qwen3_4b/q8_sft/final_checkpoint` | complete |

The Q7 development smoke used four optimizer steps and separately verified
reload, finite loss, non-zero gradients on target SID rows, and a valid Trie
constrained catalog SID.  Its artifacts are under
`outputs/qwen3_4b/q7_sft_smoke/`.

Full-test evaluation uses the same `scripts/evaluate.py` protocol for all
tracks: 7,974 rows, 20 beams, `max_new_tokens=8`, and Trie-constrained SID
generation.  The final evaluation uses deterministic generation
(`do_sample=False`, `use_model_defaults=False`) and is stored under
`outputs/qwen3_4b/q{6,7,8}_sft_eval_beam20_deterministic/`.  The earlier
sampling/evaluation attempts remain as separate artifacts and are not included
in the table below.

### Q6/Q7/Q8 full-test recommendation metrics

All values below are computed from the complete 7,974-row prediction files by
`scripts/mm/evaluate_metrics.py`.  `invalid_sid_rate` is zero for all three
tracks.

| Track | Raw collision | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q6 (`64^3`) | 0.130522 | 0.004891 | 0.007274 | 0.009782 | 0.003104 | 0.003850 | 0.004477 | 0.007340 |
| Q7 (`128^3`) | 0.055785 | **0.008277** | **0.010660** | 0.015300 | **0.004969** | **0.005736** | **0.006911** | 0.004938 |
| Q8 (`256^3`) | **0.030962** | 0.005016 | 0.009782 | **0.016805** | 0.002277 | 0.003806 | 0.005592 | 0.004271 |

The lower raw collision rate does not by itself solve collided targets.  The
collision-group sample counts are Q6/Q7/Q8 = `1,971 / 822 / 497`, and
collision-group HR@10 and NDCG@10 are `0.0 / 0.0` for every track.  Q7 has the
strongest HR@5/10 and NDCG@5/10 and is therefore the primary tokenizer for the
next RL comparison.  Q8 is retained as the higher-resolution runner-up because
it has the best HR@20.  No GRPO run was started from these checkpoints in this
round.

The complete-precision artifact is
`results/qwen3_4b_sft_summary.csv`; each row also contains head/mid/tail and
collision/non-collision fields.  SFT was trained with code commit `fa4d950`;
the deterministic evaluation fix and final metrics use commit `d77c828`.
