# MM-OneRec

MM-OneRec is a semantic-ID generative recommendation system.  It keeps the
MiniOneRec downstream task fixed—user history semantic IDs are mapped to the
next-item semantic ID—and makes the item representation and policy-training
choices explicit:

```text
Amazon item (title, description, image)
        |
        +-- Text-only representation
        +-- SigLIP text/image baseline
        +-- Frozen Qwen3-VL joint representation
        +-- Qwen3-VL + train-only recommendation alignment
        |
        v
RQ-VAE residual quantization -> hierarchical Semantic ID
        |
        v
Qwen SFT -> native verl GRPO -> Trie-constrained Top-K evaluation
```

The repository preserves the original SigLIP-MM, Text-SID, SFT, legacy RL,
Trie decoding and Amazon23 artifacts.  New representation tracks use the same
RQ/SID, generator, split and evaluator so their effects can be separated.

## 1. Tracks and implementation status

| Track | Representation | SID | SFT | RL | Evaluation |
|---|---|:---:|:---:|---|:---:|
| Text baseline | existing text embedding | yes | yes | legacy RL artifact | formal artifact |
| SigLIP-MM baseline | normalized `0.7 * text + 0.3 * image` | yes | yes | legacy RL artifact | formal artifact |
| Text/SigLIP + native verl GRPO | same baseline representations | launcher ready | data conversion ready | requires verl run | not run in this checkout |
| Qwen3VL-MM | frozen joint VLM representation | encoder ready | not run | not run | not run |
| Qwen3VL-RecAlign | frozen VLM + train-only projector | alignment code ready | not run | not run | not run |

“Formal artifact” refers only to files already present under
`outputs/formal_amazon23_1m/`; no Qwen3-VL or native-verl metric is inferred
from those files.

## 2. Representation layer

### Existing SigLIP baseline

`multimodal/text_encoder.py`, `multimodal/image_encoder.py` and
`multimodal/multimodal_encoder.py` implement the original reproducible track:

```text
title + description -> frozen SigLIP text tower
image               -> frozen SigLIP vision tower
e_mm = normalize(0.7 * normalize(e_text) + 0.3 * normalize(e_image))
```

Image URLs are downloaded once by `multimodal/image_downloader.py`.  A missing
or broken image is represented by a false mask and weighted fusion falls back
to the text vector.  This behavior is retained for the baseline.

### Frozen Qwen3-VL item representation

`multimodal/qwen3_vl_encoder.py` adds the joint track.  The default model is
`Qwen/Qwen3-VL-4B-Instruct`; development runs can select a smaller compatible
Qwen3-VL checkpoint.  The model is frozen and receives one item at a time in
the following deterministic form:

```text
Represent this product for recommendation.
Focus on category, function, appearance, material, style and product attributes.
Title: ...
Description: ...
Product representation:
```

The default `last_token` pooling takes the last valid text position using the
attention mask.  `text_mean` is also available and excludes known image-token
IDs when the processor exposes them.  A missing image uses the same Qwen3-VL
model with title and description only; it never falls back to the SigLIP
space.  The output records `has_image`, `raw_dim`, pooling, model name and
projection metadata.  Optional PCA projection is fitted only on item content
vectors, for example from the native hidden size to 768 dimensions:

```bash
python -m multimodal.qwen3_vl_encoder \
  --items data/Amazon/index/Industrial_and_Scientific.item.json \
  --manifest data/cache/amazon23_1k/images.jsonl \
  --output data/embeddings/industrial.qwen3vl.npy \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --pooling last_token --projection pca --target-dim 768 \
  --batch-size 1 --device cuda:0
```

The command writes the matrix, a `.json` metadata sidecar, aligned
`.item_ids.json`, and `.has_image.npy`.

### Recommendation-aware alignment

`multimodal/rec_alignment.py` keeps the VLM frozen and trains only:

```text
Linear -> GELU -> Linear -> L2 normalization
```

The user vector is a recency-weighted mean of history item vectors.  The
projector is optimized with sampled-negative InfoNCE using only the specified
training CSV.  Validation and test files are never opened by this module.
The resulting 128D/256D matrix can be sent through the same RQ-VAE code path:

```bash
python -m multimodal.rec_alignment \
  --embeddings data/embeddings/industrial.qwen3vl.npy \
  --items data/Amazon/index/Industrial_and_Scientific.item.json \
  --train-csv data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --output data/embeddings/industrial.qwen3vl-recalign.npy \
  --output-dim 256 --epochs 3 --negatives 32 --device cuda:0
```

## 3. Semantic IDs and common downstream path

Every representation track follows the same path:

```text
item embedding -> RQ-VAE/RQ-KMeans -> SID index -> SID CSV
             -> Qwen SFT -> RL -> Trie evaluator
```

The default RQ capacity remains 32/32/32 codebooks and latent dimension 64;
track-specific capacity changes are not silently introduced.  SID export
reports total items, raw unique SIDs, raw collisions and post-dedup
collisions.  The optional `<d_n>` token disambiguates a raw collision and is
not treated as an additional semantic RQ layer by the new reward code.

## 4. Training

### SFT

The existing `scripts/sft.py` remains the generator SFT entry point.  Its
causal objective is:

```text
P(next SID | history SID)
```

Item representation affects the SFT task only through the exported SID index;
the VLM is not inserted into the Qwen generator prompt.

### Legacy RL baseline

`scripts/rl.py` and `minionerec/trainer.py` are retained as
`legacy_group_relative_rl`.  They reproduce the old artifacts, but their
detached self-ratio policy term is not a standard old-policy clipped
surrogate.  They are no longer the recommended RL entry point.

### Native verl GRPO

The new entry point is `rl/verl/run_grpo.sh`.  It delegates rollout, old
policy log-probabilities, PPO clipping, reference-model KL and optimizer
updates to `verl.trainer.main_ppo`:

```bash
python -m rl.verl.prepare_data \
  --interactions data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --output data/verl/industrial/train.parquet \
  --dataset Amazon23 --category Industrial_and_Scientific

python -m rl.verl.run_grpo \
  --config rl/verl/configs/grpo_small.yaml --dry-run

# Requires an installed, compatible verl environment.
bash rl/verl/run_grpo.sh --config rl/verl/configs/grpo_formal.yaml
```

The generated native overrides include:

```text
algorithm.adv_estimator=grpo
actor_rollout_ref.rollout.n=8       # formal; 4 in smoke
actor_rollout_ref.actor.ppo_epochs=1
actor_rollout_ref.actor.clip_ratio=0.2
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.actor.kl_loss_coef=1e-3
```

Thus `pi_old` is the rollout policy snapshot and `pi_ref` is the separate KL
anchor.  The production implementation is native verl; the CPU helper in
`rl/verl/grpo_math.py` only verifies the ratio/clipping semantics:

$$
r_t = \exp(\log\pi_\theta - \log\pi_{\mathrm{old}}),\qquad
A_i = \frac{R_i-\mu_R}{\sigma_R+\epsilon}.
$$

The launcher accepts `ppo_epochs: 1` or `2` in the config and writes the
resolved configuration beside the run output.  It never falls back to the
legacy trainer when `verl` is unavailable.

### Reward modes

`rl/verl/reward.py` provides four modes:

| Mode | Definition |
|---|---|
| `exact` | `1` only for exact target SID |
| `sid_hier` | prefix match with normalized `0.5 / 0.3 / 0.2` layer weights |
| `semantic` | valid predicted item cosine mapped from `[-1,1]` to `[0,1]` |
| `hybrid` | exact `1`; otherwise `0.6 * sid_hier + 0.4 * semantic` |

Invalid/unmapped items have semantic reward `0`.  The reward function is
loaded by verl through `reward.custom_reward_function.path`; weights are kept
in configuration rather than trainer code.  Training rollout is intentionally
unconstrained in this version: invalid SID gets zero reward.  Evaluation keeps
the existing Trie prefix constraint, and the launcher refuses a request for a
vLLM monkey patch.

## 5. Evaluation and preserved formal artifacts

The evaluator reports HR@5/10/20, NDCG@5/10/20, coverage, invalid-SID rate
and head/mid/tail buckets.  Existing formal artifacts use Amazon23
`Industrial_and_Scientific_1m`, the complete 7,974-row test split, 20-beam Trie
decoding and the same metric script for all four checkpoints.

| Existing artifact | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | Coverage | Tail HR@10 | Invalid SID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Text-SID + SFT | 0.000878 | 0.001129 | 0.003261 | 0.000566 | 0.000641 | 0.001181 | 0.006806 | 0.000627 | 0.0 |
| Text-SID + legacy RL | 0.000627 | 0.001379 | 0.003762 | 0.000325 | 0.000554 | 0.001171 | 0.004671 | 0.001046 | 0.0 |
| SigLIP-MM-SID + SFT | 0.000251 | 0.001630 | 0.007775 | 0.000103 | 0.000530 | 0.002067 | 0.006940 | 0.000418 | 0.0 |
| SigLIP-MM-SID + legacy RL | 0.000376 | 0.002508 | 0.007650 | 0.000160 | 0.000830 | 0.002129 | 0.004137 | 0.000418 | 0.0 |

Full precision values and all head/mid/tail fields remain in
[`results/summary.csv`](results/summary.csv); prediction/config artifacts are
under `outputs/formal_amazon23_1m/`.  These numbers are not reused as
Qwen3-VL or native-verl results.

## 6. Tests and dependencies

The CPU test suite covers image/SID alignment, pooling masks, missing-image
behavior, PCA shape, reward ordering, collision suffix handling, GRPO ratio
diagnostics, parquet schema, and train-only alignment.  Run:

```bash
python -m pytest -q
```

The current checkout result is `56 passed, 3 skipped`.  Install the optional
native tracks in a compatible environment with:

```bash
pip install -r requirements.verl.txt
```

`verl` and the Qwen3-VL model weights are deliberately not bundled in the
repository.  The existing baseline environment and artifacts remain usable
without them.

## 7. Layout

```text
multimodal/
  image_downloader.py       image cache and manifest
  image_encoder.py          SigLIP baseline vision encoder
  multimodal_encoder.py     SigLIP fusion baseline
  qwen3_vl_encoder.py       frozen joint Qwen3-VL representations
  rec_alignment.py          train-only recommendation projector
rl/verl/
  prepare_data.py           interaction CSV -> parquet
  reward.py                 exact/hier/semantic/hybrid rewards
  grpo_math.py              CPU ratio/clip diagnostics
  run_grpo.py               native verl launcher
  configs/                  smoke/formal configs
rq/                         RQ-VAE and RQ-KMeans
minionerec/                 data, legacy trainer, Trie and SASRec
scripts/mm/                 baseline image/SID/evaluation commands
outputs/formal_amazon23_1m/ preserved formal artifacts
results/summary.csv         preserved comparable summary
docs/EXPERIMENT_LOG.md      run provenance
docs/INTERVIEW_GUIDE.md     implementation-focused explanations
```

## 8. Attribution

MM-OneRec is based on and adapted from MiniOneRec.  The original license and
attribution are retained in [`LICENSE`](LICENSE).
