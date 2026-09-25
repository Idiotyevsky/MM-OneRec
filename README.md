# MM-OneRec

MM-OneRec is a multimodal generative recommender built around Semantic IDs,
Qwen3-VL item representations, Qwen3-4B post-training, and constrained
decoding. It formulates sequential next-item recommendation as autoregressive
generation of a catalog item code.

~~~text
Title + Description + Image
            │
            ▼
      Frozen Qwen3-VL
            │
            ▼
        PCA / L2 norm
            │
            ▼
          RQ-VAE
            │
  <a_i><b_j><c_k>[<d_n>]
            │
            ▼
        Qwen3-4B
        SFT → GRPO
            │
            ▼
 Trie-constrained beam search
            │
            ▼
      Top-K recommendation
~~~

## Highlights

- **Multimodal item representation**: title, description, and product image
  are encoded jointly by a frozen Qwen3-VL model.
- **Hierarchical Semantic IDs**: RQ-VAE converts continuous item vectors into
  multi-level discrete codes that can be generated token by token.
- **LLM-based recommendation**: Qwen3-4B predicts the next item SID from a
  user's chronological SID history.
- **Recommendation post-training**: supervised fine-tuning and native verl
  GRPO with group-relative rewards, clipping, and reference KL.
- **Valid generative retrieval**: a prefix Trie masks invalid SID continuations
  during evaluation, so decoded paths map back to catalog items.
- **Full-catalog protocol**: deterministic beam search, ranking metrics,
  coverage, validity, and long-tail analysis are supported.

## Multimodal Item Representation

### Qwen3-VL

multimodal/qwen3_vl_encoder.py uses Qwen/Qwen3-VL-4B-Instruct as a frozen
item encoder. Each item is represented with an image when available and with
title plus description. The default last_token pooling reads the last valid
text position after the joint image-text context; text_mean is available for
controlled comparisons.

Missing images use the same Qwen3-VL model with a text-only message. They do
not switch to a different embedding space. Encodings are saved together with
item IDs, image-presence masks, model metadata, and optional PCA statistics.
PCA can project the native hidden size to 768 dimensions before RQ-VAE.

~~~bash
bash scripts/mm/encode_qwen3vl.sh \
  --items <item-metadata.json> \
  --manifest <cached-image-manifest.jsonl> \
  --output data/embeddings/items.qwen3vl.npy \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --pooling last_token \
  --projection pca --target-dim 768 \
  --batch-size 4 --device cuda:0
~~~

The repository also keeps a frozen SigLIP baseline. Its fusion is:

~~~text
e_mm = normalize(0.7 * normalize(e_text) + 0.3 * normalize(e_image))
~~~

### Recommendation-aware Alignment

multimodal/rec_alignment.py optionally trains a small projector while the
Qwen3-VL encoder stays frozen:

~~~text
Linear → GELU → Linear → L2 normalization
~~~

A recency-pooled history vector is matched to the next training item with
sampled-negative InfoNCE. Only training interactions are used. The resulting
item vectors follow the same RQ-VAE and recommendation pipeline.

## Semantic ID Tokenization

All representation tracks use the same downstream interface:

~~~text
item embedding
    ↓
RQ-VAE / RQ-KMeans
    ↓
SID index and interaction CSV
    ↓
Qwen SFT → RL post-training → Trie evaluation
~~~

RQ-VAE residual quantization produces a hierarchy such as:

~~~text
<a_i><b_j><c_k>
~~~

The implementation supports controlled codebook-capacity studies including
64^3, 128^3, and 256^3 configurations. Codebook utilization, entropy,
perplexity, prefix diversity, reconstruction quality, and raw SID collisions
can be inspected with scripts/mm/analyze_codebook.py.

If multiple items share the same raw RQ path, export adds a deterministic
collision-disambiguation suffix:

~~~text
<a_i><b_j><c_k><d_n>
~~~

<d_n> is an addressing suffix, not a fourth semantic RQ layer. Items with
unique raw SIDs terminate after the three semantic tokens, while collision
groups use only as many suffix values as required. The suffix is handled by
the tokenizer and Trie but is excluded from hierarchical semantic rewards.

## Generative Recommendation

### Qwen3-4B SFT

The core task is:

~~~text
user history SID sequence → next-item SID
~~~

The representation model is used before training to construct the catalog SID.
Images and VLM hidden states are not inserted into the Qwen prompt. SID tokens
are added as atomic tokenizer entries and the model vocabulary is resized
accordingly. The causal objective is applied only to target tokens.

The SFT entry point is scripts/sft.py; the shell wrapper is
scripts/mm/train_sft.sh.

### Trie-constrained Decoding

minionerec/logit_processor.py builds a prefix map from the catalog SID index.
At each decoding step it permits only valid child tokens. Beam search can
therefore return several catalog-mappable item candidates while preventing
invalid SID paths.

## Recommendation Post-training

### Native verl GRPO

The recommended RL entry point is rl/verl/run_grpo.py or its shell wrapper
rl/verl/run_grpo.sh. Native verl provides rollout, old-policy log
probabilities, clipped policy updates, and optimizer state management. The
configuration separates:

~~~text
pi_old  = policy that generated the rollout
pi_ref  = reference model used for KL regularization
~~~

The group-relative advantage is computed from multiple responses for the same
prompt. Available reward modes are:

| Mode | Meaning |
| --- | --- |
| exact | one only for an exact target SID |
| sid_hier | weighted match of the semantic RQ levels |
| semantic | valid-item embedding similarity mapped to a stable range |
| hybrid | exact reward, otherwise hierarchical plus semantic reward |

The training rollout remains native to verl; invalid or unmapped SIDs receive
zero reward. Evaluation uses the catalog Trie and deterministic constrained
beam search.

### Legacy Compatibility Path

scripts/rl.py and minionerec/trainer.py are retained as
legacy_group_relative_rl for reproducing the original MiniOneRec-style
training path. New experiments should use the native verl launcher.

## Evaluation

MM-OneRec provides a deterministic full-catalog evaluation pipeline. For each
held-out user state, the generator produces a beam of Semantic IDs, the Trie
maps valid paths back to catalog items, and the evaluator compares the ranked
items with the next interaction.

Supported reports include:

- HR@5, HR@10, and HR@20
- NDCG@5, NDCG@10, and NDCG@20
- catalog coverage and invalid SID rate
- head / mid / tail performance
- collision-group versus non-collision targets
- recommendation and first-prefix concentration diagnostics

Run-specific predictions, checkpoints, logs, and metric files are kept outside
the public source snapshot.

## Quick Start

The commands below show the public entry points. Replace the data paths with
local Amazon metadata, image manifests, SID files, and generator checkpoints.

1. Encode item content:

   ~~~bash
   bash scripts/mm/encode_qwen3vl.sh --help
   ~~~

2. Train or export a Semantic-ID tokenizer:

   ~~~bash
   python scripts/mm/run_rq_ablation.py --help
   bash scripts/mm/build_sid.sh --help
   ~~~

3. Build recommendation samples and run SFT:

   ~~~bash
   bash scripts/mm/train_sft.sh \
     --base_model <generator-checkpoint> \
     --train_file <sid-train.csv> \
     --eval_file <sid-valid.csv> \
     --sid_index_path <sid-index.json> \
     --output_dir outputs/local_sft
   ~~~

4. Evaluate with constrained beam search:

   ~~~bash
   bash scripts/mm/evaluate.sh \
     --base_model outputs/local_sft/final_checkpoint \
     --train_file <sid-train.csv> \
     --info_file <catalog-info.tsv> \
     --category Industrial_and_Scientific \
     --test_data_path <sid-test.csv> \
     --result_json_data outputs/local_eval/predictions.json \
     --num_beams 20 --max_new_tokens 8
   ~~~

5. Optionally prepare parquet data and launch native GRPO:

   ~~~bash
   python -m rl.verl.prepare_data --help
   bash rl/verl/run_grpo.sh --config rl/verl/configs/grpo_small.yaml
   ~~~

Every entry point supports --help; smoke helpers under scripts/mm/ and tests/
provide smaller checks before a full run.

## Project Structure

~~~text
MM-OneRec/
├── multimodal/        # Qwen3-VL, SigLIP baseline, and RecAlign
├── rq/                # RQ-VAE and RQ-KMeans tokenization
├── minionerec/        # datasets, Trie, legacy trainer, and SASRec
├── scripts/           # SFT and evaluation entry points
├── scripts/mm/        # image, SID, diagnostics, and evaluation utilities
├── rl/verl/           # native GRPO data conversion and launcher
├── tests/              # unit and regression tests
├── config/             # runtime configuration examples
└── docs/               # environment and implementation notes
~~~

## Tests

Run the CPU regression suite from the repository root:

~~~bash
python -m pytest -q
~~~

The tests cover tokenizer extension, multimodal item ordering, missing-image
handling, pooling masks, PCA shapes, SID rewards, collision suffixes, parquet
conversion, and the shared Trie/evaluator utilities.

## Attribution

MM-OneRec is based on and adapted from MiniOneRec. The original license and
attribution are retained in LICENSE.
