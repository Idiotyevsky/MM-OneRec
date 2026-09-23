# Experiment Log

## 2026-09-22 — Amazon23 real-data pipeline validation

Status: pipeline validation only; not a recommendation-quality experiment.

### Data

- Source: McAuley Lab Amazon Reviews 2023, `Industrial_and_Scientific`.
- Raw files: 2.347 GB reviews and 1.130 GB metadata, downloaded from the publisher's Hugging Face repository.
- Small input: first 200,000 reviews, iterative 5-core.
- Output: 28 users, 18 items, 202 interactions; train/valid/test = 139/17/18.

### Multimodal encoding

- Encoder: `google/siglip-base-patch16-224`, frozen.
- Text: title + description through SigLIP text tower.
- Image: first valid Amazon product image through SigLIP vision tower.
- Download: 17 valid, 1 missing, 0 terminal failures.
- Text/image shape: `[18, 768]` / `[18, 768]`.
- Fusion: L2-normalized weighted sum, alpha = 0.7; output `[18, 768]`.

### Semantic ID

- Method: existing RQ-VAE.
- Smoke parameters: 50 epochs, codebooks 8/8/8, latent dimension 32.
- Best reconstruction loss: 0.0005615.
- Total items: 18; unique SID: 11; collisions: 7; collision rate: 38.89%.

This collision rate is not representative: the run deliberately uses only 18 items and a tiny training schedule. No HR/NDCG or multimodal improvement claim is made.

## 2026-09-22 — 1k-item multimodal and SFT smoke

Status: locally trained pipeline smoke; not a recommendation-quality comparison.

### Data and images

- Parsed the first 1,000,000 Amazon23 reviews with iterative 5-core: 11,668 users, 7,493 items, 91,403 interactions.
- Deterministic dense subset: 2,564 users, 999 items, 17,073 interactions.
- Image cache: 967 valid images, 32 items without a usable URL, 0 terminal download failures.
- Frozen SigLIP text/image embeddings and weighted multimodal embedding all have shape `[999, 768]`; alpha = 0.7.

### Semantic ID

- Existing RQ-VAE, identical Text/MM configuration: 100 epochs, codebooks 32/32/32, latent dimension 64.
- Text raw SID: 638 unique, 361 collisions, collision rate 36.14%.
- MM raw SID: 764 unique, 235 collisions, collision rate 23.52%.
- Deterministic `<d_n>` collision suffix: both exported catalogs contain 999 unique SIDs and 0 collisions.
- The lower MM collision rate is a representation diagnostic only; it is not evidence of better recommendation quality.

### Qwen SFT

- Model: `Qwen/Qwen2.5-0.5B`, trained locally on one NVIDIA L40S.
- Each run samples 64 rows from each of SidSFT, SID/item alignment, and fusion sequence tasks: 192 training examples; validation has 64 examples.
- Common setup: one epoch, 12 optimizer steps, batch size 16, micro batch size 4, learning rate 1e-4, cutoff length 256.
- Text-SID: train loss 4.5846, eval loss 5.3429, runtime 111.9 seconds.
- MM-SID: train loss 4.5699, eval loss 5.1883, runtime 123.4 seconds.
- These tiny independently sampled smoke losses must not be interpreted as an MM improvement.

### 64-sample constrained decoding smoke

- Text-SFT and MM-SFT checkpoints each generated 64 validation/test smoke rows with 5-beam Trie decoding and max 8 new tokens.
- Text-SFT: HR@5/10/20 = 0/0/0; NDCG@5/10/20 = 0/0/0; coverage = 0.02803; invalid SID rate = 0.0.
- MM-SFT: HR@5/10/20 = 0.03125/0.03125/0.03125; NDCG@5/10/20 = 0.0234375/0.0234375/0.0234375; coverage = 0.02302; invalid SID rate = 0.0.
- Long-tail HR@10: Text = head/mid/tail 0/0/0; MM = 0.04/0.03448/0.0.
- These are 64-sample smoke results and must not be described as a stable multimodal lift. The full small-subset test evaluation is recorded below.

### Full 1k-item small-subset test evaluation

- Test split: 2,564 users, one held-out target per user; all four checkpoints generated exactly 5 candidates per user with 5-beam Trie decoding and max 8 new tokens. Because the run used 5 beams, HR@5/10/20 are numerically identical; the summary CSV still records all requested K values.
- Text-SFT: HR@5/10/20 = 0.005850/0.005850/0.005850; NDCG@5/10/20 = 0.002644/0.002644/0.002644; coverage = 0.044044; tail HR@10 = 0.002247; invalid SID rate = 0.0.
- Text-GRPO (stable 4-step run): HR@5/10/20 = 0.008190/0.008190/0.008190; NDCG@5/10/20 = 0.004788/0.004788/0.004788; coverage = 0.046046; tail HR@10 = 0.004494; invalid SID rate = 0.0.
- MM-SFT: HR@5/10/20 = 0.010140/0.010140/0.010140; NDCG@5/10/20 = 0.005374/0.005374/0.005374; coverage = 0.039039; tail HR@10 = 0.009112; invalid SID rate = 0.0.
- MM-GRPO (stable 4-step run): HR@5/10/20 = 0.007020/0.007020/0.007020; NDCG@5/10/20 = 0.003957/0.003957/0.003957; coverage = 0.042042; tail HR@10 = 0.006834; invalid SID rate = 0.0.
- MM-SID is higher than Text-SID under SFT on this fixed small subset, while the short MM-GRPO run is lower than MM-SFT. These are local observations from tiny checkpoints, not causal or generalizable improvement claims.

### GRPO diagnosis and corrected smoke

- The initial one-step run used 12 training prompts, 4 evaluation prompts, 2 candidates per prompt, `warmup_ratio=0.03`, and cosine scheduling. Its first logged learning rate was `0.0`; a safetensors comparison found all 290 shared tensors identical between each SFT checkpoint and its GRPO checkpoint. Therefore the earlier identical SFT/GRPO metrics were caused by a no-op update, not by a meaningful GRPO result.
- The script now exposes and records `warmup_ratio`, `lr_scheduler_type`, and `save_strategy`. The corrected run used `warmup_ratio=0`, `learning_rate=1e-6`, cosine scheduling, 4 optimizer steps, 12 prompts, 2 candidates, ranking reward, and `beta=0.04`. Text/MM each changed all 290 shared tensors; train KL was about 0.00062/0.00009 and invalid SID rate remained 0.0 at test time.
- The corrected run verifies that GRPO updates and the downstream constrained-decoding evaluation are connected. Four steps are still a smoke-scale experiment; no stable GRPO gain is claimed.

### Environment findings

- Repository-provided Qwen weight is a placeholder and bundled CSV/NPY assets are truncated.
- Base Python environment has incompatible Torch/torchao/torchaudio packages.
- SigLIP and RQ were therefore run with the existing `/home/zfs01/jiangjr/envs/opd` environment.
- Missing runtime dependencies found in the original RQ scripts: scikit-learn and polars; both are now declared.
