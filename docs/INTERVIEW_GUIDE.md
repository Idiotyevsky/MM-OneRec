# Implementation Guide

This guide follows the code in the current repository.  It focuses on the current implementation and public project structure.

## 1. Why use a generative recommender?

The model generates a compact item code rather than selecting from a flat
classification head.  Semantic IDs make item structure explicit and allow the
same autoregressive interface to produce a ranked list.

## 2. How is it different from SASRec?

`minionerec/sasrec.py` scores a bounded item-classification head from a causal
self-attention user sequence.  MM-OneRec predicts the tokens of a Semantic ID;
the Trie evaluator maps valid token paths back to catalog items.

## 3. Why not generate raw item IDs?

Raw IDs are arbitrary symbols with no useful prefix structure.  A generated
prefix cannot express item similarity and an unconstrained decoder can produce
an ID absent from the catalog.

## 4. What is a Semantic ID?

It is a sequence of discrete residual-quantization codes, for example
`<a_12><b_7><c_3>`.  The index JSON maps each catalog item to its code path.

## 5. How does RQ-VAE work here?

The encoder maps an item vector to a latent; residual vector quantizers encode
the remaining error level by level; the decoder reconstructs the input.  The
export script uses the trained code indices as SID tokens.

## 6. Why residual quantization?

Each later code models what earlier codebooks did not explain, so a prefix can
retain coarse structure while suffixes retain finer detail.

## 7. What is SID collision?

Two items can share all raw RQ codes.  `scripts/mm/sid_stats.py` reports raw
unique/collision counts.  Export can append `<d_n>` to disambiguate a path,
but that suffix is not treated as a semantic RQ layer.

## 8. How can collisions be reduced?

Measure them first, then change a common RQ capacity/configuration, train a
better representation, or use a documented disambiguation policy.  A track
must not silently receive a different capacity.

## 9. Why add images?

Product appearance, material, shape and packaging may be absent from title and
description.  The SigLIP track is a reproducible baseline; Qwen3-VL provides a
joint image-text representation for the stronger track.

## 10. How does SigLIP-MM fuse modalities?

The baseline L2-normalizes text/image vectors and computes
`normalize(0.7 * text + 0.3 * image)`.  The image cache stores a validity mask;
missing images fall back to the text vector only in this baseline track.

## 11. How does Qwen3-VL fusion differ?

`qwen3_vl_encoder.py` sends image, title and description through one frozen
Qwen3-VL forward pass.  The default representation is the last valid text
position after the joint context, not a weighted sum of two independent
vectors.

## 12. Why not use a VLM as the generator?

The project isolates the item-representation question.  Keeping the existing
Qwen generator, SID format, RQ capacity and evaluator makes an ablation
interpretable and avoids multiplying model-memory costs.

## 13. What happens when an image is missing?

The Qwen3-VL track invokes the same model with title and description only and
records `has_image=false`; it does not substitute a SigLIP vector.  The
SigLIP baseline has its own documented text fallback.

## 14. What is recommendation-aware alignment?

It is a second-stage projector trained on frozen VLM vectors.  A recency-pooled
history vector is matched to the next training item with sampled-negative
InfoNCE.  Only the training CSV is read, so valid/test targets are excluded.

## 15. What is the SFT input/output?

Input is a prompt containing history SIDs; output is the target next-item SID.
The causal LM loss is applied to target tokens.  Images are already accounted
for in the SID construction stage.

## 16. Why run RL after SFT?

SFT maximizes likelihood of the observed target.  RL can optimize a ranking-
oriented reward over multiple sampled candidates from the same user state.
Whether it improves a cutoff is an empirical question.

## 17. What did the old RL actually implement?

`ReReTrainer` is retained as `legacy_group_relative_rl`.  Its policy term uses
`exp(logp - logp.detach())`, so it does not provide a true old-policy ratio or
clipped surrogate.  Its historical metrics remain useful as a baseline.

## 18. What is standard GRPO in this repository?

`rl/verl/run_grpo.py` launches native `verl.trainer.main_ppo` with
`algorithm.adv_estimator=grpo`, rollout group size `n>1`, actor clip ratio,
`ppo_epochs`, and actor-side reference KL.  The repository does not reimplement
the optimizer loop.

## 19. How is GRPO different from PPO?

PPO normally learns a critic/value estimate.  GRPO computes a standardized
advantage from rewards of multiple responses to one prompt and therefore does
not need a critic.  Both use old-policy ratio clipping; reference KL is a
separate anchor.

## 20. What are `pi_old` and `pi_ref`?

`pi_old` is the policy that produced the rollout log probabilities used in the
importance ratio.  `pi_ref` is the fixed reference model used by the KL loss.
They are not interchangeable.

## 21. How is group advantage computed?

For group rewards `R_i`, the implementation uses
`(R_i - mean(R_group)) / (std(R_group) + epsilon)`.  The CPU helper tests this
normalization; native verl performs the training update.

## 22. What reward modes are available?

`exact` rewards only an exact SID; `sid_hier` uses 0.5/0.3/0.2 prefix weights;
`semantic` uses valid-item cosine mapped to [0,1]; `hybrid` uses exact 1 or
0.6 hierarchical + 0.4 semantic.

## 23. Why remove string similarity and completion-index reward?

Character similarity can reward malformed token strings, and a candidate's
array position is not its recommendation rank.  The new reward parses SID
layers, resolves valid catalog items, and receives rank from evaluation rather
than group ordering.

## 24. Why use KL regularization?

Actor-side reference KL keeps the policy near the SFT/reference distribution
and reduces reward hacking.  It is configured with `kl_loss_coef`, default
`1e-3`, and is logged by native verl.

## 25. Why is training rollout unconstrained?

The current verl/vLLM integration in this checkout does not expose a stable
batch Trie logits processor.  The launcher therefore keeps rollout native and
assigns zero reward to invalid SIDs.  Evaluation still uses the existing Trie;
there is no vLLM monkey patch.

## 26. How does Trie decoding work?

At each generation step, `minionerec/logit_processor.py` masks tokens that are
not children of the current valid SID prefix.  Completed valid paths are held
at EOS and beam search returns catalog-mappable candidates.

## 27. What is HR@K?

Hit Rate is the fraction of test examples whose target item appears in the top
K predictions.  It ignores the exact position inside the cutoff.

## 28. What is NDCG@K?

NDCG gives a hit at rank `r` weight `1 / log2(r + 1)` and averages it over test
examples.  It distinguishes a rank-1 hit from a rank-K hit.

## 29. What do head/mid/tail metrics measure?

Training interaction frequency defines head (top 20%), mid (middle 60%) and
tail (bottom 20%) item buckets.  The evaluator reports HR@10/NDCG@10 per
bucket to show whether representation changes alter long-tail behavior.

## 30. What is the main deployment bottleneck?

Offline image/VLM encoding is the main representation cost; online generation
is dominated by autoregressive decoding and beam expansion.  Precomputed SIDs,
cached Trie structures, a smaller generator or a two-stage candidate retriever
can reduce serving latency.
