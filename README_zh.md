# MM-OneRec：多模态 Semantic ID 生成式推荐

MM-OneRec 保持 MiniOneRec 的推荐任务定义不变：输入用户历史 Semantic ID，生成下一商品的 Semantic ID。当前仓库同时保留原有 Text-SID、SigLIP-MM、SFT、legacy RL、Trie 解码和 Amazon23 正式结果，并新增标准 verl GRPO 入口、冻结 Qwen3-VL Item Representation，以及可选的推荐感知对齐。

```text
Amazon 商品：title / description / image
        ↓
Text-only | SigLIP-MM | Qwen3-VL-MM | Qwen3-VL-RecAlign
        ↓
同一套 RQ-VAE → Semantic ID → Qwen SFT → verl GRPO
        ↓
Trie-constrained Beam Search → HR / NDCG / Coverage / Tail
```

## 实验状态

| Track | SID | SFT | RL | Eval |
|---|:---:|:---:|---|:---:|
| Text baseline | ✓ | ✓ | legacy RL | ✓ |
| SigLIP-MM baseline | ✓ | ✓ | legacy RL | ✓ |
| Text/SigLIP + native verl GRPO | ✓ | ✓ | launcher ready | 未运行 |
| Qwen3VL-MM | 编码器已完成；RQ ablation 与候选 SID 已导出 | 下游 SFT 未运行 | 未运行 | 未运行 |
| Qwen3VL-RecAlign | 对齐代码已实现 | 未运行 | 未运行 | 未运行 |

表中的正式结果只来自仓库已有 `outputs/formal_amazon23_1m/` 文件，不将旧结果代替新 track 的结果。

## 1. Item Representation

### SigLIP baseline

`multimodal/text_encoder.py`、`multimodal/image_encoder.py` 和
`multimodal/multimodal_encoder.py` 实现原有可复现 baseline：

```text
title + description → frozen SigLIP text tower
image               → frozen SigLIP vision tower
e_mm = normalize(0.7 * normalize(e_text) + 0.3 * normalize(e_image))
```

图片由 `multimodal/image_downloader.py` 下载并缓存。缺图通过 mask 记录，SigLIP weighted fusion 回退到 text vector。

### Qwen3-VL joint representation

`multimodal/qwen3_vl_encoder.py` 默认使用 `Qwen/Qwen3-VL-4B-Instruct`，模型 frozen，只做一次 representation extraction，不训练 VLM。输入 prompt 固定为商品 title、description 和可选的 category/brand 等字段。默认 `last_token` pooling 使用 attention mask 取最后一个有效文本位置，也支持 `text_mean`。

缺图时仍调用同一个 Qwen3-VL，仅输入文本，不回退到 SigLIP embedding space。输出保存：

```text
*.npy
*.npy.json
*.item_ids.json
*.has_image.npy
```

如需和 SigLIP 的 768D 下游配置公平对照，可以使用仅基于 item content 的 PCA：

```bash
python -m multimodal.qwen3_vl_encoder \
  --items data/Amazon/index/Industrial_and_Scientific.item.json \
  --manifest data/cache/amazon23_1k/images.jsonl \
  --output data/embeddings/industrial.qwen3vl.npy \
  --projection pca --target-dim 768 --pooling last_token \
  --model Qwen/Qwen3-VL-4B-Instruct
```

### Recommendation-aware alignment

`multimodal/rec_alignment.py` 冻结 Qwen3-VL，只训练轻量：

```text
Linear → GELU → Linear → L2 Normalize
```

User representation 是历史商品向量的 recency pooling，训练目标是 sampled-negative InfoNCE。行为监督只读取 `--train-csv`，不会读取 valid/test target。得到的 128D/256D 向量可命名为 `Qwen3VL-RecAlign-SID`，继续走同一 RQ-VAE 和推荐链路。

## 2. Semantic ID

所有 representation track 都使用相同的：

```text
Item embedding → RQ-VAE / RQ-KMeans → SID index → SID CSV
              → Qwen SFT → RL → Trie evaluator
```

默认 codebook capacity 仍为 32/32/32、latent dimension 64。SID 导出会记录 total items、raw unique SID、raw collision 和 deduplicated collision。碰撞消歧用的 `<d_n>` 不是语义 RQ 层，新 reward 不会给它额外层级权重。

## 3. SFT 与 RL

### SFT

`scripts/sft.py` 保持原有 generator。Item representation 只影响 Item → SID，不把图片或 VLM hidden state 直接拼入 Qwen prompt。训练目标为：

```text
P(next SID | history SID)
```

### Legacy RL

`scripts/rl.py` 和 `minionerec/trainer.py` 保留为 `legacy_group_relative_rl`，用于复现旧结果。它们的 policy term 使用 detached self-ratio，并不是严格的 `pi_theta / pi_old` clipped surrogate。旧 artifact 不删除，但新标准训练不再从这里启动。

### Native verl GRPO

新入口为 `rl/verl/run_grpo.sh`，由 `verl.trainer.main_ppo` 负责 rollout、old-policy log-prob、clipped policy loss、reference KL 和 optimizer update：

```bash
python -m rl.verl.prepare_data \
  --interactions data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --output data/verl/industrial/train.parquet \
  --dataset Amazon23 --category Industrial_and_Scientific

python -m rl.verl.run_grpo \
  --config rl/verl/configs/grpo_small.yaml --dry-run

# 安装兼容 verl 后运行正式配置
bash rl/verl/run_grpo.sh --config rl/verl/configs/grpo_formal.yaml
```

核心配置：

```text
algorithm.adv_estimator=grpo
actor_rollout_ref.rollout.n=8       # formal；smoke 为 4
actor_rollout_ref.actor.ppo_epochs=1
actor_rollout_ref.actor.clip_ratio=0.2
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.actor.kl_loss_coef=1e-3
```

$$
r_t=\exp(\log\pi_\theta-\log\pi_{\mathrm{old}}),\qquad
A_i=\frac{R_i-\mu_R}{\sigma_R+\epsilon}.
$$

`pi_old` 是 rollout policy snapshot，`pi_ref` 是独立的 KL anchor。配置支持 `ppo_epochs=1/2`，每次 run 会在输出目录保存解析后的 config。当前训练 rollout 不做 Trie monkey patch：无效 SID reward 为 0，评估阶段继续使用现有 Trie constrained decoding。

## 4. Reward

`rl/verl/reward.py` 提供四种 mode：

| mode | 定义 |
|---|---|
| `exact` | 预测 SID 与 target 完全一致为 1，否则 0 |
| `sid_hier` | 真实 RQ 层 prefix 权重 `0.5 / 0.3 / 0.2` |
| `semantic` | 合法预测商品与 GT 的 cosine 映射到 `[0,1]` |
| `hybrid` | exact 为 1，否则 `0.6 * sid_hier + 0.4 * semantic` |

reward 通过 `reward.custom_reward_function.path` 加载，不在 trainer 中 hard-code。invalid/unmapped item 的 semantic reward 为 0。

## 5. 已有 Amazon23 正式结果

以下四行来自已有 `results/summary.csv`，使用相同的 Amazon23 `Industrial_and_Scientific_1m` 完整 test split（7,974 条）、20-beam Trie decoding 和指标脚本：

| Track | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | Coverage | Tail HR@10 | Invalid SID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Text-SID + SFT | 0.000878 | 0.001129 | 0.003261 | 0.000566 | 0.000641 | 0.001181 | 0.006806 | 0.000627 | 0.0 |
| Text-SID + legacy RL | 0.000627 | 0.001379 | 0.003762 | 0.000325 | 0.000554 | 0.001171 | 0.004671 | 0.001046 | 0.0 |
| SigLIP-MM-SID + SFT | 0.000251 | 0.001630 | 0.007775 | 0.000103 | 0.000530 | 0.002067 | 0.006940 | 0.000418 | 0.0 |
| SigLIP-MM-SID + legacy RL | 0.000376 | 0.002508 | 0.007650 | 0.000160 | 0.000830 | 0.002129 | 0.004137 | 0.000418 | 0.0 |

完整浮点数、head/mid/tail 分桶、预测和配置仍保存在 `results/summary.csv` 与 `outputs/formal_amazon23_1m/`。这些数字不代表 Qwen3-VL 或 native verl 的结果。

## 6. 测试与依赖

```bash
python -m pytest -q
```

当前 checkout 的结果为 `56 passed, 3 skipped`。新增 CPU 测试覆盖：Qwen3-VL mock item order、缺图路径、pooling mask、PCA shape、reward ordering、`<d_n>` suffix、GRPO ratio/clip、verl parquet schema 和 train-only alignment。

native verl / Qwen3-VL 环境可按需安装：

```bash
pip install -r requirements.verl.txt
```

模型权重、verl 和生成的 embedding 不放入仓库；原有 SigLIP baseline 和正式 artifact 保持可用。

## 7. 目录

```text
multimodal/                 SigLIP、Qwen3-VL、RecAlign
rl/verl/                    parquet、reward、GRPO launcher/config
rq/                         RQ-VAE / RQ-KMeans / SID
minionerec/                 data、legacy trainer、Trie、SASRec
scripts/mm/                 图片、SID、评估脚本
outputs/formal_amazon23_1m/ 原有正式 artifact
results/summary.csv         原有可比结果汇总
docs/EXPERIMENT_LOG.md      实验 provenance
docs/INTERVIEW_GUIDE.md     代码对应的实现说明
```

## 8. Attribution

本项目基于并改造自 MiniOneRec，原项目 license 和 attribution 保留在 [`LICENSE`](LICENSE)。
