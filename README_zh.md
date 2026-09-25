# MM-OneRec：多模态 Semantic ID 生成式推荐系统

MM-OneRec 将序列推荐建模为 Semantic ID 的自回归生成：先用多模态模型
构建商品表征，再通过 RQ-VAE 离散化，最后由 Qwen3-4B 根据用户历史 SID
生成下一商品。

~~~text
Title + Description + Image
            │
            ▼
      冻结的 Qwen3-VL
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
 Trie-constrained Beam Search
            │
            ▼
        Top-K 推荐
~~~

## 项目要点

- **多模态商品表征**：冻结 Qwen3-VL，联合编码商品标题、描述和图片。
- **层次化 Semantic ID**：使用多级残差量化，把连续商品向量转成可生成的
  离散 token 序列。
- **生成式推荐**：Qwen3-4B 根据用户按时间排序的历史 SID 预测下一商品 SID。
- **推荐后训练**：支持监督微调和基于原生 verl 的 GRPO，包括组内相对奖励、
  clipped policy update 和 reference KL。
- **合法生成**：评估阶段使用 SID 前缀 Trie 约束 beam search，使候选能够映射
  回商品目录。
- **完整目录评估**：支持确定性 full-catalog 评估、排序指标、覆盖率、合法性和
  long-tail 分析。

## 多模态商品表征

### Qwen3-VL

multimodal/qwen3_vl_encoder.py 默认使用冻结的
Qwen/Qwen3-VL-4B-Instruct。有图片时，模型接收图片、title 和 description；
缺图时仍使用同一个 Qwen3-VL，只发送文本，不切换到另一种 embedding space。

默认 last_token pooling 读取联合图文上下文之后最后一个有效文本位置，也可以
使用 text_mean 做对照。编码结果同时保存 item ID、has_image mask、模型信息和
投影信息；可用只基于商品内容拟合的 PCA 将原始 hidden dimension 投影到 768D。

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

仓库同时保留 SigLIP baseline。其可解释的加权融合为：

~~~text
e_mm = normalize(0.7 * normalize(e_text) + 0.3 * normalize(e_image))
~~~

### Recommendation-aware Alignment

multimodal/rec_alignment.py 冻结 Qwen3-VL，只训练轻量 projector：

~~~text
Linear → GELU → Linear → L2 Normalize
~~~

用历史商品向量的 recency pooling 构造 user vector，再以 sampled-negative
InfoNCE 对齐训练集中的下一商品。得到的 item vector 继续使用同一套 RQ-VAE 和
生成式推荐链路。

## Semantic ID

所有 representation track 都遵循统一路径：

~~~text
item embedding
    ↓
RQ-VAE / RQ-KMeans
    ↓
SID index 与 interaction CSV
    ↓
Qwen SFT → RL post-training → Trie evaluation
~~~

RQ-VAE 残差量化生成类似下面的层次 code：

~~~text
<a_i><b_j><c_k>
~~~

实现支持 64^3、128^3、256^3 等受控 codebook capacity 对照，并可通过
scripts/mm/analyze_codebook.py 检查 codebook utilization、entropy、perplexity、
prefix diversity、重构质量和 raw SID collision。

当多个商品共享完整 raw RQ path 时，导出阶段增加确定性的碰撞消歧 suffix：

~~~text
<a_i><b_j><c_k><d_n>
~~~

<d_n> 是寻址后缀，不是第四个语义 RQ 层。没有碰撞的商品在三个语义 token
之后结束；碰撞组只使用所需数量的 suffix。Trie 和 tokenizer 会处理该后缀，
层次语义 reward 则不把它当作新的量化层。

## 生成式推荐

### Qwen3-4B SFT

核心任务保持为：

~~~text
用户历史 SID 序列 → 下一商品 SID
~~~

多模态信息在推荐训练前进入 Item → SID 阶段，图片和 VLM hidden state 不直接
拼入 Qwen prompt。SID token 会作为 atomic token 加入 tokenizer，并同步扩展
模型词表。训练只对目标 token 计算 causal language-model loss。

入口是 scripts/sft.py，shell wrapper 为 scripts/mm/train_sft.sh。

### Trie-constrained Decoding

minionerec/logit_processor.py 根据 catalog SID index 构造前缀映射，每一步只
允许当前合法 prefix 的 child token。这样 beam search 可以返回多个目录内商品，
同时避免生成无法映射的 SID 路径。

## 推荐后训练

### 原生 verl GRPO

推荐的 RL 入口是 rl/verl/run_grpo.py 或 rl/verl/run_grpo.sh。rollout、
old-policy log probability、clipped policy update 和 optimizer state 由原生
verl 管理。配置明确区分：

~~~text
pi_old  = 产生 rollout 的策略
pi_ref  = 用于 KL regularization 的 reference model
~~~

每个 prompt 采样多个 response，使用组内 reward 计算相对 advantage。当前提供：

| mode | 含义 |
| --- | --- |
| exact | 完整 SID 匹配时为 1 |
| sid_hier | 按真实 RQ 层计算 prefix 匹配 |
| semantic | 合法预测商品与 GT 的 embedding 相似度 |
| hybrid | exact，否则组合层次 reward 与 semantic reward |

训练 rollout 保持 verl 原生流程；无效或无法映射的 SID 得到 0 reward。评估时
继续使用 catalog Trie 和确定性 constrained beam search。

### Legacy 兼容路径

scripts/rl.py 和 minionerec/trainer.py 作为 legacy_group_relative_rl 保留，
用于复现 MiniOneRec 的旧训练路径。新的实验入口使用原生 verl launcher。

## 评估

MM-OneRec 提供确定性的 full-catalog 评估：对每个留出用户状态生成一组 SID，
Trie 将合法路径映射回目录商品，再与下一次交互进行排序比较。

评估器支持：

- HR@5、HR@10、HR@20
- NDCG@5、NDCG@10、NDCG@20
- catalog coverage 与 invalid SID rate
- head / mid / tail 分桶
- collision-group 与 non-collision target
- 推荐 item 和一级 prefix 的集中度诊断

与具体运行相关的预测、checkpoint、日志和指标文件保存在公开源码快照之外。

## Quick Start

下面列出公开入口；请将路径替换为本地 Amazon metadata、图片 manifest、SID 文件
和生成模型 checkpoint。

1. 使用 Qwen3-VL 编码商品：

   ~~~bash
   bash scripts/mm/encode_qwen3vl.sh --help
   ~~~

2. 训练或导出 Semantic-ID tokenizer：

   ~~~bash
   python scripts/mm/run_rq_ablation.py --help
   bash scripts/mm/build_sid.sh --help
   ~~~

3. 构造训练样本并运行 SFT：

   ~~~bash
   bash scripts/mm/train_sft.sh \
     --base_model <generator-checkpoint> \
     --train_file <sid-train.csv> \
     --eval_file <sid-valid.csv> \
     --sid_index_path <sid-index.json> \
     --output_dir outputs/local_sft
   ~~~

4. 使用 Trie constrained beam search 评估：

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

5. 可选：准备 parquet 并启动原生 GRPO：

   ~~~bash
   python -m rl.verl.prepare_data --help
   bash rl/verl/run_grpo.sh --config rl/verl/configs/grpo_small.yaml
   ~~~

所有入口都支持 --help；正式运行前可以使用 scripts/mm/ 下的 smoke helper
和 tests/ 完成小规模检查。

## 项目结构

~~~text
MM-OneRec/
├── multimodal/        # Qwen3-VL、SigLIP baseline、RecAlign
├── rq/                # RQ-VAE 与 RQ-KMeans tokenization
├── minionerec/        # dataset、Trie、legacy trainer、SASRec
├── scripts/           # SFT 与 evaluation 入口
├── scripts/mm/        # 图片、SID、诊断和评估工具
├── rl/verl/           # native GRPO 数据转换与 launcher
├── tests/              # 单元测试与回归测试
├── config/             # runtime 配置示例
└── docs/               # 环境和实现说明
~~~

## 测试

在仓库根目录运行：

~~~bash
python -m pytest -q
~~~

测试覆盖 tokenizer extension、多模态 item order、缺图处理、pooling mask、
PCA shape、SID reward、collision suffix、parquet conversion 以及共享 Trie/
evaluator 工具。

## Attribution

本项目基于并改造自 MiniOneRec，原项目 license 和 attribution 保留在
LICENSE 中。
