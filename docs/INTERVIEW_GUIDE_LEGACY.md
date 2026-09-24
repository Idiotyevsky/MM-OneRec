# Archived historical guide

This file is preserved for provenance. It predates the native verl/Qwen3-VL refactor; see `INTERVIEW_GUIDE.md` for the current implementation.

# MM-OneRec 面试说明

以下回答以本仓库的实际实现为准。当前多模态离线 smoke 已跑通，真实 Amazon23 + SigLIP 训练和四组推荐实验尚未完成；面试时不要声称已有提升。

## 1. 为什么做生成式推荐？

它把候选 item 表示为可生成的离散 SID，用一个自回归模型统一建模用户历史和下一 item。优点是可以复用 LLM 的序列建模与生成基础设施；代价是必须解决合法 item 约束、推理开销与 SID 质量。

## 2. 和 SASRec 有什么区别？

`minionerec/sasrec.py` 的 SASRec 用 item embedding、因果 self-attention 和全量 item 分类头打分；MM-OneRec 的 Qwen 按 token 生成层次 SID。SASRec 是有界候选分类，生成式路线需要 Trie 防止生成不存在的 item。

## 3. 为什么不直接生成 Item ID？

原始 ID 是任意编号，相似商品没有共享结构，而且长 ID 会成为没有语义的 token 序列。SID 让相似 item 有机会共享高层 code，并减少词表设计的随意性。

## 4. Semantic ID 是什么？

它是 RQ 将连续 item embedding 离散化后得到的多级 code，例如 `[<a_12>, <b_7>, <c_3>]`。仓库的 index JSON 保存 item 到 code 列表的映射。

## 5. RQ-VAE 怎么工作？

encoder 将 item embedding 压到 latent，多个 codebook 依次量化当前残差，decoder 重构原 embedding；训练同时优化重构和量化相关损失。实现入口是 `rq/rqvae.py` 和 `rq/models/rqvae.py`。

## 6. 为什么使用 residual quantization？

单层 codebook 容量有限。逐层量化残差可以用多个较小 codebook 组合出更大的离散空间，并自然产生层次 token 序列。

## 7. SID collision 是什么？

多个 item 得到完全相同的 code 序列，生成该 SID 时无法唯一还原 item。本项目报告 total items、unique SID、collision count 和 collision rate。

## 8. 如何避免 SID collision？

可增加 codebook size/层数、改善 embedding、使用 balanced/constrained RQ-KMeans，或给碰撞 item 增加末级唯一 token。先测量再调整，不能只靠增大模型掩盖问题。

## 9. 为什么加入图像？

标题可能短、模板化或缺失。图片能提供颜色、外形、包装和款式信号。新增代码只在 SID 构建前融合，不改变 Qwen 训练接口。

## 10. 多模态 fusion 怎么做？

`multimodal/multimodal_encoder.py` 对两种向量 L2 normalize，默认按 0.7/0.3 加权后再次 normalize。缺图回退 text；维度不同可用确定性 PCA 到共同维度。

## 11. 为什么没有直接使用 VLM？

目标是低成本、可复现的推荐闭环。VLM 或 cross-attention 会引入昂贵端到端训练，并把变化扩散到 Qwen；冻结 encoder + 简单融合更容易做可靠消融。

## 12. 为什么多模态可能帮助 long-tail item？

低频 item 的协同信号弱，但内容仍可用；视觉相似性可能让它共享更合理的 SID 前缀。这只是机制假设，所以代码按训练频率分别报告 head/mid/tail，不能预设提升。

## 13. SFT 输入输出是什么？

核心任务是历史 item 的 SID 序列作为上下文，目标为下一 item SID。数据构造在 `minionerec/data.py`，入口在 `scripts/sft.py`。

## 14. SFT loss 是什么？

标准 causal language-model cross entropy。prompt token label 设为 -100，只在目标 SID token 上计算 next-token loss。

## 15. 为什么 SFT 后还需要 GRPO？

SFT 只最大化真实下一 SID 的似然，不直接优化 top-K 排名或规则 reward。GRPO 可在 SFT 初始化附近，用候选组的相对 reward 调整策略；是否有效需与 SFT 实测。

## 16. 为什么选择 GRPO？

当前仓库和 TRL 已提供 GRPO 路线；它使用组内相对优势，不需要单独训练 value model，工程上比 PPO 简单。

## 17. GRPO 和 PPO 的区别是什么？

PPO 通常依赖 critic/value estimate 和 clipped probability ratio；GRPO 从同一 prompt 的多个生成 reward 计算组内标准化 advantage，可省掉 critic。当前 README 不能把这套训练误写为 PPO。

## 18. Group advantage 如何计算？

同一 prompt 的 G 个 reward 计算均值和标准差，`A_i=(r_i-mean)/(std+epsilon)`。所有候选 reward 相同会导致接近零的学习信号。

## 19. Reward 怎么设计？

首版使用稳定、可解释的 ranking reward 或 rule + ranking。原仓库还涉及 semantic、SASRec/CF 等信号，但不在第一版无限叠加；日志应说明实际启用项。

## 20. 为什么需要 KL regularization？

它限制 RL policy 过度偏离 SFT/reference model，减少 reward hacking 和语言分布崩坏。需要同时记录 KL，而非只展示 reward。

## 21. Constrained decoding 为什么必要？

LLM 的 token 组合空间远大于真实 SID 集合。没有约束会生成无法映射回 item 的序列，直接损害推荐有效性。

## 22. Trie constrained decoding 怎么实现？

合法 SID token 序列构成前缀树。每一步根据当前前缀返回 child token，`minionerec/logit_processor.py` 将其他 token logits mask 掉；叶子只允许结束。

## 23. Beam search 有什么作用？

它并行保留多个高概率合法前缀，从而形成 top-K 候选。beam 多会提高推理显存和延迟，评估脚本已有 OOM 时缩小 batch/beam 的逻辑。

## 24. HR@K 是什么？

若真实下一 item 出现在 top-K，样本 hit 为 1，否则为 0；对样本平均。它只关注是否命中，不区分第 1 和第 K。

## 25. NDCG@K 是什么？

单正例时，命中 rank r 的贡献为 `1/log2(r+1)`。越靠前贡献越大，没命中为 0。

## 26. HR 和 NDCG 区别？

HR 衡量召回到没有；NDCG 同时反映排序位置。两个指标一起报告能避免“都命中但排得很后”的信息丢失。

## 27. SASRec baseline 怎么工作？

它为历史 item 加位置 embedding，用 causal self-attention 得到序列表示，再通过 item 分类头排名。它是合理的非生成 baseline，但 P0 先完成 Text/MM 消融。

## 28. 如果 MM-SID 没有提升怎么办？

如实报告，并分解检查图片覆盖率、坏图、item 对齐、投影损失、SID collision 和 tail bucket。简单加权可能因跨模态空间未对齐而退化；这本身是有价值的工程结论。

## 29. 如果图片缺失怎么办？

下载 manifest 记录 missing/failed，image encoder 输出零向量和 false mask，weighted fusion 对该 item 完全回退 text。不会把零向量当作真实视觉特征。

## 30. 线上部署最大瓶颈是什么？

离线侧是图片抓取和全量视觉编码；在线侧主要是自回归 beam search 的延迟与显存，而不是离线 fusion。实际系统可预计算 SID、缓存 Trie，并用蒸馏/更小模型或两阶段召回降低成本。

## 建议的两分钟项目讲法

先说边界：原仓库主线完整但 SID 实际是文本侧，而且随附大文件不完整。然后说改造：冻结视觉 encoder、可靠图片缓存、缺图 mask、简单可解释融合，保持后面的 RQ/SFT/GRPO/Trie 不变。最后说验证：20-item 离线 smoke 与全仓库测试已通过，并在 Amazon23 Industrial_and_Scientific_1m 的 7,493-item、7,974-row full test split 上用同一 20-beam evaluator 完成 Text/MM SID、SFT 和 GRPO 评估，统一记录 collision、invalid rate、HR/NDCG、coverage 与 tail HR/NDCG。初始 GRPO 因 warmup 导致学习率为 0，已通过逐张量比较确认并修复。
