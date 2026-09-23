# MM-OneRec 中文说明

> **多模态 Semantic ID 驱动的生成式推荐系统**

MM-OneRec 将商品文本与图片编码为 Item embedding，经 RQ-VAE 量化成层次化 Semantic ID，再使用 Qwen 完成 history SID → next-item SID 的 SFT、GRPO 和 Trie constrained decoding。

主文档 [`README.md`](README.md) 已按开源项目技术文档重新整理，包含：

- 项目定位、设计选择与端到端架构；
- SigLIP text/image encoder、cache、mask 和加权融合；
- RQ-VAE、SID collision、SFT、GRPO、Trie 解码；
- 原 MiniOneRec Amazon18 reference data；
- MM-OneRec Text-SID / MM-SID 的统一正式评估结果；
- Amazon23 item-side 资产与完整复现命令。

## 数据与结果入口

| 内容 | 路径 |
| --- | --- |
| 原项目 Amazon18 reference data | `data/Amazon/` |
| MM-OneRec Amazon23 processed data | `data/processed/amazon23_1m/` |
| Text/MM SID index 与 CSV | `data/sid/amazon23_1m/` |
| formal metrics | `outputs/formal_amazon23_1m/` |
| 汇总表 | `results/summary.csv` |

原项目 reference data 与 MM-OneRec 结果分开记录，便于后续使用同一评估脚本追加新的 baseline 或正式结果。

## 当前结果

四个正式 checkpoint 使用完全相同的协议：Amazon23 `Industrial_and_Scientific_1m` 完整 test split（7,974 条样本）、20-beam Trie constrained decoding、`max_new_tokens=8` 和同一指标脚本。

| 模型 | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | Coverage | Tail HR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text-SID + SFT | 0.000878 | 0.001129 | 0.003261 | 0.000566 | 0.000641 | 0.001181 | 0.006806 | 0.000627 |
| Text-SID + GRPO | 0.000627 | 0.001379 | 0.003762 | 0.000325 | 0.000554 | 0.001171 | 0.004671 | 0.001046 |
| MM-SID + SFT | 0.000251 | 0.001630 | 0.007775 | 0.000103 | 0.000530 | 0.002067 | 0.006940 | 0.000418 |
| MM-SID + GRPO | 0.000376 | 0.002508 | 0.007650 | 0.000160 | 0.000830 | 0.002129 | 0.004137 | 0.000418 |

完整字段（含 head/mid/tail 分桶和 Invalid SID Rate）见 [`results/summary.csv`](results/summary.csv)，原始文件位于 `outputs/formal_amazon23_1m/`。

快捷入口：

- [系统架构](README.md#系统架构)
- [多模态 Item 表征](README.md#多模态-item-表征)
- [Semantic ID](README.md#semantic-id从连续表征到可生成目标)
- [实验结果与数据来源](README.md#实验结果与数据来源)
- [快速开始](README.md#快速开始)
