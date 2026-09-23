# MM-OneRec 中文说明

> **多模态 Semantic ID 驱动的生成式推荐系统**

MM-OneRec 将商品文本与图片编码为 Item embedding，经 RQ-VAE 量化成层次化 Semantic ID，再使用 Qwen 完成 history SID → next-item SID 的 SFT、GRPO 和 Trie constrained decoding。

主文档 [`README.md`](README.md) 已按开源项目技术文档重新整理，包含：

- 项目定位、设计选择与端到端架构；
- SigLIP text/image encoder、cache、mask 和加权融合；
- RQ-VAE、SID collision、SFT、GRPO、Trie 解码；
- 原 MiniOneRec Amazon18 reference data；
- MM-OneRec Text-SID / MM-SID 的统一 pilot 结果；
- Amazon23 item-side 资产与完整复现命令。

## 数据与结果入口

| 内容 | 路径 |
| --- | --- |
| 原项目 Amazon18 reference data | `data/Amazon/` |
| MM-OneRec Amazon23 processed data | `data/processed/amazon23_1m/` |
| Text/MM SID index 与 CSV | `data/sid/amazon23_1m/` |
| pilot metrics | `outputs/eval_full/` |
| 汇总表 | `results/summary.csv` |

原项目 reference data 与 MM-OneRec 结果分开记录，便于后续使用同一评估脚本追加新的 baseline 或正式结果。

## 当前结果

999-item dense subset、2,564 个 test users、5-beam Trie evaluator：

| 模型 | HR@10 | NDCG@10 | Coverage | Tail HR@10 |
| --- | ---: | ---: | ---: | ---: |
| Text-SID + SFT | 0.005850 | 0.002644 | 0.044044 | 0.002247 |
| Text-SID + GRPO | 0.008190 | 0.004788 | 0.046046 | 0.004494 |
| MM-SID + SFT | 0.010140 | 0.005374 | 0.039039 | 0.009112 |
| MM-SID + GRPO | 0.007020 | 0.003957 | 0.042042 | 0.006834 |

完整字段见 [`results/summary.csv`](results/summary.csv)，详细说明见主文档的[实验结果与数据来源](README.md#实验结果与数据来源)。

快捷入口：

- [系统架构](README.md#系统架构)
- [多模态 Item 表征](README.md#多模态-item-表征)
- [Semantic ID](README.md#semantic-id从连续表征到可生成目标)
- [实验结果与数据来源](README.md#实验结果与数据来源)
- [快速开始](README.md#快速开始)
