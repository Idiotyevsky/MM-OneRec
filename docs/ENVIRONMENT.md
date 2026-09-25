# MM-OneRec 实验环境

最后核验：2026-09-25
专用环境：`/home/zfs02/jiangjr/envs/mm-onerec`（Conda，Python 3.11.16）

## 已核验版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.16 |
| PyTorch | 2.6.0+cu124 |
| CUDA runtime（PyTorch） | 12.4 |
| Transformers | 4.57.1 |
| verl | 0.6.1 |
| vLLM | 0.8.5.post1 |
| qwen-vl-utils | 0.0.14 |
| Ray | 2.43.0 |
| uvicorn | 0.30.6 |
| flash-attn | 未安装（`flash_attn` 不可导入） |
| NumPy | 1.26.3 |
| Accelerate | 1.10.1 |
| PEFT | 0.21.0 |
| TRL | 0.24.0 |
| NVIDIA driver（4090-1） | 535.230.02 |

依赖完整性检查：`pip check` 通过。当前专用环境已经验证可以导入
`Qwen3VLForConditionalGeneration`、`verl` 和 `vllm`，并可执行
`verl.trainer.main_ppo --help`。远程 4090-1 上的 native verl smoke 已实际完成
4 个 optimizer step；同一环境已完成 `Qwen/Qwen3-VL-4B-Instruct` 的 7,493-item
正式编码（batch size 4、bfloat16、PCA 到 768D），并使用
`Qwen/Qwen3-4B-Instruct-2507` 完成 Q6/Q7/Q8 三套一轮全参数 BF16 SFT；仓库
CPU 测试结果为 `59 passed, 3 skipped`。

## GPU

本轮实际执行节点为 `4090-1`（`210.28.132.72`），共有 8 张 NVIDIA GeForce
RTX 4090，每张约 49 GiB，驱动 `535.230.02`。smoke 使用
`CUDA_VISIBLE_DEVICES=0`，只释放并使用了确认属于当前用户的残留进程所占用的
GPU 0；其他用户任务未被终止。Ray/Torch 临时目录使用 `/dev/shm/mmr`，避免远程
主机 `/tmp` 空间不足影响启动。Qwen3-4B SFT 分别使用 CUDA 0/1/2；GPU 3
保留给用户进程。最终推荐评估采用 HF `scripts/evaluate.py` 的 Trie-constrained
deterministic beam search，未将未经验证的 vLLM guided decoding 混入评测口径。

## 固定依赖

`requirements.verl.txt` 现在固定到本轮真实验证过的版本。基础环境与本专用环境
分开，旧的基础 `requirements.txt` 不作为 verl/Qwen3-VL 运行环境。该环境已经通过
`pip check`，且完成了 Qwen3-VL 真实输入 smoke 与 native verl 4-step smoke。
