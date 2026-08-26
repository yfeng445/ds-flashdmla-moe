# AI Infra 论文与参考资料

本目录保存与本项目相关的论文原文、中文译件和 PMPP 第四版参考资料。当前馆藏共
80 份 PDF：45 份研究原文、17 份研究译件、1 份 PMPP4 原书和 17 份 PMPP4
中文章节。

现有中文件均来自历史课程资料，尚未完成人工逐页审校。文件名中的 `ai-draft`
表示 AI 初译，`ai-summary` 表示摘要或节译；没有这些后缀的历史译件也应按
“AI 生成、未审校”理解。仓库的 MIT License 不覆盖这些外部作品。

- [完整馆藏目录](catalog.md)
- [来源与迁移清单](manifest.yaml)
- [PMPP4 原书与章节说明](books/pmpp-4e/README.md)

## 馆藏结构

| 主题 | 原文 | 中文件 | 入口 |
| --- | ---: | ---: | --- |
| Attention 与 GPU kernel | 7 | 5 | [`attention-kernels/`](attention-kernels/) |
| MLA 与 Transformer | 7 | 3 | [`mla-transformers/`](mla-transformers/) |
| MoE 与通信融合 | 10 | 8 | [`moe/`](moe/) |
| 分布式训练与分片 | 14 | 0 | [`distributed-training/`](distributed-training/) |
| 推理与 serving | 5 | 1 | [`serving/`](serving/) |
| Scaling 与低精度 | 2 | 0 | [`scaling-foundations/`](scaling-foundations/) |
| PMPP 第四版 | 1 | 17 | [`books/pmpp-4e/`](books/pmpp-4e/) |

## 当前实现主线

### CUDA、Attention 与 MLA

1. [PMPP 第四版原书](<books/pmpp-4e/original/programming-massively-parallel-processors-4e.pdf>)
   → [Online Softmax](<attention-kernels/originals/Online normalizer calculation for softmax.pdf>)
   → [FlashAttention](<attention-kernels/originals/FlashAttention - Fast and Memory-Efficient Exact Attention with IO-Awareness.pdf>)。
2. [FlashAttention-2](<attention-kernels/originals/FlashAttention-2 - Faster Attention with Better Parallelism and Work Partitioning.pdf>)
   → [Hopper/CUTLASS case study](<attention-kernels/originals/A Case Study in CUDA Kernel Fusion - Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library.pdf>)
   → [ThunderKittens](<attention-kernels/originals/ThunderKittens - Simple, Fast, and Adorable AI Kernels.pdf>)。
3. [RoFormer](<mla-transformers/originals/RoFormer - Enhanced Transformer with Rotary Position Embedding.pdf>)
   → [DeepSeek-V2](<mla-transformers/originals/DeepSeek-V2 - A Strong, Economical, and Efficient Mixture-of-Experts Language Model.pdf>)
   → [DeepSeek-V3](<mla-transformers/originals/DeepSeek-V3 Technical Report.pdf>)
   → [TransMLA](<mla-transformers/originals/TransMLA - Multi-Head Latent Attention Is All You Need.pdf>)。

### MoE、通信与分布式训练

1. [Sparsely-Gated MoE](<moe/originals/Outrageously Large Neural Networks - The Sparsely-Gated Mixture-of-Experts Layer.pdf>)
   → [GShard](<moe/originals/GShard - Scaling Giant Models with Conditional Computation and Automatic Sharding.pdf>)
   → [Switch Transformer](<moe/originals/Switch Transformers - Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.pdf>)
   → [DeepSeekMoE](<moe/originals/DeepSeekMoE - Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.pdf>)。
2. [CCFuser](<moe/originals/Harnessing Inter-GPU Shared Memory for Seamless MoE Communication-Computation Fusion.pdf>)
   → [FlashDMoE v2](<moe/originals/FlashDMoE - Fast Distributed MoE in a Single Kernel.pdf>)
   → [FlashMoE v3](<moe/originals/FlashMoE - Fast Distributed MoE in a Single Kernel.pdf>)。
3. [Megatron-LM](<distributed-training/originals/Megatron-LM - Training Multi-Billion Parameter Language Models Using Model Parallelism.pdf>)
   → [PyTorch FSDP](<distributed-training/originals/PyTorch FSDP - Experiences on Scaling Fully Sharded Data Parallel.pdf>)
   → [SimpleFSDP](<distributed-training/originals/SimpleFSDP - Simpler Fully Sharded Data Parallel with torch.compile.pdf>)
   → [Hecate](<moe/originals/Hecate - Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism.pdf>)。

### Serving 与低精度

1. [PagedAttention](<serving/originals/Efficient Memory Management for Large Language Model Serving with PagedAttention.pdf>)
   → [Attention Sinks](<serving/originals/Efficient Streaming Language Models with Attention Sinks.pdf>)
   → [AlpaServe](<serving/originals/AlpaServe - Statistical Multiplexing with Model Parallelism for Deep Learning Serving.pdf>)。
2. [Training Compute-Optimal Large Language Models](<scaling-foundations/originals/Training Compute-Optimal Large Language Models.pdf>)
   → [Scaling Laws for Precision](<scaling-foundations/originals/Scaling Laws for Precision.pdf>)。

## 版本与译件说明

- arXiv `2506.04667v2` 使用标题 **FlashDMoE**；`v3` 政名为
  **FlashMoE**。两个版本及各自中文件均保留，不互相覆盖。
- BabelDOC 和 ChatGPT/WeasyPrint 生成件仅作为阅读辅助，不作为权威术语来源。
- FlashAttention-3、ThunderKittens 和 PMPP4 第 20 章中文件属于节译或摘要。
- PMPP4 中文资料现有第 1–16 章和第 20 章；第 17–19、21–23 章缺失。

## 权威外部资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
  与 [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [NVSHMEM Memory Model](https://docs.nvidia.com/nvshmem/api/gen/mem-model.html)
- [FlashAttention 官方实现](https://github.com/Dao-AILab/flash-attention)
- [FlashMoE 官方实现](https://github.com/osayamenja/FlashMoE)

新增本地资料时，应同步更新 `catalog.md` 和 `manifest.yaml`，记录上游来源、版本、
翻译状态和权利归属。公开再分发前仍需按单份材料核对许可。
