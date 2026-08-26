# 本地论文馆藏

本页列出 `docs/papers/` 中的全部本地 PDF。研究类中文件均为未审校 AI 译稿；
“节译”表示没有覆盖原文全部附录或参考资料。

## Attention 与 GPU kernel

| 资料 | 原文 | 中文件 |
| --- | --- | --- |
| Online normalizer calculation for softmax | [PDF](<attention-kernels/originals/Online normalizer calculation for softmax.pdf>) | — |
| FlashAttention | [PDF](<attention-kernels/originals/FlashAttention - Fast and Memory-Efficient Exact Attention with IO-Awareness.pdf>) | — |
| FlashAttention-2 | [PDF](<attention-kernels/originals/FlashAttention-2 - Faster Attention with Better Parallelism and Work Partitioning.pdf>) | [AI 初译](<attention-kernels/translations/zh-CN/FlashAttention-2 - 通过更好的并行和工作划分实现更快的注意.pdf>) |
| FlashAttention-3 | [PDF](<attention-kernels/originals/FlashAttention-3 - Fast and Accurate Attention with Asynchrony and Low-precision.pdf>) | [AI 节译](<attention-kernels/translations/zh-CN/FlashAttention-3 - 利用异步与低精度实现的快速且精确的注意力机制.pdf>) |
| Hopper/CUTLASS FlashAttention-2 case study | [PDF](<attention-kernels/originals/A Case Study in CUDA Kernel Fusion - Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library.pdf>) | [AI 初译](<attention-kernels/translations/zh-CN/CUDA 内核融合 - 在 NVIDIA Hopper 架构上使用 CUTLASS 实现 FlashAttention-2 的案例研究.pdf>) |
| FlashMask | [PDF](<attention-kernels/originals/FlashMask - Efficient and Rich Mask Extension of FlashAttention.pdf>) | [AI 初译](<attention-kernels/translations/zh-CN/FLASHMASK - FlashAttention 的高效且丰富的掩码扩展.pdf>) |
| ThunderKittens | [PDF](<attention-kernels/originals/ThunderKittens - Simple, Fast, and Adorable AI Kernels.pdf>) | [AI 节译](<attention-kernels/translations/zh-CN/ThunderKittens - 简单、高速、可爱的 AI 内核.pdf>) |

## MLA 与 Transformer

| 资料 | 原文 | 中文件 |
| --- | --- | --- |
| DeepSeek-V2 | [PDF](<mla-transformers/originals/DeepSeek-V2 - A Strong, Economical, and Efficient Mixture-of-Experts Language Model.pdf>) | [AI 初译](<mla-transformers/translations/zh-CN/DeepSeek-V2 - 强大、经济且高效的混合专家语言模型.pdf>) |
| DeepSeek-V3 | [PDF](<mla-transformers/originals/DeepSeek-V3 Technical Report.pdf>) | [AI 初译](<mla-transformers/translations/zh-CN/DeepSeek-V3 技术报告.pdf>) |
| DeepSeek-V3.2 | [PDF](<mla-transformers/originals/DeepSeek-V3.2 - Pushing the Frontier of Open Large Language Models.pdf>) | — |
| RoFormer / RoPE | [PDF](<mla-transformers/originals/RoFormer - Enhanced Transformer with Rotary Position Embedding.pdf>) | [AI 初译](<mla-transformers/translations/zh-CN/RoFormer - 采用旋转位置嵌入的增强型 Transformer 架构.pdf>) |
| Linear Transformers Are Secretly Fast Weight Programmers | [PDF](<mla-transformers/originals/Linear Transformers Are Secretly Fast Weight Programmers.pdf>) | — |
| Parallelizing Linear Transformers with the Delta Rule | [PDF](<mla-transformers/originals/Parallelizing Linear Transformers with the Delta Rule over Sequence Length.pdf>) | — |
| TransMLA | [PDF](<mla-transformers/originals/TransMLA - Multi-Head Latent Attention Is All You Need.pdf>) | — |

## MoE 与通信融合

| 资料 | 原文 | 中文件 |
| --- | --- | --- |
| Sparsely-Gated MoE | [PDF](<moe/originals/Outrageously Large Neural Networks - The Sparsely-Gated Mixture-of-Experts Layer.pdf>) | [AI 初译](<moe/translations/zh-CN/极其庞大的神经网络 - 稀疏门控的专家混合层.pdf>) |
| GShard | [PDF](<moe/originals/GShard - Scaling Giant Models with Conditional Computation and Automatic Sharding.pdf>) | [AI 初译](<moe/translations/zh-CN/GShard - 通过条件计算与自动分片实现巨型模型的扩展性.pdf>) |
| Switch Transformers | [PDF](<moe/originals/Switch Transformers - Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.pdf>) | [AI 初译](<moe/translations/zh-CN/Switch Transformers - 借助简单高效的稀疏性扩展至万亿参数规模.pdf>) |
| DeepSeekMoE | [PDF](<moe/originals/DeepSeekMoE - Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.pdf>) | [AI 初译](<moe/translations/zh-CN/DeepSeekMoE - 迈向专家混合语言模型的终极专家专精.pdf>) |
| Auxiliary-Loss-Free Load Balancing | [PDF](<moe/originals/Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts.pdf>) | [AI 初译](<moe/translations/zh-CN/面向专家混合（MoE）的无辅助损失负载均衡策略.pdf>) |
| CCFuser / Inter-GPU Shared Memory | [PDF](<moe/originals/Harnessing Inter-GPU Shared Memory for Seamless MoE Communication-Computation Fusion.pdf>) | [AI 初译](<moe/translations/zh-CN/利用跨 GPU 共享内存实现无缝的 MoE 通信-计算融合.pdf>) |
| FlashDMoE v2 | [PDF](<moe/originals/FlashDMoE - Fast Distributed MoE in a Single Kernel.pdf>) | [AI 初译](<moe/translations/zh-CN/FlashDMoE - 单一内核中的快速分布式混合专家.pdf>) |
| FlashMoE v3 / NeurIPS 2025 | [PDF](<moe/originals/FlashMoE - Fast Distributed MoE in a Single Kernel.pdf>) | [AI 初译](<moe/translations/zh-CN/FlashMoE - 在单一内核中的快速分布式混合专家.pdf>) |
| Hecate | [PDF](<moe/originals/Hecate - Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism.pdf>) | — |
| MoE router weights as embeddings | [PDF](<moe/originals/Your Mixture-of-Experts LLM Is Secretly an Embedding Model for Free.pdf>) | — |

## 分布式训练与分片

- [Beyond Data and Model Parallelism](<distributed-training/originals/Beyond Data and Model Parallelism for Deep Neural Networks.pdf>)
- [Megatron-LM](<distributed-training/originals/Megatron-LM - Training Multi-Billion Parameter Language Models Using Model Parallelism.pdf>)
- [Alpa](<distributed-training/originals/Alpa - Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning.pdf>)
- [Amazon SageMaker Model Parallelism](<distributed-training/originals/Amazon SageMaker Model Parallelism - A General and Flexible Framework for Large Model Training.pdf>)
- [TeraPipe](<distributed-training/originals/TeraPipe - Token-Level Pipeline Parallelism for Training Large-Scale Language Models.pdf>)
- [On Optimizing the Communication of Model Parallelism](<distributed-training/originals/On Optimizing the Communication of Model Parallelism.pdf>)
- [PyTorch FSDP](<distributed-training/originals/PyTorch FSDP - Experiences on Scaling Fully Sharded Data Parallel.pdf>)
- [SimpleFSDP](<distributed-training/originals/SimpleFSDP - Simpler Fully Sharded Data Parallel with torch.compile.pdf>)
- [Memory and Bandwidth are All You Need for FSDP](<distributed-training/originals/Memory and Bandwidth are All You Need for Fully Sharded Data Parallel.pdf>)
- [Adjoint Sharding](<distributed-training/originals/Adjoint Sharding for Very Long Context Training of State Space Models.pdf>)
- [AutoShard](<distributed-training/originals/AutoShard - Automated Embedding Table Sharding for Recommender Systems.pdf>) — 推荐系统 embedding-table 分片
- [Model Parallelism Literature Review](<distributed-training/originals/Model Parallelism on Distributed Infrastructure - A Literature Review from Theory to LLM Case-Studies.pdf>)
- [Spectral Model Sharding](<distributed-training/originals/On Sampling Strategies for Spectral Model Sharding.pdf>) — 联邦/端侧场景
- [CNN Domain Decomposition](<distributed-training/originals/Model Parallel Training and Transfer Learning for Convolutional Neural Networks by Domain Decomposition.pdf>) — 扩展阅读

## 推理与 serving

| 资料 | 原文 | 中文件 |
| --- | --- | --- |
| PagedAttention | [PDF](<serving/originals/Efficient Memory Management for Large Language Model Serving with PagedAttention.pdf>) | [AI 重排译稿](<serving/translations/zh-CN/基于PagedAttention的大型语言模型服务的高效内存管理.pdf>) |
| StreamingLLM / Attention Sinks | [PDF](<serving/originals/Efficient Streaming Language Models with Attention Sinks.pdf>) | — |
| AlpaServe | [PDF](<serving/originals/AlpaServe - Statistical Multiplexing with Model Parallelism for Deep Learning Serving.pdf>) | — |
| Confidence-token model routing | [PDF](<serving/originals/Learning to Route LLMs with Confidence Tokens.pdf>) | — |
| Model-Agnostic Hybrid Sharding | [PDF](<serving/originals/Model Agnostic Hybrid Sharding for Heterogeneous Distributed Inference.pdf>) | — |

## Scaling 与低精度

- [Training Compute-Optimal Large Language Models](<scaling-foundations/originals/Training Compute-Optimal Large Language Models.pdf>)
- [Scaling Laws for Precision](<scaling-foundations/originals/Scaling Laws for Precision.pdf>)

## PMPP 第四版

原书及 17 份现有中文章节见 [PMPP4 说明页](books/pmpp-4e/README.md)。
