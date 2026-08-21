# HPC for AI 课程笔记

- 集合原始页面：[HPC for AI](https://distinct-capricorn-c04.notion.site/HPC-for-AI-20d88315b6b480538083fbe724df2902)
- 整理日期：2026-08-17
- 课程大纲：[INFO 7375：面向 AI 的高性能计算课程大纲](info-7375-syllabus.md)
- 英文原文整理：[HPC for AI Notes](source/hpc-for-ai.en.md)

本目录将公开 Notion 补充系列整理为十一周中文主笔记。内容保留每周的主题、公式、系统数据流、作业方向和稳定外部参考，但不复制 Notion 界面、附件或临时下载链接。英文目录保存对应的纯英文结构化版本。

> 这是 2026-08-17 的课程资料快照。课程安排、项目状态、工具版本与岗位链接都可能继续变化，应以原页面及相应官方资料为准。

## 周次导航

| 周次 | 主题 | 学习作用 |
| --- | --- | --- |
| 1 | [并行、线程与同步](notes/01-parallelism.md) | 从进程、线程和共享状态进入并行程序的正确性与性能问题。 |
| 2 | [CUDA 与 GEMM 基础](notes/02-cuda.md) | 建立 GPU 执行层次、kernel launch、矩阵乘和基本优化模型。 |
| 3 | [内存层次、Tiling 与融合](notes/03-memory.md) | 理解数据复用、shared memory、coalescing、reduction 与 fusion。 |
| 4 | [Attention 与 FlashAttention](notes/04-attention.md) | 从稳定 softmax 推导 tiled exact attention 及其 IO 优化。 |
| 5 | [Layout Algebra](notes/05-layout-algebra.md) | 用 shape、stride、composition 和 inverse 表达线程到数据的映射。 |
| 6 | [CuTe GEMM](notes/06-cute-gemm.md) | 把 layout、copy、MMA 和流水线组合成层次化 GEMM kernel。 |
| 7 | [MPI、NCCL 与 Collective](notes/07-communication.md) | 建立分布式内存模型与 collective 通信语义。 |
| 8 | [数据并行与专家并行](notes/08-data-expert-parallelism.md) | 理解 MoE routing、两次 All-to-All 和 DP/TP/EP 混合数据流。 |
| 9 | [序列并行与 Ring Attention](notes/09-sequence-parallelism.md) | 学习 blockwise exact attention、K/V 环传输及异步 tile pipeline。 |
| 10 | [推理系统](notes/10-inference-systems.md) | 连接 prefill/decode、KV cache、PagedAttention 与 continuous batching。 |
| 11 | [职业能力建设](notes/11-career-building.md) | 通过社区、开源、论文复现、系统练习和岗位分析构建长期能力。 |

## 建议阅读路线

- CUDA/kernel：1 → 2 → 3 → 4 → 5 → 6；
- 分布式 MoE：1 → 7 → 8；
- 长上下文与推理：4 → 9 → 10；
- 完成任一技术路线后阅读第 11 周，把知识转化为复现、开源和持续学习计划。

这些 Notion 周笔记保留补充页面的组织顺序，不是官方 PDF 14 周课程安排的替代或逐周展开。`docs/chapters/` 则按本知识库的数学定义、reference、CUDA 接口与验证边界重新组织专题。
