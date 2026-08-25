# NEU INFO 7375：High Performance Computing for AI

本目录把 INFO 7375 课程材料、HPC for AI 十一周补充笔记，以及本仓库围绕数学定义、reference、CUDA 接口和验证边界编写的专题讲义放在同一个课程单元中。

- [课程大纲](syllabus.md)：以官方 PDF 为准，记录课程目标、先修要求、教材、14 周计划与作业组织方式。
- [英文源文档入口](source/index.en.md)：保留公开补充材料的英文结构化版本。
- [专题练习](exercises.md)：从公式、reference 到 kernel 的递进任务。
- 集合原始页面：[HPC for AI](https://distinct-capricorn-c04.notion.site/HPC-for-AI-20d88315b6b480538083fbe724df2902)
- 整理日期：2026-08-17。

> 这是课程资料的仓库快照。课程安排、项目状态、工具版本与岗位链接都可能继续变化，应以原页面及相应官方资料为准。

## 十一周补充笔记

这些笔记保留公开 Notion 补充系列的页面顺序，不替代官方 PDF 的 14 周课程安排，也不与其逐周对应。

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

## 专题讲义

每章依次回答算子计算什么、朴素实现为什么慢、GPU 应怎样组织数据，以及如何证明优化没有改变语义。

| 章节 | 核心问题 | 配套代码 |
| --- | --- | --- |
| [0. 从并行线程到 GPU Tiling](chapters/00-gpu-execution-and-tiling.md) | grid/block/warp、内存复用与异步流水线 | `gemm.py`、`gemm_benchmarking.py` |
| [1. 从 Softmax 到在线 Softmax](chapters/01-online-softmax.md) | 如何稳定地流式归一化 | `attention.py` |
| [2. 从标准 Attention 到 FlashAttention](chapters/02-flash-attention.md) | 如何避免写出完整分数矩阵 | `attention.py`、实验 CUDA |
| [3. Multi-head Latent Attention](chapters/03-mla.md) | MLA 压缩了什么，decode 为何不同 | `mla.py` |
| [4. DeepSeekMoE](chapters/04-deepseek-moe.md) | 分组路由、SwiGLU 与 token dispatch | `moe.py`、`router_ops.py`、`expert_ops.py`、`benchmarks/router.py`、`benchmarks/experts.py` |
| [5. Expert Parallelism](chapters/05-expert-parallelism.md) | 如何跨 rank 保持 token 身份、顺序与梯度 | `expert_parallel.py` |
| [6. PyTorch 自定义算子](chapters/06-pytorch-custom-operators.md) | dispatcher、FakeTensor、autograd 与 stream | `ops.py`、`csrc/` |
| [7. Benchmark 与 Roofline](chapters/07-benchmarking-and-roofline.md) | 如何得到单卡与多 rank 的可复核性能证据 | `benchmarking.py`、`benchmarks/` |
| [8. 对称内存与 One-sided MoE](chapters/08-one-sided-symmetric-memory.md) | PGAS、data/flag 协议、时间缓冲与内存代价 | `symmetric_memory.py` |
| [9. FP8 E4M3FN 与 INT8 量化](chapters/09-fp8-int8-quantization.md) | 显式 scale、饱和、反量化 linear 与验证边界 | `quantization.py`、`quantized_benchmarking.py` |
| [10. 单卡可验证的 One-sided 协议与 TP](chapters/10-logical-one-sided-and-tp.md) | rank 双射、generation 状态机、route identity 与 logical TP | `parallel_topology.py`、`one_sided_protocol.py`、`fake_distributed.py`、`tensor_parallel.py` |
| [练习](exercises.md) | 从公式、reference 到 kernel 的递进任务 | 全仓库 |

## 验证与复现入口

- [2026-08-22 RTX 5090 单卡证据](../../../validation/single-gpu/2026-08-22-rtx5090-next-phase/README.md)：hosted CUDA build/reference workflow、installed-wheel 数值测试、MoE Kineto aggregate activity 与分析中间张量清单。
- [机器可读单卡摘要](../../../validation/single-gpu/2026-08-22-rtx5090-next-phase/summary.json)：明确区分原生执行、Kineto 聚合观测、分析字节数和未采集的 Nsight 证据。
- [Logical EP/TP 示例输出](../../../validation/logical/2026-08-22-ep-tp-reference.json)：固定为 `simulated=true`，不表示真实远程传输或多卡验证。

证据页中的 Kineto count 是完整 profiling harness 下的聚合 activity occurrence，不保证一行对应一次物理 kernel launch。本轮没有 Nsight 报告，也不由这些结果宣称稳定加速。

## 建议阅读路线

- CUDA/kernel：1 → 2 → 3 → 4 → 5 → 6；
- 分布式 MoE：1 → 7 → 8；
- 长上下文与推理：4 → 9 → 10；
- 完成任一技术路线后阅读第 11 周，把知识转化为复现、开源和持续学习计划；
- 阅读专题讲义时，先运行 `examples/reference_demo.py`，再手算极小输入、修改 shape 或 mask 让测试失败，最后进入 kernel 优化。

## 符号约定

- `B`：batch size。
- `H`：attention head 数；MoE 章节中隐藏维度写作 `D_h`。
- `S_q, S_k`：query 与 key/value 序列长度。
- `D`：head dimension 或 model dimension，由上下文说明。
- `E`：routed expert 数量。
- `K`：每个 token 激活的 expert 数量。
- 所有行向量默认写在张量的倒数第二维。
