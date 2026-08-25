# 第 7 周：分布式内存、MPI/NCCL 与 Collective

- 原始页面：[Week 7: Communication](https://distinct-capricorn-c04.notion.site/Week-7-Communication-32088315b6b4808983bdf39dd7ca922a)
- 整理日期：2026-08-17
- 英文版：[Week 7: Communication](../source/notes/07-communication.en.md)

## 分布式内存系统

分布式内存系统由多个 processor/node 和 interconnect 组成。不同进程拥有独立地址空间，
异步执行，并通过显式 point-to-point 或 collective 操作交换数据。Message passing 是编程
模型，MPI 是该模型的一套接口规范；Open MPI、MPICH 等才是具体实现。

node 可以指 core、单台多核主机或一个含多 GPU 的服务器。link 可以是板内 GPU 互连，也
可以是跨机器的光纤网络。因此必须区分：

- intra-node：单机板内连接，例如 PCIe、NVLink/NVSwitch；
- inter-node：服务器之间的网络，例如 Ethernet、InfiniBand 或外部 NVLink fabric。

课程以 HGX H100 拓扑说明：8-GPU 节点可通过 NVSwitch 提供 any-to-any 连接，更大规模 pod
则需要专用高速网络连接多个节点。带宽数字和产品拓扑会随代际变化，设计时应查询目标系统
而不是把课程示例当作永久规格。

## MPI 最小程序

典型 MPI 程序按以下顺序建立进程上下文：

1. `MPI_Init` 初始化环境；
2. `MPI_Comm_rank` 获取当前 rank；
3. `MPI_Comm_size` 获取 communicator 中的进程数；
4. 每个 rank 执行相同程序中的不同数据分片；
5. `MPI_Finalize` 结束环境。

多个进程的输出顺序不确定，因为 rank 独立、异步推进。`mpicc` 是编译 wrapper，`mpirun -np N`
启动 N 个进程；MPI 本身不要求 GPU，因此也适用于纯 CPU 分布式程序。

## Collective 语义

| Collective | 输入到输出的关系 | 典型用途 |
| --- | --- | --- |
| All-Reduce | 所有 rank 归约，完整结果返回每个 rank | 数据并行梯度同步 |
| Reduce | 所有 rank 归约，结果只到 root | 汇总指标或集中处理 |
| All-Gather | 收集所有 rank 分片，完整拼接结果返回每个 rank | 参数/activation 重组 |
| Gather | 收集所有分片，结果只到 root | 集中保存或分析 |
| Scatter | root 的大 buffer 分片发给各 rank | 分发输入或 shard |
| Reduce-Scatter | 先归约，再把结果分片给各 rank | 分片梯度/状态更新 |
| All-to-All | 每个 rank 给每个其他 rank 发送独立分片 | MoE token dispatch/combine |

Collective 的价值不只是 API 简短。实现会根据 topology 选择 ring、tree、分层算法或其他
调度，避免所有 rank 同时把大数据灌入一个 endpoint，并利用多条 link 并行传输。

## MPI 与 NCCL

- MPI 覆盖通用 message passing、CPU buffer 和广泛的分布式控制模式；
- NCCL 针对 NVIDIA GPU collective，理解 NVLink、PCIe 和网络拓扑，并能在 CUDA stream
  上排队通信；
- 两者可以共存：MPI 负责进程启动/控制，NCCL 负责 GPU tensor collective；
- 正确性取决于所有 rank 以兼容的顺序、shape、dtype 和 communicator 参与操作。

## DeepSeekMoE Reference 测试生成

课程作业要求先用 Hugging Face DeepSeek-V3 实现生成小型 deterministic test case，再用纯 C
实现 MoE operator。推荐流程是：

- 从 `modeling_deepseek_v3.py` 提取 MLP、router 等 block 到独立文件；
- 缩小 hidden size 和 expert 数量，让测试可读且运行快速；
- 固定随机种子并启用 deterministic 模式；
- 权重与输入使用不同种子，避免改权重配置时意外改变输入；
- 把 weights、inputs 和 reference outputs 一起保存为测试样例；
- 连续生成两次，确认生成流程自身可复现；
- 分别验证 MLP、top-k router、dispatch、expert 和 combine，再测试整体数据流。

真实 DeepSeek 权重规模过大，不适合作为单元测试 fixture；小型随机权重的目标是锁定算子语义，
而不是模拟最终模型质量。

## 延伸阅读

- [HGX H100 Platform Overview](https://developer.nvidia.com/blog/introducing-nvidia-hgx-h100-an-accelerated-server-platform-for-ai-and-high-performance-computing)；
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl)；
- [*Demystifying NCCL*](https://arxiv.org/pdf/2507.04786v1)；
- [*DeepSeekMoE*](https://arxiv.org/pdf/2401.06066)；
- [Hugging Face DeepSeek-V3 Modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py)；
- [第 4 章：DeepSeekMoE](../chapters/04-deepseek-moe.md)；
- [第 5 章：Expert Parallelism](../chapters/05-expert-parallelism.md)。
