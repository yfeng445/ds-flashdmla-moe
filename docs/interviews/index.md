# AI Infra 面试资料与准备指南

本目录采用“一份主题题库对应一个 `*-qa.md` 文件”的方式组织面试内容，方便按专题复习和持续补题。当前题库是 [MLA、CUDA 与 MoE Q&A](ai-infra-mla-cuda-moe-qa.md)。

下面的准备指南把 attention、MLA、MoE、分布式通信、推理系统和 GPU 性能分析整理成一条通用学习路线，不把任何特定代码库作为求职陈述材料。职业能力建设与社区资源见 [HPC for AI 第 11 周：Career Building](../courses/NEU_INFO_7375/notes/11-career-building.md)；课程笔记按源页面顺序保留，本指南则按准备流程重新编排。

## 1. 先确定方向与能力地图

AI Infra 岗位大致可分为：

| 方向 | 核心问题 | 需要深入的能力 |
| --- | --- | --- |
| CUDA / Kernel | 单算子正确性、IO 与并行效率 | CUDA、数值稳定性、memory hierarchy、profiling |
| 训练系统 | 模型状态、并行策略与容错 | DDP/FSDP、TP/PP/EP、collective、checkpoint |
| 推理系统 | TTFT、ITL、吞吐与显存容量 | batching、KV cache、scheduler、quantization |
| Compiler / Runtime | 图优化、代码生成与设备执行 | IR、layout、fusion、Triton/OpenXLA、runtime |
| GPU Cluster | 资源调度、网络拓扑与可靠性 | Kubernetes/Slurm、RDMA、observability、故障定位 |

选择一个主方向和一个相邻方向。主方向要求能推导、实现、测量和定位问题；相邻方向要求能解释接口和系统权衡。阅读 JD 时拆出问题域、编程语言、硬件与分布式要求、验证方法和加分项，再用它们调整学习权重。

## 2. Online Softmax 与 FlashAttention

### 2.1 数值稳定性

对一行 score `x`，稳定 softmax 为：

```text
m = max_i x_i
p_i = exp(x_i - m) / sum_j exp(x_j - m)
```

Online softmax 可以分块扫描。将已有状态 `(m, l)` 与新 tile 的 `(m_t, l_t)` 合并：

```text
m_new = max(m, m_t)
l_new = exp(m - m_new) * l + exp(m_t - m_new) * l_t
```

若还维护未归一化输出 `o`，历史输出也要乘 `exp(m-m_new)`，当前 `P@V` 乘 `exp(m_t-m_new)` 后再累加，最后除以 `l`。应能解释 row max、denominator、output accumulator 和 mask 四种状态；全 mask 行必须避免 NaN。

### 2.2 FlashAttention 的核心

FlashAttention 用 tiling 与 online softmax 避免在 HBM 中物化完整 `S×S` score/probability matrix。FA2 进一步调整 work partition，常用 sliced-Q 降低 warp 间交换与同步，减少非矩阵乘工作，并增加 sequence 方向的并行度。是否更快仍取决于 head dimension、sequence、dtype、GPU 与 kernel 配置。

Causal mask 应在 score 进入 max/exp/sum 前生效。kernel 根据绝对 `(q_position, k_position)` 判断 `k_position <= q_position`，无需物化完整 mask。

## 3. MHA、GQA、MLA 与 KV Cache

### 3.1 共享与压缩

- MHA：每个 query head 使用独立 K/V head，KV cache 最大；
- MQA：所有 query heads 共享一组 K/V；
- GQA：一组 query heads 共享 K/V；
- MLA：把历史状态压缩到共享 latent，再重建 K/V 或把投影吸收到 query/output 路径。

比较 cache 时要计算每 token 的实际元素数和 dtype，不能只看 latent rank。

### 3.2 Naive 与 absorbed MLA

设单个 head 的 up-projection 为 `W_K`、`W_V`：

```text
q_nope: [d_nope]
c_t:    [r_kv]
W_K:    [d_nope, r_kv]
W_V:    [d_v, r_kv]

q_nope · (W_K c_t) = (q_nope W_K) · c_t
sum_t p_t (W_V c_t) = W_V (sum_t p_t c_t)
```

因此可以先得到 latent query，在 latent 空间完成 weighted sum，再做 value up-projection。前提是投影对 token 共享且 softmax 概率不变。RoPE 与位置相关，不能随意吸收；常见设计单独保存 positional key。

### 3.3 KV cache、在线重建与分页

KV cache 保存历史 token 可复用状态，避免 decode 重算完整前缀。在线重建则是不长期物化完整 per-head K/V，而从 latent 按 tile 临时重建，或用 absorbed 形式绕过显式重建。

Static cache 预分配连续空间；paged cache 把逻辑 token block 映射到非连续物理页。Causal 语义属于逻辑 token position，而不是 physical slot。完整 serving 系统还需要 allocator、eviction、prefix sharing 和 continuous-batching scheduler。

PagedAttention 主要解决动态 KV cache 的分配、碎片和共享；FlashAttention 主要优化 attention 计算的 IO。两者不是替代关系。

## 4. DeepSeek MoE 与路由

标准数据流是：

```text
gate score -> group-limited Top-K -> dispatch/pack
           -> expert SwiGLU -> weighted combine -> token order
```

DeepSeek 风格 SwiGLU expert：

```text
W2(SiLU(W1(x)) * W3(x))
```

Routing bias 可以影响 expert 选择，但 combine weight 通常仍从原始 score gather，避免把负载控制混入模型概率语义。

### 4.1 Expert-major pack

token-major route 跨 expert 交错。按 expert 重排后，每个 expert 的 active rows 连续，可用 offsets 调度 GEMM，减少 padding。必须保留 inverse permutation、token identity、top-k slot 与 route weight，combine 才能恢复原顺序。

### 4.2 Capacity 与负载

常见初始容量估计：

```text
capacity = ceil(tokens * top_k / experts * capacity_factor)
```

处理过载可选择动态 load bias、增大 capacity factor、drop、reroute 或 fallback。每种策略都改变吞吐、数值语义、均衡效果与复杂度，训练和推理还可能使用不同策略。

## 5. Expert Parallel 与通信

普通 EP 的两次 All-to-All 分别负责 dispatch token 到 expert owner，以及把 expert output 送回源 rank。每条 route 至少携带 source rank、token index、top-k slot、expert owner 和 weight。非线性 expert 计算前不能随意提前乘 route weight。

NCCL collective 适合批量、规则的 GPU 通信；NVSHMEM 提供 one-sided put/get 与 signal，可用于更细粒度的 actor/persistent-kernel 设计。概念上的 tile 生命周期为：

```text
route/pack -> dispatch packet -> remote signal
-> decode task -> schedule -> expert compute
-> return packet -> source signal -> combine
```

证明 communication-computation overlap 需要 Nsight Systems timeline 与 serialized baseline；总 latency 变小本身不能说明物理重叠。Nsight Compute 则用于分析单 kernel 的 occupancy、memory throughput、tensor-core utilization 和 stall reasons。

## 6. GPU 内存层次

- register：线程私有，最快；过量使用会降低 occupancy 或造成 spill；
- shared memory / L1：SM 内资源，shared memory 由 CTA 显式协作；
- L2：全 GPU 共享的片上 cache；
- HBM：设备全局内存；
- PCIe：CPU-GPU 或部分 GPU-GPU 通路；
- NVLink：高带宽 GPU 互连；
- NVSwitch：连接多个 NVLink endpoint 的交换 fabric。

常见追问包括 coalescing、bank conflict、shared-memory 容量与 occupancy 权衡、tile padding，以及为什么 `__syncthreads()` 必须由整个 block 一致到达。

## 7. 分布式训练与推理系统

### 7.1 DDP 与 FSDP

- DDP 通常让每个 rank 常驻完整参数、梯度和 optimizer state，并在反向传播中 All-Reduce 梯度；
- FSDP 分片参数、梯度和 optimizer state，在计算 wrapped module 前 All-Gather 参数，反向时 Reduce-Scatter 梯度。它节省显存，但增加通信频率与调度复杂度。

### 7.2 Serving

- vLLM 重点是推理 serving、continuous batching、PagedAttention/KV 管理与吞吐；
- DeepSpeed 重点是训练并行、ZeRO 和内存管理，同时也提供推理能力；
- prefill 通常更偏计算密集，以 TTFT 衡量；decode 通常更受显存带宽限制，以 ITL 衡量；
- static batching 会留下空闲 slot，continuous batching 在 iteration 边界接纳新请求；
- scheduler 必须联合考虑 KV 容量、prefill 干扰、吞吐与尾延迟。

## 8. C++ 基础

- 左值有稳定身份并可取地址；右值通常是临时值或可被移动的对象；
- `T&` 绑定可修改左值，`const T&` 可绑定左值和临时量，`T&&` 绑定右值；
- 大型只读对象常传 `const T&`，需要取得所有权时可按值传入再 move，小型标量按值最清楚；
- 拷贝构造复制资源，移动构造转移资源；移动后的对象仍有效，但其值通常未指定。

还应掌握 RAII、智能指针、容器失效规则、模板、线程同步和 CUDA host 侧资源生命周期。

## 9. 手写与口述练习

### 9.1 多头注意力

```python
def mha(x, wq, wk, wv, wo, heads, mask=None):
    batch, seq, model = x.shape
    dim = wq.shape[0] // heads
    q = torch.nn.functional.linear(x, wq).view(batch, seq, heads, dim).transpose(1, 2)
    k = torch.nn.functional.linear(x, wk).view(batch, seq, heads, dim).transpose(1, 2)
    v = torch.nn.functional.linear(x, wv).view(batch, seq, heads, dim).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) / dim**0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = (probs @ v).transpose(1, 2).contiguous().view(batch, seq, heads * dim)
    return torch.nn.functional.linear(out, wo)
```

口述时主动说明 shape、FP32 softmax、mask 语义和全 mask 行处理。

### 9.2 Reduction

若 `1,240,000` 个元素、每 block 256 threads、每 thread 加载两个元素：

```text
items_per_block = 512
grid = ceil(1,240,000 / 512) = 2,422
```

第一阶段输出 2,422 个 partial，后续还需递归规约。边界加载置零；成熟实现常用 grid-stride loop 和 warp shuffle 减少 shared memory 与同步。

### 9.3 Transformer 前向

```text
x -> RMSNorm -> Attention -> residual add
  -> RMSNorm -> FFN or MoE -> residual add
```

应能在每一步给出 tensor shape、参数规模、activation 生命周期，以及 dense FFN 和 sparse MoE 在计算与通信上的区别。

## 10. 性能证据方法

任何“更快”结论都至少需要：

- GPU 型号、数量、互连、power/clock 条件；
- driver、CUDA、framework 和编译架构；
- batch、sequence、head、rank、expert、top-k、dtype；
- baseline 以及数学语义是否相同；
- warmup、样本数、median/p95 和原始 latency；
- output、gradient 或 loss 的误差标准；
- 声称 overlap 或瓶颈改善时的 profiler timeline/counters。

建议采用如下闭环：

```text
定义语义 -> 建立 reference -> 覆盖边界输入 -> 测量 baseline
-> profiler 定位 -> 单一变量优化 -> 复测正确性和性能 -> 记录边界
```

结果尚不充分时，只陈述“实现了某条路径”“通过数值对照”或“设计目标是”，不把小 shape、单机或单一 baseline 外推成普遍结论。

## 11. 社区、开源与自学路线

### 11.1 社区

优先选择与方向匹配的长期社区：SC、ISC、hpc.social、PyTorch、OpenXLA、MLCommons、NVIDIA Developer、HPC-AI Society 和 HPC Carpentry。参与方式应从可复现问题、技术讨论、workshop 和公开记录开始。

### 11.2 开源

成熟生态可关注 PyTorch、vLLM、Megatron-LM、DeepSpeed 和 OpenXLA；快速演进方向可关注 SGLang、llm-d、SkyPilot、TensorRT-LLM 与 Colossal-AI。先阅读 contribution guide 和近期 issue，再从文档、测试、复现或最小修复开始。

### 11.3 阅读与构建

- 广度：浏览 SC/ISC、NeurIPS/ICLR/ICML、SOSP/EuroSys 的 systems 论文；
- 深度：定期完整复现一篇论文，保留正确性、环境、profile 与失败记录；
- 训练：用 Nanotron 与 Ultra-Scale Playbook 理解并行维度；
- 集群：练习 Kubernetes/Slurm、GPU 拓扑和 vLLM autoscaling；
- kernel：用 Nsight 建立并验证瓶颈假设；
- compiler：从 Triton 延伸到 layout、IR 与代码生成；
- 数值：系统学习 mixed precision 与 quantization。

## 12. 面试准备循环

### 12.1 技术题

每个主题按五层准备：

1. 问题：开销或约束是什么；
2. 原理：公式与数据流怎样成立；
3. 实现：shape、线程/进程划分、内存和同步；
4. 验证：reference、边界、误差、benchmark、profile；
5. 权衡：适用条件、失败模式和替代方案。

先闭卷口述或手写，再查资料补缺口；随后用追问测试“为什么”“如果 shape 改变”“如何证明”。只讲概念会缺少落地，只讲 kernel 会缺少系统视角，只报数字而不讲 baseline 会缺少可信度。

### 12.2 经历与行为题

只选能够解释决策和复盘的真实经历。使用简洁的 `背景/约束 -> 目标 -> 行动与取舍 -> 可核验结果 -> 反思` 结构，明确团队协作边界，不虚构规模、性能或职责。准备以下类型各一例：

- 定位复杂故障；
- 在性能与正确性之间做取舍；
- 与他人处理技术分歧；
- 需求不清或资源受限时推进；
- 实验失败后改变假设。

### 12.3 向面试官提问

- 团队的核心 workload、规模和主要性能指标是什么？
- kernel、runtime、scheduler 和 cluster 层分别由谁负责，接口怎样协作？
- correctness、benchmark 和 regression 的标准是什么？
- 当前最大瓶颈是计算、显存、网络、调度还是可靠性？
- 新成员前三个月最希望解决什么问题？
- 团队如何使用 profiler、线上 tracing 和故障复盘？

这些问题用于判断工作内容、工程成熟度和学习空间，也能把讨论落到真实系统约束上。
