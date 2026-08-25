# 第 8 周：数据并行与专家并行

- 原始页面：[Week 8: Data & Expert Parallelism](https://distinct-capricorn-c04.notion.site/Week-8-Data-Expert-Parallelism-32688315b6b480c6b66bf9830dfc3cc6)
- 整理日期：2026-08-17
- 英文版：[Week 8: Data & Expert Parallelism](../source/notes/08-data-expert-parallelism.en.md)

## 从 Dense FFN 到 MoE

Dense Transformer 对每个 token 激活同一组 FFN 参数。Mixture of Experts（MoE）则用 router 为每个 token 选择少量 expert，只激活模型总参数的一部分。因此，MoE 可以在单 token 计算量近似受控的同时扩大总参数规模；代价是 routing、负载均衡、token 重排和跨设备通信。

一个包含 `N` 个 expert 的 MoE 层通常由 router 与 experts 组成。对 token `x_i`，router 先产生 affinity：

```text
r_i = Router(x_i) ∈ R^N
SoftmaxRouter(x_i) = softmax(x_i W + b)
TopKRouter(x_i) = softmax(KeepTopK(x_i W + b))
```

若选择集合为 `T_i`，输出是被选 expert 输出的加权和：

```text
y_i = Σ_{e ∈ T_i} r_{i,e} Expert_e(x_i)
```

`K=1` 时常称为 Switch-style routing。实际系统还需定义 capacity、溢出 token 的处理方式，以及用于避免少数 expert 过载的辅助损失。

## DeepSeekMoE 的 expert 组织

DeepSeekMoE 使用两项关键设计：

- **细粒度 expert 切分**：把传统 expert 的中间维度缩小为原来的 `1/m`，同时把 expert 数扩为 `mN`，并把每个 token 的选择数扩为 `mK`。总激活规模近似不变，但组合空间更丰富；
- **shared expert isolation**：设置总是激活的 shared experts，承载跨 token 的共性知识；routed experts 则学习更有区分度的模式。

## DP、TP 与 EP

| 并行方式 | 主要切分对象 | 典型通信 | 主要约束 |
| --- | --- | --- | --- |
| Data Parallelism（DP） | batch/token 样本 | 梯度 All-Reduce 或 Reduce-Scatter | 每个 rank 通常保留模型副本 |
| Tensor/Model Parallelism（TP） | 单层权重与矩阵乘 | All-Reduce、All-Gather、Reduce-Scatter | 高频通信位于层内部 |
| Expert Parallelism（EP） | experts | 两次 All-to-All | 路由负载和网络带宽 |

EP 通常让 router 在各 rank 上复制，而把 experts 分片到不同 GPU。以下描述假设单个 expert 能装入一张 GPU；若不能，还要在 expert 内叠加 TP。

## Expert Parallel 前向数据流

一次典型前向过程为：

1. 每个 rank 对本地 token 执行 routing，得到目标 expert 和权重；
2. 按目标 rank/expert 对 token 做本地 permutation；
3. 第一次 All-to-All 把 token dispatch 到持有相应 expert 的 rank；
4. 各 rank 对收到的 token 执行本地 expert 计算；
5. 第二次 All-to-All 把 expert 输出送回 token 的原始 rank；
6. 本地 unpermute，并按 router 权重 combine 多个 expert 输出。

因此，MoE 性能不仅取决于 GEMM，也取决于 dispatch/combine 的带宽、消息粒度、负载偏斜、padding/capacity 浪费，以及通信能否与计算重叠。

## 混合 EP、TP 与 DP

大模型往往同时使用多种并行维度。一个简化的数据流是：attention 在 TP group 内计算并归约；各 DP/EP rank 本地 routing；EP group 通过 All-to-All 交换 token；expert 内部可再用 TP 完成矩阵乘与归约；最后执行逆向 All-to-All 并恢复原 token 顺序。

设计 process groups 时要明确每一种 collective 属于哪个维度，避免把 DP、TP、EP 的 rank 语义混在一起。通信优化也应针对真实拓扑：例如让高频 TP 留在节点内，把 EP 或 DP 映射到更合适的跨节点 fabric。

## 课程作业方向

本周作业要求用 CUDA/NCCL 实现多 GPU DeepSeekMoE，组合数据并行与专家并行，复用已生成的确定性测试，并与 Transformers reference 对比正确性与性能。这里记录的是课程任务及系统设计要点，不表示当前仓库已经实现该多 GPU 路径。

## 延伸阅读

- [*Ring Attention with Blockwise Transformers for Near-Infinite Context*](https://arxiv.org/pdf/2310.01889)；
- [*DeepSpeed-Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models*](https://arxiv.org/pdf/2309.14509)；
- [*Efficient Training of MoE Models at Scale with Pytorch*](https://arxiv.org/pdf/2303.06318)；
- [*MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models*](https://arxiv.org/pdf/2505.11432)；
- [第 4 章：DeepSeekMoE](../chapters/04-deepseek-moe.md)；
- [第 5 章：Expert Parallelism](../chapters/05-expert-parallelism.md)。
