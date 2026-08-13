# 第二章：从标准 Attention 到 FlashAttention

## 2.1 标准 Attention 的数据流

单个 head 的 scaled dot-product attention 为：

```math
S = \frac{QK^T}{\sqrt{D}}, \qquad
P = \operatorname{softmax}(S), \qquad
O = PV.
```

若序列长度为 `N`，中间矩阵 `S` 和 `P` 都是 `N x N`。朴素 GPU 实现通常启动三个
阶段：GEMM、Softmax、GEMM。计算复杂度本来就是 `O(N^2D)`，但额外把 `O(N^2)`
中间量写入并读回 HBM，会让长序列 attention 被显存容量和带宽限制。

FlashAttention 的关键不是近似，而是改变循环和存储层次：让 `Q/K/V` tile 进入
shared memory 或寄存器，在片上完成 score、在线 Softmax 与 value 累加，避免将完整
`S/P` 写入 HBM。

## 2.2 二维分块

把 query 行分为高度 `B_r` 的块，把 key/value 行分为高度 `B_c` 的块。一个 CTA
负责一个 query 块，并遍历所有 key/value 块：

```text
for each Q tile i:
    initialize m_i, l_i, accumulator_i
    for each K/V tile j:
        load K_j, V_j
        S_ij = Q_i K_j^T * scale
        apply mask
        update online softmax state
    write normalized O_i
```

这对应第一章的行级递推，只是一次处理 `B_r x B_c` 个 score。

## 2.3 Causal mask 的坐标

当 `S_q == S_k` 时，causal 条件是 `key_position <= query_position`。decode 时常见
`S_q = 1, S_k = cache_length`；此时唯一 query 代表序列末尾，必须能看到整个历史。
因此本仓库使用右下对齐规则：

```math
k \le q + S_k - S_q.
```

这类语义必须先在 reference 中锁定。若 kernel 只用 tile 内局部坐标做三角 mask，
相同长度测试可能通过，decode 却会悄悄出错。

## 2.4 从 FA1 到 FA2

第一代实现的主要思想是 IO-aware tiling。FlashAttention-2 进一步改善工作划分：

- 减少非矩阵乘法 FLOP；
- 在序列维上增加并行度；
- 在 warp 间重新分配 Q/K/V tile 和输出工作；
- 减少 shared-memory 读写与不必要同步。

课程原型使用 WMMA 演示 Tensor Core，但“调用了 WMMA”不等于形成高效 FA2。真正的
kernel 还必须回答：一个 warp 负责哪些行、累加器是否发生重复写、tail tile 是否
越界、shared memory 是否超过设备限制，以及数据布局是否允许合并访问。

## 2.5 Forward 的验证矩阵

每个 CUDA forward 至少应覆盖以下组合：

- `S_q/S_k`：相等、cross-attention、单 token decode；
- `D`：16/32 的倍数和非倍数尾部；
- value dimension 与 head dimension 相等和不等；
- causal 与 non-causal；
- boolean mask、additive mask、全遮挡行；
- FP16、BF16、FP32；
- 非连续张量或明确拒绝非连续张量。

比较对象不是另一个 CUDA 原型，而是 FP32/FP64 materialized reference。除了最大绝对
误差，还应记录相对误差，并将 kernel 误差与低精度 PyTorch baseline 的误差比较。

## 2.6 第一版正确性 kernel

在二维 tile 和 Tensor Core 优化之前，先建立最小的原生算子边界。仓库第一版 CUDA
forward 令一个 CTA 负责一个 `(batch, head, query)` 行，CTA 内线程共同完成：

1. 将当前 query 搬入 shared memory；
2. 依次遍历可见 key，对 `D` 维做 block reduction；
3. 每得到一个 score，就用第一章的在线 Softmax 递推更新 `(m,l)`；
4. 同步缩放 value 分子累加器；
5. 遍历完成后除以 `l` 并写回输出。

这还不是高性能 FlashAttention：它没有二维 score tile，也没有 Tensor Core，并会为每个
query 重复读取 K/V。它的价值是让 PyTorch dispatcher、张量约束、当前 CUDA stream、
causal 坐标和数值 reference 首先形成一个完整可测闭环。此后可以替换 kernel 内部的
工作划分，而不改变公开算子语义。

原生入口的第一版约束是：

- contiguous FP32，形状为 `[B,H,S,D]`；
- Q/K 的 head dimension 相同，K/V 的序列长度相同；
- 支持 `S_q != S_k` 与右下对齐 causal mask；
- 暂不支持显式 mask、FP16/BF16 和 backward；
- shared memory 需求超过设备上限时明确报错。

Python 的 `backend="auto"` 在请求满足这些条件时进入 CUDA forward；不支持的
shape/dtype/mask 会回退到可微 reference。原生 backward 完成之前，autograd 在反向时
用 reference 重算梯度，因此语义正确但不是最终训练性能。`backend="cuda"` 用于验证和
基准，它会对不支持的输入约束直接报错，避免把回退耗时误当成 kernel 性能。

## 2.7 Backward 的核心恒等式

令 `dO` 为上游梯度，`P = softmax(S)`，则：

```math
dV = P^T dO,
```

```math
D_i = \sum_d dO_{id} O_{id},
```

```math
dS_{ij} = P_{ij}\left((dO_i\cdot V_j)-D_i\right),
```

```math
dQ = \text{scale}\, dS K, \qquad
dK = \text{scale}\, dS^T Q.
```

Backward 可以利用 forward 保存的 log-sum-exp 重算 `P`，无需保存完整概率矩阵。
重算必须遍历完整 `D` 维；若一个 `16 x 16` WMMA 只计算了第一个 K tile，却被当成
完整 `QK^T`，梯度在 `D > 16` 时会错误。

仓库中的 `scaled_dot_product_attention_backward_reference` 直接实现以上恒等式，而不是
调用另一个 CUDA kernel 作为“参考”。它会把 mask 应用在 score 上，再由稳定 Softmax
生成 P。全遮挡行的 P 为零，因此该行对 dQ、dK 和 dV 的贡献也自然为零。

验证 backward 时需要区分三类证据：

1. 解析公式与 autograd 对同一 FP64 forward 的一阶梯度一致；
2. `gradcheck` 用有限差分检查一阶导数；
3. `gradgradcheck` 检查 backward 本身仍由可微运算组成。

若 Q/K/V 引用同一个张量，三个角色的偏导必须分别计算，再由 autograd 按 alias 关系
相加。把同一 graph node 重复传给一次 `autograd.grad` 容易先得到总梯度，再在外层重复
累计；这是算子测试必须单独覆盖的边界。

第一版 CUDA backward 与 forward 一样采用一个 query row 一个 CTA。它进行三次 key
遍历：第一次计算稳定 Softmax 的 `(m,l)`，第二次计算行校正项 `D_i`，第三次形成 `dS`
并累加梯度。`dQ` 只属于当前 query row，可在 CTA 内累加；不同 query 都可能更新同一
key/value row，所以 `dK/dV` 使用 `atomicAdd`。

原子累加的加法顺序由 GPU 调度决定，浮点结果不保证 bitwise deterministic。因此
`torch.use_deterministic_algorithms(True)` 时必须走解析 reference，而不能悄悄调用原生
backward。高阶梯度也使用解析公式，因为第一版原生 backward 没有继续注册它自己的
backward。这些回退是语义边界，不是性能声明。

## 2.8 性能测量

正确性通过后再测性能：

1. 预热 kernel 与 allocator。
2. 使用 CUDA events，而不是 CPU wall clock 包围异步 launch。
3. 固定 shape、dtype、causal 模式和 GPU clock 环境。
4. 报告中位数和尾延迟，不只报告最好一次。
5. 同时记录峰值显存与有效吞吐。

Nsight Compute 的第一轮问题应很朴素：是 launch-bound、memory-bound，还是
Tensor Core 没有吃满？不要在没有 profile 证据时先写 megakernel。
