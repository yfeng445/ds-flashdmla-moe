# 第 4 周：Attention、Online Softmax 与 FlashAttention

- 原始页面：[Week 4: Attention](https://distinct-capricorn-c04.notion.site/Week-4-Attention-26a88315b6b480f1b26ffba505ff0677)
- 整理日期：2026-08-17
- 英文版：[Week 4: Attention](../source/notes/04-attention.en.md)

## 本周目标

第四周把 GEMM tiling 的数据复用思路扩展到 attention，并用 online softmax 在分块处理
`K/V` 时维护精确归一化状态。课程先实现 FlashAttention-2 论文算法 1 的正确版本，再把
Tensor Core、warp 分工和更底层优化留到后续阶段。

## Attention 定义

对 `Q,K,V in R^(N x d)`：

\[
O = \operatorname{softmax}\left(\frac{QK^T}{\sqrt d}\right)V.
\]

朴素实现包含矩阵乘法、转置、缩放、softmax 和第二次矩阵乘法，并会把 `N x N` 的 score 或
probability 矩阵写入 HBM。FlashAttention 的关键不是近似，而是改变计算顺序，避免这份中间
矩阵落盘。

## Q Tile 决定输出 Ownership

把序列维切成 query tile `Q_i` 和 key/value tile `K_j,V_j`：

- 一个 thread block 负责一个 `Q_i` 以及对应的输出 `O_i`；
- 该 block 依次遍历全部 `K_j,V_j`；
- 每个 block 只写自己的 `O_i`，因此输出 ownership 清晰；
- 不同 block 会重复读取 `K/V` tile，但不会争用同一输出行。

这里 chunk、block 和 tile 都是在描述分块，但 CUDA thread block 与数学 tile 不应在所有语境
下机械等同。

## Shared Memory 约束

若元素为 4 字节，最基本的 `Q_i,O_i,K_j,V_j` 驻留需求约为：

\[
2B_r d \cdot 4 + 2B_c d \cdot 4.
\]

实际算法还需要 score tile、概率 tile、行最大值与归一化因子。原课程给出的完整 tile 结构
导出如下元素数量约束：

\[
M < 2B_c d + 2B_r d + 6B_r + 2B_rB_c,
\]

其中 `M` 表示可用于这些中间量的 shared-memory 元素预算。真实 kernel 还必须扣除 register、
对齐、动态 shared memory 限制和实现中额外状态。

## Online Softmax 状态

对每一行分块读取 score 时维护：

- `m`：截至当前 tile 的行最大值；
- `l`：以 `m` 为基准的指数和；
- `O`：与同一归一化状态对应的未最终除法输出累加。

新 tile 的最大值为 `m_new` 时，旧状态必须乘以 `exp(m_old - m_new)` 后再与新 tile 合并。
这使任意 tile 顺序下的结果与一次性 softmax 数学等价，同时避免保存完整 score 行。

## `diag` 记号的含义

`diag(x)` 把长度为 `m` 的向量变成 `m x m` 对角矩阵，因此 `diag(x)Y` 等价于用 `x[i]`
缩放 `Y` 的第 `i` 行；`diag(x)^(-1)Y` 则逐行除以 `x[i]`。论文用这一矩阵记号紧凑表示按行
rescale，并不要求实现真的构造对角矩阵。

## 原练习要求

1. 用未并行的 C 实现 FlashAttention-2 第 3.1 节算法 1；
2. 用 CUDA 并行实现同一算法；
3. 此阶段优先保证 correctness，不要求立即使用 Tensor Core、coalescing 或复杂 warp 分工；
4. 使用普通 attention reference 比较输出，并覆盖非整 tile、不同序列长度和数值范围。

## 延伸阅读

- [*Online Normalizer Calculation for Softmax*](https://arxiv.org/pdf/1805.02867)；
- [*FlashAttention*](https://arxiv.org/pdf/2205.14135)，建议先读第 1-5 页；
- [*FlashAttention-2*](https://arxiv.org/pdf/2307.08691)，建议先读第 1-6 页；
- [How to Read a Paper](https://6826.csail.mit.edu/2020/papers/howtoread.pdf)；
- [How to Read an Engineering Research Paper](https://cseweb.ucsd.edu/~wgg/CSE210/howtoread.html)；
- [第 1 章：Online Softmax](../chapters/01-online-softmax.md)；
- [第 2 章：FlashAttention](../chapters/02-flash-attention.md)。
