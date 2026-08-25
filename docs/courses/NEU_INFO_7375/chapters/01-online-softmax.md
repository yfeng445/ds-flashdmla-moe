# 第一章：从 Softmax 到在线 Softmax

## 1.1 为什么先研究 Softmax

Attention 的核心并不只是矩阵乘法。对每一行 query，我们还必须把一整行 score
转换为概率：

```math
p_j = \frac{e^{x_j}}{\sum_t e^{x_t}}.
```

直接计算指数很容易溢出。稳定实现先减去行最大值
`m = max_j x_j`：

```math
p_j = \frac{e^{x_j-m}}{\sum_t e^{x_t-m}}.
```

减去同一个常数不会改变结果，却保证最大的指数为 1。问题是：这仍然需要先读完整
行求最大值，再读一遍求指数和。若 score 行不能完整驻留在片上存储中，我们能否按块
处理，并在看到新块时修正旧结果？

## 1.2 在线最大值与归一化因子

假设已经处理过一段元素，保存两个状态：

```math
m = \max_j x_j, \qquad
l = \sum_j e^{x_j-m}.
```

新块的对应状态为 `m_t, l_t`。合并后的最大值是：

```math
m' = \max(m, m_t).
```

旧分母和新块分母使用的基准不同，必须缩放到共同基准 `m'`：

```math
l' = e^{m-m'}l + e^{m_t-m'}l_t.
```

若新块直接以 `m'` 为基准计算指数，则第二项也可写成
`sum_j exp(x_j - m')`。这条递推只需要常数大小的行状态，因此可以流式处理任意长的
score 行。

## 1.3 同时维护 Attention 输出

Attention 一行输出为：

```math
o = \frac{\sum_j e^{x_j-m}v_j}{l}.
```

定义未归一化累加器：

```math
a = \sum_j e^{x_j-m}v_j.
```

看到新块后，与分母同理：

```math
a' = e^{m-m'}a + \sum_{j\in tile} e^{x_j-m'}v_j.
```

遍历完全部 key/value 块后，输出 `o = a/l`。于是实现不再需要保存概率矩阵；每个
query 行只保留 `m`、`l` 和长度为 value dimension 的 `a`。

仓库中的 `blockwise_attention` 就是这条递推的可执行规格。它使用 PyTorch 循环而非
CUDA，目的不是快，而是让每个状态转移都可以被测试。

## 1.4 全遮挡行

如果一行所有位置都被 mask，score 全为负无穷：

```math
m=-\infty, \qquad l=0.
```

此时不能计算 `-inf - -inf`，也不能除以零。本讲义采用明确契约：

- 输出向量为 0；
- log-sum-exp 为负无穷；
- 不产生 NaN。

CUDA kernel 不能依赖“实际输入一般不会这样”来回避边界条件。padding、稀疏窗口或
错误组合的 mask 都可能产生全遮挡行。

## 1.5 并行规约

在 GPU 上，一行的最大值和求和通常由 warp 或 CTA 合作完成：

1. 每个线程处理若干 score。
2. 使用 shuffle 做 warp 内 max/sum。
3. 多个 warp 的部分结果写入 shared memory。
4. 一个 warp 完成 CTA 级规约。

实现时要同时检查：

- shuffle mask 是否只包含仍然活跃的 lane；
- 非 32 整倍数的尾部是否填充为 `-inf` 或 0；
- shared memory 在下一次复用前是否有 CTA 级同步；
- FP16/BF16 输入是否在 FP32 中进行 max、exp 和累加。

## 1.6 本章检查点

读完后应能解释：

- 为什么减最大值不改变 Softmax；
- 为什么新最大值出现时必须重标定旧分母和旧输出；
- 为什么只保存归一化后的 `o` 也可以更新，但保存未归一化 `a` 更容易推导；
- 全遮挡行怎样避免 NaN；
- 在线算法减少的是中间存储，而不是数学运算的阶数。
