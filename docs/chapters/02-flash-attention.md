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

把 query 行分为高度 `B_r` 的块，把 key/value 行分为高度 `B_c` 的块。下面先写成
query 块在外层的形式；2.4 节会说明 FA1 如何交换两层循环，以及这种交换如何改变
CTA 和 warp 的工作划分：

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

## 2.4 从 FA1 到 FA2：工作划分而不是版本标签

[FlashAttention-1 论文](https://arxiv.org/abs/2205.14135)建立了 IO-aware exact
attention；[FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)在相同目标上进一步
调整循环、并行网格和 warp 分工。本仓库把两者实现为可执行的教学 kernel；直接对应的
production 源码是共享约束与 causal 坐标的
[`attention_common.cuh`](../../csrc/attention/attention_common.cuh)、
KV-outer 的 [`fa1_forward_cuda.cu`](../../csrc/attention/fa1_forward_cuda.cu) 和
Q-tile-owned 的 [`fa2_forward_cuda.cu`](../../csrc/attention/fa2_forward_cuda.cu)。

### 2.4.1 循环次序

FA1 先固定一个 K/V tile，再在 kernel 内遍历所有 Q tile：`K/V outer -> Q inner`。
同一份 K/V tile 因而能被一个 batch-head CTA 中的多个 Q tile 使用。FA2 则让 CTA
先拥有一个 Q tile，Q 常驻该 CTA，再顺序遍历 K/V tile：`Q owner -> K/V inner`。

### 2.4.2 launch grid

FA1 的 grid 只有 batch-head 维，每个 `(batch, head)` 启动一个 CTA；Q tile 循环留在
CTA 内。FA2 把 query-tile 也展开到 grid，每个 `(batch, head, query_tile)` 启动一个
CTA。后者在 query 序列较长、batch/head 较小时提供更多独立 CTA，但这里只描述并行
结构，不据此承诺某个 backend 必然更快。

### 2.4.3 warp 分工

FA1 的四个 warp 对同一 Q 行拆分当前 tile 的 key 位置，各自产生局部 `m/l` 与 value
分子，随后合并四份局部输出状态。FA2 则按 Q 行拆分 warp：一个 warp 独占 Q tile 中
的一行，所有 lane 协作完成该行的 dot reduction 和输出更新，不需要跨 warp 合并同一
输出行。

### 2.4.4 在线 Softmax 递推

FA1 在每个 K/V tile 后把归一化的 `O`、行最大值 `m` 和分母 `l` 写入 FP32 workspace；
下一个 K/V tile 会重新读出三者，把旧的归一化 `O` 还原到共同尺度后再合并并归一化。
FA2 让每个 warp 将未归一化的 value 分子、`m` 和 `l` 保持在片上，遍历完所有 K/V
tile 后只除以 `l` 一次并写出 FP16 结果。

两者都可以在 causal 模式下跳过整个不可见的 K/V tile，也都在 tile 内用右下对齐坐标
过滤单个 key。这是共同的 causal 优化，不是区分 FA1 与 FA2 的版本定义。课程原型中
是否调用 WMMA 或 Tensor Core 同样不能单独决定版本；版本差异要落实到上述循环、grid、
warp 所有权和递推状态。

## 2.5 Forward 的验证矩阵

Forward 验证必须按 facade 中各 backend 的能力边界分组，不能把所有组合都当作每个
backend 的支持范围。对于支持的输入，至少覆盖：

- `S_q/S_k`：相等、cross-attention、单 token decode；
- `D`：16/32 的倍数和非倍数尾部；
- value dimension 与 head dimension 相等和不等；
- causal 与 non-causal；
- `reference`/`blockwise` 的浮点 dtype，`cuda_rowwise` 的 FP16/BF16/FP32，以及
  `fa1`/`fa2` 的 FP16；
- 仅对 `reference`/`blockwise` 覆盖 boolean mask、additive mask 和全遮挡行。

支持组合的比较对象不是另一个 CUDA 原型，而是 FP32/FP64 materialized reference。
除了最大绝对误差，还应记录相对误差，并将 kernel 误差与低精度 PyTorch baseline 的
误差比较。

不支持的组合应单独做 strict rejection 测试：`cuda_rowwise`、`fa1`、`fa2` 必须拒绝
任何显式 mask 和非连续 Q/K/V；`fa1`/`fa2` 还必须拒绝 BF16/FP32，以及超过其上限的
`D` 或 `D_v`。这类测试验证的是 backend contract，而不是数值误差。

## 2.6 统一 facade、backend 矩阵与第一版正确性 kernel

公开入口统一为 `flash_attention_forward`；显式 backend 的实现和当前能力如下：

| Backend | Implementation | Dtype/device | Gradient behavior |
|---|---|---|---|
| `reference` | materialized PyTorch specification | floating CPU/CUDA | differentiable |
| `blockwise` | online-softmax PyTorch specification | floating CPU/CUDA | differentiable |
| `cuda_rowwise` | one query row per CTA | FP16/BF16/FP32 CUDA | existing native/reference backward policy |
| `fa1` | formal KV-outer teaching kernel | FP16 CUDA, `D,D_v <= 128` | forward-only |
| `fa2` | formal Q-tile-owned teaching kernel | FP16 CUDA, `D,D_v <= 128` | forward-only |

下面的脚本可直接运行；两次调用除 `backend` 外使用完全相同的 facade 参数：

```python
import torch

from ds_flash_mla_moe import flash_attention_forward

q = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.float16)

out_fa1 = flash_attention_forward(
    q, k, v, causal=True, scale=None, attn_mask=None, backend="fa1"
)
out_fa2 = flash_attention_forward(
    q, k, v, causal=True, scale=None, attn_mask=None, backend="fa2"
)
torch.testing.assert_close(out_fa1, out_fa2, atol=2e-2, rtol=2e-2)
```

三个原生 backend 都要求 contiguous `[B,H,S,D]` 张量、Q/K/V 相同 dtype、Q/K 的
`D` 相同、K/V 的序列长度相同，且不接受显式 `attn_mask`；右下对齐 causal 还要求
`S_q <= S_k`。此外，`fa1`/`fa2` 只接受 FP16 CUDA、`D > 0`、非空 key、
`D,D_v <= 128`，并拒绝任何 `requires_grad=True` 的 Q/K/V。它们是严格的显式选项：
条件不满足就报错，不会静默回退。

`backend="auto"` **不会**选择 `fa1` 或 `fa2`。它只在满足约束时选择
`cuda_rowwise`，否则回退到可微的 `blockwise` specification。旧名 `backend="cuda"`
只是 `cuda_rowwise` 的 deprecated alias，不应出现在新的调用中。

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

当前原生入口的约束是：

- contiguous FP16/BF16/FP32，Q/K/V 使用相同 dtype，形状为 `[B,H,S,D]`；
- Q/K 的 head dimension 相同，K/V 的序列长度相同；
- 支持 `S_q != S_k` 与右下对齐 causal mask；
- dot、在线 Softmax 状态和 value numerator 在 FP32 中计算；
- 支持 native forward/backward，但暂不支持显式 mask；
- shared memory 需求超过设备上限时明确报错。

普通一阶反向在允许非确定性原子累加时进入 `cuda_rowwise` 的 native backward；
deterministic 模式与高阶梯度仍使用解析 reference。显式 `cuda_rowwise` 会对不支持的输入
约束直接报错，避免把回退耗时误当成 kernel 性能。

## 2.7 Backward 的核心恒等式

本节的 production CUDA backward 只属于 `cuda_rowwise`，不属于新的 formal `fa1`/`fa2`
backend；后两者是 forward-only，并在 facade 边界拒绝需要梯度的输入。仓库
`csrc/experimental/attention/` 下的 backward/WMMA 文件是未注册的课程实验，也不能当作
`fa1` 或 `fa2` 的可用 backward。

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

`cuda_rowwise` 或实验性 backward 可以利用 forward 保存的 log-sum-exp 重算 `P`，无需
保存完整概率矩阵。
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

`cuda_rowwise` 的第一版 CUDA backward 与 forward 一样采用一个 query row 一个 CTA。它进行三次 key
遍历：第一次计算稳定 Softmax 的 `(m,l)`，第二次计算行校正项 `D_i`，第三次形成 `dS`
并累加梯度。`dQ` 只属于当前 query row，可在 CTA 内累加；不同 query 都可能更新同一
key/value row，所以 `dK/dV` 使用 FP32 `atomicAdd`。FP16/BF16 输入也先写入 FP32
`dQ/dK/dV` workspace，kernel 完成后再 cast 回输入 dtype，避免把长归约直接压进低精度原子操作。

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
