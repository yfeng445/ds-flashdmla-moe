# 第 6 周：CuTe GEMM

- 原始页面：[Week 6: CuTe GEMM](https://distinct-capricorn-c04.notion.site/Week-6-CuTe-GEMM-30a88315b6b4804b8b8fcfcbdc36554f)
- 整理日期：2026-08-17
- 英文版：[Week 6: CuTe GEMM](../source/notes/06-cute-gemm.en.md)

## 从 Layout Algebra 到 GEMM

CuTe 用 layout algebra 统一表达问题 shape、global-memory tensor、CTA tile、shared-memory
tile 和线程分工。它能组合算法、dtype、layout 与性能选项，从同一组抽象生成大量 kernel
变体。

逻辑除法 `A div B` 的直觉是用 layout `B` 对 `A` 分块，让新坐标写成
`(tile 内坐标, tile 坐标)`。这正是 GEMM 中“一个 CTA 负责一个 M/N tile，并沿 K tile
迭代”的索引需求。

## Kernel Launch 前的四类对象

CuTe GEMM 示例在 launch 前定义：

1. `A/B/C` 的 GMEM tensor view；
2. CTA tiler；
3. `A/B/C` 的 SMEM layout；
4. 把 tile 元素分给线程的 thread layout。

### GMEM Tensor View

问题 shape `(M,N,K)` 可选择出 `(M,K)`、`(N,K)` 与 `(M,N)`。pointer、shape 和 stride
组合形成 `mA,mB,mC`，因此同一 kernel 可以处理转置或不同 leading dimension，而不用修改
核心计算循环。

### CTA Tiler

例如 `(128,128,8)` 表示一个 CTA 处理 `M x N x K` 问题中的 `128 x 128 x 8` 子块。
grid 的前两维覆盖 `ceil(M/128)` 与 `ceil(N/128)` 个输出 tile，K 维则在 main loop 中遍历。

### SMEM Layout

SMEM layout 描述 `A` 的 `128x8` tile、`B` 的 `128x8` tile 和输出/累加相关 tile 怎样存储
在 shared memory。它需要同时服务 coalesced copy、bank-conflict 控制和 MMA 读取模式。

### Thread Layout

示例使用 256-thread CTA。`(32,8)` 可以把第 0 维解释为 warp 内 lane、第 1 维解释为
8 个 warp。一个好的 thread layout 应满足：

- 相邻 lane 尽量访问连续 global address；
- 每个 warp 的工作范围可预测；
- ownership 与 MMA load/accumulator pattern 对齐。

## `local_tile`：选择 CTA 的数据

`local_tile` 把完整 tensor 除成 tile grid，再由 CTA coordinate 选择局部 view。坐标中的整数
固定某一 tile，而 `_` 保留该维供后续遍历。例如 M、N 固定为 `blockIdx.x/y`，K 使用 `_`，
结果便保留一个尾部 K-tile 维度供 main loop 迭代。

这比手写 pointer arithmetic 更清楚地表达：

- 当前 CTA 拥有哪个输出 tile；
- A 只固定 M tile、B 只固定 N tile；
- K 是遍历维，而不是 grid 中另一个独立输出 ownership 维。

## `local_partition`：选择线程的数据

`local_partition(tensor, thread_layout, threadIdx.x)` 进一步把一个 CTA tile 划分给线程。
每个线程获得不重叠的 subtensor，并在整个 tile 上重复其 thread-tile pattern。于是所有线程
可以并行执行从 GMEM 到 SMEM 的 copy，而不是由单线程串行搬运。

## Main Loop 与异步 Copy

每个 K tile 的基本顺序是：

```text
copy GMEM -> SMEM
commit async-copy group
wait for the required groups
block-wide barrier
GEMM SMEM x SMEM -> registers
block-wide barrier before reusing SMEM
```

CuTe 在适用时可把 GMEM-to-SMEM copy 降低为 `cp.async`。`cp_async_fence()` 提交当前
线程发出的 copy group，`cp_async_wait<0>()` 等待该线程先前提交的全部 group。此时仍需要
`__syncthreads()`：前者只保证当前线程的异步 copy，后者保证 block 内所有线程的数据都已
可见。计算后的第二次 barrier 防止某些线程提前覆盖仍被其他线程读取的 SMEM。

## Predication 与架构演进

实际 tile 位于矩阵边界时可能超出合法 shape，需要 predicate 避免越界；TMA tensor 则进一步
抽象 Hopper 及之后架构的大块异步搬运。CuTe 的静态 layout/tiler 使同一 GEMM 结构能够针对
不同 Tensor Core 指令、dtype 和 copy pipeline 特化。

## 延伸阅读

- [CuTe GEMM Tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html)；
- [CuTe Predication](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0y_predication.html)；
- [CuTe TMA Tensors](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html)；
- [CUTLASS `sgemm_1.cu`](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_1.cu)；
- [*FlashAttention-3*](https://arxiv.org/pdf/2407.08608)；
- [NVIDIA Tensor Core Evolution](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)；
- [GTC 2025 CuTe Session](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72720/)。
