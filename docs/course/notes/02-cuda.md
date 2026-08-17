# 第 2 周：CUDA 与朴素 GEMM

- 原始页面：[Week 2: CUDA](https://distinct-capricorn-c04.notion.site/Week-2-CUDA-26388315b6b480d480aec7e22cde5776)
- 整理日期：2026-08-17
- 英文版：[Week 2: CUDA](../source/notes/02-cuda.en.md)

## 本周目标

第二周把 CPU 上的矩阵乘法迁移到 CUDA，建立 kernel launch、grid/block/thread、host/device
内存和错误检查的基本模型。练习只要求全局内存上的朴素 GEMM；tiling 和 shared memory 留到
下一周。

## CUDA 执行模型

一个 `__global__` kernel 由 host 发起：

```cpp
kernel<<<blocks_per_grid, threads_per_block>>>(...);
```

- `threadIdx`：线程在 block 内的坐标；
- `blockIdx`：block 在 grid 内的坐标；
- `blockDim`：每个 block 的线程布局；
- 二维输出常使用二维 block，让一个线程映射到一个输出元素；
- kernel launch 是异步的，调试或读取结果前需要在正确边界同步。

常见的最小错误检查链包括 launch 后调用 `cudaGetLastError` 或
`cudaPeekAtLastError`，并在需要观察执行期错误时进行同步。错误字符串可通过
`cudaGetErrorString` 获得。

## Host 与 Device 内存

- host memory 是 CPU 地址空间；
- device/global/HBM 是 GPU 可访问的全局内存；
- `cudaMalloc` 和 `cudaFree` 管理 device allocation；
- `cudaMemcpy` 按 Host-to-Device、Device-to-Host 或 Device-to-Device 方向搬运数据。

二维矩阵通常在一段连续的一维内存中按 row-major 保存。宽度为 `W` 时，坐标 `(row, col)`
对应 `row * W + col`。例如宽度为 4 的矩阵中 `(2, 1)` 位于线性索引 9。

## 从矩阵乘法到 GEMM

普通矩阵乘法为：

\[
C = AB,
\]

其中 `A` 为 `m x k`、`B` 为 `k x n`、`C` 为 `m x n`。GEMM 进一步计算：

\[
C \leftarrow \alpha\,op(A)op(B) + \beta C,
\]

`op(A)` 和 `op(B)` 可以分别是原矩阵或转置。完整接口应覆盖：

- `A B`；
- `A^T B`；
- `A B^T`；
- `A^T B^T`。

实现时不要真的生成转置副本，而应根据 transpose flag 改变读取索引。每个线程计算一个
`C[row, col]`，沿 `k` 维累加点积，最后一次性执行 `alpha * acc + beta * C[...]`。

## 硬件并行规模

课程用 CPU 与 GPU 的数量级对比说明线程映射的重要性：高端 CPU 可能有约百个 core 和数百
硬件线程，而 B200 级 GPU 具有上百个 SM，每个 SM 可驻留大量线程，总并发线程数量可达到
数十万量级。GPU 的优势来自大量相似工作，而不是单个线程更强。

## 调试原则

- 如果一个问题长时间无法定位，优先拆分组件并为每个组件编写独立测试；
- CUDA 中 `printf` 可用但代价高、输出顺序混乱，不能代替清晰的数据流；
- 用 shape、索引和 ownership 让代码尽量“显然正确”；
- assert、launch error check 和同步点通常比无结构日志更有效；
- 同一套测试必须覆盖普通尺寸和非 block 整数倍尺寸。

## 原练习要求

先实现只读写 global memory 的朴素 GEMM，不调用 cuBLAS 或 cuDNN；验证基本版本后，再加入
`A/B` 可选转置和 `C` 原地更新。此阶段目标是语义正确与接口完整，不是性能优化。

## 延伸阅读

- *Programming Massively Parallel Processors*：第 5 章 Memory Architecture and Data Locality；
- [CUDA C++ Programming Guide：Introduction](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction)；
- [CUDA C++ Programming Guide：Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)。

