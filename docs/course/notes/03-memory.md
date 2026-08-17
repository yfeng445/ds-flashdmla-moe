# 第 3 周：内存层次、Tiling 与 Benchmark

- 原始页面：[Week 3: Memory](https://distinct-capricorn-c04.notion.site/Week-3-Memory-26388315b6b48016a19bc6f451f9e1eb)
- 整理日期：2026-08-17
- 英文版：[Week 3: Memory](../source/notes/03-memory.en.md)

## 本周目标

第三周从“能运行的 GEMM”转向“能解释性能的 GEMM”。核心是计算 FLOPs、正确计时、理解
HBM/shared memory/register 的容量与作用，并通过 tiling 增加数据复用。

## GEMM 的 FLOPs

令 `A` 的形状为 `(M, K)`，`B` 为 `(K, N)`，输出 `C` 为 `(M, N)`。每个输出元素需要
`K` 次乘法和 `K-1` 次加法，因此精确操作数为：

\[
(2K-1)MN,
\]

在大矩阵分析中通常近似为：

\[
2MNK.
\]

只有同时给出问题规模、平均 kernel 时间和 FLOPs 计数，TFLOP/s 才有意义。

## 可靠的 CUDA Benchmark 流程

1. 确保 GPU 上没有干扰被测结果的并发负载；
2. 为被测工作创建明确的 CUDA stream；
3. 先运行约 3-10 次 warmup；
4. 在同一 stream 上记录 start event；
5. 连续执行约 50 次 kernel；
6. 记录 stop event，并等待 stop event 或 stream 完成；
7. 检查 launch 与执行期错误；
8. 用 event elapsed time 除以迭代次数得到平均时间；
9. 用 FLOPs 除以平均秒数，换算成 TFLOP/s；
10. 同时报告 shape、dtype、GPU 和编译配置。

`cudaStreamSynchronize(stream)` 等待该 stream 之前排队的全部操作，并向 host 暴露相关执行
错误；`cudaEventSynchronize(event)` 等待事件触发，而事件只会在同一 stream 中排在它之前的
操作完成后触发。二者的等待对象不同，但都可用于建立可信的测量边界。

## GPU 内存层次

| 层次 | 可见范围 | 特征 | 典型用途 |
| --- | --- | --- | --- |
| HBM / global memory | 全 GPU | 容量最大、延迟最高 | 输入、输出和大张量 |
| shared memory / SMEM | 一个 thread block 所在 SM | 容量小、显式管理、低延迟 | tile 复用与线程协作 |
| registers / RMEM | 单线程 | 最快、数量有限 | 累加器与局部标量 |

课程材料用 A100 的数量级举例：HBM 可达数十 GB，每个 SM 的 shared memory 和 register
file 则只有数百 KB。具体容量随架构和配置变化，优化前必须查询目标设备，而不能把示例数字
写死在算法假设中。

## 为什么朴素 GEMM 浪费带宽

朴素 kernel 让每个线程独立从 HBM 读取一整行 `A` 和一整列 `B`。相邻输出元素会重复读取
大量相同数据。Tiling 将 `A`、`B` 的子块搬入 shared memory，让 block 内多个线程复用同一
批数据，从而提高 arithmetic intensity，减少每次 FLOP 对应的 HBM 流量。

性能分析必须同时观察：

- 全局内存访问是否合并；
- tile 是否在 block 内被充分复用；
- shared memory 容量是否限制 tile；
- register 使用是否降低 occupancy；
- 边界条件是否造成分支或越界。

## 原练习要求

阅读并在 H100 上复现
[*How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog*](https://siboehm.com/articles/22/CUDA-MMM)
中的代码运行与计算。重点不是照抄最终 kernel，而是逐步记录每次变更怎样改变数据搬运、并行
映射和实测性能。

## 延伸阅读

- [CUDA Matmul Optimization Worklog](https://siboehm.com/articles/22/CUDA-MMM)；
- [第 0 章：GPU Tiling](../../chapters/00-gpu-execution-and-tiling.md)；
- [第 7 章：Benchmark 与 Roofline](../../chapters/07-benchmarking-and-roofline.md)。
