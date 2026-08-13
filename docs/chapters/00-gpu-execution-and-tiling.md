# 第零章：从并行线程到 GPU Tiling

Attention、MLA 和 MoE kernel 看起来各不相同，但它们反复面对同一组底层问题：如何划分
工作、如何让相邻线程访问相邻数据、如何复用片上数据，以及如何在同步与并行度之间取舍。
本章建立后续章节共用的执行与内存模型，并用 GEMM 作为最小实验对象。

## 0.1 并行不等于自动加速

设串行程序中可并行部分占比为 `p`，使用 `N` 个执行单元时，Amdahl 上界为：

```math
S(N)\leq\frac{1}{(1-p)+p/N}.
```

线程数量增加以后，串行部分、调度、同步和内存竞争不会消失。若多个线程在没有同步的情况
下读写同一位置，还会产生 data race：结果不仅慢，而且语义不确定。因此优化的第一步不是
“多开线程”，而是先写出每个输出元素由谁负责、哪些数据被共享、何处需要 happens-before。

CUDA 把大量轻量线程组织成层级：

```text
grid
└── thread block
    └── warp
        └── thread
```

grid 中的 blocks 可按任意顺序调度到 SM；不同 block 一般不能依赖全局执行顺序。同一 block
可以通过 shared memory 交换数据，并用 block-wide barrier 协调。硬件以 warp 为基本发射
单位；一个 warp 内出现不同控制流时，分支路径会被分别执行，因此“有很多 CUDA threads”
和“所有 lane 同时做有效工作”是两回事。

## 0.2 从逻辑坐标到线性地址

对 row-major 的 `M×N` 矩阵，元素 `(row,col)` 的线性 offset 是：

```math
offset=row\cdot N+col.
```

二维 kernel 常用：

```text
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```

若相邻 lane 改变 `col`，它们访问相邻地址，global-memory transaction 更容易合并。若相邻
lane 改变 `row`，访问间隔变成 `N`，即使算法 FLOPs 完全相同，也可能需要更多 memory
transactions。张量 shape 只描述逻辑维度；stride、alignment 和 lane-to-element mapping
共同决定物理访问质量。

所有 global load/store 都必须先验证逻辑坐标。`M`、`N` 不是 block size 的整数倍时，最后
一个 block 仍会启动完整线程数；越界线程应该被 predicate 掉，而不是要求调用者悄悄 pad
输入。tail shape 是 kernel contract 的一部分。

## 0.3 朴素 GEMM 的工作划分

一般 GEMM 写作：

```math
D=\alpha AB+\beta C,
```

其中 `A∈R^{M×K}`、`B∈R^{K×N}`，输出为 `M×N`。最直接的 CUDA 映射让一个 thread 计算
一个 `D[m,n]`：

```math
D_{mn}=\alpha\sum_{k=0}^{K-1}A_{mk}B_{kn}+\beta C_{mn}.
```

矩阵乘的常用工作量约定是：

```math
F_{GEMM}=2MNK.
```

它把 multiply 与 add 各算一个 FLOP，并忽略 epilogue。这个 kernel 有足够的输出并行度，
但相邻输出会重复从 global memory 读取相同的 A 行或 B 列。按“每个输出 thread 独立加载”
建模，输入流量为 `2MNK` 个元素；这不是硬件实测值，因为 cache 可能截获部分重复读取。

## 0.4 Tiling 为什么有效

令一个 block 负责 `T_M×T_N` 输出 tile，并沿 reduction 维按 `T_K` 前进。每个阶段协作加载：

```text
A tile: [T_M, T_K]
B tile: [T_K, T_N]
```

到 shared memory，再让 block 内线程重复使用。忽略 cache 和 padding，并准确计入边界 tile
时，模型中的输入元素加载数为：

```math
N_{tile,input}
=\left\lceil\frac{N}{T_N}\right\rceil MK
+\left\lceil\frac{M}{T_M}\right\rceil KN.
```

相比朴素线程模型，输入复用比为：

```math
R_{reuse}=\frac{2MNK}{N_{tile,input}}.
```

若 `M,N` 都远大于 tile 且 `T_M=T_N=T`，该比值接近 `T`。这就是 shared-memory tiling
提高算术强度的核心，不是减少 GEMM 本身的 FLOPs。`tiled_gemm_reference` 用 PyTorch 明确
写出 M/N/K 三层 tile 循环，并支持任意 tail shape；它解释算法语义，不模拟 lane、bank
conflict 或真实 CUDA 时序，也不以击败 vendor GEMM 为目标。

仓库还提供第一版 native CUDA teaching kernel：一个 `16×16` thread block 计算一个
`16×16` 输出 tile，沿 K 维每次协作加载两个 `16×16` shared-memory tiles。边界 load 写
零，边界 store 用 predicate，因此 M/N/K 都无需整除 16。这个版本只接受 contiguous FP32
二维矩阵；`backend="cuda"` 对其他输入直接报错，`backend="auto"` 则回退到可微 reference。
它仍使用普通 CUDA cores，不应与 cuBLAS/Tensor Core GEMM 当作同级性能实现。

## 0.5 Barrier 的两个职责

典型的单缓冲 CUDA tile 循环包含两次 block barrier：

1. 生产者线程把本阶段 A/B tile 写入 shared memory 后，等待所有写入完成；
2. 所有线程消费完当前 tile 后，再允许下一阶段覆盖这块 shared memory。

少第一次 barrier 会读到尚未写好的数据，少第二次会让快线程覆盖慢线程仍在读取的数据。
barrier 只协调同一 block 中参与它的线程；把 barrier 放在只有部分线程进入的分支里，可能
导致死锁。边界 tile 常采用“所有线程都到达 barrier，越界 load 写零”的结构。

## 0.6 Tile 越大不一定越快

一个 stage 的理想 shared-memory 容量为：

```math
M_{smem,stage}=e(T_M T_K+T_K T_N),
```

其中 `e` 是元素字节数。若使用 double/triple buffering，还要乘以 stage 数；为避免 bank
conflict 添加的 padding 也会增加实际容量。更大的 tile 提高数据复用，却同时增加：

- 每 block 的 shared memory；
- 每 thread 或 warp 的 accumulator registers；
- barrier 前后等待不均衡的风险；
- tail tile 中无效 lane 的比例。

这些资源会限制一个 SM 同时驻留的 blocks/warps。occupancy 是隐藏延迟的手段，不是独立
性能目标：一个高复用、低 occupancy kernel 可能胜过高 occupancy 但反复访问 HBM 的
kernel。最终应同时检查吞吐、memory traffic、active warps、stall reasons 和数值误差。

## 0.7 从同步 tile 到异步流水线

同步 tile 循环的阶段时间近似为：

```math
t_{stage}\approx t_{load}+t_{compute}.
```

若硬件与数据依赖允许把第 `i+1` 阶段搬运和第 `i` 阶段计算重叠，稳态可接近：

```math
t_{stage}\approx\max(t_{load},t_{compute}).
```

这要求至少两组 buffer、清晰的 producer/consumer 所有权，以及“数据可读”和“buffer 可
覆盖”两类同步。Hopper 的 Tensor Memory Accelerator (TMA) 可异步搬运 global/shared
memory tiles；WGMMA 让 warpgroup 发起异步 Tensor Core 矩阵乘。warp specialization 则
让部分 warps 专门搬运，其他 warps 专门计算。

异步不是删除等待，而是把等待推迟到真正消费结果之前，并用别的独立工作填充空隙。如果
生产者快于消费者，有限 buffer 最终仍会 back-pressure；如果寄存器或 shared-memory
占用让驻留并行度过低，复杂流水线也可能更慢。FlashAttention-3 的 ping-pong scheduling
正是把这种 producer/consumer 思路应用到 GEMM、Softmax 与下一 tile 的重叠。

## 0.8 低精度输入与高精度累加

FP16、BF16、TF32 和 FP8 降低输入带宽并提高 Tensor Core 峰值，但它们的指数范围和尾数
位数不同。高性能 GEMM 常使用低精度乘数、FP32 accumulator：这能显著降低长 dot product
中的舍入累积，却不能恢复输入量化时已经丢失的信息。

因此 dtype 是算子语义和 benchmark config 的一部分。验证时至少分别报告：

- 输入与输出 dtype；
- accumulator dtype；
- reference dtype；
- `rtol/atol` 与最大误差；
- 是否使用 scaling、block quantization 或其他数值补偿。

仓库的 GEMM reference 对 FP16/BF16/FP32 输入使用 FP32 累加，对 FP64 输入使用 FP64，
并保持输出 dtype 与输入一致。

## 0.9 可复核实验

运行一个包含三个 tail 的小实验：

```bash
python benchmarks/gemm.py \
  --device cpu --dtype float64 --implementation tiled \
  --m 37 --n 29 --k 23 \
  --tile-m 16 --tile-n 8 --tile-k 7 \
  --warmup 1 --iterations 5 \
  --output benchmark-results/gemm-tail.json
```

在构建 native extension 的 CUDA 主机上，可显式测量教学 kernel：

```bash
python benchmarks/gemm.py \
  --device cuda --dtype float32 --implementation cuda \
  --m 127 --n 95 --k 63 \
  --tile-m 16 --tile-n 16 --tile-k 16 \
  --warmup 10 --iterations 100 \
  --output benchmark-results/gemm-cuda.json
```

`implementation=cuda` 要求 tile 参数精确为 `16×16×16`，防止报告配置与实际 kernel 静默
不一致。native forward 使用当前 PyTorch CUDA stream；一阶 backward 由 dispatcher 注册的
解析 GEMM 公式计算，二阶梯度使用 PyTorch 运算继续构图。

报告同时给出：

- `2MNK` 矩阵 FLOPs；
- compulsory tensor-I/O lower bound；
- 每输出 thread 独立加载的教学流量模型；
- 每输出 tile 协作加载的教学流量模型；
- tile counts、最后一个 tile 的尺寸和单 stage shared-memory 模型；
- 与独立 GEMM reference 的误差、环境信息和 raw latency samples。

这些 byte counts 都不是 profiler 测得的 DRAM traffic。Python tiled reference 的 latency
主要反映解释器循环和许多小 `matmul` 的调度成本，也不能外推到 CUDA tile kernel。它的
价值是让流量公式、边界语义和验证协议先变成可执行合同；真正优化时再逐项替换实现。
