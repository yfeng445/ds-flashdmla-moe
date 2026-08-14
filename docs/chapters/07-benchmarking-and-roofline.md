# 第七章：Benchmark、算术强度与 Roofline

kernel 输出正确以后，下一步不是直接写“比 baseline 快多少”，而是先定义可复现的测量
协议。没有 shape、dtype、backend、硬件、软件版本和原始样本的单个延迟数字，无法被复核，
也无法解释性能变化来自算法、编译参数还是运行环境。

## 7.1 异步执行与计时边界

CUDA kernel launch 对 CPU 是异步的。下面这种写法主要测到 launch 开销：

```text
t0 = cpu_clock()
kernel()
t1 = cpu_clock()
```

GPU 延迟应使用同一 CUDA stream 上记录的 start/end event，并在读取结果前同步。为了保留
抖动信息，benchmark 记录每一次迭代的 event 时间，而不是把很多次执行包进一个总区间后
只留下平均值。

预热用于排除 lazy initialization、allocator 建立和首次 dispatch 等冷启动成本。预热
次数和正式迭代次数本身也是实验条件，必须进入报告。

CPU reference 使用单调的高精度 wall clock。它适合验证 benchmark 基础设施和比较算法
趋势，但不能拿来推断 CUDA kernel 的速度。

## 7.2 延迟分布

报告至少保留：

- `min`：观察理想运行时，但容易受偶然条件影响；
- `median`：比均值更不容易被少数长尾样本拖动；
- `mean`：用于总吞吐估算，但必须和分布一起看；
- `p90/p99/max`：观察调度、温控或系统噪声造成的尾延迟；
- raw samples：允许以后改变统计方法而无需重跑实验。

不要只挑“最好的一次”，也不要在报告里删除异常值却不说明规则。如果需要剔除系统性
干扰，应保留原始数据并明确写出过滤条件。

## 7.3 Attention 的矩阵 FLOPs

对于 `[B,H,S_q,D]` 的 Q、`[B,H,S_k,D]` 的 K 和 value dimension `D_v`，只计算两个
矩阵乘的常用 FLOP 估计是：

```math
F_{QK^T}=2BHS_qS_kD,
```

```math
F_{PV}=2BHS_qS_kD_v,
```

```math
F_{matrix}=2BHS_qS_k(D+D_v).
```

因子 2 把一次乘法和一次加法各算一个 FLOP。这个估计没有包含 scale、mask、max、exp、
sum 和 normalization，因此报告把它命名为 `matrix_flops`，而不是宣称它代表全部指令。

用 median latency `t` 可以计算：

```math
\text{matrix TFLOP/s}=F_{matrix}/t/10^{12}.
```

这个值适合在相同计数约定下比较实现，不等于硬件 profiler 给出的所有浮点指令吞吐。

## 7.4 最低数据流量与算术强度

若每个输入只从 HBM 读取一次、输出只写一次，Q/K/V/O 的强制张量流量下界为：

```math
M_{min}=eBH(S_qD+S_kD+S_kD_v+S_qD_v),
```

其中 `e` 是每个元素的字节数。于是理想化算术强度为：

```math
I_{ideal}=F_{matrix}/M_{min}.
```

这是下界而非实测 DRAM bytes。逐 query 正确性 kernel 会重复读取 K/V，实际流量远高于
`M_min`；二维 tiled kernel 则通过 shared memory 重用让实际流量靠近下界。报告中的
`compulsory_bandwidth_gb_s` 只表示“若达到下界，当前延迟对应多大带宽”，不能标成实测
显存带宽。真实 bytes 应用 Nsight Compute 的 DRAM 指标获取。

Roofline 上限可写为：

```math
P \le \min(P_{peak}, I\times BW_{peak}).
```

如果实测算术强度低且吞吐接近带宽 roof，优先减少 HBM 流量；若强度高却远离计算 roof，
再检查 Tensor Core 利用率、occupancy、同步和指令混合。

## 7.5 结构化报告

仓库 benchmark 输出 JSON，包含：

- 完整 Attention config 与随机种子；
- Python、PyTorch、package、CUDA 和设备 metadata；
- source revision 与 worktree 是否包含未提交改动；
- 输出 shape/dtype/device；
- FLOPs 与 compulsory bytes 的计数约定；
- latency 统计和所有 raw samples；
- 计时前 materialized reference 校验产生的误差与容差；
- 由 median 推导的吞吐。

示例：

```bash
python benchmarks/attention.py \
  --device cuda --backend cuda --dtype float32 \
  --batch 2 --heads 16 --query-length 1024 --key-length 1024 \
  --head-dim 64 --value-dim 64 --causal \
  --warmup 10 --iterations 100 \
  --output benchmark-results/a100-attention.json
```

性能实验必须显式使用 `backend=cuda`。若使用 `auto`，不支持的输入可能正确地回退到
reference，却让测量对象发生变化。

## 7.6 公平比较

比较自定义 kernel、PyTorch SDPA 和其他库时，应固定：

1. 同一 Q/K/V 数值、shape、dtype、causal/mask 语义；
2. 相同的梯度模式——inference forward 不能和 forward+backward 混比；
3. 相同 stream 和同步边界；
4. 相同 GPU、功耗/时钟策略和软件栈；
5. 足够预热与相同迭代次数；
6. 在计时前完成正确性验证。

若一个实现不支持某种输入，应报告“不支持”，不能悄悄改 dtype、pad shape 或删除 mask 后
继续把数字放在同一张表里。

## 7.7 代表性单 GPU shape matrix

单个规整 shape 只能证明该点能够运行，不能说明 tail、decode 或负载倾斜时的行为。仓库的
`benchmarks/matrix.py` 把五类 native 算子与各自的 PyTorch baseline 组织为固定矩阵：

| Family | Case 数 | 覆盖重点 | Baseline |
| --- | ---: | --- | --- |
| GEMM | 4 | 规整、M/N/K 尾块、单行 decode、alpha/beta epilogue | PyTorch/cuBLAS |
| Attention | 4 | prefill、decode、非 2 次幂序列、不同 QK/V 宽度 | PyTorch SDPA |
| MLA | 5 | 完整 prefill、static-cache decode、direct/LoRA query、尾块 rank | absorbed PyTorch |
| Experts | 4 | FP32/FP16、空 expert、行数与维度尾块、负载倾斜 | padded PyTorch |
| Router | 3 | 规整、尾块、hot-expert skew | PyTorch reference |

运行完整矩阵：

```bash
python benchmarks/matrix.py \
  --device cuda --profile representative \
  --warmup 5 --iterations 20 --seed 20260814 \
  --output benchmark-results/operator-matrix-representative.json
```

每一对配置只能修改 backend selector；seed、shape、dtype、预热、迭代数和验证策略必须完全
相同。native 与 baseline 分别先对 operator reference 做数值验证，再进入比较。相邻 case
交替先运行 native 或 baseline，以减小固定执行顺序造成的偏差；单个 case 失败会被写入报告，
其余 case 继续执行，CLI 最终以非零状态退出。`--list-cases` 可在没有 GPU 时检查 manifest，
`--family` 与 `--case` 用于缩小复现实验范围。

不同 family 使用的 baseline 并不相同，工作量与算子边界也不同。因此每个
`native_median / baseline_median` 只能解释对应 case。报告中的跨 case minimum、median、
geometric mean 和 maximum 是未加权描述统计，不是总体 speedup；不应把 GEMM/cuBLAS、
Attention/SDPA 和 MoE/reference 的比值混成一个性能结论。

## 7.8 分布式算子的计时边界

多 rank 算子不能只测 rank 0。对第 `n` 次迭代，先让所有 rank 在相同边界开始，再取各
rank 完成时间的最大值：

```math
t_n=\max_{r=0}^{P-1}t_{n,r}.
```

这个定义衡量一次同步训练 step 何时真正可以继续。平均 rank latency 会隐藏 straggler，
而只在计时后做 barrier 会把较快 rank 等待较慢 rank 的时间错误归入下一次迭代。

为了既保留 step 指标又能诊断不均衡，应先记录矩阵 `T[n,r]`，再派生：

```math
t_n^{max}=\max_r T[n,r],\qquad
t_n^{mean}=\frac{1}{P}\sum_r T[n,r].
```

`t_max` 用于端到端吞吐，`t_mean` 与逐 rank 分布只用于解释等待和 straggler；不能用 mean
替代同步 step latency。采样值应在各 rank 本地 operation 完成后截取，再执行 gather，
否则报告会把收集统计本身算进被测算子。

Expert Parallel 的通信 payload 取决于实际 routing counts。令 `C[i,j]` 是 rank `i` 发往
rank `j` 的行数，则真正跨 rank 的 route 行数为：

```math
R_{cross}=\sum_{i\ne j}C[i,j].
```

若 model dimension 为 `D`、元素宽度为 `e` 字节，forward 的 activation payload 是：

```math
M_{EP,fwd}=2R_{cross}De,
```

因子 2 对应 dispatch 和 restore。训练时两次 collective 的 backward 又产生相同规模的
逆向 activation 通信，所以 forward+backward 的这部分 payload 为 `2M_EP,fwd`。这个
公式没有包含 expert-id、counts、collective 协议和底层链路封包，报告中必须保持这个
命名边界。

对总 token 数 `T`、routed expert 数 `E`、每 token 选择数 `K`、model dimension `D`、
单 expert hidden dimension `D_h`，MoE 前向的矩阵 FLOPs 可分开记为：

```math
F_{router}=2TDE,
```

```math
F_{routed}=6TKDD_h,
```

```math
F_{shared}=6TD(n_sD_h).
```

第二式的因子 6 来自 SwiGLU 的三个矩阵乘，每个矩阵乘把 multiply 与 add 各算一次；若
执行了 capacity drop，应把 `TK` 换成实际执行的 routed rows。第三式对应 `n_s` 个 shared
experts 拼成一个中间宽度为 `n_sD_h` 的 SwiGLU。总量为三者之和，但仍不包含 sigmoid、
Top-K、SiLU、逐元素乘、排序、通信和 combine。padded expert backend 还应另报包含 padding
槽位的 executed FLOPs，不能用 ideal routed FLOPs 掩盖无效计算。

独立 expert benchmark 应直接保存 `n_e` 原始向量，而不是只保存总行数。对 active-row
kernel，executed row 数为 `sum_e n_e`；对统一 padding baseline，则为
`E max_e n_e`。两者的 FLOPs 比值等于 padding 利用率的倒数。若 benchmark 同时测 backward，
除非完整计入所有梯度矩阵乘，否则吞吐字段应写成 forward-equivalent，而不能把 forward
FLOPs 除以 forward+backward 时间后简称“训练 TFLOP/s”。

grouped tiled backend 还要区分两种“无效行”。统一 padding 的无效行为
`E max_e(n_e)-sum_e(n_e)`；独立 16-row tiles 的 inactive lanes 为：

```math
16\sum_e\left\lceil\frac{n_e}{16}\right\rceil-\sum_e n_e.
```

后者只发生在每个 expert 的最后一个 tile，不执行额外的完整 capacity rows，但 inactive
threads 仍占 launch 与指令槽位。报告两者才能判断收益来自消除跨 expert padding，还是 tile
本身已经有较高 lane utilization。

同一个 tile count 也不能说明使用了哪类算术。expert 报告另存 `native_numeric_model`：
FP32 对应 shared-memory CUDA cores；FP16 对应 WMMA Tensor Cores、FP32 accumulator 和
FP16 materialized hidden。只有前向 engine 相同、dtype contract 相同，latency 才适合横向
比较。WMMA 的误差阈值应相对 FP32 放宽并随报告保存，但“在阈值内”不等于吞吐已接近
Tensor Core 峰值；后者需要 profiler counter 证明。

独立 router benchmark 同样不能只留下一个延迟。投影矩阵的常用计数为：

```math
F_{router}=2TDE,
```

但 sigmoid、group score、Top-K、gather 和归一化不在这个 FLOP 数里。报告应另存
`T*G` 个 group-score candidates、保留组中的 expert candidates、`T*K` 条最终 routes，
以及完整 per-expert load 向量。这样可以区分“投影矩阵更大”和“选择或负载更难”两类变化。
CUDA 验证还必须要求 indices 与稳定 tie-break reference 完全一致；只比较归一化 weights
可能让选错但分数相同的 expert 漏过测试。

## 7.9 阶段分解不是端到端求和

分布式流水线可以为每个阶段记录独立 latency，但需区分两种 max：

```math
t_{e2e}=\max_r\sum_s t_{r,s},
```

```math
t_{stage,s}=\max_r t_{r,s}.
```

一般情况下：

```math
t_{e2e}\ne\sum_s t_{stage,s},
```

因为每个 stage 的最慢 rank 可能不同；通信与计算发生重叠时，阶段边界本身也会改变时间
线。端到端 pass 用于报告用户可见 latency，插入同步的 profiling pass 用于归因。二者
必须分开运行、分别保留 raw samples。

## 7.10 负载分布与流水线模型也要保留口径

EP 报告不能用 `total_routes / world_size` 代替每 rank 的实际负载。至少分别保存 send、
receive、cross-rank send、cross-rank receive 和 per-expert counts，再从原始整数计算
peak-to-mean、CV 与零负载数。这样即使以后更换统计指标，仍可由原始分布重算。

capacity factor 报告应说明它是实际 dispatch 行为还是反事实模型。只把
`C=ceil(f R/E)` 写进报告而仍执行全部 routes 时，字段必须标记 `analytical_only`；否则
读者会把估算的 dropped rows 误认为模型输出已经发生变化。

同理，对每次迭代应先在同一 rank 内计算 `max(t_comm,r,t_compute,r)`，再对 ranks 取最大；
先对每阶段取 rank max 会丢失阶段间相关性。这个值也只有在通信与计算资源可以并行、chunk
依赖满足且 steady state 足够长时才是乐观下界。真实性能结论仍需真实流水实现、端到端
rank max 样本，以及 profiler 时间线共同支持。

真实 chunked 配置不能沿用同步分段的 stage 名称：它只记录 route/count、合并的
`pipelined_core`、combine 和可选 backward/shared stages。要计算实测流水收益，必须用完全
相同的 seed、shape、dtype、backend 和迭代数分别运行 `pipeline_chunks=1` 与 `>1`，再比较
两组端到端 rank-max raw samples；不能把 chunked core latency 与另一轮串行 stage max 直接
相减。还应同时报告 chunk 数与 `chunked_aggregate` tile tasks，因为 fill/drain、每 chunk
固定开销和每 expert 重复出现的 row-tail lanes 共同决定最优粒度。
