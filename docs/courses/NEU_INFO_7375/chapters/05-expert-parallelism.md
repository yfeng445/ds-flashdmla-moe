# 第五章：Expert Parallelism

## 5.1 为什么需要跨 GPU 路由

当全部 experts 无法或不应复制到每张 GPU 时，将 expert 集合切分给多个 rank。每个
rank 先为本地 token 运行全局 router，然后把 token 发送给拥有目标 expert 的 rank。
典型前向路径是：

```text
local tokens
├→ route and pack → all-to-all dispatch → routed expert compute
│                  → all-to-all restore → combine on source rank
└→ replicated shared expert on the source rank
→ add routed and shared outputs
```

这与 data parallel 不同：DP 复制模型并同步参数/梯度；EP 切分 experts 并在一层内部
搬运 activation。

## 5.2 三类身份

一次路由至少涉及：

- `source_rank`：token 原本属于哪个 rank；
- `token_index`：它在 source rank 本地 batch 中的位置；
- `expert_id` 与 `topk_slot`：由哪个 expert 处理、对应哪项 routing weight。

只传 activation 而不传或重构这些元数据，返回时就无法确定 contribution 应累加到哪里。
一个稳妥的 packed row metadata 可包含：

```text
(source_rank, token_index, topk_slot, expert_id)
```

也可以压缩编码，但必须保留双射关系。

## 5.3 变长 All-to-All

每个 source rank 发往不同 destination 的 token 数不同。通信前先交换 count：

```math
c_{ij} = \text{rank }i\text{ 发往 rank }j\text{ 的 packed rows}.
```

rank `j` 的接收计数是矩阵第 `j` 列。得到 counts 后计算 send/recv displacement，再进行
变长数据交换。NCCL 没有单独名为 `all_to_allv` 的高级调用时，可以在一个
`ncclGroupStart/End` 中为每个 peer 配对 `ncclSend/ncclRecv`。

所有 rank 必须以兼容顺序提交成组操作，并保证对应 send/recv 的 count 与 dtype 一致，
否则可能死锁或数据截断。

把所有 counts 写成矩阵 `C`，其中第 `i` 行由 source rank `i` 产生：

```text
                 destination rank
              0      1      ...    P-1
source 0    C[0,0] C[0,1]   ... C[0,P-1]
source 1    C[1,0] C[1,1]   ... C[1,P-1]
   ...
source P-1  C[P-1,0]         ... C[P-1,P-1]
```

rank `i` 的 `send_counts` 是第 `i` 行，rank `j` 的 `recv_counts` 是第 `j` 列。
`all_to_all_single` 可以先交换每个 rank 的 count 向量，再用这些 split sizes 交换 packed
rows。count 以“行”为单位最不易混淆；若底层通信按元素或字节计数，应在边界处显式乘以
`D` 或 `sizeof(dtype)`，不要让三个单位混用。

## 5.4 两次通信为何能原路恢复

source rank 先按 destination rank 连续打包：

```text
[to rank 0 | to rank 1 | ... | to rank P-1]
```

第一次 All-to-All 后，destination rank 收到的缓冲按 source rank 分段：

```text
[from rank 0 | from rank 1 | ... | from rank P-1]
```

通信保持每个 peer segment 的内部顺序。专家输出不改变行数时，恢复通信只需交换两组
splits：

```text
restore_send_counts = dispatch_recv_counts
restore_recv_counts = dispatch_send_counts
```

返回 source rank 的行便与原始 packed rows 一一对齐。因此 `(token_index, topk_slot)`、
route weight 可以留在 source rank，不必随 activation 往返；destination 只需要
`expert_id` 来选择本地 expert。若中间发生 capacity drop、重排后未撤销 permutation，
或每行产生可变数量的结果，这个简化不再成立，必须传递完整 route identity。

## 5.5 接收端不一定是 expert-major

source 端即使在每个 destination 段内按 expert 排序，第一次 All-to-All 的接收结果仍是
source-major 的多个段：

```text
from source 0: [expert 1 | expert 3]
from source 1: [expert 1 | expert 3]
```

仅根据每个 expert 的总 count 构造 offsets，然后直接调用 grouped GEMM 是错误的，因为
相同 expert 的行尚未全局连续。可选做法有三种：

1. 根据收到的 `expert_id` 再做一次稳定 permutation，计算后应用逆 permutation；
2. 为每个 local expert 用 index list 收集输入，再把输出 scatter 回接收顺序；
3. 使用能直接消费 row-to-expert mapping 的 grouped GEMM kernel。

参考实现采用第二种，便于检查；高性能实现通常采用第一种或第三种。

## 5.6 路由权重必须在非线性专家之后

对 route weight `alpha`，MoE contribution 是：

```math
\alpha E(x),
```

不是：

```math
E(\alpha x).
```

SwiGLU expert 是非线性的，两式一般不等。dispatch 应发送原始 activation，expert 输出
返回 source rank 后，combine 再读取本地保存的 route weight。提前乘权重虽然可能减少
一处 combine 乘法，却改变了模型定义。

## 5.7 Host staging baseline

课程中的 MPI 路径通过：

```text
device → host → MPI_Alltoallv → host → device
```

建立了容易调试的语义基线，但它包含同步、PCIe 拷贝和 host allocation，不应被当成
高性能 GPU backend。它仍然有教学价值：

- 验证 counts/displacements；
- 打印每个 rank 的 token 身份；
- 在没有 CUDA-aware MPI 时建立正确闭环；
- 与 NCCL 路径逐元素比较。

## 5.8 可求导通信

训练时，dispatch 的 backward 是一个方向相反、split sizes 互换的 All-to-All；restore
同理。若前向写作：

```text
y = A2A(x, send_counts, recv_counts)
```

则其向量—Jacobian product 为：

```text
dx = A2A(dy, recv_counts, send_counts)
```

所有 rank 必须以相同顺序进入 backward collective。一个 rank 即使本地 token 数为 0，
仍可能拥有其他 rank 选中的 experts，并且必须参与前向和反向通信。不能只用“本地输出
是否需要梯度”决定是否进入 collective；应在 process group 范围统一 autograd 开关，
或让每个 rank 始终构建相同的 collective graph。

Expert Parallel 只负责搬运激活和 expert 权重的局部梯度。replicated router 与 shared
expert 的参数梯度还需要由 data-parallel 通信求和；不要把这一步误认为两次 EP
All-to-All 会自动完成。

shared expert 不参与 routed dispatch：每个 source rank 直接对自己的 token shard 执行同一
组 replicated 权重。它与 routed 分支只在最终加法处相遇，因而具备并行执行机会；但是否
真正重叠取决于 stream、GEMM 资源竞争和依赖调度。当前 correctness reference 按顺序执行，
并把 `shared_expert` 作为独立 stage 测量。benchmark 为验证反向正确性会额外聚合各 rank 的
shared 参数梯度，这次校验通信不包含在被测 step 中。

## 5.9 通信与计算重叠

当前 NCCL 路径可用 `pipeline_chunks>1` 把 packed rows 分 chunk：当第一个 chunk 到达时
开始 expert compute，同时提交后续 chunk 的通信。每个 destination/source peer segment
独立等分，不能直接对整张 rank-major buffer 做 `tensor.chunk()`，否则 chunk 边界会跨 peer，
split sizes 不再对应。PyTorch `async_op=True` 把 NCCL collective 排到独立 CUDA stream；
`Work.wait()` 在当前 compute stream 建立依赖，输出可用后并不要求 CPU 等到 GPU 通信完全
结束。`TORCH_NCCL_BLOCKING_WAIT=1` 会把 `wait()` 改成 host-blocking，适合特定超时调试，
却会改变这里要测的执行语义；流水性能实验应保持该开关关闭。概念时间线是：

```text
NCCL stream: dispatch chunk n+1 ───────────┐
compute stream: wait dispatch n → expert n ├→ restore chunk n
NCCL stream: ──────────────────────────────┘
```

代码保留 async `Work`、输入与输出 storage 直到 wait，并在所有 chunk restore 提交后才等待
它们完成。返回行按原 packed route index 回填，因此 combine 仍读取原本地 route weight 和
token id。反向中每个 chunk 执行 splits 互换的 All-to-All；额外的零值 autograd order token
把 collectives 串成全 rank 一致的逆序，防止 autograd 对独立分支的任意调度造成死锁。
不要用 `cudaDeviceSynchronize()` 代替这些依赖；它会让重叠消失，也会掩盖错误 stream 上的
读写。

设独立 profiling pass 得到 dispatch、expert compute、restore 的阶段时间
`t_d,t_c,t_r`。完全串行的核心区模型是：

```math
t_{serial}=t_d+t_c+t_r.
```

若 dispatch 与 restore 共用同一通信资源，而足够细的 chunk 可以让通信与 expert compute
进入稳定流水，乐观下界是：

```math
t_{lower}=\max(t_d+t_r,t_c).
```

于是最多可隐藏的时间为 `t_serial-t_lower`。这是无限细 chunk 的 steady-state 下界，不含
首尾 fill/drain、event、调度、buffer 和 route 依赖成本，也不代表代码已经实现 overlap。
报告为每次迭代保留所有 rank 的阶段样本，先在同一 rank 内组合通信与计算资源时间，再取
rank maximum；不能先把三个 stage 各自取 max 后相加，因为最慢 rank 可能不同。即使如此，
该模型只用于判断机会大小。`pipeline_chunks=1` 报告该分析下界；`pipeline_chunks>1` 报告
实际执行的合并 `pipelined_core` stage，不再伪造互相可加的 dispatch/compute/restore 时间。
判断加速必须另跑相同配置的 `pipeline_chunks=1` baseline，并比较端到端 rank-max raw
samples；仅看到 async API 或 profiler 上局部重叠不能证明 step 更快。

课程中的 tile-level 调度还提示一个更细的依赖边界：一个 token 只需等待它选择的 experts
完成，不必等待本批全部 experts。不过提前释放 token 要保留 route identity 和完成计数，
并防止 output buffer 在最后一个 contribution 到达前被消费。

## 5.10 两层负载不均衡

只看全局平均 route 数会隐藏两种不同瓶颈：

- expert skew：`n_e` 不均，决定 grouped GEMM 小矩阵分布、padding 与 capacity overflow；
- rank skew：同一 owner rank 上 experts 的总 route 数不均，决定同步 collective 与 step 的
  straggler。

对任一负载向量 `x_1,...,x_m`，报告至少保留完整 counts，并可派生：

```math
\text{peak-to-mean}=\frac{\max_i x_i}{\bar x},
```

```math
CV=\frac{\sqrt{\frac{1}{m}\sum_i(x_i-\bar x)^2}}{\bar x}.
```

peak-to-mean 直接描述最忙执行单元相对平均值的倍率；CV 同时反映整个分布。还应记录零负载
expert/rank 数，因为同一个 ratio 不能区分“全部略有差异”和“部分完全空闲”。当总负载为
0 时，本教材约定 peak-to-mean 为 1、CV 为 0。

rank send rows 主要由本地 token 数与 `K` 决定；rank receive rows 由 expert owner 和路由
选择共同决定。两者必须分别统计。cross-rank send/receive 再排除 self-route，用来分析真正
经过互联的通信压力；它与 owner rank 的总 expert compute rows 也不是同一个量。

## 5.11 零 count 不是特殊退出条件

合法输入可以产生以下情况：

- rank 没有本地 token，但拥有被其他 rank 选中的 experts；
- rank 有本地 token，但没有任何 local expert；
- 某个 peer segment 为 0 行；
- 某个 expert 在当前 batch 中为 0 行；
- 所有 routes 都发往同一个 rank。

这些 rank 仍须参加 counts、dispatch 和 restore 三组 collective。允许分配形状为
`[0, D]` 的 tensor，允许 split size 为 0，但不能让该 rank 提前返回。通信实现应设置
timeout，让不匹配的 collective 以诊断错误结束，而不是无限等待。

## 5.12 分布式测试

EP 测试应包含：

- 1 rank 与单设备 reference 等价；
- 2/4 rank 与全量单设备 reference 等价；
- token 数不能整除 rank 数；
- 某个 rank 没有输入 token；
- 某个 expert 没有 token；
- 所有 token 路由到同一 rank 的极端不均衡；
- 不同 rank 上输入 shape 合法但 token count 不同；
- 连续多次运行没有陈旧 metadata 或 stream race。

数值等价测试应把各 rank 的本地 token shard 拼成一个全局输入，用持有全部 expert
weights 的单设备 reference 计算期望结果，再逐 rank 比较对应 slice。反向至少比较：

- 本地输入梯度；
- 每个 owner rank 上 local expert weights 的梯度；
- 各 rank router/shared expert 梯度求和后的全局梯度。

性能报告除了总 latency，还应拆分 router、pack、dispatch communication、expert GEMM、
restore communication 与 combine。只给端到端数字无法判断下一步应优化哪里。

## 5.13 可复核实验

先用两个 CPU process 检查协议和梯度：

```bash
torchrun \
  --master-addr=127.0.0.1 --master-port=29572 \
  --nproc-per-node=2 \
  benchmarks/expert_parallel.py \
  --backend gloo --route-backend reference --expert-backend padded --dtype float64 \
  --tokens-per-rank 3 --token-skew 1 \
  --model-dim 4 --hidden-dim 5 --shared-experts 1 \
  --experts 4 --topk 1 --hot-expert-bias 100 --capacity-factor 1 \
  --symmetric-cell-capacity 2 \
  --warmup 0 --iterations 1 --backward \
  --output benchmark-results/gloo-ep.json
```

在一台至少有两张 GPU 的机器上，用相同输入生成规则切换到 NCCL：

```bash
torchrun \
  --master-addr=127.0.0.1 --master-port=29573 \
  --nproc-per-node=2 \
  benchmarks/expert_parallel.py \
  --backend nccl --route-backend cuda --expert-backend cuda --dtype float32 \
  --tokens-per-rank 16 --token-skew 3 \
  --model-dim 32 --hidden-dim 64 --shared-experts 1 \
  --experts 4 --topk 2 \
  --warmup 2 --iterations 5 --backward \
  --output benchmark-results/nccl-ep.json
```

在保持其他参数相同的前提下，再加：

```text
--pipeline-chunks 4 --output benchmark-results/nccl-ep-pipelined.json
```

可验证真实 chunk pipeline。过细 chunk 会增加 collective launch、临时 gather、expert
permutation 与小 GEMM 开销；同一 expert 在每个 chunk 都可能产生独立的 16-row tail，
`native_grouped_tile_model.chunked_aggregate` 因而另报实际分块后的 row tiles/tasks/lanes，
以及 `active_route_rows` 守恒检查；不能沿用未分块的总 count 推导。所以 `4` 只是实验点，
不是通用最优值。

报告中的每个 latency sample 是所有 rank 完成 operation 所需时间的最大值，而不是 rank
0 的局部时间。若要分析 straggler，`rank_latency` 还保留 `[iteration,rank]` 原始样本、
逐 rank 摘要、每次 rank mean 与 rank max；兼容字段 `raw_samples_ms` 与其中的 rank-max
序列完全相同。分析报告时至少同时读取：

- `counts_matrix[i][j]`：source `i` 发往 destination `j` 的 route 行数；
- `cross_rank_route_rows`：去掉矩阵对角线后的 route 总数；
- `forward_cross_rank_activation_bytes`：dispatch 与 restore 的激活 payload；
- `max_tolerance_ratio`：实际误差除以 `atol + rtol * abs(reference)` 的最大值；
- `raw_samples_ms`：未经丢弃的逐次全局 latency。
- `rank_latency.per_iteration_rank_samples_ms`：未经丢弃的逐次、逐 rank latency；
- `expert_counts`：每个 owner rank 收到的 local expert 行数；
- `theoretical_padding_utilization`：将各 rank local experts 补到本 rank 最大 count 后的
  有效槽位比例。
- `shared_expert_compute`：replicated shared 分支的有效中间维、token 行数与前向矩阵
  FLOPs；其参数梯度归约不属于 EP 两次 All-to-All；
- `load_balance`：rank/expert 完整计数、peak-to-mean、CV、零负载数，以及 capacity 的
  反事实 drop/padding 模型；
- `overlap_model`：串行配置保存乐观流水下界；chunked 配置明确标记
  `async_pipeline_executed=true` 并引用实测 `pipelined_core` samples；在 profiler 证据加入
  报告前，`hardware_overlap_verified` 保持 `false`；
- `symmetric_buffer_model`：按 source—expert cell 建模的对称张量 shape、bytes、利用率与
  overflow；它不代表已执行 one-sided 通信或真实 drop。

报告还用一次独立测量 pass 给出：

```text
route_and_pack → exchange_counts → dispatch → expert_compute
→ restore → combine → shared_expert（若启用）→ backward（若启用）
```

每个阶段先在各 rank 上测局部时间，报告同时保留 `rank_raw_samples_ms`，并另给该阶段的
rank max。不同 stage 的 max 可能来自不同 rank，因此不能把这些 max 相加并声称等于端到端
max；overlap 模型必须先在同一 rank 内组合相关阶段，再取 rank max。端到端延迟仍以单独
的完整 operation pass 为准。GPU 阶段边界会同步当前 device，所以 profiling pass 会破坏
原本可能存在的阶段重叠，不应拿它的总和当作正常执行延迟。

`aggregate_cross_rank_activation_gb_s_at_median` 是协议 payload 除以端到端时间，只能作为
可复核的等效吞吐量，不能当作 NCCL 链路带宽。它没有计算 self-route、counts、barrier、
协议 metadata、kernel launch 和实际网络封包开销。

`expert-backend=cuda` 让 NCCL 路径使用原生 active-row SwiGLU。FP32 配置可同时选择原生
router、route pack/combine、expert-major permutation 和 CUDA-core expert；FP16 配置当前
必须让 router 与 route 走 PyTorch reference，只把 expert-major 纯搬运和 expert compute
交给 native，后者使用 WMMA+FP32 accumulation。报告中的 backend 字段必须把这两种证据
分开，不能把“FP16 expert 已原生化”改写成“整条 FP16 MoE 已原生化”。

两条路径都按 expert offsets 调度 `16x16x16` tiles，空 expert 不生成任务，各 expert 的 row
tail 独立处理；backward 仍由 reference 重算。chunked NCCL 已提供真实通信—计算 overlap
执行路径，但仍需 GPU CI 与 profiler 验证正确性和收益。WMMA 版本没有 Hopper WGMMA/TMA
或 tile-level 异步 copy。只有补齐原生 backward、测出稳定的端到端 rank-max 改善并解释
profiler 时间线后，才有资格讨论生产级 backend。

要显式验证空 local shard，可运行：

```bash
torchrun --nproc-per-node=2 benchmarks/expert_parallel.py \
  --backend gloo --route-backend reference --dtype float64 \
  --tokens-per-rank 0 --token-skew 3 \
  --model-dim 4 --hidden-dim 5 --experts 2 --topk 1 \
  --warmup 0 --iterations 1 --backward
```

此时 rank 0 有 0 个本地 token，rank 1 有 3 个；全作业仍至少有一个 token。全 rank 都必须
进入 route counts、dispatch、restore 和 backward collective。
