# 第四章：DeepSeekMoE

## 4.1 稀疏 FFN

Dense Transformer 为每个 token 执行同一组 FFN 权重。MoE 准备 `E` 组 routed
experts，但每个 token 只选择 `K` 个，因此模型总参数量可以增长，而每 token 的激活
计算量保持相对有限。

DeepSeek 风格一层输出可写为：

```math
y_t = \sum_{k=1}^{K} \alpha_{tk} E_{e_{tk}}(x_t) + S(x_t),
```

其中 `S` 是始终执行的 shared experts。

## 4.2 Expert 是三矩阵 SwiGLU

每个 expert 不是普通两层 MLP，而是：

```math
E(x)=W_2\left(\operatorname{SiLU}(W_1x)\odot(W_3x)\right).
```

`W1` 与 `W3` 都从 model dimension 投影到 hidden dimension，逐元素门控后再由 `W2`
投影回来。若实现只包含 `W1/W2`，即使输入输出 shape 相同，也不再是 DeepSeek expert
语义。

### 4.2.1 Shared experts 的 widened SwiGLU

DeepSeek-V3 风格实现不必为 `n_s` 个 shared experts 启动 `n_s` 次独立 MLP。可以把它们沿
中间维拼成一个 widened SwiGLU：

```text
W1_shared, W3_shared: [n_s * D_h, D]
W2_shared:            [D, n_s * D_h]
```

将中间维按 `n_s` 段切开后，逐元素 SiLU 和乘法不会跨段混合，而 `W2_shared` 会把各段的
输出相加。因此，一个宽度为 `n_s D_h` 的 SwiGLU 与 `n_s` 个宽度为 `D_h`、输出求和的
独立 shared experts 数学等价。这里的 `n_s` 表示 shared expert 倍数，不是 routed Top-K，
每个 token 都会执行全部 shared 宽度。

## 4.3 Scoring 与选择偏置

router 首先计算：

```math
z = xW_g^T.
```

score 可以是 Softmax 或 sigmoid。DeepSeek-V3 风格路由允许加入每 expert 的 correction
bias，但它只服务于“选择谁”：

```math
s^{select}=s+b.
```

Top-K index 从 `s_select` 得到，最终权重却从无偏的 `s` 中 gather。若把 bias 直接带入
组合权重，就会把负载均衡控制量混入模型输出。

对 sigmoid score，选中的权重需要重新归一化：

```math
\alpha_{tk}=r\frac{s_{t,e_{tk}}}{\sum_j s_{t,e_{tj}}},
```

其中 `r` 是 route scale。

## 4.4 Group-limited Top-K

将 `E` 个 experts 均分成 `G` 组，先为每组计算 group score，只保留 `G_k` 个组，
再从这些组内选全局 Top-K。无 bias 时 group score 可取组内最大值；有 correction
bias 的 DeepSeek-V3 参考路径使用组内前两个 selection score 之和。

实现必须满足：

- `E` 能被 `G` 整除；
- `K <= G_k * (E/G)`；
- 未选组的 experts 在第二次 Top-K 前被置为负无穷；
- 精确定义相同 score 的 tie 行为。

本仓库规定完全同分时较小 group id 优先，同一保留组集合内较小 expert id 优先。
FP32 sigmoid 在大正 logit 上可能精确饱和为 `1.0`，所以 tie 不是只会出现在手工构造输入
里的理论边界。`deepseek_grouped_topk` 使用 stable descending sort 实现这一可复现规则，
也是其他 backend 的可执行规格。

### 4.4.1 CUDA grouped router 的首个正确性边界

原生 `grouped_topk` 路径把工作拆成两段：先由 PyTorch 当前 stream 上的矩阵乘和 sigmoid
得到 `[T,E]` scores，再由 CUDA selector 为每个 token 执行 group score、保留组、expert
选择、无偏 score gather 与归一化。可选 correction bias 只参与前两次选择，不进入输出
权重，也不接收梯度。

首版 selector 为每个 token 分配一个 thread，并串行扫描 groups 与 experts。它避免把
selection 拉回 host，支持空 token 维度、group tail 组合和明确 tie-break，也注册了
FakeTensor 与 reference-recompute autograd；但最坏 `O(TKGE)` 的重复扫描不是性能终点。后续可
在不改变输出契约的前提下，用 warp/block reduction、分层 selection 或 fused projection
替换内部调度。

独立 benchmark 可把 router 投影、选择与负载分布从整个 MoE 中拆出来：

```bash
python benchmarks/router.py \
  --device cuda --backend cuda --dtype float32 \
  --tokens 4096 --model-dim 7168 --experts 256 --topk 8 \
  --n-groups 8 --topk-groups 4 --hot-expert-bias 0.5 \
  --warmup 10 --iterations 100 --backward \
  --output benchmark-results/router.json
```

这里的 forward+backward 仍使用 reference-recompute backward，不能把计时结果解释成原生
router backward 的性能。

## 4.5 从 token-major 到 expert-major

router 输出通常是：

```text
indices: [T, K]
weights: [T, K]
```

expert GEMM 更希望输入按 expert 连续排列。因此 dispatch 需要：

1. 对 expert id 做 histogram，得到 `count[e]`。
2. 对 count 做 exclusive scan，得到 `offset[e]`。
3. 把每个 `(token, slot)` 写入对应 expert 区间。
4. 保存原 token index、slot 或 routing weight，供 combine 恢复。

dispatch 必须发送原始 `x`，routing weight 在 expert 输出后再乘：

```math
\alpha E(x) \neq E(\alpha x).
```

SwiGLU 是非线性的，把 `alpha` 提前乘进 token 会改变模型函数。这也是课程原型迁移到
正式实现时必须修正的语义边界。

关键不变量：dispatch 后的每一行必须能唯一追溯到原始 `(token, topk_slot)`；combine
后每个 token 恰好累加其所有有效 expert contribution。

## 4.6 Capacity 是策略，不是默认语义

某些 MoE 系统使用 capacity factor，为每个 expert 设置容量并丢弃 overflow token。
这有利于静态缓冲和负载控制，却会改变输出语义。DeepSeek reference 的 grouped
Top-K 不等价于任意一种 capacity drop。

因此本项目把 capacity 视为可选 dispatch policy。启用时必须明确：

- 容量公式；
- 哪些 token 优先保留；
- 丢弃后权重是否重新归一化；
- dropped token 的输出和梯度；
- 统计多少 token 被丢弃。

## 4.7 优化阶梯

建议依次建立：

```text
Python token loop
→ vectorized router
→ histogram + scan + pack
→ grouped GEMM experts
→ fused activation/gating
→ fused combine
→ persistent or mega-kernel experiment
```

每一级都与同一 reference 比较。否则多个优化同时引入时，很难判断错误来自路由、
打包、expert 数学还是恢复顺序。

## 4.8 CUDA route 原语

将 dispatch 与 combine 拆成独立原生算子后，pack 的最小输出契约可以写成：

```text
packed_x, packed_alpha, packed_route_id, packed_expert_id,
counts_per_expert, counts_per_rank
```

其中 `packed_x` 仍是未乘权重的原始 activation。`packed_route_id = token * K + slot`，
因此可以无损恢复 `token = route_id // K` 和 `slot = route_id % K`。按
`(destination_rank, expert_id)` 分桶足以服务 EP 通信；同一个 expert 区间内部不要求稳定
排序，只要所有 metadata 使用同一 permutation。

一个简单正确性 kernel 可分四步：

1. 对每个 route 的 `(owner[e], e)` key 做 histogram；
2. 对 key counts 做 exclusive scan，得到每个 bucket 的起点；
3. 每个 route 用 atomic cursor 领取 bucket 内的 row，并复制 activation/metadata；
4. combine 用 `atomicAdd(y[token, d], alpha[row] * contribution[row, d])`。

这两个 atomic 都带来非确定顺序：pack 的同 bucket 行序不固定，combine 的浮点累加顺序
不固定。它们不改变 route 双射与数学定义，但可能造成末位舍入差异。启用 deterministic
algorithms 时，应切换到稳定 sort/scan 与确定性 reduction，不能继续调用 atomic kernel
却宣称结果确定。

上述实现只消除了 Python 索引和恢复开销。串行 scan、每 route 一个 block、FP32-only 和
atomic combine 都不是性能终点；下一步才是并行 scan、向量化 copy、expert-major
permutation 与 grouped GEMM。

## 4.9 变长 experts 与 padding 利用率

batched GEMM 要求 batch 中矩阵 shape 一致，但不同 expert 的 route 数 `n_e` 通常不同。
一个容易验证的 baseline 是把每个 expert 补到：

```math
M=\max_e n_e,
```

再执行 shape 为 `[E,M,D]` 的三次 batched GEMM。有效槽位率为：

```math
U_{pad}=\frac{\sum_e n_e}{E M}.
```

当路由均匀时 `U_pad` 接近 1；一个热门 expert 主导时，其余 experts 的 padding 会造成大量
无效 FLOPs。单个 rank 没有 route 时约定利用率为 1，因为它既没有有效工作，也没有分配
padding 工作。

padded baseline 的价值是把 Python 循环合并为规则的 batched GEMM，并提供容易计算的性能
上界。真正的 grouped GEMM 接收每个 expert 独立的 `M_e=n_e`，无需执行 padding rows，
但 kernel 的 tile scheduling、空 expert 与极小矩阵处理更复杂。比较两者时必须同时报告
latency 与 `U_pad`，否则无法区分 kernel 效率和路由均衡度。

### 4.9.1 从 active-row 契约到 grouped tiled GEMM

若 activation 已按 local expert 连续排列，可以用 offsets：

```text
offsets = [0, n_0, n_0+n_1, ..., sum_e n_e]
```

把每一行映射到其 expert 权重，再只对 `sum_e n_e` 个有效行计算三矩阵 SwiGLU。这样的
active-row kernel 不执行 padding，也不需要 host 逐 expert 启动 GEMM；空 expert 对应相邻
offset 相等，不能被当成错误。

首版 scalar reduction 建立上述语义后，当前 FP32 CUDA forward 已改为 grouped tiled GEMM。
每个任务描述一个 expert 的 `16x16` 输出 tile；K 维也按 16 分段，activation 与 weight 由
`16x16` threads 协作装入 shared memory。权重保持 row-major 存储，相邻 lanes 先连续读取 K
维，再在 shared memory 内转置成计算需要的 `[K,N]` tile。每个 stage 在“装载完成”和
“消费完成”后各有一次 block barrier，tail load 写零、tail store 用 predicate。

调度 metadata 完全在 device 上由 offsets 生成：

```math
N_{row\_tiles}=\sum_e\left\lceil\frac{n_e}{16}\right\rceil,
```

```math
N_{hidden\_tasks}=N_{row\_tiles}\left\lceil\frac{D_h}{16}\right\rceil,
\qquad
N_{down\_tasks}=N_{row\_tiles}\left\lceil\frac{D}{16}\right\rceil.
```

空 expert 贡献零个 tasks，非空 expert 各自承担最后一个 row-tail tile，因此没有把所有
experts 补到同一个 `max_e n_e` capacity。与此同时，硬件仍会为每个 row tile 启动 16 行
lanes；报告把 `16 N_row_tiles - sum_e n_e` 记为 inactive tail row lanes。这个数是 kernel
tile 边界成本，不等于 padded batched GEMM 的全局 padding rows。

FP32 路径中，W1/W3 共用 activation tile，但分别加载权重并保留两个 FP32 accumulator；
SiLU 与逐元素乘写出 materialized hidden state，随后 W2 使用相同 tiled 映射完成 down projection。实现覆盖
任意 `D/D_h/n_e` tails、current stream 和空 expert；backward 仍是 PyTorch segmented
reference-recompute。它是 grouped、tiled、无 capacity padding 的 CUDA-core forward。

### 4.9.2 FP16 WMMA：混合精度必须写成接口契约

同一 task metadata 也驱动 FP16 Tensor Core 路径。每个 block 恰好一个 32-thread warp，整个
warp 共同负责一个 `16x16` 输出 tile；K 维仍以 16 为一段。每段先由 lanes 把 FP16
activation 和 weight 搬进 32-byte aligned shared tiles，tail 写零，再让所有 lanes 一致调用
`load_matrix_sync` 与 `mma_sync`。这不是代码风格约束：WMMA 是 warp-synchronous API，若
分支条件在 warp 内不一致，行为无定义；shared pointer 与 leading dimension 也必须满足
WMMA 的对齐约束。

三次投影的数值阶段为：

```text
FP16 x/W1/W3 → WMMA FP32 gate/up accumulators
             → FP32 SiLU(gate) * up
             → round to FP16 materialized hidden
FP16 hidden/W2 → WMMA FP32 output accumulator → round to FP16 output
```

因此 reference 不能只写“所有中间值一直 FP32”，否则 down projection 的输入已经和 kernel
不同。当前 reference 在 FP16 下显式执行 hidden 的 `FP32 → FP16 → FP32`，让量化边界、
autograd 重算和 benchmark reference 一致。BF16 暂不进入 native expert 路径；FP16 还要求
compute capability 7.0 以上。

这里使用的是 warp-level WMMA，不是 Hopper 的 warpgroup-level WGMMA。当前 global/shared
搬运也是同步的，没有 TMA、double buffering、warp specialization 或异步 MMA。课程中
H100/FlashAttention-3 的 WGMMA+TMA 流水线应视为下一阶段设计空间，而不是给当前 kernel
贴上的性能标签。即便 WMMA 数值测试通过，也必须再用 Nsight Compute 检查 Tensor Core
指令、eligible warps、memory stalls 和 tile-tail 成本，才能判断它是否真的比 FP32 路径快。

可以用同一份 counts 单独复核 expert compute：

```bash
python benchmarks/experts.py \
  --device cuda --backend cuda --dtype float16 \
  --expert-counts 17,0,5,31 --model-dim 64 --hidden-dim 128 \
  --warmup 5 --iterations 50 --backward \
  --output benchmark-results/experts-skewed.json
```

报告同时保留 `forward_active_row_matrix_flops` 与
`forward_padded_matrix_flops`。前者描述 native active-row kernel 的理想矩阵工作，后者是
把所有 experts 补到最大 count 的反事实 baseline；两者不能互换。启用 `--backward` 时，
latency 是 forward+reference-recompute backward，而报告中的矩阵 FLOPs 仍明确标为 forward。
`native_grouped_tile_model` 另存 row/output task counts 与 tail-lane utilization；它是按 shape
推导的 launch-work 模型，不是 profiler counter。`native_numeric_model` 则保存选中的
CUDA-core/WMMA engine、multiplicand/accumulator/materialized-hidden dtype 与最低计算能力；
两种模型都不能替代 profiler 证据。

### 4.9.3 Whole-layer single-device forward milestone

公开入口 `deepseek_moe_forward(..., backend="reference"|"cuda"|"auto")` 把本章已经
分别验证过的阶段串成一条完整路径：

```text
route -> pack -> offsets -> expert -> combine
```

CUDA v1 的编排位于
[`csrc/moe/deepseek_moe_forward_cuda.cu`](../../csrc/moe/deepseek_moe_forward_cuda.cu)。
它先调用 grouped Top-K 和 route-pack，随后用 device 上的 `zeros`、`cumsum` 与 `cat`
建立 expert offsets，再调用 active-row SwiGLU expert，最后从 packed route id 导出 token
index 并 combine。一次公开调用因此仍包含多个 ATen/kernel launch，也会物化 scores、packed
rows、offsets、hidden state 与 contributions；“一个 raw operator”不等于“一个 kernel”。

这个里程碑应准确称为 **single-device、staged、correctness-first** whole-layer forward。
CUDA 路径只接受 contiguous CUDA FP32、sigmoid scoring、无 `requires_grad` 的输入，并要求
deterministic algorithms 关闭。`reference` 总是执行 packed PyTorch 规范，`cuda` 对不满足契约
的输入报错，`auto` 才允许在不满足 CUDA v1 契约时回退。

真正的 FlashMoE 后续目标还包括 persistent scheduling、跨 expert 的 tile scheduling，以及
one-sided multi-GPU communication；当前实现没有这些机制，不能据此宣称 fused 或持久化
FlashMoE 性能。前文 router/expert 的 backward 只描述各自既有的实验性 stage 语境；新的
whole-layer raw operator 是 forward-only，本节不把它描述为可训练算子。

## 4.10 Capacity factor 的丢弃—填充平衡

令一次全局路由共有 `R=T K` 行、`E` 个 routed experts，则平均每 expert 的 route 数为：

```math
\bar n=\frac{R}{E}.
```

给定 capacity factor `f>0`，一个常见的统一容量模型是：

```math
C=\left\lceil f\bar n\right\rceil.
```

对实际计数 `n_e`，该模型最多接收：

```math
R_{keep}=\sum_e\min(n_e,C),
```

并产生：

```math
R_{drop}=R-R_{keep},\qquad
R_{pad}=EC-R_{keep}.
```

这里 drop 与 padding 可以同时出现：热门 expert 溢出，而冷门 expert 的容量未用满。槽位
利用率应以实际保留行计算：

```math
U_C=\frac{R_{keep}}{EC}.
```

`f=1` 只表示容量等于平均负载，不表示不会丢弃。对一组已经观测到的 counts，令
`n_max=max_e n_e`，则 `f=n_max/\bar n` 一定足以使取整后的容量不小于 `n_max`；更小的
factor 也可能因 `ceil` 恰好足够，所以这个比值是易复核的充分界，不应误称为唯一最小值。

当前 benchmark 的 `uniform_capacity_model` 只对已有 routes 做上述反事实分析，不执行
drop，也不改变输出。若未来把 capacity 变成真实 dispatch policy，还必须补齐 4.6 节列出
的保留顺序、权重与梯度语义。
