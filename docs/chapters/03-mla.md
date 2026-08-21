# 第三章：Multi-head Latent Attention

## 3.1 MLA 解决什么问题

标准多头注意力在自回归推理时为每个历史 token 缓存每个 head 的 K 和 V。随着层数、
head 数与上下文增长，KV cache 会成为显存和带宽瓶颈。MLA 的目标是把跨 head 的 KV
信息压缩到更低维的 latent 表示，并在需要时通过上投影恢复或吸收到 query/output
计算中。

MLA 不是简单地分别执行 `K @ W_K` 和 `V @ W_V`。DeepSeek 风格 MLA 还包含：

- query 的可选低秩投影；
- KV 的联合低秩表示；
- 不参与位置编码的 NoPE 子空间；
- 使用 RoPE 的位置子空间；
- prefill 与 decode 不同的缓存和计算路径。

## 3.2 Query 路径

记模型输入为 `x`。若启用 query 低秩分解：

```math
c^Q = W^{DQ}x,
```

```math
q = W^{UQ}\operatorname{RMSNorm}(c^Q).
```

随后把每个 head 的 query 拆成：

```math
q_h = [q_h^{N}; q_h^{R}],
```

其中 `N` 表示 NoPE 部分，`R` 表示应用 RoPE 的部分。只有 `q_h^R` 经过旋转位置编码。

## 3.3 KV 路径

KV 下投影同时产生 latent 内容和独立的位置 key：

```math
[c^{KV}; k^R] = W^{DKV}x.
```

`c^{KV}` 经过 RMSNorm 后，由上投影生成各 head 的 NoPE key 与 value：

```math
[k_h^N; v_h] = W_h^{UKV}\operatorname{RMSNorm}(c^{KV}).
```

位置 key `k^R` 在各 head 间共享，再与 `k_h^N` 拼接。朴素路径因此可以写成标准
attention，适合作为正确性 reference。

## 3.4 两种缓存

朴素缓存保存展开后的每 head K/V：

```text
K cache: [B, S, H, D_qk]
V cache: [B, S, H, D_v]
```

压缩缓存只保存：

```text
latent KV cache: [B, S, R_kv]
positional cache: [B, S, D_rope]
```

其中 `R_kv` 通常远小于 `H(D_nope + D_v)`。decode 每次读取历史缓存时，带宽节省
来自这项压缩，而不是 Softmax 本身。

## 3.5 Weight absorption

在 decode 路径中，可以利用矩阵乘法结合律，把 KV 上投影权重吸收到 query 和输出
投影中。概念上：

```math
q_h^N (W_h^{UK})c^{KV}
= (q_h^N W_h^{UK}) c^{KV}.
```

这样无需为每个历史 token 展开所有 head 的 `k_h^N`。对 value 聚合也可先在 latent
空间完成，再经相应权重映射回各 head 输出。

吸收路径和朴素路径必须数值等价，但布局、计算顺序和累加误差不同。因此 MLA 的第一
批测试应比较：

- naive prefill 与 absorbed prefill；
- naive decode 与 compressed-cache decode；
- 从空 cache 开始逐 token decode 与一次性 causal prefill；
- NoPE/RoPE 拆分和不同 rank；
- cache 写入位置、长度边界与 batch slot 复用。

## 3.6 Kernel 划分建议

第一版不要把所有投影、RoPE、attention 与 output projection 融成一个 kernel。更稳妥
的递进是：

1. PyTorch naive MLA reference。
2. PyTorch absorbed decode reference。
3. CUDA RoPE 与 cache 写入。
4. compressed-cache decode kernel。
5. MLA prefill 的 tiled attention。
6. profile 后选择值得融合的边界。

prefill 偏计算密集，decode 偏带宽与 launch 延迟敏感；两者共享数学语义，但不应被
强迫使用同一个 launch configuration。

## 3.7 Cache 容量模型

设 batch 为 `B`、序列长度为 `S`、head 数为 `H`，元素宽度为 `e` 字节。展开 K/V 的
payload 元素数为：

```math
N_{expanded}=BSH(D_{nope}+D_{rope}+D_v).
```

latent cache 的 payload 元素数为：

```math
N_{latent}=BS(R_{kv}+D_{rope}).
```

payload 压缩比为：

```math
C=\frac{H(D_{nope}+D_{rope}+D_v)}{R_{kv}+D_{rope}}.
```

位置 key `D_rope` 不能被省略，因为 RoPE 后的 key 与绝对位置相关；但它只保存一份并在
heads 间共享。若实现还保存 `[S]` 的 int64 positions，额外容量为 `8S` 字节。容量报告
必须区分 payload 与 position metadata，且二者都是存储大小，不是实测 HBM traffic。

## 3.8 Prefill 与 decode 的计时边界

同一个“MLA latency”至少包含六种不同实验：

| workload | query 长度 | cache 操作是否计时 |
| --- | ---: | --- |
| `prefill_attention` | `S` | cache 已构建，只测 attention |
| `prefill_with_cache` | `S` | 重建 `S` 个 latent entries |
| `decode_attention` | `1` | cache 已含当前 token |
| `decode_with_append` | `1` | append 当前 token 后做 attention |
| `decode_with_static_write` | `1` | 原位写入当前 token 后做 attention |
| `decode_with_paged_write` | `1` | 覆盖一个物理 slot 后按 block table 做 attention |

六者不能放在同一列直接比较。特别是函数式 cache 使用 `torch.cat`，一次 append
会读取旧 prefix 并写出完整新 cache，拷贝量随上下文长度线性增长。这是易验证的语义实现，
不是生产 decode cache。`MLAStaticCache` 则预分配 KV、RoPE 和 position storage，维护
一个 valid cursor，并将新 chunk 原位写入 `[cursor:cursor+L]`。storage 地址在 decode
过程中保持不变，单 token 的 cache 写入量为：

```math
B(R_{kv}+D_{rope})e + 8\quad\text{bytes}.
```

这把 cache 搬运从随 `S` 增长的 `O(S)` 降为 `O(1)`；attention 读取历史 cache 的成本仍然
是 `O(S)`。当前 reference 为整个 batch 维护一个共享 cursor，适合等长 batch 和实验验证。
多请求 continuous batching 需要为每个 batch slot 分别维护有效长度、请求生命周期和 page
映射；本章 3.14 节给出最小实验控制面，但生产系统还需要更完整的执行层。静态写入会修改 storage，因此 API 明确限定在 `torch.no_grad()` 或
`torch.inference_mode()` 下使用，而不伪装成可微操作。

`MLAPagedCache` 进一步把 latent payload 拆成固定大小的物理页，并用 per-row length 与
block table 描述逻辑序列。它允许同一 batch 中各行长度不同，也允许 logical page 在物理上
不连续；decode 只覆盖一个全局 physical slot。该路径同样是 inference-only，但不再要求整个
batch 共享一个连续 cursor。

## 3.9 两条路径的 FLOPs 边界

naive 路径每次调用都会把 `S_k` 个 latent entries 上投影到所有 heads，主要额外矩阵
FLOPs 为：

```math
2BS_kR_{kv}H(D_{nope}+D_v).
```

absorbed 路径改为把 NoPE query 投到 latent 维、在 latent 维计算 score/value reduction，
再把聚合后的 latent output 投到 `D_v`。decode 的 `S_q=1` 时，这避免对整个历史 cache
做 head-wise 上投影；prefill 时是否更快则依赖 shape、实现和 GEMM 效率，不能只凭 FLOPs
下结论。

仓库的结构化实验可直接运行：

```bash
python benchmarks/mla.py \
  --device cpu --dtype float64 \
  --implementation absorbed \
  --workload decode_with_static_write \
  --sequence-length 128 --iterations 10 \
  --output benchmark-results/mla-decode.json
```

计时前会用另一条 naive/absorbed 路径逐元素校验；报告保留完整 config、cache 容量、矩阵
FLOPs 计数约定和所有 raw samples。若要观察函数式 cache 复制造成的差异，应以相同参数再
运行一次 `decode_with_append`，而不是把两种更新策略混入同一个计时区间。

## 3.10 当前 staged CUDA correctness backend

仓库当前没有把整层 MLA 伪装成一个超大 fused kernel，而是沿用 3.6 节的可验证分层，注册
以下 native FP16/BF16/FP32 算子：

1. direct query projection + RoPE；
2. LoRA query projection + RMSNorm + RoPE；
3. latent KV projection + RMSNorm + RoPE；
4. projection 后直接写入预分配 static cache；
5. projection 后按 slot mapping 写入 paged cache；
6. 连续 latent cache 上的 absorbed score、causal mask、online softmax、latent value reduction；
7. 直接按 block table 读取 paged latent cache 的 absorbed attention；
8. head output projection。

连续 staged 路径要求原有六个阶段全部可用；paged 路径复用 query/output projection，并要求
两个 paged 原生算子可用，不再把 payload 临时 materialize 成连续 K/V。out-of-place 算子的
backward 通过可追踪 PyTorch specification 重计算，以便先固定一阶梯度语义；static cache
和 paged cache 写入会修改 storage，因此仍严格限定为 inference-only。该实现证明了完整 prefill/decode
数据流和 dispatcher 契约。同一次 native 请求中的输入、权重、cache 和 stage 输出使用统一
storage dtype；linear reduction、RMSNorm 统计、RoPE、在线 Softmax 和 latent/value 累积使用
FP32，每个公开 stage 再写回 storage dtype。连续 absorbed-attention 已有按 head 维度选择的 warp
specialization；paged attention 目前是 correctness-first 的 one-CTA-per-query/head kernel。两者都
还不是生产内核：prefill/decode 尚未使用完整的专用调度，也没有 continuous-batching 请求/page
生命周期、prefix-sharing 策略或 profiler 驱动的跨算子融合。

## 3.11 位置校验也可能成为 GPU 同步边界

Python 层的参数检查并不天然“免费”。若 positions 位于 CUDA，下面这种条件最终需要把一个
GPU boolean 读回主机：

```python
if not torch.all(positions[1:] > positions[:-1]):
    raise ValueError(...)
```

一次检查很小，但 MLA 原先会在 cache 构建、attention 请求、query projection 和 RoPE 边界
重复检查同一个 Tensor。每次 `bool(cuda_tensor)` 都可能产生 DtoH scalar copy 和 stream
synchronize，使 CPU launch 序列被切碎。

当前实现仍在公开 API 边界检查非负、严格递增和 prefix 顺序，但在一次失败前先把多个谓词
合并，只做一次正常路径标量读取。已经验证的 cache/query positions 以 Tensor identity 和
version counter 为依据复用；任何原地修改都会改变 version 并触发重新检查。static cache
另外记录 position storage 的已验证 prefix 长度和版本，`truncate` 可以复用仍然有效的前缀，
外部修改 storage 后则不能继续信任。inference-mode Tensor 若不提供 version counter，也不会
跨 API 调用复用验证结果。

RTX 5090 上对 representative MLA case 的同口径 capture 包含一次输出、5 次 warmup 和 20 次
正式调用，即 26 次主路径调用。修改前后的同步事件计数为：

| Case | `_local_scalar_dense` 修改前 | 修改后 | `cudaStreamSynchronize` 修改前 | 修改后 |
| --- | ---: | ---: | ---: | ---: |
| `mla_prefill_regular` | 212 | 28 | 220 | 36 |
| `mla_decode_regular` | 162 | 29 | 170 | 37 |

这证明的是 Python-side 隐式同步减少，不等价于宣称 kernel 延迟按相同比例下降。修改后的聚合
报告保存在 `validation/single-gpu/2026-08-14-rtx5090-cu128/`。第一轮快照中，absorbed-attention
分别占 prefill/decode custom-operator self-device 时间的 67.2% 和 81.3%，因此它成为下一轮
CUDA 优化目标。

## 3.12 小 head 维度的 warp-partition attention

原 generic kernel 为每个 `[batch, head, query]` 行启动一个 128-thread block。每处理一个 key，
它都需要在整个 block 上归约 latent dot 与 RoPE dot，并同步在线 Softmax 状态和 latent numerator。
当 `R_kv=32`、`D_rope=16`、`D_v=32` 时，大量线程不持有有效分量，而每个 key 仍承担多次
block-wide barrier。

当前 specialized 路径在 `R_kv`、`D_rope`、`D_v` 都不超过 32 时采用四个 warp：

1. warp 0 先计算并共享一次 absorbed latent query；
2. 四个 warp 分别处理 `key_index = warp_id + 4n` 的 key 子序列；
3. 每个 warp 用 shuffle 完成 dot reduction，并维护自己的 online-softmax maximum、denominator
   和 latent numerator；
4. key 循环结束后，warp 0 用各 partition maximum 做一次数值稳定合并，再完成 latent-to-value
   projection。

这样每个 key 的循环内不再需要 block-wide barrier。任一受管维度大于 32 时仍选择 generic
kernel，因此优化没有缩小公开算子的 shape/stride 契约。CUDA 回归覆盖 32 临界点、非整齐维度、
causal/non-causal、非连续 last-dimension stride，以及 latent、RoPE、value 三种独立 fallback。

同一 RTX 5090 上，包含 26 次主路径调用的 Kineto capture 将 absorbed kernel self-device 总时间
从 prefill/decode 的 2.632/2.699 ms 降到 0.429/0.491 ms；它在当前 custom-op self-device 时间中
所占比例也降到 41.8%/44.3%。这只是相同固定 case 的本机 kernel 观察：它不包含可泛化的 Nsight
counter，也不能当作端到端或其它 shape/hardware 的加速比。下一步仍需用 Nsight Compute 检查
occupancy、memory traffic 和指令分布，再决定 prefill/decode 专用调度与融合边界。

## 3.13 Paged latent cache 的数据与错误契约

物理 storage 不再带 batch 维，而是：

```text
kv_storage:       [num_pages, page_size, R_kv]
pe_storage:       [num_pages, page_size, D_rope]
position_storage: [num_pages, page_size]
```

全局 slot `s` 对应 `page=s//page_size`、`offset=s%page_size`。一次
`write_mla_paged_cache` 接受 `[B,S]` slot mapping；同一次调用中的 slot 必须互不重复且位于
容量内，避免并行 scatter race。后续调用可以有意覆盖旧 slot，latent、RoPE payload 与 absolute
position 会一起替换。

读取侧使用 `block_table[B,max_logical_pages]` 与 `sequence_lengths[B]`。每行实际使用的 page id
必须在范围内且不得重复，未使用表项必须为 `-1`；不同 batch 行可以共享物理页。有效 logical
slot 必须已经写入，absolute positions 必须在每行严格递增。causal mask 比较的是这些 absolute
positions，而不是物理 slot id。

CUDA kernel 直接在 key loop 中把 logical token 映射到 `(physical_page,page_offset)`，以 FP32
完成 absorbed query、score、online softmax、latent numerator 和 value projection，再写回 storage
dtype。测试覆盖 ragged batch、非连续物理页、重复/越界 slot、覆盖语义、未写入和非单调位置，
以及 `S=257,page_size=16` 的尾页。Python 公共 API 校验后，native op 可跳过重复的 host-side
防御检查；校验缓存只在 tensor identity/version 未变化时有效，直接原地修改会触发重新检查。
## 3.14 CUDA Graph bucket 与最小 continuous batching 控制面

`SingleOutputCUDAGraphRunner` 不直接捕获调用者的临时 tensor。capture 时它为每个输入建立
runner-owned static buffer；replay 先核对 shape、dtype、device，再把新值复制进固定地址，最后
重放 graph。输出也是固定地址，下一次 replay 会覆盖其内容。也就是说，一个 bucket 的静态
合同是“地址由 runner 固定、值可变、shape/dtype/device 不变”，而不是要求调用者永久持有同一
个输入 tensor。

`MLAPagedDecodeGraphRunner` 把 bucket 进一步限定为固定的 batch size、`model_dim` 和
`max_logical_pages`，且 query length 恒为 1。cache 与 weights 必须维持 capture 时的地址；每次
replay 的 `block_table`、`sequence_lengths`、已写入 slot 和 absolute query positions 都会在
copy 前校验。graph 内只运行已经预校验的 raw query projection、paged absorbed attention 和
output projection，cache projection/write 不在本 graph 内。新的 batch shape 或 block-table
宽度需要另建 bucket。

CPU 侧的 `ContinuousBatchingScheduler` 提供 `FixedPageAllocator` 及
`submit/schedule/complete/abort/cancel`：

- FIFO 请求只在 iteration 边界进入 active set；
- prefill batch 与 decode batch 保持同质，当前 prefill 一次处理完整 prompt；
- 每个 decode iteration 对每个 active request 最多预留一个 token；
- `schedule` 先计算整个 batch 的页需求，再一次性预留；失败不改变任何请求或 allocator 状态；
- `complete` 提交长度并回收完成请求，`abort` 精确恢复页顺序、请求状态与 FIFO admission；
- in-flight 请求必须先 complete/abort，才能 cancel。

这条路径补上了固定页分配、请求回收和最小多请求 continuous batching，但仍没有 eviction、
prefix sharing、chunked prefill、优先级、speculative decoding、网络层或模型执行器。它把
storage/compute primitive 与可测试控制面接起来，仍不是完整 serving runtime。

聚焦验证中，CPU scheduler 契约测试为 `20 passed`；RTX 5090 installed-wheel
的 CUDA graph 子集为 `7 passed, 6 deselected`，覆盖 stable output address、bucket
不兼容时 copy 前拒绝、paged-MLA raw replay、host reentrancy、cross-stream event
串行化和 canonical int64 positions。以下命令分别复现原生 graph 和最小调度器
lifecycle：

```bash
python -m pytest -o addopts= -ra -m cuda tests/test_cuda_graph.py
python benchmarks/cuda_graph.py --batch 32 --width 256 --warmup 5 --iterations 20
python benchmarks/continuous_batching.py \
  --requests 8 --prompt-length 8 --max-new-tokens 4 \
  --page-size 4 --num-pages 64 --max-batch-size 4
```

Graph benchmark 保留 eager/replay 原始样本但不做 speedup claim；scheduler benchmark
不启动模型 kernel，只记录请求/页状态迁移。完整环境见
[单卡证据快照](../../validation/single-gpu/2026-08-22-rtx5090-next-phase/README.md)。
