# AI Infra 面试整理：DeepSeek MLA、FlashAttention 与 MoE

本文根据根目录 `AI INFRA.ipynb` 重新组织，原 notebook 保持不变。目标不是背诵一套
“漂亮答案”，而是把项目事实、算法原理和仍需实验验证的内容分开，方便面试时经得住追问。

配套练习见 [AI Infra 高压模拟面试](infra-mock-interview.md)。建议先只看问题并录音作答，
再展开参考答案；不要先背答案。本文默认目标岗位是 **AI Infra / CUDA Kernel / 推理系统
工程师**。如果实际 JD 更偏训练系统、Serving 或通用 C++，应再调整问题权重。

回答项目题时统一使用下面的证据链：

```text
真实动作 -> 形成的系统能力 -> 为什么有价值 -> 可复核证据 -> 当前边界
```

只有自己能够脱稿推导、复现实验并修改的部分，才说“我实现了”。尚未确认个人所有权的
内容用“项目中实现了”；不能把 AI 辅助生成但自己尚未掌握的代码直接当作个人能力。

## 0. 先统一项目口径

### 30 秒版本

这是一个 correctness-first 的 DeepSeek MLA + MoE 学习与实现项目。它先用 PyTorch
reference 固定数值语义，再逐步替换为 CUDA 和分布式实现。目前包括：

- 分块 online-softmax attention，以及 FP16/BF16/FP32 CUDA forward/backward；
- MLA 的 naive/absorbed prefill、compressed static/paged cache decode，以及直接消费 latent
  cache 的 FP16/BF16/FP32 staged CUDA pipeline；
- DeepSeek 风格 grouped Top-K、route pack/combine、expert-major SwiGLU；
- FP32 CUDA-core 与 FP16 WMMA expert kernel；
- 两 rank Gloo reference，以及代码层面的 NCCL variable All-to-All/chunk pipeline。

### 更适合开口的 45 秒版本

我在做一个 correctness-first 的 DeepSeek MLA + MoE 算子项目。我的方法不是先写一个看起来
很快的 kernel，而是先用 PyTorch reference 固定 forward、backward、mask、路由身份和
determinism 语义，再把已验证边界接入 PyTorch dispatcher 和 CUDA。当前最完整的一条链路是
MLA：从 naive/absorbed 等价推导、compressed static cache，到 per-slot paged cache，再到由 native
query/cache projection、RoPE、absorbed attention 和 output projection 组成的 staged
FP16/BF16/FP32 CUDA pipeline。paged attention 直接按 block table 读取 latent pages，不先展开连续
K/V。它已经在单张
RTX 5090、CUDA 12.8 环境中完成原生构建和数值测试；out-of-place backward 仍是 reference
recompute，多卡 NCCL 与 NVSHMEM 也不能说成已经实机验证。这个项目主要证明的是我能把算法
语义、PyTorch 算子契约和 CUDA 执行边界连起来，而不是宣称已经做出生产级 FA3。

上面使用“我”之前，必须确认这些工作确实属于自己的可解释贡献；否则改成“这个项目”。

### 稳妥版与进取版定位

- **稳妥版**：具备 PyTorch reference、CUDA correctness kernel 和算子测试经验的 AI Infra
  候选人，能够解释 MLA/MoE 数据流及单卡实现边界。
- **进取版**：能够从代数推导到 PyTorch custom op 再到 CUDA kernel 完成验证闭环的 Kernel
  Engineer 候选人。只有能现场手推 absorbed MLA、解释 kernel 每块 shared memory、复现构建和
  修改 bug 时才使用此定位。
- **暂不使用**：“生产级高性能算子负责人”“多卡通信优化专家”“FA3/WGMMA 实现者”。当前
  证据不足以支撑这些称呼。

### 一分钟版本

项目的主线是减少两类开销。Attention 侧避免落盘完整的 `S×S` score/probability
矩阵；MLA 进一步把 K/V 压缩进 latent cache，并在需要时吸收或重建投影。MoE 侧则把
路由、dispatch、expert compute 和 combine 的数据身份与梯度契约先做正确，再研究通信计算
重叠。所有“更快”的结论都必须带硬件、shape、dtype、误差和原始 latency 样本。

### 绝对不要混说的内容

- 当前本地环境是单卡 RTX 5090，不是四卡 H100。
- 不能把 FlashMoE/NVSHMEM 的论文或原型设计说成此仓库已经实现。
- 没有保存可复核 benchmark 前，不说“训练 step 提升 1.2×”。
- NCCL chunk pipeline 已有软件协议，但物理 overlap 仍需 Nsight timeline 证明。
- 当前 MLA CUDA 是 correctness-first 的同 dtype staged pipeline，低精度 storage 配合 FP32
  accumulation；不是 FA3、TMA 或 WGMMA 实现。
- 普通 Attention 已支持 FP16/BF16 storage 与 FP32 accumulation，但仍是 row-wise scalar
  kernel；四组同 dtype FA4 正式快照中 FA4 三组 median 更低、native 一组更低，且原始样本
  波动明显，不能称为高性能实现或给出普适排序。

## 1. Online Softmax 与 FlashAttention

### 1.1 为什么普通 Softmax 不稳定

对一行分数 `x`，直接计算 `exp(x_i)` 可能溢出。稳定形式是：

```text
m = max_i x_i
p_i = exp(x_i - m) / sum_j exp(x_j - m)
```

Online softmax 不要求一次看到整行。对当前累计状态 `(m, l)` 和新 tile 的局部状态
`(m_t, l_t)`：

```text
m_new = max(m, m_t)
l_new = exp(m - m_new) * l + exp(m_t - m_new) * l_t
```

若还维护未归一化输出 `o`，旧输出也必须乘 `exp(m-m_new)`，当前 tile 的 `P@V`
乘 `exp(m_t-m_new)` 后再累加；扫完所有 K/V tile，最后除以 `l`。

### 1.2 手推时要能说清楚的状态

- row max `m`：当前扫描范围的最大 score；
- denominator `l`：以 `m` 为参考尺度的指数和；
- output accumulator `o`：同一尺度下尚未最终归一化的加权 V；
- mask：必须在 max/exp/sum 前生效，被 mask 的项等价于 `-inf`。

若整行全部被 mask，工程实现应返回全零概率/输出，不能产生 NaN。

### 1.3 FA1 和 FA2 的核心区别

两者都利用 tiling + online softmax 避免将完整 attention matrix 写回 HBM。面试时不要只说
“FA2 更快”，应回答：

- FA2 重新划分 work partition，常用 sliced-Q，让 warp 负责不同 query rows，减少 warp
  间 shared-memory 交换和同步；
- 减少非矩阵乘 FLOPs，如重复 rescale、bounds check 和 mask bookkeeping；
- 增加 sequence 方向并行度，改善 batch/head 较小时的占用率；
- 是否更快仍取决于 head dim、sequence、dtype、GPU 和 kernel 配置。

### 1.4 因果 mask 如何融合进 kernel

kernel 已知 tile 中元素的绝对 `(q_position, k_position)`。在 score 进入 row-max 更新前判断
`k_position <= q_position`；不满足的 score 按 `-inf` 处理，不需要物化完整 mask 矩阵。
增量 decode 使用 absolute positions 比简单的 tile-local 下标更稳妥。

## 2. MHA、GQA、MLA 与 KV Cache

### 2.1 三者分别共享或压缩什么

- MHA：每个 query head 有独立 K/V head，表达力强，KV cache 最大。
- GQA：多个 query heads 共享一组 K/V heads；MHA 与 MQA 之间的折中。
- MLA：先将输入压缩成共享 latent `c_t^KV`，缓存 latent 和 RoPE positional key，再通过
  up-projection 获得各 head 的非位置 K/V，或把投影权重吸收到 query/output 路径。

压缩比不能只看 latent rank：需要比较每 token 的
`r_kv + d_rope` 与展开后的 `H*(d_nope + d_rope + d_v)`，并计入 dtype。

### 2.2 Naive 与 absorbed MLA

设每个 head 的 K/V up-projection 为 `W_K^h`、`W_V^h`：

```text
naive:    k_t^h = W_K^h c_t,  v_t^h = W_V^h c_t
absorbed: q_latent^h = q_nope^h W_K^h
          score_content = q_latent^h · c_t
          latent_out = sum_t p_t c_t
          head_out = latent_out (W_V^h)^T
```

两条路径在代数上等价。Absorbed decode 不必为历史 token 展开完整 K/V，尤其适合
`query_length=1` 的带宽受限场景。

### 2.3 “KV 在线重建”和 KV cache 不是同一概念

- KV cache：保存历史 token 可复用的状态，避免 decode 每步重算历史投影。
- 在线重建：不将完整 per-head K/V 长期物化；从 latent 按 tile 临时重建，或使用 absorbed
  形式完全绕过显式重建。

本项目缓存 `latent kv + positional pe + absolute positions`。Functional append 用
`torch.cat` 会复制前缀；static cache 预分配连续 storage，只写新增条目；paged cache 则将 payload
存成 `[physical_page,page_offset,latent_dim]`，写入由 slot mapping 指定，读取由 block table 和
per-row sequence length 指定。

### 2.4 当前 CUDA MLA kernel 怎么讲

输入是 `q_nope/q_pe`、compressed `kv/pe`、absorbed `key_up/value_up` 和 absolute
positions。一个 block 负责一个 `(batch, head, query)` row：

1. 计算 `q_nope @ key_up` 得到 latent query；
2. 流式扫描 compressed cache，合并 content score 与 RoPE score；
3. 融合 causal mask 和 online softmax；
4. 直接累积 latent value；
5. 乘 `value_up` 写出 head output。

它没有写出 score matrix，也没有展开完整 K/V。目前 staged MLA pipeline 支持相同 dtype 的
FP16/BF16/FP32 storage；query/cache projection、RoPE、attention 和最终 `W_O` 已分别接入
native CUDA stage，
out-of-place backward 走可追踪的 absorbed reference recompute。

paged kernel 保持同一数学核心，但 key loop 先把 logical token 通过 block table 映射到物理页。
causal 比较仍使用 absolute position，而不是 slot id；因此页在显存中的顺序不会改变 attention
语义。当前实现支持 ragged row length 和尾页，但 page allocator、eviction、prefix sharing 与
continuous-batching scheduler 仍不属于这个仓库。

### 2.5 用维度证明 absorbed 等价

只说“把权重吸收到 Q 上”不够。对单个 head，设：

```text
q_nope: [d_nope]
c_t:    [r_kv]
W_K:    [d_nope, r_kv]
W_V:    [d_v, r_kv]
```

naive content score 为：

```text
q_nope · (W_K c_t) = (q_nope W_K) · c_t
```

右边先得到 `[r_kv]` 的 latent query。对 value：

```text
sum_t p_t (W_V c_t) = W_V (sum_t p_t c_t)
```

所以 softmax 概率不变时，可以先在 latent 空间做加权和，再做一次 value up-projection。
RoPE 部分不能随便吸收，因为旋转与位置相关；本实现单独缓存 `pe` 并单独计算 positional
score。面试官若问前提，要说清这是线性结合律/分配律，且 `W_K/W_V` 对 token 共享。

### 2.6 paged cache 贡献怎么按“动作—能力—证据—边界”表达

- **动作**：定义 physical page、global slot、block table 与 per-row length 契约，并实现 reference、
  per-slot CUDA projection write 和直接 paged absorbed-attention kernel。
- **系统能力**：decode 不再要求 batch 共享连续 cursor，也不用为了计算把 latent cache 先 gather
  成连续 K/V；同一调用拒绝重复/越界 slot，读取拒绝无效页、未写入和非单调 absolute position。
- **结果证据**：CPU/CUDA 对照覆盖 ragged batch、非连续物理页、覆盖写和 257-token 尾页；Kineto
  还暴露并推动消除了重复 D2H metadata validation。
- **个人边界**：这是 cache primitive 与 correctness kernel，不是完整 vLLM-style allocator/runtime，
  也没有 Nsight counter 支撑生产吞吐结论。

面试官若追问“为什么需要 positions”，回答 causal 语义属于逻辑 token 时间轴；physical slot
只是存储地址。若追问“为什么同一次写禁止重复 slot、跨调用却允许覆盖”，回答前者会产生并行
scatter race，后者是明确的生命周期操作，会一起替换 latent、RoPE 和 position 三类字段。

### 2.7 当前单卡性能证据怎么说

可以准确说：在 RTX 5090、PyTorch `2.10.0+cu128`、CUDA 12.8、FP32，配置
`B=1, S=128, D=128, H=4, r_kv=32`，5 次 warmup、20 次采样：

- fused CUDA prefill median 为约 `1.904 ms`；同一 benchmark harness 下 absorbed PyTorch
  路径约 `2.304 ms`，这个特定 shape 下约 `1.21x`；
- static-cache decode median 为约 `2.722 ms`；absorbed PyTorch 路径约 `3.038 ms`，约
  `1.12x`；
- 对 absorbed reference 的最大 tolerance ratio 分别约 `0.356` 与 `0.012`，均在设置的
  FP32 组合运算容差内。

必须紧接着补充：这是小 shape、单卡、source-dirty 开发态 smoke，不是生产模型吞吐，也没有
与 FlashMLA、FlashAttention、Triton、cuDNN SDPA 或 CUTLASS 做同等语义对比。decode 结果还
包含 Python/PyTorch 周边操作，不能单独归因于 CUDA core。报告位于 `benchmark-results/`；
没有这些限定时，不要只说“提升 1.21x”。

### 2.8 如果被问“为什么目前只快一点”

合理回答不是找借口，而是指出当前结构：一个 block 负责一个 `(batch, head, query)` row；
每个 key 都进行 block reduction 和同步；latent query、numerator 与 reduction buffer 放在 shared
memory；score/value 累积主要是标量 FP32 FMA；query projection、RoPE、attention 与 `W_O`
仍是多个独立 native launch。小 batch/decode 下 block 数少，kernel launch 与框架开销占比也高。

下一步优化顺序应先用 Nsight Compute/Systems 确认瓶颈，再考虑 warp-level reduction、减少
每 key 同步、向量化/低精度、更多 query/head 并行、projection fusion 或更适合 decode 的
persistent 组织。没有 profiler 数据时，只能称这些为假设，不能声称瓶颈已经确定。

## 3. DeepSeek MoE 与路由

### 3.1 标准数据流

```text
gate score -> group-limited Top-K -> dispatch/pack
           -> expert SwiGLU -> combine(weighted sum) -> token order
```

DeepSeek SwiGLU expert 是：

```text
W2(SiLU(W1(x)) * W3(x))
```

路由 correction/load bias 只影响选择，combine weight 仍从 unbiased score gather，避免把
负载控制直接混入模型概率语义。

### 3.2 为什么要 expert-major pack

token-major 的路由结果跨 expert 交错。按 expert 重排后，每个 expert 的 active rows 连续，
可直接用 offsets 调度 GEMM tile，避免为容量上限计算大量 padding row。必须保留 inverse
permutation/token identity，combine 才能恢复原 token 与 slot。

### 3.3 容量、溢出和训练/推理对齐

常见初始容量估计：

```text
capacity = ceil(tokens * top_k / experts * capacity_factor)
```

但不要声称本项目已有“0.1% 以下溢出”数据，除非拿出报告。可讨论的策略包括：动态 load
bias、增大 capacity factor、dropping、reroute 或退回非融合路径。每个策略在吞吐、数值语义、
负载均衡和实现复杂度间有不同权衡。

## 4. Expert Parallel、NCCL 与 NVSHMEM

### 4.1 普通 EP 为什么有两次 All-to-All

- dispatch：源 rank 将 token activation 发给拥有目标 expert 的 rank；
- combine/return：expert output 回到源 rank，再按 routing weight 累加到原 token。

每条 route 必须携带足够身份信息：源 rank、token index、top-k slot、expert id/owner 和
route weight。非线性 expert 之前不能错误地先乘 route weight。

### 4.2 FlashMoE 设计该如何准确表述

FlashMoE 风格原型用 actor/persistent-kernel 思路：Subscriber 观察远端 signal 并解码
packet，Scheduler 分配 ready task，Processor 执行通信 tile、expert FFN 或 combine。
NVSHMEM 提供 one-sided put/get 与 signal，不等于 NCCL All-to-All。

单个 token tile 生命周期：

```text
route/pack -> dispatch packet -> remote signal
-> subscriber decode -> scheduler assign -> processor expert compute
-> combine packet -> source signal -> combine/scale/accumulate
```

这部分是值得研究的目标，不是本仓库当前已经完成的 backend。

### 4.3 怎样证明 overlap

不能只用总 latency 猜。Nsight Systems 中要看到不同 stream/通信轨道在时间上重叠，并比较
serialized baseline；同时观察 SM Active、copy/NVLink 活跃度和关键依赖。Nsight Compute
用于看单 kernel 的 occupancy、memory throughput、tensor-core utilization 和 stall reason。

## 5. GPU 内存层次与 shared memory

从线程到系统可这样回答：

- register：线程私有，最快，过多会降低 occupancy 或 spill；
- shared memory / L1：SM 内，shared memory 由同 CTA 线程显式协作，生命周期随 block；
- L2：全 GPU 共享的 on-chip cache；
- HBM：设备全局内存，容量/延迟/带宽与片上存储不同；
- PCIe：CPU-GPU 或 GPU-GPU 通路；
- NVLink：更高带宽 GPU 互连；
- NVSwitch：连接多个 NVLink endpoint 的交换 fabric。

常见追问：bank conflict、coalescing、shared-memory 容量与 occupancy 权衡、为什么 tile 要加
padding、为什么 `__syncthreads()` 必须由整个 block 一致到达。

## 6. 手写题模板

### 6.1 多头注意力（PyTorch）

```python
def mha(x, wq, wk, wv, wo, heads, mask=None):
    batch, seq, model = x.shape
    dim = wq.shape[0] // heads
    q = torch.nn.functional.linear(x, wq).view(batch, seq, heads, dim).transpose(1, 2)
    k = torch.nn.functional.linear(x, wk).view(batch, seq, heads, dim).transpose(1, 2)
    v = torch.nn.functional.linear(x, wv).view(batch, seq, heads, dim).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) / dim**0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = (probs @ v).transpose(1, 2).contiguous().view(batch, seq, heads * dim)
    return torch.nn.functional.linear(out, wo)
```

要主动说明 shape、softmax 用 FP32、boolean mask 语义，以及全 mask row 的处理。

### 6.2 1,240,000 元素 reduce

若每 block 256 threads、每 thread 先加载两个元素：

```text
items_per_block = 512
grid = ceil(1,240,000 / 512) = 2,422
```

第一阶段每 block 输出一个 partial，因此还需后续 kernel 递归 reduce 2,422 个 partial，或由
最后阶段执行。边界加载置零；树形规约每轮同步。更成熟实现会用 warp shuffle 减少 shared
memory 和同步，并使用 grid-stride loop 控制 block 数。

### 6.3 Transformer 前向

Pre-norm block：

```text
x -> RMSNorm -> Attention -> residual add
  -> RMSNorm -> FFN/MoE -> residual add
```

回答时把 dense/sparse layer 的区别放在 FFN：dense FFN 每 token 都过同一组权重；MoE
通过 gate 只激活 top-k routed experts，另可有 shared experts。

## 7. 分布式训练常见追问

### DDP vs FSDP

- DDP：每 rank 常驻完整参数、梯度和 optimizer state；反向 All-Reduce 梯度。
- FSDP：参数、梯度和 optimizer state 分片；计算某 wrapped module 前 All-Gather 参数，反向
  Reduce-Scatter 梯度。省显存但通信更频繁、调度更复杂。

### PagedAttention vs FlashAttention

- FlashAttention 优化 attention 计算本身的 IO，核心是 tiling + online softmax；训练与
  prefill 常见，decode kernel 也可采用相关思想。
- PagedAttention 主要解决 serving 中动态 KV cache 的分页分配、碎片与共享；它是 cache
  管理/访问布局方案，不是 softmax 算法的替代品。

### vLLM、DeepSpeed 各自重点

- vLLM：推理 serving、continuous batching、KV cache 管理与吞吐。
- DeepSpeed：训练系统、ZeRO 状态分片、并行和显存管理；也包含推理能力，但面试中先说
  清主定位。

## 8. C++ 基础：左值、右值与参数传递

- 左值有稳定身份、可取地址；右值通常是临时值或即将被移动的对象。
- `T&` 绑定可修改左值，`const T&` 可绑定左值和临时量，`T&&` 绑定右值。
- 大对象只读参数通常传 `const T&`；需要取得所有权时可按值传入再 move，或提供 `T&&`；
  小型标量按值最清楚。
- 拷贝构造复制资源，移动构造转移资源；移动后的对象仍有效但状态未指定。

## 9. 反问自己：每个性能结论需要哪些证据

回答任何“快多少”前至少准备：

- GPU 型号、数量、互连和 power/clock 条件；
- CUDA、driver、PyTorch 与编译架构；
- batch/sequence/head/rank/expert/top-k/dtype；
- baseline 名称与是否同等数学语义；
- warmup、迭代数、median/p95 和原始样本；
- output、gradient、loss 的误差/一致性；
- Nsight timeline 或 kernel metrics（若声称 overlap 或瓶颈改善）。

没有这些证据时，使用“实现了”“验证了数值一致性”“设计目标是”这样的措辞，不使用
“高性能”“1.2×”“无损”等结论。

## 10. 强主张的表达校准

| 容易失分的说法 | 建议说法 | 当前证据 | 个人边界 / 风险 |
| --- | --- | --- | --- |
| “我实现了 FlashMLA” | “项目实现了一个 correctness-first、FP16/BF16/FP32 staged absorbed MLA CUDA pipeline” | 原生 CUDA 源码、dispatcher、CUDA 测试和 smoke 报告 | 不是官方 FlashMLA，也没有 FA3/TMA/WGMMA |
| “MLA 快了 1.21x” | “在 RTX 5090 的一个 B1/S128 FP32 smoke shape 下，median 相对项目 absorbed baseline 为约 1.21x” | 两份同 harness benchmark JSON | shape 很小；非生产模型；开发态源码 |
| “做了多卡通信计算重叠” | “实现了 NCCL-only chunk/async 软件协议；物理 overlap 尚待多卡 timeline 验证” | async All-to-All 与 chunk pipeline 代码；Gloo 两 rank语义测试 | 本地只有单卡，不能声称多卡性能 |
| “支持反向” | “CUDA forward 的一阶梯度通过可追踪 absorbed reference recompute 获得” | custom-op autograd 注册及梯度对照测试 | 不是 fused native MLA backward |
| “支持任意输入” | “高层 API 可 fallback；原生 MLA 当前限定同 dtype CUDA FP16/BF16/FP32、无显式 mask，并提供 paged latent-cache primitive” | 输入契约、paged 边界测试和 CUDA 对照 | 不支持 mixed dtype、任意 mask、完整 page allocator/continuous batching 或生产尺寸调优 |
| “低精度 Attention 已经很快” | “native Attention 已完成 FP16/BF16 correctness contract；正式四组 paired 快照里 FA4 三组 median 更低、native 一组更低，样本波动使其不能外推” | CUDA forward/backward 测试与可选 FA4 matrix | 单机小 shape；尚无 Nsight/CUTLASS 或生产调度 |
| “实现了 NVSHMEM FlashMoE” | “实现的是 symmetric-buffer 分析模型；NVSHMEM actor backend 是研究目标” | `symmetric_memory.py` 与讲义 | 没有 one-sided runtime backend |

## 11. 面试官评分视角

一次项目回答至少应让面试官听到下面五件事：

1. **问题**：为什么 expanded KV、完整 score matrix 或 route padding 会产生开销；
2. **选择**：为什么采用 absorbed MLA、online softmax、expert-major pack 等方案；
3. **实现**：张量维度、block/warp 工作划分、shared memory 和同步点；
4. **验证**：reference、forward/backward、边界输入、stream、determinism、benchmark；
5. **边界**：尚未实现或尚未实机验证的部分，以及下一步如何验证。

只讲概念通常像读论文；只讲 kernel 像没有系统视角；只报数字而不讲 baseline/shape 像营销。
配套模拟题会围绕这五层连续追问。
