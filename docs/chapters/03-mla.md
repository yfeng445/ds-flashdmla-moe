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

同一个“MLA latency”至少包含五种不同实验：

| workload | query 长度 | cache 操作是否计时 |
| --- | ---: | --- |
| `prefill_attention` | `S` | cache 已构建，只测 attention |
| `prefill_with_cache` | `S` | 重建 `S` 个 latent entries |
| `decode_attention` | `1` | cache 已含当前 token |
| `decode_with_append` | `1` | append 当前 token 后做 attention |
| `decode_with_static_write` | `1` | 原位写入当前 token 后做 attention |

五者不能放在同一列直接比较。特别是函数式 cache 使用 `torch.cat`，一次 append
会读取旧 prefix 并写出完整新 cache，拷贝量随上下文长度线性增长。这是易验证的语义实现，
不是生产 decode cache。`MLAStaticCache` 则预分配 KV、RoPE 和 position storage，维护
一个 valid cursor，并将新 chunk 原位写入 `[cursor:cursor+L]`。storage 地址在 decode
过程中保持不变，单 token 的 cache 写入量为：

```math
B(R_{kv}+D_{rope})e + 8\quad\text{bytes}.
```

这把 cache 搬运从随 `S` 增长的 `O(S)` 降为 `O(1)`；attention 读取历史 cache 的成本仍然
是 `O(S)`。当前 reference 为整个 batch 维护一个共享 cursor，适合等长 batch 和实验验证。
生产 continuous batching 还需为每个 batch slot 分别维护有效长度、请求生命周期和 page
映射。静态写入会修改 storage，因此 API 明确限定在 `torch.no_grad()` 或
`torch.inference_mode()` 下使用，而不伪装成可微操作。

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
以下 native FP32 算子：

1. direct query projection + RoPE；
2. LoRA query projection + RMSNorm + RoPE；
3. latent KV projection + RMSNorm + RoPE；
4. projection 后直接写入预分配 static cache；
5. absorbed score、causal mask、online softmax、latent value reduction；
6. head output projection。

`backend="cuda"` 要求这六个阶段全部可用，不再只替换 attention core。out-of-place 算子的
backward 通过可追踪 PyTorch specification 重计算，以便先固定一阶梯度语义；static cache
写入会修改 storage，因此仍严格限定为 inference-only。该实现证明了完整 prefill/decode
数据流和 dispatcher 契约，但还不是生产内核：当前只有 FP32，prefill/decode 尚未使用各自
专用调度，也没有 paged cache、continuous-batching slot 生命周期或 profiler 驱动的融合。

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
报告保存在 `validation/single-gpu/2026-08-14-rtx5090-cu128/`。该次快照中，absorbed-attention
分别占 prefill/decode custom-operator self-device 时间的 67.2% 和 81.3%，因此下一轮 CUDA
工作应先用 Nsight Compute 分解它的访存、occupancy 和指令瓶颈，再决定专用 prefill/decode
调度或融合边界。
