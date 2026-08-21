# 第九章：FP8 E4M3FN 与 INT8 前向量化实验

这一章把量化作为独立的 linear/GEMM 实验，而不是悄悄塞进 MLA 或 MoE。这样可以分别回答三件事：
量化后的值是什么、scale 沿哪个轴定义、原生 backend 到底执行了什么。当前实现只覆盖 forward，不提供
backward，也不声称使用 Tensor Core 或获得加速。

## 9.1 数据与 scale 契约

激活矩阵采用 `[M, K]`，每一行一个 FP32 scale；linear 权重采用
`[N, K] = [out_features, in_features]`，每个输出通道一个 FP32 scale。两者在物理上都由 axis 0
索引 scale，并沿 axis 1 的 `K` 维求绝对值最大值：

```math
s_i=\begin{cases}
1,&\max_j|x_{ij}|=0,\\
\max_j|x_{ij}|/q_{max},&\text{otherwise}.
\end{cases}
```

全零行使用 `scale=1`，避免产生零除；非零但极小的行把 scale 下限固定为 FP32 smallest normal，避免
scale 本身下溢成 0。所有输入必须是有限的、连续的 FP32 二维矩阵。返回的
`QuantizedMatrixMetadata` 是冻结 dataclass，明确保存 format、shape、源/value/scale/accumulator dtype、
scale index/reduction axis、连续行主序布局和饱和范围。

## 9.2 两种格式

### symmetric INT8

INT8 使用 `[-127, 127]`，不使用 `-128`，从而保持正负对称。归一化值使用 round-to-nearest-even 后饱和：

```math
q_{ij}=\operatorname{clamp}(\operatorname{round}(x_{ij}/s_i),-127,127).
```

payload dtype 是 `torch.int8`。

### FP8 E4M3FN

FP8 使用 E4M3FN 的有限范围 `[-448, 448]`。仓库把编码位保存在 `torch.uint8` payload 中，避免把格式
语义藏在隐式 cast 里；解码时按 `torch.float8_e4m3fn` 的位模式解释。NaN 编码不属于合法 payload。
这里的“FN”表示有限值范围与特殊编码不同于 IEEE 风格的 Inf/NaN 分配，不能和 E5M2 或其他 FP8
变体混用。

## 9.3 dequantized linear oracle

`dequantized_linear` 接受一份 per-row 激活和一份 per-output-channel 权重，计算：

```math
Y=(Q_x\odot s_x)(Q_w\odot s_w)^T.
```

输出和累加器均为 FP32。reference 路径先显式反量化，再调用 PyTorch matmul；原生路径用标量 CUDA
kernel 在同一数学边界内解码和累加。它不是 FP8/INT8 Tensor Core GEMM，也没有吞吐量承诺。

```python
import torch
from ds_flash_mla_moe import (
    dequantized_linear,
    quantize_activations,
    quantize_weights,
)

x = torch.randn(32, 64)
w = torch.randn(128, 64)
qx = quantize_activations(x, format="int8", backend="reference")
qw = quantize_weights(w, format="int8", backend="reference")
y = dequantized_linear(qx, qw, backend="reference")
```

`backend="cuda"` 只接受已构建扩展支持的格式和 CUDA 张量；不满足条件时直接报错。`auto` 才允许退回
reference。原生 quantize 为了拒绝 NaN/Inf，会在 eager 阶段检查有限性；已经量化的 linear raw op
不做 host round-trip，因此可以单独 capture/replay。二者都不接受 `requires_grad`。

## 9.4 可复核 benchmark

benchmark 把一次性的激活/权重量化放在计时区间外，只测 dequantized linear，并同时保存两类误差：

- native/selected backend 对同一量化 payload 的 paired dequantized reference 误差；
- 量化输出相对原始 FP32 linear 的量化误差。

```bash
python benchmarks/quantized_gemm.py \
  --device cpu --backend reference --format fp8_e4m3fn \
  --m 128 --n 128 --k 128 --warmup 2 --iterations 20
```

报告保留 raw latency samples、payload/scale/output 的分析字节数，并写入
`performance_claim=false`。只有 CUDA build、同流执行、graph replay、数值配对和 profiler 证据都来自
同一环境后，才适合讨论实现层面的性能。
