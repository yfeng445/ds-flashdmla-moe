# 2026-08-22 单 GPU next-phase 验证快照

这份快照记录 FA3 teaching forward、staged/fused/persistent MoE、CUDA Graph、
FP8/INT8 forward 与最小 continuous batching 在本轮实际得到的证据。它是
correctness/profiler 基线，不是生产性能报告。

## 环境与构建来源

- NVIDIA GeForce RTX 5090，compute capability 12.0
- WSL2 Linux x86_64
- Python 3.12.13
- PyTorch 2.10.0+cu128，CUDA runtime 12.8
- 本轮用于 RTX 5090 installed-wheel 复核的 artifact head：
  `b6383068802bd2927c29fae191d3f3e30eb651c3`
- [CUDA build run 32530555834](https://github.com/yfeng445/ai-infra-kernel-lab/actions/runs/32530555834)：
  success，产出 CPython 3.12 CUDA wheel
- [Reference tests run 32530556015](https://github.com/yfeng445/ai-infra-kernel-lab/actions/runs/32530556015)：
  success

wheel 由 hosted CUDA 12.8 development container 编译，随后不在本机重编译，直接安装到
WSL2 环境。从 `/tmp` 使用测试文件的绝对路径执行，避免 worktree `src`
覆盖 installed wheel。

## 原生正确性结果

上述 validated native artifact 上的综合 CUDA 回归：

```text
179 passed, 291 deselected in 35.92s
```

它覆盖 `tests/test_moe_backends.py`、`tests/test_attention_backends.py`、
`tests/test_cuda_graph.py`、`tests/test_quantization.py` 与
`tests/test_quantized_benchmarking.py` 中的 CUDA-marked 用例。为了让各个子系统的
边界可读，开发过程中还单独保留了以下聚焦结果：

| 子系统 | 实际结果 | 证据边界 |
| --- | --- | --- |
| FA1/FA2/FA3 attention backends | `82 passed, 46 deselected in 15.51s` | 数值、strict dispatch、tail/decode、current stream |
| CUDA Graph / paged MLA replay | `7 passed, 6 deselected in 15.39s` | stable address、metadata-before-copy、cross-stream serialization |
| FP8/INT8 | `6 passed, 59 deselected in 15.29s` | paired oracle、`K=513` exact lanes、current stream + graph replay |
| staged/fused/persistent MoE | `tests/test_moe_backends.py: pass (100%)` | 数值、tie/skew/empty expert、tail、current stream、raw opcheck |

这 179 项综合回归使用同一个 `b6383068` wheel；表中子集计数是在各功能
收敛时记录的聚焦运行，用于说明覆盖内容，不应与综合计数相加。
该 artifact 覆盖本轮 CUDA production source；之后的 protocol/docs-only commit 不应
被写成由该 wheel 验证，它们由后续 hosted reference workflow 与 CPU full suite 复核。

## MoE Kineto 与中间张量

同一 RTX 5090 上对 `T=128, D=64, D_h=128, E=8, K=2`、FP32 workload
分别选择三个显式 backend，执行两次 warmup 和三次 timed iteration。
该 capture 来自已 review 的 `a7b366e` MoE 实现；从该实现到 validated native artifact
之间，MoE/profiler production source 未改变。

| Backend | Observed aggregate custom-kernel activities | Observed device activities | Analytical intermediate bytes | Analytical metadata bytes | Observed allocator peak delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cuda_staged` | 66 | 676 | 281200 | 10864 | 1423360 |
| `cuda_fused` | 42 | 586 | 211080 | 6280 | 1423360 |
| `cuda_persistent` | 42 | 604 | 211088 | 6288 | 1423360 |

Kineto 数值是完整 profiling harness 中 key averages 的 aggregate activity
occurrence：它们不是物理 kernel launch count，parent operator 和 child kernel
也不可相加。中间张量值是按 shape/dtype 统计的 materialization model，不是
DRAM traffic。三个 backend 本次的 allocator peak delta 相同。

## FP8/INT8 结构化 smoke

原生 `backend=cuda`、`M=5, N=7, K=513`、一次 warmup 和两次采样：

| Format | Paired dequantized-reference max abs error | Max tolerance ratio |
| --- | ---: | ---: |
| INT8 | 1.71661376953125e-05 | 0.16004451299051473 |
| FP8 E4M3FN | 2.193450927734375e-05 | 0.1622789392032281 |

原始 timing 样本只是 smoke 运行的一部分，本快照不用它们比较性能。

## 复现命令

### Hosted workflow 与 artifact

`CUDA build` 在修改 `csrc/**`、`src/**`或 build 配置的 branch push 上自动
触发，`Reference tests` 在每次 push 上触发。对已有 run 可直接复核：

```bash
gh run view 32530555834
gh run view 32530556015
gh run download 32530555834 --name cuda-wheel --dir artifacts/cuda-wheel
```

### Installed-wheel CUDA 回归

```bash
REPO=/absolute/path/to/ai-infra-kernel-lab
PYTHON="$REPO/.venv/bin/python"

uv pip install --python "$PYTHON" --force-reinstall --no-deps \
  artifacts/cuda-wheel/*.whl

cd /tmp
"$PYTHON" -m pytest -o addopts= -ra -m cuda \
  "$REPO/tests/test_moe_backends.py" \
  "$REPO/tests/test_attention_backends.py" \
  "$REPO/tests/test_cuda_graph.py" \
  "$REPO/tests/test_quantization.py" \
  "$REPO/tests/test_quantized_benchmarking.py"
```

### MoE Kineto 与分析 inventory

```bash
cd "$REPO"
for backend in cuda_staged cuda_fused cuda_persistent; do
  "$PYTHON" benchmarks/moe.py \
    --mode kineto --device cuda --backend "$backend" --dtype float32 \
    --tokens 128 --model-dim 64 --hidden-dim 128 \
    --experts 8 --topk 2 --n-groups 1 --topk-groups 1 \
    --warmup 2 --iterations 3 \
    --output "benchmark-results/moe-${backend}-kineto.json"
done
```

每份报告的 `intermediate_bytes.inventories` 保留 staged/fused/persistent 的命名
buffer 清单。如果只需在 CPU 上复核分析字节数，可用相同 shape 运行
`--mode benchmark --device cpu --backend reference`；该命令不会产生 CUDA profiler
证据。

### Graph、scheduler 与量化 smoke

```bash
"$PYTHON" benchmarks/cuda_graph.py --batch 32 --width 256 \
  --warmup 5 --iterations 20
"$PYTHON" benchmarks/continuous_batching.py \
  --requests 8 --prompt-length 8 --max-new-tokens 4 \
  --page-size 4 --num-pages 64 --max-batch-size 4
"$PYTHON" benchmarks/quantized_gemm.py \
  --device cuda --backend cuda --format int8 \
  --m 5 --n 7 --k 513 --warmup 1 --iterations 2
"$PYTHON" benchmarks/quantized_gemm.py \
  --device cuda --backend cuda --format fp8_e4m3fn \
  --m 5 --n 7 --k 513 --warmup 1 --iterations 2
```

## 证据边界

- `performance_claim=false`；没有从上述 smoke 得出稳定 speedup。
- 本轮没有采集 Nsight Systems/Compute；不报告 occupancy、DRAM traffic、
  physical launch count 或计算/搬运 overlap。
- 快照只有一张 GPU；没有验证 NCCL/NVSHMEM 远程可见性、EP/TP transport、
  deadlock freedom 或多卡通信计算 overlap。
- `fa3` 是 asynchronous double-buffer teaching backend，不等于 production
  FlashAttention-3；FP8/INT8 kernel 是标量 correctness path，不等于 Tensor-Core GEMM。

同目录的 [`summary.json`](summary.json) 保留以上结果的机器可读版本。
