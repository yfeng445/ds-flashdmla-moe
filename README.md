# DeepSeek Flash MLA + MoE

A correctness-first study and implementation of the two sparse-compute building
blocks used by DeepSeek-style language models:

- Multi-head Latent Attention (MLA), with FlashAttention-inspired kernels.
- Group-limited Mixture-of-Experts (MoE), including a correctness-first
  Expert Parallel reference over `torch.distributed`.

The project grows out of the INFO 7375 *High Performance Computing for AI*
course. Course exercises are not copied into the library wholesale. Instead,
each optimized operator is developed from an executable PyTorch specification,
then tested and benchmarked before it becomes part of the supported API.

The textbook begins with GPU execution, tiled GEMM, memory reuse, and
producer-consumer pipelines before applying those ideas to attention and MoE.

## Project status

| Area | Reference | Tested | CUDA | Distributed |
| --- | --- | --- | --- | --- |
| Tiled GEMM teaching model | materialized + explicit tiles | CPU | FP32 16×16 source | no |
| Scaled dot-product attention | yes | CPU | FP32 forward/backward source | no |
| Blockwise online attention | yes | CPU | FP32 forward/backward source | no |
| DeepSeek grouped Top-K gate | yes | CPU | FP32 sigmoid source | replicated in EP |
| DeepSeek SwiGLU MoE | token-loop + packed | CPU | FP32 CUDA-core + FP16 WMMA active-row experts | 2-rank Gloo reference |
| MLA prefill/decode | naive + absorbed | CPU | not yet | no |
| Expert parallelism | variable All-to-All | CPU forward/backward | native route + async chunk pipeline | Gloo verified; NCCL CI pending |
| One-sided EP layout | symmetric-buffer cost model | CPU | no NVSHMEM backend | analytical only |

Files under `csrc/experimental/` are teaching prototypes. They are intentionally
excluded from the importable wheel and supported API, and must not be treated as
validated kernels.

The CUDA MoE primitives remain correctness-first. The grouped router and native
route pack/combine currently accept FP32 only. The router uses a dense PyTorch
gate projection followed by a native group-limited selector;
exact score ties prefer the smaller group/expert id. Route packing uses atomic
row assignment within each destination/expert segment, and combine uses atomic
accumulation by token; deterministic mode selects the PyTorch route reference.
The native expert-major SwiGLU schedules independent `16x16x16` tiles from each
expert's row segment, handles empty experts and arbitrary tails, and keeps a
traceable PyTorch-recompute backward. FP32 uses shared-memory CUDA-core GEMM;
FP16 uses one converged WMMA warp per output tile, FP32 accumulators, and an FP16
materialized hidden state. Neither path has asynchronous copies, Hopper WGMMA/TMA,
or profiler-driven tuning against cuBLAS/CUTLASS.

## Quick start

The current supported layer is pure PyTorch and works on CPU or CUDA tensors.
Python 3.10+ and PyTorch 2.4+ are required.

```bash
python -m pip install -e '.[test]'
pytest
```

```python
import torch
from ds_flash_mla_moe import blockwise_attention

q = torch.randn(2, 4, 128, 64)
k = torch.randn(2, 4, 128, 64)
v = torch.randn(2, 4, 128, 64)

out = blockwise_attention(q, k, v, causal=True, block_size=32)
```

`flash_attention_forward(..., backend="auto")` selects the optional CUDA
kernel only for its currently supported input contract and otherwise falls
back to the PyTorch specification. Build the native extension on a CUDA machine
with:

```bash
python -m pip install '.[test,cuda-build]'
DS_FLASH_BUILD_CUDA=1 python -m pip install --no-build-isolation .
pytest -ra
```

The first kernel accepts contiguous FP32 tensors shaped `[B, H, S, D]`, supports
right-aligned causal attention and `S_q != S_k`, but does not yet support
explicit masks or low-precision inputs. The correctness-first native backward
uses atomic accumulation for `dK/dV`; deterministic mode and higher-order
gradients use the analytic PyTorch specification. `backend="cuda"` rejects
unsupported input contracts instead of silently changing semantics.

`tiled_gemm(..., backend="cuda")` exposes the first shared-memory teaching
kernel: contiguous FP32 rank-2 matrices, fixed 16x16x16 tiles, arbitrary M/N/K
tails, optional `alpha * A @ B + beta * C` epilogue, and an analytic PyTorch
backward. It is a correctness milestone rather than a cuBLAS competitor.

`swiglu_experts_expert_major(..., backend="cuda")` accepts contiguous FP16 or FP32
expert-major rows, an int64 offsets vector, and local `[E_l,D_h,D]`/
`[E_l,D,D_h]` weights. It launches on PyTorch's current stream, executes no
global capacity-padding rows, and uses a registered reference-recompute backward.
Each non-empty expert receives its own row-tail tile, whose inactive lanes are
reported separately from padded-baseline rows. The explicit CUDA backend fails
loudly on an unsupported device, dtype, or layout. FP16 requires compute
capability 7.0 or newer; BF16 is not currently supported by this native kernel.

`grouped_topk(..., backend="cuda")` accepts contiguous FP32 CUDA activations,
gate weights, and optional correction bias with sigmoid scoring. The bias changes
selection only; returned weights and their gradients come from unbiased scores.
The current one-thread-per-token selector uses serial candidate scans, so it
establishes routing, stream, dispatcher, and autograd semantics rather than a
production-performance claim.

## Repository layout

```text
.
├── csrc/
│   ├── attention/                # native CUDA operator source
│   ├── gemm/                     # fixed-tile CUDA teaching kernel
│   ├── moe/                      # route and active-row SwiGLU kernels
│   └── experimental/attention/   # unverified course-era CUDA prototypes
├── benchmarks/                   # structured latency and environment reports
├── docs/                         # textbook-style notes and reading guide
├── examples/                     # runnable reference examples
├── src/ds_flash_mla_moe/         # supported Python specifications
└── tests/                        # numerical and semantic contracts
```

## Numerical contracts

- GEMM computes `alpha * A @ B + beta * C`; `C` may be omitted only when
  `beta=0`. The reference widens FP16/BF16/FP32 accumulation to FP32.
- Attention inputs use `[..., query_length, head_dim]`; values may have a
  different final dimension.
- Boolean attention masks use `True` for positions that participate in
  attention. Causal masking is right-aligned when query and key lengths differ.
- Reference attention and MoE reductions accumulate in FP32 for FP16/BF16
  inputs and return the value/input dtype. FP16 routed experts explicitly round
  the materialized SwiGLU hidden state to FP16 before the FP32-accumulated down
  projection, matching the native WMMA stage boundary.
- The materialized attention backward specification exposes analytic
  `dQ/dK/dV` and is checked against autograd, gradcheck, and gradgradcheck.
- The DeepSeek gate uses its correction bias only to choose experts. The
  returned routing weights are gathered from the unbiased scores.
- Experts use the DeepSeek SwiGLU form `W2(SiLU(W1(x)) * W3(x))`.
- Expert Parallel dispatch sends unweighted activations; routing weights are
  applied after nonlinear expert outputs return to their source rank.
- The distributed reference supports uneven and empty local token shards and
  differentiates through both variable All-to-All exchanges. Replicated router
  and shared-expert gradients still require an external data-parallel reduction.
- The symmetric-memory helper models a per-PE
  `[peer, round, buffer, local_expert, capacity, feature]` layout and its
  route-cell overflow/storage cost. It does not allocate NVSHMEM memory or
  imply that a one-sided backend has been implemented.

## Development order

1. Lock down mathematical references and adversarial shape tests.
2. Replace the attention prototypes with a verified forward kernel, followed by
   backward.
3. Implement MLA prefill and compressed-cache decode.
4. Implement the unfused MoE route/dispatch/expert/combine pipeline.
5. Migrate the verified Gloo Expert Parallel protocol to NCCL, then explore
   fusion and overlap without changing its route-identity contract.

Performance claims will be added only with reproducible benchmark inputs,
hardware/software metadata, and raw results.

Benchmark CLIs emit self-describing JSON reports. The tail-safe tiled GEMM
teaching reference can run as a normal Python process:

```bash
python benchmarks/gemm.py --device cpu --dtype float64 \
  --implementation tiled --m 37 --n 29 --k 23 \
  --tile-m 16 --tile-n 8 --tile-k 7 --iterations 5
```

Attention uses the same report conventions:

```bash
python benchmarks/attention.py --device cpu --backend reference \
  --query-length 128 --key-length 128 --iterations 20
```

MLA reports separate prefill/decode and attention-only/cache-update timing:

```bash
python benchmarks/mla.py --device cpu --dtype float64 \
  --implementation absorbed --workload decode_with_static_write \
  --sequence-length 128 --iterations 10 \
  --output benchmark-results/mla-decode.json
```

Use `decode_with_append` with the same shape to expose the linear prefix-copy
cost of a functional cache baseline.

Expert-major SwiGLU has a standalone benchmark whose comma-separated counts
make skew and empty experts explicit:

```bash
python benchmarks/experts.py \
  --device cpu --backend reference --dtype float64 \
  --expert-counts 17,0,5,31 --model-dim 64 --hidden-dim 128 \
  --warmup 2 --iterations 20 --backward \
  --output benchmark-results/experts-skewed.json
```

On a native CUDA build, use `--device cuda --backend cuda --dtype float32` for
the CUDA-core path, or `--dtype float16` for WMMA. The report preserves the
selected forward engine, multiplicand/accumulator/hidden dtypes, per-expert counts, ideal
active-row and padded-baseline FLOPs, grouped row/output tile counts, row-tail
lane utilization, raw latency samples, and output/gradient error.

The grouped router also has an isolated benchmark. It records exact selected
indices, output/gradient error, the full per-expert load vector, and projection
FLOPs separately from selection candidates:

```bash
python benchmarks/router.py \
  --device cpu --backend reference --dtype float64 \
  --tokens 128 --model-dim 64 --experts 8 --topk 2 \
  --n-groups 4 --topk-groups 2 --hot-expert-bias 0.5 \
  --warmup 2 --iterations 20 --backward \
  --output benchmark-results/router.json
```

On a native build, use `--device cuda --backend cuda --dtype float32`. The
reported TFLOP/s-equivalent counts only the dense gate projection; sigmoid,
group scoring, selection, gather, and normalization remain outside that count.

The Expert Parallel validator is launched with `torchrun`; rank zero writes a
single report containing the route-count matrix, per-rank metadata, global
maximum latency samples, load-skew/capacity diagnostics, an overlap contract,
a symmetric-buffer footprint model, and
reference errors. It also keeps the full `[iteration, rank]` latency matrix so
stragglers are not hidden behind the rank maximum:

```bash
torchrun --master-addr=127.0.0.1 --master-port=29572 \
  --nproc-per-node=2 benchmarks/expert_parallel.py \
  --backend gloo --router-backend reference --route-backend reference --dtype float64 \
  --expert-backend padded \
  --tokens-per-rank 3 --token-skew 1 \
  --model-dim 4 --hidden-dim 5 --shared-experts 1 --experts 4 --topk 1 \
  --hot-expert-bias 100 --capacity-factor 1 --symmetric-cell-capacity 2 \
  --warmup 0 --iterations 1 --backward \
  --output benchmark-results/gloo-ep.json
```

The same validator accepts `--backend nccl --router-backend cuda
--route-backend cuda --expert-backend cuda` on a multi-GPU host. This verifies
the unfused NCCL protocol with the native router, route, and active-row expert
kernels in FP32. For FP16, select `--router-backend reference --route-backend
reference --expert-backend cuda --dtype float16`: communication stays NCCL while
only expert-major packing and expert compute are native, with WMMA used for the
three projections. NCCL runs may additionally set `--pipeline-chunks N` with
`N>1`; each peer segment is split independently, dispatch and restore use
asynchronous collectives, and expert compute for one chunk can overlap another
chunk's communication. The report marks the asynchronous chunk pipeline as executed and
stores the combined `pipelined_core` stage, while leaving physical hardware overlap
unverified until profiler evidence is available; use a separate `N=1` run as the serialized
baseline. This remains an unfused research pipeline, not a production backend.
`--hot-expert-bias` affects selection only and makes an expert-skew
stress case reproducible; `--capacity-factor` models drop/padding but does not
drop benchmark routes. `--symmetric-cell-capacity` independently models a
per-source/per-expert cell limit for the symmetric layout; it is not the same
capacity policy. `--shared-experts N` adds the replicated shared branch as one
SwiGLU with effective hidden dimension `N * hidden_dim`; its latency and FLOPs
are reported separately from routed expert compute.

## Learning material

The [`docs/`](docs/index.md) directory is organized as a compact textbook. It
starts from stable online softmax, derives MLA and DeepSeekMoE, and then maps the
equations to CUDA and distributed execution. It also contains a curated reading
list rather than copies of course PDFs or books.

## Attribution

The operator semantics are checked against the public DeepSeek-V3 reference
implementation. The attention validation strategy follows the FlashAttention
project: compare outputs and gradients against a high-precision framework
reference over varied shapes, masks, and dtypes.

This repository is licensed under the [MIT License](LICENSE). External papers,
books, and projects retain their own licenses.
