# Single-GPU validation snapshot

This directory records one local correctness and smoke-latency run of the native CUDA
extension. It is evidence that the fixed workloads executed successfully; it is not a
performance comparison or a sustained self-hosted CI result.

## Environment

- NVIDIA GeForce RTX 5090, compute capability 12.0
- WSL2 Linux x86_64
- Python 3.12.13
- PyTorch 2.10.0 with CUDA 12.8
- Native extension loaded with all 14 CUDA dispatcher kernels

The complete CUDA-aware test suite passed with `354 passed`. Each JSON report contains the
exact configuration, environment metadata, numerical verification, latency summary, and all
20 raw post-warmup samples. Native and baseline reports use identical inputs and shapes.

| Workload | Native median (ms) | Baseline | Baseline median (ms) | Native / baseline |
| --- | ---: | --- | ---: | ---: |
| GEMM, `127x63 @ 63x95`, FP32 | 0.181248 | PyTorch GEMM / cuBLAS | 0.219136 | 0.827 |
| Causal attention, `B=1,H=2,S=128,D=64`, FP32 | 0.121696 | PyTorch SDPA | 0.063584 | 1.914 |
| MLA attention-only prefill, prior partial pipeline, FP32 | 2.506240 | absorbed PyTorch | 1.678608 | 1.493 |
| MLA static-cache decode, prior partial pipeline, FP32 | 2.296320 | absorbed PyTorch | 2.588192 | 0.887 |
| MLA full prefill with cache projection, FP32 | 1.155536 | absorbed PyTorch | 2.636864 | 0.438 |
| MLA full static-cache decode, FP32 | 1.028400 | absorbed PyTorch | 2.058480 | 0.500 |
| Expert-major SwiGLU, counts `17,0,5,31`, FP32 | 0.942144 | padded PyTorch | 2.729504 | 0.345 |
| Expert-major SwiGLU, counts `17,0,5,31`, FP16 WMMA | 1.078240 | padded PyTorch | 3.930464 | 0.274 |
| Grouped Top-K router, 256 tokens and 8 experts, FP32 | 0.907840 | PyTorch reference | 1.229312 | 0.738 |

A ratio below one means the native median was lower in this one run; a ratio above one means
the baseline median was lower. These single-shape samples are diagnostic inputs for profiling,
not general speedup claims.

## Representative shape matrix

[`operator-matrix-representative.json`](operator-matrix-representative.json) extends the
fixed snapshots to 20 independently verified pairs with regular, tail, decode, and skew shapes.
Every nested native and baseline report retains 20 post-warmup samples.

| Family | Cases | Coverage | Baseline | Native lower | Baseline lower | Native / baseline range |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| GEMM | 4 | regular, decode, tails | PyTorch/cuBLAS | 3 | 1 | 0.076–1.957 |
| Attention | 4 | prefill, decode, tails | PyTorch SDPA | 0 | 4 | 1.272–5.559 |
| MLA | 5 | prefill, decode, direct/LoRA, tails | absorbed PyTorch | 5 | 0 | 0.369–0.529 |
| Experts | 4 | FP32/FP16, tails, skew/empty experts | padded PyTorch | 4 | 0 | 0.037–0.096 |
| Router | 3 | regular, tail, hot-expert skew | PyTorch reference | 3 | 0 | 0.205–0.226 |

These ratios are only paired observations for this machine and configuration. Baselines and
operator boundaries differ across families, so the aggregate ratio statistics in the JSON are
unweighted descriptors, not an overall speedup or a cross-family ranking.

The commands and fixed shapes mirror `.github/workflows/cuda-tests.yml`. Re-run that workflow
on a registered single-GPU runner before treating this snapshot as continuously reproducible.
