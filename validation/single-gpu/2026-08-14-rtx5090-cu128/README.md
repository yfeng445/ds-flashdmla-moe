# Single-GPU validation snapshot

This directory records one local correctness and smoke-latency run of the native CUDA
extension. It is evidence that the fixed workloads executed successfully; it is not a
performance comparison or a sustained self-hosted CI result.

## Environment

- NVIDIA GeForce RTX 5090, compute capability 12.0
- WSL2 Linux x86_64
- Python 3.12.13
- PyTorch 2.10.0 with CUDA 12.8
- Native extension loaded with all nine CUDA dispatcher kernels

The complete CUDA-aware test suite passed with `343 passed`. Each JSON report contains the
exact configuration, environment metadata, numerical verification, latency summary, and all
20 raw post-warmup samples.

| Workload | Median latency (ms) | Verification |
| --- | ---: | --- |
| GEMM, `127x63 @ 63x95`, FP32 | 0.181248 | max tolerance ratio 0.055916 |
| Causal attention, `B=1,H=2,S=128,D=64`, FP32 | 0.121696 | passed against the PyTorch reference |
| MLA prefill, `B=1,S=128,H=4`, FP32 | 2.506240 | max tolerance ratio 0.356430 |
| MLA static-cache decode, prefix 128, FP32 | 2.296320 | max tolerance ratio 0.012265 |
| Expert-major SwiGLU, counts `17,0,5,31`, FP32 | 0.942144 | max tolerance ratio 0.087917 |
| Expert-major SwiGLU, counts `17,0,5,31`, FP16 WMMA | 1.078240 | max tolerance ratio 0.020092 |
| Grouped Top-K router, 256 tokens and 8 experts, FP32 | 0.907840 | exact indices; weights and gradients passed |

The commands and fixed shapes mirror `.github/workflows/cuda-tests.yml`. Re-run that workflow
on a registered single-GPU runner before treating this snapshot as continuously reproducible.
