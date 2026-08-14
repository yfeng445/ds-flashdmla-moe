# Single-GPU validation snapshot

This directory records one local correctness and smoke-latency run of the native CUDA
extension. It is evidence that the fixed workloads executed successfully; it is not a
performance comparison or a sustained self-hosted CI result.

## Environment

- NVIDIA GeForce RTX 5090, compute capability 12.0
- WSL2 Linux x86_64
- Python 3.12.13
- PyTorch 2.10.0 with CUDA 12.8
- Native extension loaded with all 16 CUDA dispatcher kernels

The current CUDA-aware test suite passed with `520 passed`. Each paired latency JSON contains
the exact configuration, environment metadata, numerical verification, latency summary, and
all 20 raw post-warmup samples. Native and baseline reports use identical inputs and shapes.

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
| GEMM | 4 | regular, decode, tails | PyTorch/cuBLAS | 2 | 2 | 0.122–2.650 |
| Attention | 4 | prefill, decode, tails | PyTorch SDPA | 1 | 3 | 0.380–7.349 |
| MLA | 5 | prefill, decode, direct/LoRA, tails | absorbed PyTorch | 5 | 0 | 0.059–0.258 |
| Experts | 4 | FP32/FP16, tails, skew/empty experts | padded PyTorch | 4 | 0 | 0.060–0.212 |
| Router | 3 | regular, tail, hot-expert skew | PyTorch reference | 3 | 0 | 0.228–0.299 |

These ratios are only paired observations for this machine and configuration. Baselines and
operator boundaries differ across families, so the aggregate ratio statistics in the JSON are
unweighted descriptors, not an overall speedup or a cross-family ranking.

## Optional FlashAttention-4 pairs

[`operator-matrix-fa4.json`](operator-matrix-fa4.json) records four exact-dtype Attention pairs
against the optional `flash-attn-4==4.0.0b22` backend. Both sides use the same generated inputs,
causal semantics, warmup, iteration count, and numerical reference. The timed FA4 boundary
includes its BHSD/BSHD layout adapter and output copy.

| Case | Dtype | Native median (ms) | FA4 median (ms) | Native / FA4 | Lower median |
| --- | --- | ---: | ---: | ---: | --- |
| Prefill, `B=1,H=4,S=128,Dq=Dv=64` | BF16 | 0.123712 | 0.015472 | 7.996 | FA4 |
| Tail prefill, `B=2,H=3,S=127,Dq=40,Dv=48` | FP16 | 0.125920 | 0.043712 | 2.881 | FA4 |
| Decode, `B=2,H=4,Sq=1,Sk=128,Dq=Dv=64` | BF16 | 0.121680 | 0.006912 | 17.604 | FA4 |
| Tail decode, `B=1,H=3,Sq=7,Sk=129,Dq=40,Dv=24` | FP16 | 0.123680 | 0.166176 | 0.744 | native |

All eight outputs passed the materialized FP32-reference check. Three FA4 medians and one native
median were lower in this particular run, but several raw samples show substantial desktop-WSL
variation. The JSON is therefore a reproducible correctness and profiling baseline, not evidence
that either implementation wins generally.

## Staged MLA low-precision pairs

[`operator-matrix-mla-low-precision.json`](operator-matrix-mla-low-precision.json) records four
same-dtype staged MLA pairs. The native path and absorbed PyTorch baseline receive identical
inputs, weights, cache configuration, warmup, iteration count, and verification policy. Both
implement the public stage writeback semantics for FP16/BF16 storage; the native kernels use
FP32 for projection, normalization, RoPE, online softmax, and value accumulation internally.

| Case | Dtype | Native median (ms) | PyTorch absorbed median (ms) | Native / baseline | Lower median |
| --- | --- | ---: | ---: | ---: | --- |
| Prefill, `B=1,S=128,H=4,M=128` | BF16 | 0.899696 | 2.681008 | 0.336 | native |
| Tail prefill, `B=1,S=127,H=3,M=96` | FP16 | 0.304128 | 1.651296 | 0.184 | native |
| Static decode, `B=1,Sk=128,H=4,M=128` | BF16 | 0.240704 | 1.584032 | 0.152 | native |
| Tail static decode, `B=1,Sk=129,H=3,M=96` | FP16 | 0.439440 | 1.658736 | 0.265 | native |

All eight outputs passed the alternate naive/absorbed MLA reference check, and the report was
generated from a clean source tree. The native median was lower in all four runs, but these are
small, unscaled-random-weight workloads on a desktop WSL session. The result is a correctness
and profiler baseline, not a claim about production models, long contexts, or other hardware.

## Paged MLA decode pairs

[`operator-matrix-mla-paged.json`](operator-matrix-mla-paged.json) records two same-dtype decode
pairs whose timed boundary contains one physical-slot projection write followed by absorbed
attention directly through a block table. Both reports use `page_size=16`, five warmups, twenty
samples, and a clean source tree.

| Case | Dtype | Native median (ms) | PyTorch paged median (ms) | Native / baseline | Lower median |
| --- | --- | ---: | ---: | ---: | --- |
| `B=2,S=257,H=4,Rkv=32`, long tail page | BF16 | 0.336480 | 2.139152 | 0.157 | native |
| `B=1,S=129,H=3,Rkv=19`, tail dimensions | FP16 | 0.524192 | 1.846000 | 0.284 | native |

All four outputs passed the alternate naive/absorbed verification. The BF16 pair had zero observed
absolute error for this generated input; the FP16 native result had maximum absolute error `0.0625`
and tolerance ratio `0.428`. These two small cases validate the storage/compute path and provide
profiling targets; they do not establish production serving throughput or a general performance
ordering.

[`torch-profiler-mla-paged.json`](torch-profiler-mla-paged.json) captures the BF16 native side with
one output call, five warmups, and twenty timed calls. Across 26 paged-attention calls it records
five `_local_scalar_dense` events and seventeen stream synchronizations; validation metadata is
reused after the first successful identity/version check. The paged-attention kernel contributes
`6.501 ms` aggregate self-device time and remains the primary CUDA optimization target. This is a
Kineto aggregate, not an Nsight counter report; parent operator and child kernel rows are correlated.

## MLA profiler triage

[`torch-profiler-mla-prefill.json`](torch-profiler-mla-prefill.json) and
[`torch-profiler-mla-decode.json`](torch-profiler-mla-decode.json) are structured
PyTorch/Kineto aggregates for one native side of the representative matrix. Each capture has
one unprofiled preflight and then records fresh setup, one output call, 5 warmup calls, and 20
timed calls. No large Chrome trace is checked in.

| Case | Calls | DtoH reads before/after | Stream sync before/after | Absorbed kernel before/after (ms) | Current custom-op share |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mla_prefill_regular` | 26 | 212 / 28 | 220 / 36 | 2.632 / 0.429 | 41.8% |
| `mla_decode_regular` | 26 | 162 / 29 | 170 / 37 | 2.699 / 0.491 | 44.3% |

The pre-change counts and kernel times were captured with the same cases before the corresponding
optimization; the checked-in JSON files contain the current aggregates. Position validation is
reused only while Tensor identity/version is unchanged, with mutation regressions for latent and
static-cache storage. For latent, RoPE, and value dimensions at most 32, absorbed attention now
partitions keys across four warps, maintains one online-softmax state per warp, and performs one
stable merge. Larger dimensions retain the generic block-wide kernel. The current reports name
`mla_absorbed_attention_warp_partition_float_kernel` explicitly; the 6.13x/5.50x reductions above
are profiler kernel self-device observations, not end-to-end speedups or Nsight counter results.
Parent operator and child kernel rows are correlated and must not be summed.

The commands and fixed shapes mirror `.github/workflows/cuda-tests.yml`. Re-run that workflow
on a registered single-GPU runner before treating this snapshot as continuously reproducible.
