# Week 3: Memory Hierarchy, Tiling, and Benchmarking

- Original page: [Week 3: Memory](https://distinct-capricorn-c04.notion.site/Week-3-Memory-26388315b6b48016a19bc6f451f9e1eb)
- Snapshot prepared: 2026-08-17
- Chinese version: [Week 3 notes](../../notes/03-memory.md)

## Weekly Focus

Week 3 moves from a functioning GEMM to an explainable GEMM. The core topics are
FLOP counting, trustworthy timing, the HBM/shared-memory/register hierarchy, and
using tiles to increase data reuse.

## GEMM FLOPs

For `A` shaped `(M, K)` and `B` shaped `(K, N)`, each of the `M*N` output values
requires `K` multiplications and `K-1` additions. The exact count is

\[
(2K-1)MN,
\]

which is normally approximated as `2MNK` for large matrices. A TFLOP/s result is
meaningful only when it is paired with the shape, average kernel time, and FLOP
count.

## Trustworthy CUDA Timing

1. avoid unrelated concurrent GPU work;
2. enqueue the measured work on an explicit CUDA stream;
3. run roughly 3-10 warmup iterations;
4. record a start event on the same stream;
5. run about 50 measured iterations;
6. record and wait for the stop event or stream;
7. check launch and execution errors;
8. divide event elapsed time by the iteration count;
9. divide FLOPs by average seconds and convert to TFLOP/s;
10. report shape, dtype, GPU, and compilation configuration.

`cudaStreamSynchronize` waits for prior work in one stream and surfaces its
errors. `cudaEventSynchronize` waits for one event, which fires only after the
earlier operations in that event's stream finish.

## GPU Memory Hierarchy

| Level | Visibility | Character | Typical use |
| --- | --- | --- | --- |
| HBM / global memory | whole GPU | largest and highest latency | inputs, outputs, large tensors |
| shared memory / SMEM | one thread block on an SM | small, explicit, low latency | tile reuse and cooperation |
| registers / RMEM | one thread | fastest and limited | accumulators and local scalars |

The course uses A100-scale examples: tens of gigabytes of HBM versus hundreds of
kilobytes of shared memory and registers per SM. Exact capacities vary by
architecture and configuration and must be queried on the target device.

## Why Naive GEMM Wastes Bandwidth

In a naive kernel, each thread reloads an entire row of `A` and column of `B`
from HBM. Neighboring outputs repeatedly consume the same data. Tiling stages
submatrices in shared memory so threads in a block reuse them, raising arithmetic
intensity and reducing HBM traffic per FLOP.

Analysis should include global-memory coalescing, reuse per tile, shared-memory
capacity, register pressure and occupancy, and boundary behavior.

## Source Exercise

Read and reproduce the runs and calculations in
[*How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog*](https://siboehm.com/articles/22/CUDA-MMM)
on an H100. The purpose is to connect each code change to data movement,
parallel mapping, and measured performance, rather than copying only the final
kernel.

## Further Reading

- [CUDA Matmul Optimization Worklog](https://siboehm.com/articles/22/CUDA-MMM);
- [GPU Execution and Tiling](../../../chapters/00-gpu-execution-and-tiling.md);
- [Benchmarking and Roofline](../../../chapters/07-benchmarking-and-roofline.md).

