# Week 2: CUDA and Naive GEMM

- Original page: [Week 2: CUDA](https://distinct-capricorn-c04.notion.site/Week-2-CUDA-26388315b6b480d480aec7e22cde5776)
- Snapshot prepared: 2026-08-17
- Chinese version: [Week 2 notes](../../notes/02-cuda.md)

## Weekly Focus

Week 2 moves matrix multiplication from the CPU to CUDA and introduces kernel
launches, grids, blocks, threads, host/device memory, `nvcc`, and error checks.
The exercise intentionally uses global memory only; tiling and shared memory are
deferred to Week 3.

## CUDA Execution Model

A host launches a `__global__` kernel with a grid and block configuration:

```cpp
kernel<<<blocks_per_grid, threads_per_block>>>(...);
```

- `threadIdx` locates a thread within its block;
- `blockIdx` locates a block within the grid;
- `blockDim` describes the block dimensions;
- a two-dimensional output often maps naturally to a two-dimensional block;
- launches are asynchronous, so observation and error boundaries need explicit
  synchronization.

Launch checks use `cudaGetLastError` or `cudaPeekAtLastError`. Runtime failures
become visible at a synchronization boundary, and `cudaGetErrorString` converts
an error code to readable text.

## Host and Device Memory

Host memory belongs to the CPU address space. Device, global, or HBM memory is
visible to the GPU and is managed with `cudaMalloc` and `cudaFree`. Transfers use
`cudaMemcpy` with Host-to-Device, Device-to-Host, or Device-to-Device direction.

A row-major matrix of width `W` maps `(row, col)` to `row * W + col`. With width
4, coordinate `(2, 1)` has linear index 9.

## From Matrix Multiplication to GEMM

For `A` shaped `m x k` and `B` shaped `k x n`, ordinary multiplication is

\[
C = AB.
\]

GEMM computes

\[
C \leftarrow \alpha\,op(A)op(B) + \beta C,
\]

where each operand may be read normally or transposed. A complete interface
supports `AB`, `A^TB`, `AB^T`, and `A^TB^T` without materializing transpose
copies. Each thread accumulates one output element across `k`, then applies the
alpha and beta terms once.

## Hardware Scale

The course contrasts a many-core CPU with a B200-class GPU. A CPU may expose a
few hundred hardware threads, while a GPU has more than one hundred streaming
multiprocessors and can keep hundreds of thousands of threads resident in
aggregate. GPU throughput comes from mapping large amounts of similar work, not
from making an individual thread stronger.

## Debugging Principles

- split a problem into independently testable components when debugging stalls;
- do not use device `printf` as a substitute for explicit ownership and shape
  reasoning;
- combine assertions, launch checks, and deliberate synchronization points;
- test ordinary dimensions and dimensions that are not block-size multiples;
- keep the same correctness suite when extending the interface.

## Source Exercise

Implement a naive global-memory GEMM without cuBLAS or cuDNN. After the base
version is correct, add optional transposition for both operands and update `C`
in place. The target for this week is semantic correctness and interface
coverage, not low-level optimization.

## Further Reading

- *Programming Massively Parallel Processors*, Chapter 5;
- [CUDA C++ Programming Guide: Introduction](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction);
- [CUDA C++ Programming Guide: Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).

