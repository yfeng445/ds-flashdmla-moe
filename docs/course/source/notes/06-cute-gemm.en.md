# Week 6: CuTe GEMM

- Original page: [Week 6: CuTe GEMM](https://distinct-capricorn-c04.notion.site/Week-6-CuTe-GEMM-30a88315b6b4804b8b8fcfcbdc36554f)
- Snapshot prepared: 2026-08-17
- Chinese version: [Week 6 notes](../../notes/06-cute-gemm.md)

## From Layout Algebra to GEMM

CuTe uses layout algebra to express problem shapes, global-memory tensors, CTA
tiles, shared-memory tiles, and thread ownership. Combinations of algorithms,
dtypes, layouts, and performance policies can share the same abstractions.

Logical division tiles a layout `A` by another layout `B`, producing coordinates
that separate a position within a tile from the tile's position in the grid.
GEMM needs exactly this relationship: one CTA owns an M/N output tile and
iterates over K tiles.

## Four Pre-Launch Object Groups

A CuTe GEMM defines:

1. global-memory tensor views for A, B, and C;
2. a CTA tiler;
3. shared-memory layouts;
4. thread layouts that assign tile elements to threads.

### Global-Memory Views

The `(M,N,K)` problem shape yields `(M,K)`, `(N,K)`, and `(M,N)` views. Combining
each pointer with a shape and stride creates `mA`, `mB`, and `mC`, allowing the
same main loop to handle transpose and leading-dimension variants.

### CTA Tiler

A tiler such as `(128,128,8)` assigns a `128 x 128 x 8` problem subtile to one
CTA. Grid dimensions cover the M/N output tiles, while K remains a main-loop
dimension.

### Shared-Memory Layouts

Shared-memory layouts describe storage for A and B input tiles and any C-related
tile. They must support efficient copies, avoid harmful bank conflicts, and
match the expected MMA load pattern.

### Thread Layouts

A `(32,8)` thread layout naturally identifies a lane and one of eight warps in a
256-thread CTA. Useful layouts promote contiguous lane accesses, predictable
per-warp work, and alignment with MMA operands and accumulators.

## `local_tile`: Select CTA Data

`local_tile` divides a full tensor into a tile grid and selects a local view with
a CTA coordinate. An integer fixes one tile along a mode, while `_` preserves a
mode for iteration. Fixing M and N with block indices but preserving K produces
the K-tile dimension needed by the GEMM main loop.

## `local_partition`: Select Thread Data

`local_partition(tensor, thread_layout, threadIdx.x)` divides a CTA tile among
threads. Each thread obtains a non-overlapping subtensor, allowing all threads
to participate in global-to-shared copies.

## Main Loop and Asynchronous Copy

Each K-tile iteration follows this dependency chain:

```text
copy GMEM -> SMEM
commit the asynchronous-copy group
wait for required groups
block-wide barrier
GEMM SMEM x SMEM -> registers
block-wide barrier before SMEM reuse
```

When possible, CuTe can lower global-to-shared copies to `cp.async`.
`cp_async_fence()` commits the current thread's copy group and
`cp_async_wait<0>()` waits for that thread's committed groups. A following
`__syncthreads()` is still required to make every thread's shared-memory writes
visible. The second barrier prevents early overwrite while other threads still
read the current tile.

## Predication and Architecture Evolution

Boundary tiles need predicates to suppress invalid accesses. TMA tensors extend
the same organization to large asynchronous transfers on Hopper and newer
architectures. Static layouts and tilers make it possible to specialize one
GEMM structure for different tensor-core instructions, dtypes, and pipelines.

## Further Reading

- [CuTe GEMM Tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html);
- [CuTe Predication](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0y_predication.html);
- [CuTe TMA Tensors](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html);
- [CUTLASS `sgemm_1.cu`](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_1.cu);
- [*FlashAttention-3*](https://arxiv.org/pdf/2407.08608);
- [NVIDIA Tensor Core Evolution](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell);
- [GTC 2025 CuTe Session](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72720/).

