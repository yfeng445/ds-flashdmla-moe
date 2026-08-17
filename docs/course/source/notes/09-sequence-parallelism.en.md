# Week 9: Sequence Parallelism, Ring Attention, and ThunderKittens

- Original page: [Week 9: Sequence Parallelism](https://distinct-capricorn-c04.notion.site/Week-9-Sequence-Parallelism-32d88315b6b480fc8718f6c1b1e5a6fa)
- Snapshot prepared: 2026-08-17
- Chinese version: [Week 9 notes](../../notes/09-sequence-parallelism.md)

## Why Partition the Sequence Dimension

Long contexts increase attention computation as well as memory for activations
and KV state. Sequence parallelism distributes tokens across devices so each
device retains only part of the sequence state. The central problem is how to
exchange required K/V blocks without approximating attention and how to hide
that communication behind computation.

## Blockwise Parallel Transformer

A Blockwise Parallel Transformer does not wait for the complete attention
output before running the FFN. It computes one query block and immediately feeds
that result through the FFN, avoiding simultaneous materialization of the full
attention output and all FFN intermediates.

The computation maintains online-softmax running maxima, normalizers, and output
accumulators. When a new K/V block changes the maximum, prior accumulators are
rescaled before adding the new contribution. Blockwise execution therefore
changes scheduling and retained state, but remains equivalent to complete
attention when implemented correctly.

The paper reports context lengths up to roughly 32 times those of its vanilla
Transformer baseline and around 2--4 times those of its FlashAttention baseline.
These are results for particular experiments, not fixed ratios across systems.

## Ring Attention

Ring Attention partitions the sequence across devices. Each device retains its
Q block while K/V blocks rotate around a device ring:

1. compute attention for the current K/V block and update online-softmax state;
2. concurrently send that block forward and receive the next block;
3. repeat until every Q block has visited every K/V block.

K/V communication can overlap each blockwise attention step. With correct
accumulation and masking, the result is exact attention. Under favorable
conditions, additional devices provide both sequence storage and work that can
hide communication, although actual scaling depends on bandwidth, block size,
and balance.

## ThunderKittens Tile Abstractions

[ThunderKittens](https://github.com/HazyResearch/ThunderKittens) represents GPU
kernel data with typed, fixed-shape tiles:

- register and shared-memory tiles describe their respective storage levels;
- dtype, shape, and layout information guides loads, stores, MMA, and layout
  operations at compile time;
- global layouts model `[batch, depth, rows, cols]`, using constants for fixed
  dimensions and `-1` for dynamic dimensions;
- `shared_allocator` is a shared-memory bump allocator for pipeline tiles and
  synchronization objects;
- warp-level MMA and register/shared loads compose through common tile APIs.

For example, `gl<bf16, 1, 1, -1, -1, st_bf<32, 32>>` describes a BF16 global
layout with fixed batch and depth, dynamic rows and columns, and a 32 by 32 base
tile. The abstraction retains explicit memory-hierarchy and warp decomposition
while reducing manual indexing and layout errors.

## TMA and Asynchronous Pipelines

The Tensor Memory Accelerator is a hardware data-movement engine on Hopper and
newer architectures. A tensor map describes a multidimensional global-memory
tensor, allowing asynchronous transfers into shared memory without assigning
compute threads to per-element address calculation. TMA can also apply swizzle.

A typical load pipeline uses an mbarrier or semaphore:

1. `init_semaphore` initializes shared synchronization state;
2. `expect_bytes` records the transfer size expected for the stage;
3. `load_async` starts the TMA load;
4. a consumer calls `wait` for the current phase and toggles the phase with XOR
   before reusing the barrier.

One semaphore can accumulate arrivals from multiple transfers. For asynchronous
stores, an operation such as `store_async_read_wait` must complete before the
source shared-memory tile is reused. Double-buffered and multistage pipelines
are correct only when phases, tile lifetimes, and expected byte counts agree.

## Assignment Direction

The source assignment asks students to reimplement DeepSeekMoE with
ThunderKittens, explore WMMA or Tensor Cores and TMA on B200, and compare
performance. The main lesson is mapping MoE data flow onto typed tiles, MMA, and
asynchronous transfer rather than treating a framework wrapper as a performance
result by itself.

## Further Reading

- [*Blockwise Parallel Transformer for Large Context Models*](https://arxiv.org/pdf/2305.19370);
- [*Ring Attention with Blockwise Transformers for Near-Infinite Context*](https://arxiv.org/pdf/2310.01889);
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens);
- [K/V communication overlap animation](https://coconut-mode.com/KV-overlap-large.gif);
- [K/V ring rotation animation](https://coconut-mode.com/KV-rotate.gif).
