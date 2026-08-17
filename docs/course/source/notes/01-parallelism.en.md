# Week 1: Parallelism and pthreads

- Original page: [Week 1: Parallelism](https://distinct-capricorn-c04.notion.site/Week-1-Parallelism-26288315b6b4808583c0ecee574eca71)
- Snapshot prepared: 2026-08-17
- Chinese version: [Week 1 notes](../../notes/01-parallelism.md)

> This note preserves the technical topics and exercise intent of the public
> course page. Time-sensitive workspace invitations and credit limits are not
> reproduced.

## Weekly Focus

Week 1 introduces CPU concurrency through multithreading, shared memory, and
race conditions. The important question is not whether more threads exist, but
how work is partitioned, which state is shared, and when synchronization costs
erase the expected speedup.

## Parallel Ownership and Races

Matrix multiplication offers a natural first decomposition: separate threads
can compute separate output elements. Writes are independent when every thread
owns its output region. Races appear when threads update the same state, reuse a
buffer without synchronization, or consume data before a producer finishes.

A parallel design should make five things explicit:

1. the output region owned by each thread;
2. which data is read-only and which data has multiple writers;
3. the required happens-before relationships;
4. the state protected by each synchronization operation;
5. whether thread creation, scheduling, and synchronization outweigh the work.

## Exercise Path: Single Thread to pthreads

The source assignment uses matrix multiplication in C:

- implement a single-threaded row-major reference;
- test shapes such as `1x1 @ 1x1`, `1x1 @ 1x5`, `2x1 @ 1x3`, and
  `2x2 @ 2x2`;
- partition the output with pthreads and reuse the same correctness tests;
- time runs with `1, 4, 16, 32, 64, 128` threads;
- use matrices large enough that computation dominates launch and timer noise;
- report absolute time and speedup relative to one thread.

Speedup is limited by serial work, thread-management overhead, memory bandwidth,
and load imbalance. Oversubscribing useful hardware concurrency normally adds
context switching rather than useful parallel work.

## Scaling Laws and HPC

[*Scaling Laws for Neural Language Models*](https://arxiv.org/pdf/2001.08361)
motivates the course. Predictable relationships between model size, data, and
training compute imply that progress can require much more computation. Hardware
efficiency, scalable parallel execution, and communication control therefore
become core AI capabilities rather than optional tuning.

## Further Reading

- *Programming Massively Parallel Processors*, Chapters 1-3;
- [Modal Hello World](https://modal.com/docs/examples/hello_world);
- [Modal Images](https://modal.com/docs/guide/images);
- [Modal GPU Guide](https://modal.com/docs/guide/gpu);
- [Modal CUDA Guide](https://modal.com/docs/guide/cuda);
- [Modal Resource Management](https://modal.com/docs/guide/resources).
