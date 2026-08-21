# HPC for AI Course Notes

- Parent source: [HPC for AI](https://distinct-capricorn-c04.notion.site/HPC-for-AI-20d88315b6b480538083fbe724df2902)
- Snapshot prepared: 2026-08-17
- Course syllabus: [INFO 7375: High Performance Computing for AI](info-7375-syllabus.en.md)
- Chinese collection: [HPC for AI course notes](../hpc-for-ai.md)

This directory preserves an English-only structured version of eleven public
supplemental Notion weekly pages. It retains topic hierarchy, equations, system
data flows, assignment directions, and stable references without copying Notion
interface artifacts, attachments, or temporary download links.

> This is a snapshot prepared on August 17, 2026. Course organization, project
> status, tool versions, and job links may change; consult the original pages
> and official sources for current information.

## Weekly Navigation

| Week | Topic | Learning role |
| --- | --- | --- |
| 1 | [Parallelism, Threads, and Synchronization](notes/01-parallelism.en.md) | Introduces correctness and performance in processes, threads, and shared state. |
| 2 | [CUDA and GEMM Fundamentals](notes/02-cuda.en.md) | Establishes GPU execution hierarchy, kernel launch, matrix multiplication, and basic optimization. |
| 3 | [Memory Hierarchy, Tiling, and Fusion](notes/03-memory.en.md) | Covers data reuse, shared memory, coalescing, reductions, and fusion. |
| 4 | [Attention and FlashAttention](notes/04-attention.en.md) | Derives tiled exact attention and IO optimization from stable softmax. |
| 5 | [Layout Algebra](notes/05-layout-algebra.en.md) | Uses shape, stride, composition, and inverse to map threads onto data. |
| 6 | [CuTe GEMM](notes/06-cute-gemm.en.md) | Composes layouts, copies, MMA, and pipelines into a hierarchical GEMM kernel. |
| 7 | [MPI, NCCL, and Collectives](notes/07-communication.en.md) | Establishes distributed-memory and collective-communication semantics. |
| 8 | [Data and Expert Parallelism](notes/08-data-expert-parallelism.en.md) | Explains MoE routing, two All-to-All operations, and hybrid DP, TP, and EP flow. |
| 9 | [Sequence Parallelism and Ring Attention](notes/09-sequence-parallelism.en.md) | Covers blockwise exact attention, K/V ring rotation, and asynchronous tile pipelines. |
| 10 | [Inference Systems](notes/10-inference-systems.en.md) | Connects prefill and decode, KV cache, PagedAttention, and continuous batching. |
| 11 | [Career Building](notes/11-career-building.en.md) | Builds a long-term path through communities, open source, reproduction, systems practice, and job analysis. |

## Suggested Paths

- CUDA and kernels: 1, 2, 3, 4, 5, 6;
- distributed MoE: 1, 7, 8;
- long context and inference: 4, 9, 10;
- after any technical path, use Week 11 to convert knowledge into reproduction,
  open-source participation, and a continuing study plan.

These weekly pages preserve the organization of the supplemental Notion
collection; they are neither a replacement for nor a week-by-week expansion of
the official PDF's 14-week schedule. The topic chapters elsewhere in the
documentation reorganize material around mathematical definitions, references,
CUDA interfaces, and verification boundaries.
