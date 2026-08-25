# INFO 7375: High Performance Computing for AI

- Official syllabus: [PDF](http://newton.neu.edu:8080/syllabusrepo/37287.pdf?t=1770744869205)
- Supplemental Notion page: [Syllabus page](https://distinct-capricorn-c04.notion.site/Syllabus-High-Performance-Computing-for-AI-1f788315b6b48050946ede67ec5c086f)
- Notion snapshot prepared: 2026-08-17
- Official PDF checked: 2026-08-21
- Chinese version: [INFO 7375 course syllabus](../syllabus.md)

> Textbook entries and the 14-week schedule follow the official PDF. The public
> Notion page and eleven-week notes are supplemental; their week numbers and
> topics do not map one-to-one to the official schedule. Grading, lateness, and
> submission rules describe the source course and are not current repository
> requirements or current Northeastern University policy.

## Course Description

INFO 7375 is a practical course in GPU programming and distributed systems for
modern AI workloads. Students write CUDA, combine kernels, use tensor cores,
quantize models, and expand training and inference from individual GPUs to
multi-node systems. The course emphasizes measurement, bottleneck diagnosis,
and maintaining correctness when failures occur.

By the end of the course, students should be able to:

- relate training workloads to GPU hardware and its execution hierarchy;
- implement CUDA kernels that are both correct and performant;
- use tensor cores and mixed-precision arithmetic safely;
- operate continuous batching in large-scale serving systems;
- fuse kernels with the GPU memory hierarchy in mind;
- distribute training workloads over multiple GPUs;
- apply data, tensor, model, and pipeline parallelism;
- overlap communication with computation.

## Prerequisites

- Strong general programming ability is expected.
- Students are expected to acquire the necessary AI background during the first
  two weeks. The source syllabus recommends the August 2025 edition of Jurafsky
  and Martin's [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/),
  particularly:
  - [Chapter 4, Logistic Regression](https://web.stanford.edu/~jurafsky/slp3/4.pdf),
    sections 4.0-4.4;
  - [Chapter 5, Embeddings](https://web.stanford.edu/~jurafsky/slp3/5.pdf),
    sections 5.0-5.7;
  - [Chapter 6, Neural Networks](https://web.stanford.edu/~jurafsky/slp3/6.pdf),
    sections 6.0-6.5;
  - [Chapter 7, Large Language Models](https://web.stanford.edu/~jurafsky/slp3/7.pdf),
    sections 7.0-7.4;
  - [Chapter 8, Transformers](https://web.stanford.edu/~jurafsky/slp3/8.pdf),
    sections 8.0-8.10.
- Course projects use C. The source recommends Kernighan and Ritchie's
  *The C Programming Language*, second edition, and estimates that a strong
  programmer can learn the required C through about twenty hours of practice.

## Textbooks

- Jesper Larsson Träff, *Lectures on Parallel Computing*.
- Hwu, Kirk, and El Hajj,
  [*Programming Massively Parallel Processors: A Hands-On Approach*](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311/).
- The supplemental Notion page also lists Jason Sanders and Edward Kandrot,
  [*CUDA by Example*](https://edoras.sdsu.edu/~mthomas/docs/cuda/cuda_by_example.book.pdf#page=43.00).

## Course Approach

- Learn by constructing systems from scratch in weekly projects.
- Use live coding in lectures to work through concepts needed by those projects.
- The supplemental Notion page also recommends recent high-performance-computing
  research, including:
  - [*FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*](https://arxiv.org/pdf/2407.08608);
  - [*FlashMoE: Fast Distributed MoE in a Single Kernel*](https://arxiv.org/pdf/2506.04667).

## Grading

- The final grade is the average of eight to ten weekly programming projects.
- Each late day reduces the affected project grade by ten percent.
- The supplemental Notion page describes the intended overall distribution as follows:
  - 20% A: consistently near-perfect work;
  - 30% A-: every project submitted, with code working in roughly 70% of cases;
  - 30% B+: every project submitted, with code working in roughly 50% of cases;
  - 10% B: most projects submitted, with code working in roughly 30% of cases;
  - 10% other: most projects missing or code rarely working.

## Course Schedule

### Part 1: GPU Programming

- Week 1: parallelism, multithreading, and pthreads;
- Week 2: CUDA fundamentals and GEMM;
- Week 3: memory hierarchy, tiling, softmax, and kernel fusion;
- Week 4: FlashAttention 1, 2, and 3;
- Week 5: warp-level programming, intrinsics, and asynchronous execution with streams;
- Week 6: tensor cores and mixed-precision FP16, BF16, and FP8 computation;
- Week 7: collective communication with NCCL and CUDA Graphs.

### Part 2: HPC for Inference

- Week 8: INT8/FP8 quantization and graph compilers;
- Week 9: continuous batching and large-scale serving.

### Part 3: HPC for Training

- Week 10: data, tensor, model, and pipeline parallelism;
- Week 11: PyTorch FSDP and Megatron;
- Week 12: gradient compression, communication overlap, and scaling efficiency;
- Week 13: SLURM resource management, checkpointing, fault tolerance, and recovery.
- Week 14: I/O bottlenecks and topology-aware communication.

## Assignment Submission

The source course used the following submission workflow:

- create a private repository named `neu-hpc-for-ai` under the student's GitHub account;
- add the instructor, `@suhabe`, and the teaching assistant's GitHub account as collaborators;
- use the same repository for every assignment;
- place each assignment in a top-level directory, with the first named `week_01`;
- include a `README.md` in each assignment directory that answers the questions or links to the relevant code;
- submit the GitHub URL of the assignment directory through Canvas.

This repository does not reproduce that assignment layout. It reorganizes the
related material into verifiable references, CUDA implementations, benchmarks,
and textbook-style documentation.
