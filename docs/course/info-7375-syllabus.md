# INFO 7375：面向 AI 的高性能计算课程大纲

- 课程英文名：*High Performance Computing for AI*
- 官方课程大纲：[PDF](http://newton.neu.edu:8080/syllabusrepo/37287.pdf?t=1770744869205)
- Notion 补充页面：[课程大纲页面](https://distinct-capricorn-c04.notion.site/Syllabus-High-Performance-Computing-for-AI-1f788315b6b48050946ede67ec5c086f)
- Notion 快照整理：2026-08-17
- 官方 PDF 核对：2026-08-21
- 英文整理版：[English source version](source/info-7375-syllabus.en.md)

> 本文的教材与 14 周课程安排以官方 PDF 为准；公开 Notion 页面及十一周课程笔记是
> 补充材料，其编号和主题不与官方周计划一一对应。评分、迟交和作业提交规则只用于
> 记录来源课程的组织方式，不代表本仓库当前要求，也不应视为东北大学的最新课程政策。

## 课程简介

INFO 7375 是一门围绕现代 AI 工作负载展开的实践型课程，主题覆盖 GPU 编程与分布式
系统。学生会从零编写 CUDA、融合 kernel、使用 Tensor Core、进行模型量化，并把训练与
推理扩展到多 GPU 和多节点环境。课程强调用数据衡量性能、定位瓶颈，并在故障条件下保持
计算正确性。

完成课程后，学生应当能够：

- 把训练工作负载映射到 GPU 硬件及其执行层次；
- 编写正确且有实际性能的 CUDA kernel；
- 安全地使用 Tensor Core 与混合精度；
- 在大规模服务中运行 continuous batching；
- 面向 GPU 内存层次进行 kernel fusion；
- 把训练工作负载分布到多个 GPU；
- 使用数据、张量、模型与流水线并行；
- 重叠计算与通信。

## 先修要求

- 具备较强的编程能力。
- 学生需要在课程前两周内补齐所需背景。推荐阅读 Jurafsky 与 Martin 的
  [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/)
  2025 年 8 月版本中的以下内容：
  - [第 4 章：Logistic Regression](https://web.stanford.edu/~jurafsky/slp3/4.pdf)，
    4.0-4.4 节；
  - [第 5 章：Embeddings](https://web.stanford.edu/~jurafsky/slp3/5.pdf)，
    5.0-5.7 节；
  - [第 6 章：Neural Networks](https://web.stanford.edu/~jurafsky/slp3/6.pdf)，
    6.0-6.5 节；
  - [第 7 章：Large Language Models](https://web.stanford.edu/~jurafsky/slp3/7.pdf)，
    7.0-7.4 节；
  - [第 8 章：Transformers](https://web.stanford.edu/~jurafsky/slp3/8.pdf)，
    8.0-8.10 节。
- 课程项目使用 C。原始大纲推荐 Kernighan 与 Ritchie 的
  *The C Programming Language*（第二版），并指出编程基础扎实的学生通常可通过约
  20 小时练习掌握项目所需的 C 语言知识。

## 教材

- Jesper Larsson Träff，*Lectures on Parallel Computing*；
- Hwu、Kirk 与 El Hajj，
  [*Programming Massively Parallel Processors: A Hands-On Approach*](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311/)；
- Notion 补充页面另列出可选读物：[Jason Sanders 与 Edward Kandrot，*CUDA by Example*](https://edoras.sdsu.edu/~mthomas/docs/cuda/cuda_by_example.book.pdf#page=43.00)。

## 教学方式

- 通过每周项目从零构建系统，在实现中学习；
- 课堂以 live coding 为主，集中处理每周项目中的困难概念；
- Notion 补充页面还建议阅读并实现高性能计算领域的近期研究工作，例如：
  - [*FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*](https://arxiv.org/pdf/2407.08608)；
  - [*FlashMoE: Fast Distributed MoE in a Single Kernel*](https://arxiv.org/pdf/2506.04667)。

## 评分方式

- 最终成绩取 8-10 个每周编程项目的平均分；
- 每迟交一天扣除该次作业成绩的 10%；
- Notion 补充页面给出的总体成绩分布与判定参考为：
  - 20% 为 A：持续提交接近满分的作业；
  - 30% 为 A-：提交全部项目，代码约 70% 的情况下能够正确运行；
  - 30% 为 B+：提交全部项目，代码约 50% 的情况下能够正确运行；
  - 10% 为 B：提交大部分项目，代码约 30% 的情况下能够正确运行；
  - 10% 为其他成绩：多数项目未提交，或代码很少能够正确运行。

## 课程安排

### 第一部分：GPU 编程

- 第1周：并行、multithreading 与 pthreads；
- 第2周：CUDA 基础与 GEMM；
- 第3周：内存层次、tiling、softmax 与 kernel fusion；
- 第4周：FlashAttention 1、2 与 3；
- 第5周：warp 级编程、intrinsics，以及使用 stream 进行异步执行；
- 第6周：Tensor Core 与 FP16、BF16、FP8 混合精度计算；
- 第7周：使用 NCCL 的集合通信与 CUDA Graphs。

### 第二部分：面向推理的 HPC

- 第8周：INT8/FP8 量化与计算图编译器；
- 第9周：continuous batching 与大规模服务。

### 第三部分：面向训练的 HPC

- 第10周：数据、张量、模型与流水线并行；
- 第11周：PyTorch FSDP 与 Megatron；
- 第12周：梯度压缩、通信重叠与 scaling efficiency；
- 第13周：使用 SLURM 进行资源管理，以及 checkpoint、容错与恢复。
- 第14周：I/O 瓶颈与拓扑感知通信。

## 作业提交方式

原始课程使用如下提交流程：

- 在个人 GitHub 账号下创建名为 `neu-hpc-for-ai` 的私有仓库；
- 将授课教师 `@suhabe` 和助教的 GitHub 账号添加为协作者；
- 所有作业使用同一个仓库；
- 每次作业使用一个顶层目录，例如第一周使用 `week_01`；
- 每个作业目录包含 `README.md`，逐项回答问题，或给出仓库中相关代码的链接；
- 在 Canvas 中提交对应作业目录的 GitHub URL。

本仓库没有照搬上述作业目录，而是把相关知识重组为可验证的 reference、CUDA 实现、
benchmark 与教材式文档。
