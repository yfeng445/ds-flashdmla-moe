# 第 1 周：并行与 pthreads

- 原始页面：[Week 1: Parallelism](https://distinct-capricorn-c04.notion.site/Week-1-Parallelism-26288315b6b4808583c0ecee574eca71)
- 整理日期：2026-08-17
- 英文版：[Week 1: Parallelism](../source/notes/01-parallelism.en.md)

> 本文按原始课程页整理概念与练习目标，不保留课程 workspace 邀请、额度等时效性管理信息。

## 本周目标

第一周从 CPU 并发建立并行计算直觉，重点是多线程、共享内存和 race condition。这里的核心
不是“线程越多越快”，而是理解工作如何拆分、线程如何共享状态，以及同步成本何时超过并行
收益。

## 并行、共享内存与竞争

以矩阵乘法为例，输出矩阵的不同元素可以由不同线程计算；如果每个线程只写自己负责的输出
位置，写入之间天然独立。问题通常出现在多个线程更新同一变量、复用共享缓冲区或依赖未完成
的数据时：执行顺序不再确定，结果便可能随调度变化。

检查并行实现时应逐项回答：

1. 每个线程负责哪些输出？
2. 哪些内存只读、哪些内存会被多个线程写？
3. 是否存在必须发生在另一个操作之前的依赖？
4. 同步原语保护的是哪一段状态？
5. 线程数量增加后，创建、调度与同步开销是否抵消了加速？

## 练习路线：从单线程到 pthreads

原课程用 C 语言矩阵乘法作为第一项练习：

- 先实现单线程版本，明确 row-major 索引和维度约束；
- 用不同形状覆盖边界情况，例如 `1x1 @ 1x1`、`1x1 @ 1x5`、
  `2x1 @ 1x3` 和 `2x2 @ 2x2`；
- 再用 pthreads 划分输出工作，并复用同一组 correctness tests；
- 对 `1、4、16、32、64、128` 个线程分别计时；
- 使用足够大的矩阵，让被测计算明显长于线程创建和计时噪声；
- 同时记录绝对耗时和相对单线程的 speedup，而不只报告“更快”。

理想加速比受串行部分、线程管理、内存带宽与负载不均衡限制。线程数超过有效硬件并发度后，
继续增加线程通常只会增加上下文切换和同步成本。

## 为什么 Scaling Law 需要 HPC

课程把 [*Scaling Laws for Neural Language Models*](https://arxiv.org/pdf/2001.08361)
作为背景阅读。其启发是：模型规模、数据量和训练计算量之间存在可预测关系；当进一步提升模型
能力需要显著更多计算时，高效利用硬件、扩展并行执行和控制通信开销便成为 AI 进展的基础
条件，而不只是工程上的附加优化。

## 延伸阅读

- *Programming Massively Parallel Processors*：第 1 章 Introduction；
- *Programming Massively Parallel Processors*：第 2 章 Heterogeneous Data Parallel Computing；
- *Programming Massively Parallel Processors*：第 3 章 Multidimensional Grids and Data；
- [Modal Hello World](https://modal.com/docs/examples/hello_world)；
- [Modal Images](https://modal.com/docs/guide/images)；
- [Modal GPU Guide](https://modal.com/docs/guide/gpu)；
- [Modal CUDA Guide](https://modal.com/docs/guide/cuda)；
- [Modal Resource Management](https://modal.com/docs/guide/resources)。

