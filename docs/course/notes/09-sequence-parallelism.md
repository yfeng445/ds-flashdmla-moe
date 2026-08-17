# 第 9 周：序列并行、Ring Attention 与 ThunderKittens

- 原始页面：[Week 9: Sequence Parallelism](https://distinct-capricorn-c04.notion.site/Week-9-Sequence-Parallelism-32d88315b6b480fc8718f6c1b1e5a6fa)
- 整理日期：2026-08-17
- 英文版：[Week 9: Sequence Parallelism](../source/notes/09-sequence-parallelism.en.md)

## 为什么沿序列维切分

长上下文会同时放大 attention 的计算量以及 activation、KV 状态的内存占用。序列并行把 token 序列分到多个设备，使单设备只保留部分序列状态；关键问题随之变成：如何在不近似 attention 的前提下交换必要的 K/V 块，并把通信隐藏在计算后面。

## Blockwise Parallel Transformer

Blockwise Parallel Transformer 不等待完整 attention 输出落入显存后再运行 FFN，而是按 query block 计算 attention，并立即把该 block 的结果送入 FFN。这样可以避免同时物化整段 attention 输出和完整 FFN 中间激活，降低峰值内存。

计算需要维护在线 softmax 的 running maximum、normalizer 与输出累加值。每处理一个新的 K/V block，先根据新旧最大值重标定历史累加项，再加入当前 block 的贡献。因此 blockwise 只是改变执行顺序和中间状态的保存方式，在正确实现下仍与完整 attention 等价。

原论文报告其最长序列能力相对 vanilla Transformer 可提高到约 32 倍、相对 FlashAttention 可提高约 2–4 倍；这些是特定实验配置下的结果，不应当视作所有硬件和模型上的固定比例。

## Ring Attention

Ring Attention 把序列切成与设备数对应的块。每个设备固定保留自己的 Q block，而 K/V blocks 沿设备环依次传递：

1. 使用当前本地 K/V block 计算一块 attention，并更新在线 softmax 状态；
2. 同时把 K/V block 发送给下一个设备、从上一个设备接收下一块；
3. 重复直到每个 Q block 都访问过全部 K/V blocks。

每一步都能把 K/V 通信与 blockwise attention 计算重叠。只要累计过程与 mask 处理正确，这仍是 exact attention。理想条件下，增加设备既增加总序列存储，也增加可与通信重叠的计算，使可处理上下文随设备数扩展；实际伸缩性取决于网络带宽、block 大小和负载均衡。

## ThunderKittens 的 tile 抽象

[ThunderKittens](https://github.com/HazyResearch/ThunderKittens) 用带类型的固定形状 tile 表达 GPU kernel 数据：

- register tile 与 shared-memory tile 分别描述寄存器和共享内存中的数据；
- 类型携带 dtype、shape 和 layout，让 load/store、MMA 与布局变换在编译期获得更多信息；
- global layout 用 `[batch, depth, rows, cols]` 描述张量，固定维可写为常量，动态维用 `-1`；
- `shared_allocator` 是共享内存上的 bump allocator，用于按 pipeline 阶段分配 tile 和同步对象；
- warp-level MMA、register/shared load 等操作通过统一 tile 接口组合。

例如 `gl<bf16, 1, 1, -1, -1, st_bf<32, 32>>` 表示 batch/depth 固定为 1、行列动态、基本 tile 为 32×32 的 BF16 global layout。该抽象的目标是让 kernel 仍然显式表达层次化内存和 warp 工作划分，同时减少手写索引与布局错误。

## TMA 与异步流水线

Tensor Memory Accelerator（TMA）是 Hopper 及更新架构上的硬件数据搬运引擎。它借助 tensor map 描述多维 global-memory 张量，在不占用计算线程做逐元素地址计算的情况下，将数据异步搬运到 shared memory，并可应用 swizzle。

典型 load pipeline 使用 mbarrier/semaphore：

1. `init_semaphore` 初始化共享同步对象；
2. `expect_bytes` 登记本阶段预期到达的字节数；
3. `load_async` 发起 TMA load；
4. consumer 用 `wait` 等待当前 phase，随后以 XOR 翻转 phase 以复用 barrier。

一个 semaphore 可以累计多个传输的到达字节。异步 store 则需要在复用源 shared-memory tile 前调用类似 `store_async_read_wait` 的操作，确认 TMA 已完成读取。双缓冲或多阶段 pipeline 只有在 producer/consumer phase、tile 生命周期和 barrier 字节数完全匹配时才正确。

## 课程作业方向

本周作业要求用 ThunderKittens 重新实现 DeepSeekMoE，在 B200 上探索 WMMA/Tensor Core 与 TMA，并比较性能。其学习重点是把上一周的 MoE 数据流映射到 typed tiles、MMA 和异步搬运，而不是把某个框架封装本身当作性能结论。

## 延伸阅读

- [*Blockwise Parallel Transformer for Large Context Models*](https://arxiv.org/pdf/2305.19370)；
- [*Ring Attention with Blockwise Transformers for Near-Infinite Context*](https://arxiv.org/pdf/2310.01889)；
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens)；
- [K/V communication overlap animation](https://coconut-mode.com/KV-overlap-large.gif)；
- [K/V ring rotation animation](https://coconut-mode.com/KV-rotate.gif)。
