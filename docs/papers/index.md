# AI Infra 论文与参考资料

本目录用于保存 AI Infra 相关论文的可合法分发原文、翻译件、阅读笔记和索引。当前先按主题列出原始论文、官方代码和适合补基础的教材，便于建立可追溯的实现依据。

新增材料时应记录论文标题、作者、原始链接、版本或发布日期，以及原文授权信息。翻译件还应标注对应原文版本和翻译状态；无法确认再分发权限时，只保留外部链接和原创阅读笔记。

## 并行计算与 CUDA 基础

1. David B. Kirk, Wen-mei W. Hwu, Izzat El Hajj，*Programming Massively
   Parallel Processors: A Hands-on Approach*，第 4 版。重点阅读线程组织、内存层次、
   tiling、reduction、scan 和性能分析章节。
2. Jesper Larsson Träff，*Lectures on Parallel Computing*。用于理解并行代价模型、
   MPI 与集合通信。
3. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
   与 [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)。
4. [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)，只在需要核对
   指令语义时查阅，不建议从头通读。
5. [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) 与
   [Tensor Memory Accelerator programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma)，
   用于继续学习 TMA、异步 transaction barrier 与 Hopper 资源限制。
6. CUDA Programming Guide 的
   [Warp Matrix Functions（固定 CUDA 12.0 文档）](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-c-programming-guide/index.html#warp-matrix-functions)，
   用于核对 WMMA 的 warp 一致执行、内存对齐、leading dimension 与支持的混合精度 tile。
7. [Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)，
   用于区分同步 shared staging、`memcpy_async` 与 Hopper TMA，不把 WMMA 本身误写成异步
   数据流水线。

## Attention 与在线 Softmax

1. Milakov, Gimelshein，
   [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)。
2. Dao et al.，
   [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)。
3. Dao，
   [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)。
4. Shah et al.，
   [FlashAttention-3](https://arxiv.org/abs/2407.08608)。
5. [FlashAttention 官方实现与测试](https://github.com/Dao-AILab/flash-attention)。

## Transformer 与 MLA

1. Jurafsky, Martin，
   [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)，重点补充
   embeddings、neural networks、LLM 与 Transformer 章节。
2. DeepSeek-AI，
   [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)。
3. DeepSeek-AI，
   [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)。
4. [DeepSeek-V3 官方 inference reference](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)，
   用于核对 MLA 的 NoPE/RoPE 拆分、latent cache 与 weight absorption。

## MoE 与分布式执行

1. Shazeer et al.，
   [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)。
2. Lepikhin et al.，
   [GShard](https://arxiv.org/abs/2006.16668)。
3. Fedus et al.，
   [Switch Transformers](https://arxiv.org/abs/2101.03961)。
4. Dai et al.，
   [DeepSeekMoE](https://arxiv.org/abs/2401.06066)。
5. Wang et al.，
   [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)。
6. [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)，重点阅读
   point-to-point、group calls、stream semantics 与 collective 操作。
7. [NVSHMEM Memory Model](https://docs.nvidia.com/nvshmem/api/gen/mem-model.html)、
   [Memory Ordering](https://docs.nvidia.com/nvshmem/api/gen/api/ordering.html) 与
   [Signaling Operations](https://docs.nvidia.com/nvshmem/api/gen/api/signal.html)，用于核对
   symmetric address、fence/quiet、put-with-signal 和 wait/test 语义。
8. Aimuyo, Oh, Singh，
   [FlashMoE: Fast Distributed MoE in a Single Kernel](https://arxiv.org/abs/2506.04667)，
   以及[官方实现](https://github.com/osayamenja/FlashMoE)。
9. [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)，用于继续研究
   tile-centric 通信—计算重叠、task scheduling 与 distributed megakernel。

## PyTorch 扩展与验证

1. [PyTorch Custom C++ and CUDA Operators](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html)。
2. [`torch.utils.cpp_extension`](https://docs.pytorch.org/docs/stable/cpp_extension.html)。
3. [`torch.library.opcheck`](https://docs.pytorch.org/docs/stable/library.html#torch.library.opcheck)
   与 `torch.autograd.gradcheck`，用于验证注册、fake tensor、autograd 和数值梯度。
4. [PyTorch C++/CUDA extension 官方示例](https://github.com/pytorch/extension-cpp)，用于核对
   dispatcher 注册、动态库加载和 ahead-of-time build 的最小结构。
5. [PyTorch Distributed：同步与异步 collective](https://docs.pytorch.org/docs/stable/distributed.html#synchronous-and-asynchronous-collective-operations)，
   用于核对 NCCL `async_op`、`Work.wait()`、CUDA stream 依赖和多 communicator 顺序约束。
6. [ProcessGroupNCCL 环境变量](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html)，
   用于区分默认的 stream dependency 与 `TORCH_NCCL_BLOCKING_WAIT` 所启用的 host-blocking
   调试行为。

## 推荐路径

- CUDA 初学者：PMPP → 在线 Softmax → FlashAttention 1。
- 熟悉 CUDA、准备实现 MLA：FlashAttention 2 → DeepSeek-V2 → DeepSeek-V3 reference。
- 准备实现 EP：DeepSeekMoE → MPI collective 基础 → NCCL point-to-point/group calls。
- 准备实现 one-sided EP：已验证的 EP reference → NVSHMEM memory model/ordering/signaling
  → FlashMoE → 最小双 GPU data/flag 实验。
- 做性能报告前：CUDA Best Practices → Nsight Compute/Nsight Systems 官方指南。
