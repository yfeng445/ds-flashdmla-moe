# DeepSeek MLA 与 MoE 高性能实现讲义

这组讲义从可以逐行验证的数学定义出发，推导 CUDA 与多 GPU 实现。每章围绕一个
算子，依次回答四个问题：

1. 算子在数学上计算什么？
2. 朴素实现为什么慢？
3. GPU 实现应怎样组织数据和并行工作？
4. 如何证明优化没有改变语义？

## 阅读顺序

| 章节 | 核心问题 | 配套代码 |
| --- | --- | --- |
| [0. 从并行线程到 GPU Tiling](chapters/00-gpu-execution-and-tiling.md) | grid/block/warp、内存复用与异步流水线 | `gemm.py`、`gemm_benchmarking.py` |
| [1. 从 Softmax 到在线 Softmax](chapters/01-online-softmax.md) | 如何稳定地流式归一化 | `attention.py` |
| [2. 从标准 Attention 到 FlashAttention](chapters/02-flash-attention.md) | 如何避免写出完整分数矩阵 | `attention.py`、实验 CUDA |
| [3. Multi-head Latent Attention](chapters/03-mla.md) | MLA 压缩了什么，decode 为何不同 | `mla.py` |
| [4. DeepSeekMoE](chapters/04-deepseek-moe.md) | 分组路由、SwiGLU 与 token dispatch | `moe.py`、`router_ops.py`、`expert_ops.py`、`benchmarks/router.py`、`benchmarks/experts.py` |
| [5. Expert Parallelism](chapters/05-expert-parallelism.md) | 如何跨 rank 保持 token 身份、顺序与梯度 | `expert_parallel.py` |
| [6. PyTorch 自定义算子](chapters/06-pytorch-custom-operators.md) | dispatcher、FakeTensor、autograd 与 stream | `ops.py`、`csrc/` |
| [7. Benchmark 与 Roofline](chapters/07-benchmarking-and-roofline.md) | 如何得到单卡与多 rank 的可复核性能证据 | `benchmarking.py`、`benchmarks/` |
| [8. 对称内存与 One-sided MoE](chapters/08-one-sided-symmetric-memory.md) | PGAS、data/flag 协议、时间缓冲与内存代价 | `symmetric_memory.py` |
| [练习](exercises.md) | 从公式、reference 到 kernel 的递进任务 | 全仓库 |

面试准备可先阅读[AI Infra 面试整理](infra-interview-guide.md)，再使用
[AI Infra 高压模拟面试](infra-mock-interview.md)限时作答。两者基于根目录 notebook 和当前
仓库事实，把已验证结果、个人贡献与设计目标分开。

第零章只要求线性代数和基本并发概念；第一、二章在此基础上进入在线 Softmax 与 tiled
Attention。第三至七章还需要了解 Transformer、MPI/NCCL 集合通信、PyTorch autograd
与 dispatcher；第八章在 EP 基线上进一步讨论 one-sided 通信与显式同步。

## 建议学习方式

- 先运行 `examples/reference_demo.py`，确认环境和张量形状。
- 阅读一节推导后，找到对应 reference 实现并手算一个极小输入。
- 修改 shape、mask 或路由参数，让测试先失败，再解释失败原因。
- 只有在 reference 和梯度检查都通过后，才开始 kernel 优化。
- 每次优化记录输入 shape、dtype、硬件和误差；“更快”不能脱离这些条件。

## 符号约定

- `B`：batch size。
- `H`：attention head 数；MoE 章节中隐藏维度写作 `D_h`，避免混淆。
- `S_q, S_k`：query 与 key/value 序列长度。
- `D`：head dimension 或 model dimension，由上下文说明。
- `E`：routed expert 数量。
- `K`：每个 token 激活的 expert 数量。
- 所有行向量默认写在张量的倒数第二维。

完整书目和推荐阅读顺序见[参考资料](references.md)。
