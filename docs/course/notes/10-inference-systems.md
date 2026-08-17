# 第 10 周：推理系统、KV Cache 与 PagedAttention

- 原始页面：[Week 10: Inference Systems](https://distinct-capricorn-c04.notion.site/Week-10-Inference-Systems-33488315b6b48083b81ecea51648dd7c)
- 整理日期：2026-08-17
- 英文版：[Week 10: Inference Systems](../source/notes/10-inference-systems.en.md)

## Prefill 与 Decode

自回归 LLM 推理分为两个性质不同的阶段：

| 阶段 | 工作 | 常见瓶颈 | 主要延迟指标 |
| --- | --- | --- | --- |
| Prefill | 并行处理全部 prompt token，生成首 token 和各层 KV cache | 长 prompt 下计算密集 | TTFT（time to first token） |
| Decode | 每一步读取权重与已有 KV，只生成一个新 token | 通常受显存带宽限制 | ITL（inter-token latency） |

端到端还会关注 E2EL/总延迟、吞吐量与 token generation latency。推理系统必须在请求级吞吐和单请求延迟之间取舍；continuous batching、prefill/decode 调度和 speculative decoding 都服务于这个目标。

## KV Cache 为什么必要

若不缓存，每生成一个 token 都要重新计算此前所有 token 的 K/V。KV cache 保存各层历史状态，使 1000-token prompt 加 100-token 输出大体只需处理约 1100 个新 token 的状态，而不是在 100 个 decode step 中反复计算不断增长的完整前缀。

缓存的代价是显存容量和带宽。对普通 multi-head attention，近似容量为：

```text
KV bytes = 2 × bytes_per_element × n_layers × batch
           × n_kv_heads × d_head × sequence_length
```

前面的 `2` 对应 K 和 V。课程以 LLaMA 2 13B、FP16、40 层、40 个 KV heads、`d_head=128` 为例：每 token 约 `0.78125 MB`，4096 tokens 约 `3.125 GB`。这里应是 GB，而不是源页面某一处误写的 MB。

不同 attention 结构通过减少 KV heads 或压缩表示降低 cache：

- MHA 为每个 query head 保留独立 K/V；
- MQA 让所有 query heads 共享一组 K/V；
- GQA 让一组 query heads 共享 K/V，在质量与缓存规模之间折中；
- MLA 把历史状态压缩为低维 latent，需要时再投影；
- GLA 进一步组织 grouped latent，便于并行切分。

## Fragmentation 与 PagedAttention

传统做法常为每个请求预留一段连续、按最大长度估算的 KV 空间。这会产生：

- **internal fragmentation**：已分配块中因请求提前结束或预留过大而未使用的空间；
- **external fragmentation**：空闲空间总量足够，但无法提供所需的大连续区间；
- 由此限制可并发 batch 和可支持序列长度。

PagedAttention 借鉴虚拟内存分页：把每个请求的逻辑 KV 序列切成固定大小的逻辑 blocks，通过 block table 映射到不连续的物理 blocks。物理空间可以按需增长和回收，不必为最大长度预留连续 buffer；相同前缀或 beam 的只读 blocks 也更容易安全共享。代价是额外的地址映射、block metadata 和能够按 block table 访问 KV 的 attention kernel。

## Static 与 Continuous Batching

Static batching 先凑齐固定 batch，再一起执行到所有请求结束。短请求完成后，其 slot 仍可能等待最长请求，且新请求不能立即加入。

Continuous batching 在 decode iteration 边界重组 active batch：请求结束后马上释放 slot，让排队请求进入。它显著提高 GPU 利用率，但 scheduler 需要处理到达时间、优先级、KV block 分配、prefill chunking，以及吞吐与尾延迟的权衡。

## 推理优化的系统视角

- **continuous batching** 减少空闲 slot；
- **chunked prefill / prefill-decode scheduling** 控制长 prompt 对 decode 延迟的干扰；
- **speculative decoding** 用较便宜的 draft 生成候选，再由 target model 并行验证；
- **KV 管理** 决定能容纳多少活跃请求；
- **kernel 与调度协同** 决定理论节省能否转化为真实 TTFT、ITL 和吞吐提升。

评估时应同时报告 workload、输入/输出长度分布、并发度、精度与硬件，不能只给一个峰值 tokens/s。

## 课程作业方向

本周作业入口是 [Nano Sglang](https://github.com/lixiaohua-neu/nano-sglang)，目标是从小型代码库理解请求调度、批处理、KV cache 和推理执行的衔接。

## 延伸阅读

- [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/pdf/2309.06180)；
- [*Orca: A Distributed Serving System for Transformer-Based Generative Models*](https://www.usenix.org/system/files/osdi22-yu.pdf)；
- [Inside vLLM](https://www.aleksagordic.com/blog/vllm)；
- [*Fast Transformer Decoding: One Write-Head is All You Need*（MQA）](https://arxiv.org/pdf/1911.02150)；
- [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/pdf/2305.13245)；
- [*DeepSeek-V2*（MLA）](https://arxiv.org/pdf/2405.04434)；
- [*Grouped Latent Attention*](https://arxiv.org/pdf/2505.21487)。
