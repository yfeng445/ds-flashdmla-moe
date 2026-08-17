# Week 10: Inference Systems, KV Cache, and PagedAttention

- Original page: [Week 10: Inference Systems](https://distinct-capricorn-c04.notion.site/Week-10-Inference-Systems-33488315b6b48083b81ecea51648dd7c)
- Snapshot prepared: 2026-08-17
- Chinese version: [Week 10 notes](../../notes/10-inference-systems.md)

## Prefill and Decode

Autoregressive LLM inference has two stages with different behavior:

| Stage | Work | Common bottleneck | Primary latency metric |
| --- | --- | --- | --- |
| Prefill | process all prompt tokens in parallel, produce the first token and per-layer KV cache | compute-intensive for long prompts | TTFT, or time to first token |
| Decode | read weights and existing KV state to generate one new token per step | usually memory-bandwidth-bound | ITL, or inter-token latency |

Systems also track end-to-end latency, throughput, and token-generation
latency. Continuous batching, prefill/decode scheduling, and speculative decoding
balance request-level latency against aggregate throughput.

## Why KV Cache Is Necessary

Without caching, every generated token would recompute K/V for the complete
prefix. KV cache retains historical state at every layer. A 1,000-token prompt
followed by 100 output tokens then processes roughly 1,100 new token states
rather than repeatedly recomputing a growing prefix over 100 decode steps.

The cost is GPU memory capacity and bandwidth. For ordinary multi-head
attention, approximate capacity is:

```text
KV bytes = 2 * bytes_per_element * n_layers * batch
           * n_kv_heads * d_head * sequence_length
```

The leading factor accounts for K and V. The source's LLaMA 2 13B example uses
FP16, 40 layers, 40 KV heads, and `d_head=128`: approximately `0.78125 MB` per
token and `3.125 GB` for 4,096 tokens. One source bullet labels the latter as
MB; the arithmetic yields GB.

Attention variants reduce cache size through sharing or compression:

- MHA retains separate K/V for each query head;
- MQA shares one K/V head across query heads;
- GQA shares K/V within groups of query heads;
- MLA compresses history into a low-dimensional latent representation;
- GLA organizes grouped latent state for parallel partitioning.

## Fragmentation and PagedAttention

Conventional allocation may reserve a contiguous KV region sized for each
request's maximum length. This causes internal fragmentation from unused space
inside allocations and external fragmentation when total free capacity cannot
satisfy a large contiguous allocation. Both reduce batch size and supported
sequence length.

PagedAttention divides a request's logical KV sequence into fixed-size logical
blocks and maps them through a block table to noncontiguous physical blocks.
Physical capacity grows and is reclaimed on demand, with no need to reserve a
maximum-length contiguous buffer. Read-only blocks can also be shared across
common prefixes or beams. The tradeoffs are mapping and metadata overhead plus
an attention kernel that follows the block table.

## Static and Continuous Batching

Static batching collects a fixed batch and runs it until every request
finishes. Slots for shorter requests can remain idle while waiting for the
longest request, and queued work cannot enter immediately.

Continuous batching rebuilds the active batch at decode-iteration boundaries.
When a request finishes, another request can take its slot. Utilization improves,
but the scheduler must manage arrivals, priority, KV blocks, prefill chunking,
and throughput versus tail latency.

## A Systems View of Inference Optimization

- continuous batching reduces idle slots;
- chunked prefill and prefill/decode scheduling limit interference from long
  prompts;
- speculative decoding generates candidates with a cheaper draft and verifies
  them in parallel with the target model;
- KV management controls active-request capacity;
- kernel and scheduler coordination determines whether theoretical savings
  improve TTFT, ITL, and throughput in practice.

Evaluation should report workload, input and output length distributions,
concurrency, precision, and hardware rather than only peak tokens per second.

## Assignment Direction

The source assignment points to [Nano Sglang](https://github.com/lixiaohua-neu/nano-sglang)
as a compact codebase for studying the connection between request scheduling,
batching, KV cache, and model execution.

## Further Reading

- [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/pdf/2309.06180);
- [*Orca: A Distributed Serving System for Transformer-Based Generative Models*](https://www.usenix.org/system/files/osdi22-yu.pdf);
- [Inside vLLM](https://www.aleksagordic.com/blog/vllm);
- [*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/pdf/1911.02150);
- [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/pdf/2305.13245);
- [*DeepSeek-V2*](https://arxiv.org/pdf/2405.04434);
- [*Grouped Latent Attention*](https://arxiv.org/pdf/2505.21487).
