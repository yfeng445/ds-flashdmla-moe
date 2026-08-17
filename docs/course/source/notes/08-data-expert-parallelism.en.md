# Week 8: Data and Expert Parallelism

- Original page: [Week 8: Data & Expert Parallelism](https://distinct-capricorn-c04.notion.site/Week-8-Data-Expert-Parallelism-32688315b6b480c6b66bf9830dfc3cc6)
- Snapshot prepared: 2026-08-17
- Chinese version: [Week 8 notes](../../notes/08-data-expert-parallelism.md)

## From Dense FFNs to MoE

A dense Transformer activates the same FFN parameters for every token. A
Mixture of Experts layer uses a router to select a small subset of experts for
each token. It can therefore increase total parameter count while keeping
active computation per token approximately bounded, at the cost of routing,
load balancing, token permutation, and distributed communication.

For `N` experts, a router first produces affinities:

```text
r_i = Router(x_i) in R^N
SoftmaxRouter(x_i) = softmax(x_i W + b)
TopKRouter(x_i) = softmax(KeepTopK(x_i W + b))
```

For selected set `T_i`, the result is a weighted sum:

```text
y_i = sum_{e in T_i} r_{i,e} Expert_e(x_i)
```

Selection with `K=1` is commonly called Switch-style routing. Real systems must
also define capacity, overflow behavior, and balancing losses.

## DeepSeekMoE Expert Organization

DeepSeekMoE introduces two central ideas:

- **fine-grained expert segmentation** divides the intermediate dimension of a
  conventional expert by `m`, expands `N` experts to `mN`, and expands `K`
  selections to `mK`; active capacity stays similar while the combination space
  grows;
- **shared expert isolation** keeps shared experts active for common knowledge
  while routed experts learn more differentiated patterns.

## DP, TP, and EP

| Parallel mode | Primary partition | Common communication | Main constraint |
| --- | --- | --- | --- |
| Data parallelism | batch or token samples | gradient All-Reduce or Reduce-Scatter | each rank usually holds a model replica |
| Tensor/model parallelism | layer weights and matrix products | All-Reduce, All-Gather, Reduce-Scatter | frequent intra-layer communication |
| Expert parallelism | experts | two All-to-All operations | routing balance and network bandwidth |

EP commonly replicates the router and shards experts across GPUs. This assumes
one expert fits on one GPU; larger experts may use TP internally.

## Expert-Parallel Forward Data Flow

A typical forward pass performs:

1. local routing to select destination experts and weights;
2. local permutation by destination rank and expert;
3. an All-to-All that dispatches tokens to expert owners;
4. local expert computation;
5. a second All-to-All that returns outputs to original token owners;
6. local unpermutation and router-weighted combination.

Performance depends on both GEMM and dispatch/combine bandwidth, message size,
load skew, padding or capacity waste, and communication-computation overlap.

## Hybrid EP, TP, and DP

Large systems combine parallel dimensions. A simplified flow applies attention
inside a TP group, routes locally, exchanges tokens across an EP group, executes
experts that may themselves use TP, and then reverses the All-to-All before
restoring token order.

Each collective must be associated with the correct process-group dimension.
Topology mapping can keep frequent TP traffic within a node while assigning EP
or DP to a suitable inter-node fabric.

## Assignment Direction

The source assignment asks students to implement multi-GPU DeepSeekMoE with
CUDA and NCCL, combine data and expert parallelism, reuse generated deterministic
tests, and compare against a Transformers reference. This is a course task and
does not claim that the repository implements the multi-GPU path.

## Further Reading

- [*Ring Attention with Blockwise Transformers for Near-Infinite Context*](https://arxiv.org/pdf/2310.01889);
- [*DeepSpeed-Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models*](https://arxiv.org/pdf/2309.14509);
- [*Efficient Training of MoE Models at Scale with Pytorch*](https://arxiv.org/pdf/2303.06318);
- [*MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models*](https://arxiv.org/pdf/2505.11432);
- [DeepSeekMoE](../../../chapters/04-deepseek-moe.md);
- [Expert Parallelism](../../../chapters/05-expert-parallelism.md).
