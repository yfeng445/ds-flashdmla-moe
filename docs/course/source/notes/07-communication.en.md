# Week 7: Distributed Memory, MPI/NCCL, and Collectives

- Original page: [Week 7: Communication](https://distinct-capricorn-c04.notion.site/Week-7-Communication-32088315b6b4808983bdf39dd7ca922a)
- Snapshot prepared: 2026-08-17
- Chinese version: [Week 7 notes](../../notes/07-communication.md)

## Distributed-Memory Systems

A distributed-memory system contains processors or nodes joined by an
interconnect. Processes own separate address spaces, run asynchronously, and
exchange data through explicit point-to-point or collective operations. Message
passing is the programming model; MPI is an interface specification with
multiple concrete implementations.

Connections may be intra-node, such as PCIe or NVLink/NVSwitch inside one
server, or inter-node, such as Ethernet, InfiniBand, or an external NVLink
fabric. The source uses HGX H100 examples to illustrate an eight-GPU any-to-any
node and larger multi-node fabrics. Product bandwidth and topology are
generation-specific and should be checked against the target system.

## Minimal MPI Program

A typical MPI program:

1. calls `MPI_Init`;
2. obtains its rank with `MPI_Comm_rank`;
3. obtains communicator size with `MPI_Comm_size`;
4. runs the same program over rank-specific data;
5. calls `MPI_Finalize`.

Output order is not deterministic because ranks progress independently.
`mpicc` is a compiler wrapper and `mpirun -np N` starts N processes. MPI does
not require GPUs and applies to CPU-only distributed programs as well.

## Collective Semantics

| Collective | Data relationship | Common use |
| --- | --- | --- |
| All-Reduce | reduce across ranks and return the full result to every rank | data-parallel gradient synchronization |
| Reduce | reduce across ranks and return the result only to a root | metric aggregation |
| All-Gather | concatenate every shard and return the full result to every rank | parameter or activation reconstruction |
| Gather | concatenate every shard only at a root | centralized saving or analysis |
| Scatter | split a root buffer among ranks | input or shard distribution |
| Reduce-Scatter | reduce, then distribute equal result shards | sharded gradient or state updates |
| All-to-All | send a different shard from every rank to every rank | MoE token dispatch and combine |

Collectives are valuable because their implementations account for topology and
endpoint limits. Ring, tree, hierarchical, and other schedules distribute large
transfers instead of flooding one receiver with every sender simultaneously.

## MPI and NCCL

- MPI covers general message passing, CPU buffers, and broad distributed control
  patterns;
- NCCL focuses on NVIDIA GPU collectives and can enqueue communication on CUDA
  streams while using NVLink, PCIe, and network topology;
- MPI can manage processes while NCCL transports GPU tensors;
- all ranks must participate with compatible order, shapes, dtypes, and
  communicators.

## DeepSeekMoE Reference Test Generation

The source assignment first uses Hugging Face DeepSeek-V3 blocks to create small,
deterministic test cases before implementing the MoE operator in sequential C:

- isolate MLP, router, and related blocks from `modeling_deepseek_v3.py`;
- shrink hidden dimensions and expert counts for readable tests;
- fix seeds and enable deterministic execution;
- use different seeds for weights and inputs;
- record weights, inputs, and reference outputs together;
- run generation twice to verify the generator is deterministic;
- test MLP, top-k routing, dispatch, experts, combine, and then the end-to-end
  data flow.

Full model weights are too large for unit-test fixtures. Small random weights
lock down operator semantics rather than model quality.

## Further Reading

- [HGX H100 Platform Overview](https://developer.nvidia.com/blog/introducing-nvidia-hgx-h100-an-accelerated-server-platform-for-ai-and-high-performance-computing);
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl);
- [*Demystifying NCCL*](https://arxiv.org/pdf/2507.04786v1);
- [*DeepSeekMoE*](https://arxiv.org/pdf/2401.06066);
- [Hugging Face DeepSeek-V3 Modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py);
- [DeepSeekMoE](../../../chapters/04-deepseek-moe.md);
- [Expert Parallelism](../../../chapters/05-expert-parallelism.md).
