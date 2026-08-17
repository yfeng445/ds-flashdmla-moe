# Week 11: Career Building for AI Infrastructure

- Original page: [Week 11: Career Building](https://distinct-capricorn-c04.notion.site/Week-11-Career-Building-34388315b6b4808fa70de9b0502a2d15)
- Snapshot prepared: 2026-08-17
- Chinese version: [Week 11 notes](../../notes/11-career-building.md)

## Join Professional Communities

Professional communities connect coursework to real engineering practice. They
provide access to domain experts, faster problem solving, current tools and
methods, practical use cases, learning events, and long-term relationships.

The source lists these entry points:

1. [SC](https://supercomputing.org/), a major HPC and AI conference with a job
   fair, Birds-of-a-Feather sessions, workshops, posters, and community events;
2. [hpc.social](https://hpc.social/), an always-on community with a jobs board,
   Slack or Discord, and the HPC Hallway call;
3. [ISC](https://isc-hpc.com/), a major European HPC conference and community;
4. [PyTorch Community Hub](https://pytorch.org/community-hub), with forums,
   Slack, events, and contribution paths for applied AI systems, training
   infrastructure, and ML platforms;
5. [OpenXLA](https://openxla.org/), focused on compilers, runtimes, StableHLO,
   and systems infrastructure;
6. [MLCommons](https://mlcommons.org/get-involved/), an open consortium for AI
   benchmarks, datasets, and engineering practices;
7. [NVIDIA Developer Discord](https://discord.com/invite/nvidiadeveloper) and
   [Developer Forums](https://forums.developer.nvidia.com/latest) for CUDA, GPU
   tooling, and platform questions;
8. [HPC-AI Society](https://hpc-ai-society.org/) and
   [HPC Carpentry](https://www.hpc-carpentry.org/) for events and practical
   learning resources.

The durable strategy is to ask reproducible questions, share solutions, join
technical discussion, and build a visible record over time.

## Contribute to Open Source

The source separates established projects from newer, fast-moving systems.

Established projects include PyTorch, vLLM, Megatron-LM, DeepSpeed, and OpenXLA.
Together they cover frameworks, serving, distributed training, state sharding,
compilers, and runtimes.

Emerging directions include the SGLang ecosystem, post-training and
reinforcement-learning frameworks such as Miles, llm-d for distributed
Kubernetes inference, SkyPilot for cross-cloud or cross-cluster orchestration,
TensorRT-LLM for optimized NVIDIA inference, and Colossal-AI for distributed
training.

Ownership, funding, and hiring status change over time. Before contributing,
check current maintenance, contribution guides, issue quality, and community
response. Documentation, tests, minimal bug fixes, benchmark reproduction, and
small tools are useful starting points.

## Self-Study: Breadth and Depth

For breadth, regularly scan titles, abstracts, introductions, diagrams, and
evaluations from:

- HPC venues such as SC and ISC;
- AI venues such as NeurIPS, ICLR, and ICML when they publish systems work;
- systems venues such as SOSP and EuroSys.

Breadth reading builds a map of problems, constraints, methods, baselines, and
claim boundaries. Depth comes from reproducing strong papers. A reproduction
should record the claim and assumptions, workload and environment, correctness
criteria, profiler evidence, and any result that could not be reproduced.

## Build Systems for Practice

### Distributed Training

- study or rebuild [Nanotron](https://github.com/huggingface/nanotron) as a
  compact Megatron-style training system;
- use the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
  to study DP, TP, PP, ZeRO or FSDP, and communication scheduling;
- begin with small models and two-process correctness before adding parallel
  dimensions and performance analysis.

### Cluster-Wide Orchestration

Kubernetes is common for enterprise cluster resource and service management,
but distributed training and topology-aware GPU scheduling need additional
layers. Ray addresses parts of the distributed AI workload experience. One
practice direction is autoscaling vLLM instances over a Kubernetes GPU cluster
while handling queues, cold starts, KV capacity, and traffic allocation. Slurm
remains common in traditional HPC and scientific-computing environments.

The layer between distributed kernels and cluster orchestration remains a rich
engineering area.

## Specialized Skills

### Profiling

Use NVIDIA Nsight Systems and Compute to form a bottleneck hypothesis for a
kernel, collect timelines or counters, make a change, and measure again. The
skill is evidence-based separation of compute, memory, synchronization, launch,
and communication bottlenecks.

### HPC Networking

Study multi-GPU server and cluster topology, bandwidth hierarchy, and failure
diagnosis across PCIe, NVLink or NVSwitch, InfiniBand or RoCE, collective
algorithms, congestion, and topology mapping. Exact bandwidth is specific to
the target hardware.

### Compilers

[Triton](https://triton-lang.org/main/index.html) is a natural extension from
CUDA into compiler-generated GPU kernels. Effective use still requires an
understanding of program instances, layout, coalescing, shared memory,
occupancy, and autotuning.

### Mixed Precision and Quantization

Study numerical range, accumulation dtype, scales and zero points, calibration,
error propagation, hardware instruction support, and the tradeoff among
precision, bandwidth, and throughput in training and inference.

## Using Job Descriptions

The source uses one SGLang-ecosystem opening as an example of deriving a study
plan from a real job description. Links change quickly, so refresh them through
project sites, community job boards, and company career pages. Decompose each
description into problem domain, programming stack, distributed or hardware
requirements, performance-validation expectations, and optional skills.

## Relation to the General Guide

The [AI Infrastructure Knowledge and Interview Guide](../../../infra-interview-guide.md)
reorganizes these career topics together with attention, MLA, MoE,
communication, inference, and performance analysis. This file preserves the
source page's topic order and coverage.
