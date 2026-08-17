# 第 11 周：AI Infra 职业能力建设

- 原始页面：[Week 11: Career Building](https://distinct-capricorn-c04.notion.site/Week-11-Career-Building-34388315b6b4808fa70de9b0502a2d15)
- 整理日期：2026-08-17
- 英文版：[Week 11: Career Building](../source/notes/11-career-building.en.md)

## 加入专业社区

专业社区可以把课程知识连接到真实工程环境：直接向领域从业者学习、加快问题定位、观察工具与最佳实践、了解不同应用场景，并通过 workshop、会议、社区 call 和开源协作建立长期联系。

课程列出的入口包括：

1. [SC（Supercomputing）](https://supercomputing.org/)：HPC/AI 的大型年度会议，包含 job fair、Birds-of-a-Feather、workshop、poster 与社区活动；
2. [hpc.social](https://hpc.social/)：常态化 HPC 社区，提供 jobs board、Slack/Discord 与 HPC Hallway call；
3. [ISC](https://isc-hpc.com/)：欧洲重要 HPC 社区与会议；
4. [PyTorch Community Hub](https://pytorch.org/community-hub)：适合 applied AI systems、training infra 和 ML platform 方向，可通过论坛、Slack、活动和开源贡献参与；
5. [OpenXLA](https://openxla.org/)：面向 compiler/runtime、StableHLO 与系统基础设施；
6. [MLCommons](https://mlcommons.org/get-involved/)：围绕 AI benchmark、dataset 和工程最佳实践的开放联盟；
7. [NVIDIA Developer Discord](https://discord.com/invite/nvidiadeveloper) 与 [Developer Forums](https://forums.developer.nvidia.com/latest)：CUDA、GPU 工具和平台问题；
8. [HPC-AI Society](https://hpc-ai-society.org/) 与 [HPC Carpentry](https://www.hpc-carpentry.org/)：社区活动与实践型学习资源。

加入社区后的重点不是一次性“求内推”，而是持续提出可复现的问题、分享解决过程、参与讨论，并逐渐形成可信的技术记录。

## 参与开源

课程把可参与的生态分成成熟项目与新兴项目。

成熟项目包括：

- PyTorch：framework、dispatcher、compiler、distributed 与 kernel 生态；
- vLLM：LLM serving、continuous batching 与 KV cache；
- Megatron-LM：大规模分布式训练；
- DeepSpeed：训练并行、ZeRO 与推理；
- OpenXLA：面向 AI 的开放 compiler/runtime 基础设施。

新兴或快速演进的方向包括：

- SGLang 及其生态；
- Miles 等 post-training / reinforcement-learning framework；
- llm-d：Kubernetes 上的分布式推理、调度与 KV-cache 优化；
- SkyPilot：跨 cloud/cluster 的 AI workload 调度与编排；
- TensorRT-LLM：NVIDIA 平台上的高性能推理；
- Colossal-AI：分布式训练。

项目归属、融资和岗位会变化；这里保留课程列举的技术方向，实际参与前应检查当前维护状态、贡献指南、issue 质量和社区响应速度。适合从文档、测试、最小 bug 修复、benchmark 复现或小型工具开始，而不是一开始就提交大规模重构。

## 自学：广度与深度并行

### 论文广度

定期浏览顶会论文的 title、abstract、introduction、图表和 evaluation，建立“当前社区在解决什么问题”的地图：

- HPC：SC、ISC；
- AI：NeurIPS、ICLR、ICML 中的 systems/HPC 工作；
- Systems：SOSP、EuroSys 等会议中的训练、推理与调度论文。

广度阅读不等于每篇精读。目标是识别问题、约束、方法、baseline 和结论边界。

### 复现深度

定期复现高质量论文能形成更稀缺的能力。一个有效复现应记录：

1. 论文主张与必要前提；
2. baseline、数据、shape、dtype、硬件和软件版本；
3. correctness 标准与误差；
4. profiler 证据与性能差异；
5. 未复现结论及可能原因。

## 为练习而构建系统

### 分布式训练

- 阅读或重建 [Nanotron](https://github.com/huggingface/nanotron)，理解一个较小的 Megatron 风格训练系统；
- 配合 [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) 学习 DP、TP、PP、ZeRO/FSDP 与通信调度；
- 从小模型和两进程 correctness 开始，再增加并行维度和性能分析。

### 集群任务编排

- Kubernetes 是企业集群常见的资源与服务编排基础，但分布式训练与 GPU 拓扑感知仍需要额外控制层；
- Ray 等系统尝试改善分布式 AI workload 的调度体验；
- 一个实践方向是在 GPU Kubernetes 集群上实现 vLLM 实例的 autoscaling，并处理排队、冷启动、KV 容量和流量分配；
- Slurm 仍广泛用于传统 HPC 与科学计算环境。

分布式 kernel 与集群编排之间的层次仍有大量工程问题，适合作为系统练习方向。

## 专项能力

### Profiling

使用 NVIDIA Nsight Systems/Compute 对一个 kernel 建立瓶颈假设、收集 timeline/counter、实施修改并复测。重点不是会打开工具，而是能用证据区分计算、内存、同步、launch 和通信瓶颈。

### HPC Networking

学习多 GPU 服务器与集群的拓扑、带宽层次和故障定位，包括 PCIe、NVLink/NVSwitch、InfiniBand/RoCE、collective 算法、拥塞与 topology mapping。课程以高带宽 GPU 集群为例；具体带宽应以目标硬件为准。

### Compilers

[Triton](https://triton-lang.org/main/index.html) 是从 CUDA 知识延伸到编译器生成 GPU kernel 的合适入口。学习时仍需理解 program instance、layout、coalescing、shared memory、occupancy 和 autotuning，而不是只记 API。

### Mixed Precision 与 Quantization

需要同时掌握数值范围、accumulation dtype、scale/zero-point、校准、误差传播、硬件指令支持，以及训练/推理中的精度—带宽—吞吐权衡。

## 岗位信息的使用方式

源页面以 SGLang 生态的一则岗位为例，说明应从真实 JD 反推能力需求。岗位链接变化很快，可从项目官网、社区 jobs board 和公司招聘页重新查找。阅读 JD 时建议拆成：核心问题域、必备编程栈、分布式/硬件要求、性能验证方式和加分项，再据此安排学习与产出。

## 与通用指南的关系

[AI Infra 知识与面试准备指南](../../infra-interview-guide.md) 将本页的职业建设内容与 attention、MLA、MoE、通信、推理和性能分析知识重新组织为可执行的准备路线；本页则保留课程源页面的原始顺序和覆盖范围。
