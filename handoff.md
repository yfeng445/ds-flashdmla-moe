# Project Handoff

更新时间：2026-08-14

当前基线：`main`，已包含 staged end-to-end MLA CUDA、稳定输出布局契约和单 GPU 成对基线快照。

远端：<https://github.com/yfeng445/ds-flashdmla-moe>

## 1. 项目目标

本仓库把 INFO 7375 *High Performance Computing for AI* 的课程内容整理成一个
correctness-first 的 DeepSeek MLA + MoE 学习与实现项目。核心原则是：

1. 先用可读的 PyTorch reference 固定数学与数值语义；
2. 用单元测试、梯度检查和边界 shape 验证 reference；
3. 再接入 CUDA、自定义算子和多 GPU 通信；
4. 只有获得可复现实机数据后，才提出性能结论。

课程源材料位于另一个本地仓库的 `3/INFO 7375`，不要误用 `4/INFO7375`。
课程作业源码没有被原样搬入 supported API；旧 CUDA 示例已迁移到
`csrc/experimental/attention/`，仅用于学习和对照。

## 2. 当前完成度

当前可视为 **v0.1 correctness + local single-GPU smoke milestone**。`main` 的
CPU/reference 路线持续通过 Python 3.10/3.12 CI，CUDA wheel 能编译并注册 14 个 native
算子；RTX 5090 本地环境也已跑通完整 CUDA 测试、固定 shape smoke benchmark 和 20-case
代表性 shape matrix。尚未完成持续 self-hosted GPU CI、双 GPU NCCL、profiler 和主流
第三方实现对照，因此仍不具备生产性能结论。

粗略进度约为 **85%**。这里的百分比衡量的是学习/研究仓库的完成度，不代表生产可用性。

| 方向 | 状态 | 说明 |
| --- | --- | --- |
| 课程讲义 | 基本完成 | `docs/chapters/00`–`08`，覆盖 tiling、在线 Softmax、FlashAttention、MLA、MoE、EP、自定义算子、benchmark/roofline、对称内存 |
| PyTorch reference | 基本完成 | GEMM、Attention、MLA、grouped Top-K、SwiGLU MoE、路由、Expert Parallel |
| CPU/Gloo 验证 | 已完成 | 单元测试、梯度检查、非规则 shape、空 expert、空 rank、两 rank Gloo |
| CUDA 算子源码 | `main` 有 14 个算子 | Attention forward/backward、tiled GEMM、router、route pack/combine、expert-major pack、SwiGLU experts，以及 6 个 staged MLA 算子 |
| FP16 Tensor Core | 已实现并完成本地单卡 smoke | expert-major SwiGLU 使用 WMMA、FP32 accumulation 和显式 FP16 hidden boundary；仍缺持续 GPU CI |
| NCCL Expert Parallel | 代码已实现，待双 GPU 验证 | 包括可微 All-to-All 和异步 chunk pipeline |
| MLA CUDA | end-to-end correctness backend 已进入 `main` | direct/LoRA query、KV projection/static write、absorbed attention、output projection 均有 FP32 native op；低精度、paged cache 与生产级调优仍未实现 |
| One-sided/NVSHMEM | 未实现 | 当前只有 symmetric-buffer 成本与布局模型 |
| 性能结论 | 尚不可下结论 | 已有单卡固定 shape 与 20-case matrix 原始样本，但仍缺持续 runner、Nsight trace 和 CUTLASS/主流 FlashAttention 对照 |

## 3. 代码地图

- `src/ds_flash_mla_moe/attention.py`：materialized 与 blockwise online attention reference。
- `src/ds_flash_mla_moe/gemm.py`：GEMM 数值契约与 tiled teaching reference。
- `src/ds_flash_mla_moe/mla.py`：MLA prefill、decode、compressed cache 与 absorbed 路径。
- `src/ds_flash_mla_moe/moe.py`：DeepSeek-style grouped routing 和 MoE reference。
- `src/ds_flash_mla_moe/router_ops.py`：group-limited Top-K API 与 backend selection。
- `src/ds_flash_mla_moe/route_ops.py`：route pack/combine 的 reference/native 接口。
- `src/ds_flash_mla_moe/expert_ops.py`：active-row expert-major SwiGLU 接口。
- `src/ds_flash_mla_moe/expert_parallel.py`：Gloo/NCCL Expert Parallel 协议与 autograd。
- `src/ds_flash_mla_moe/ops.py`：PyTorch dispatcher、FakeTensor、autograd 和 CUDA 注册。
- `src/ds_flash_mla_moe/*_benchmarking.py`：结构化 benchmark、成对 shape matrix 与报告模型。
- `csrc/`：supported CUDA/C++ extension 源码。
- `csrc/experimental/`：未验证的课程时期原型，不属于 supported API。
- `benchmarks/`：GEMM、Attention、MLA、router、experts、成对 matrix、Expert Parallel CLI。
- `tests/`：数值、梯度、dispatcher、benchmark schema 和 distributed contract 测试。
- `validation/`：带环境、误差和原始 latency 的硬件验证快照。
- `docs/`：讲义、练习、阅读顺序和参考资料。
- `AI INFRA.ipynb`：已经进入 `main` 的面试笔记，包含项目介绍、FA1/FA2、FlashMoE、
  token tile 生命周期和公司面试记录等主题。
- `handoff.md`：当前事实状态、阻塞项与下一步执行顺序；修改状态时应同步更新。

## 4. 已验证状态

2026-08-14 本地 Windows Python 3.12 / PyTorch 2.10 运行：

```text
pytest -ra --strict-markers -W error::UserWarning
295 passed, 69 skipped in 9.56s
```

另已完成：

- `ruff format --check`：通过；
- `ruff check`：通过；
- `git diff --check`：通过；
- GitHub Actions `Reference tests` 的 Python 3.10/3.12 matrix：通过：
  <https://github.com/yfeng445/ds-flashdmla-moe/actions/runs/31767920497>；
- GitHub Actions `CUDA build / wheel`：通过，并检查 14 个 CUDA dispatch kernel；wheel
  同时携带 forward-compatible PTX：
  <https://github.com/yfeng445/ds-flashdmla-moe/actions/runs/31767920481>。

### 4.1 已解决的 MLA output-layout contract

此前 `mla_absorbed_attention` 的 Python composite 继承了 `einsum` 的非连续 stride，而
FakeTensor 和原生 CUDA kernel 返回 contiguous `[B,S,H,Dv]`。当前 composite 已显式
`.contiguous()`，并新增 concrete-layout 回归断言；完整 `torch.library.opcheck` 以及
Python 3.10/3.12 CI 均已通过。

### 4.2 本地单 GPU 验证

WSL2 / RTX 5090 / PyTorch 2.10 + CUDA 12.8 环境运行：

```text
pytest -ra --strict-markers -W error::UserWarning
354 passed in 25.76s
```

同一环境完成 GEMM、Attention、MLA attention-only、完整 MLA prefill、完整 MLA
static-cache decode、FP32 experts、FP16 WMMA experts 和 grouped router 的 9 组
native/PyTorch 成对 20-sample benchmark。全部数值验证通过，环境、固定 shape、误差、
latency 汇总和原始样本保存在
`validation/single-gpu/2026-08-14-rtx5090-cu128/`。这仍是本地快照，不是持续
self-hosted CI 或普适性能结论。

同一目录还保存了 20-case representative matrix：GEMM 4 组、Attention 4 组、MLA 5 组、
experts 4 组、router 3 组，共覆盖 5 个 regular、8 个 tail、4 个 decode 和 3 个 skew case。
20 组 native/baseline 均各自通过数值验证并保留 20 个 post-warmup raw samples。不同 family
使用 cuBLAS、SDPA、absorbed MLA、padded experts 或 PyTorch router reference，因此跨 case
汇总比值只是未加权描述统计，不是总体加速比。

当前固定 shape 的完整 MLA 数据为：`prefill_with_cache` native 1.155536 ms、absorbed
PyTorch 2.636864 ms；`decode_with_static_write` native 1.028400 ms、absorbed PyTorch
2.058480 ms。这里只记录本机诊断结果，不外推到其他 shape、dtype 或硬件。

## 5. 当前阻塞与已知缺口

### 5.1 持续 GPU workflows 尚未运行

- `.github/workflows/cuda-tests.yml` 依赖标签为
  `[self-hosted, linux, x64, cuda]` 的 runner；
- `.github/workflows/nccl-expert-parallel.yml` 依赖额外带 `multi-gpu` 标签的双 GPU runner；
- 两者目前都是手动触发。本地单 GPU 结果已经固化为验证快照，但仓库仍没有注册并持续
  运行的 self-hosted 单 GPU/双 GPU runner，也没有 NCCL benchmark artifact。

### 5.2 双 GPU 通信验证缺失

当前机器只有一张可见 GPU，无法证明 NCCL FP32、FP16 WMMA 和 chunked pipeline 的
forward/backward 正确性，也无法用 profiler 证明通信计算发生了物理 overlap。

### 5.3 算法与性能缺口

- Attention CUDA 仍是 correctness-first FP32 路径，不是 FA2/FA3 级实现；
- grouped router 使用 one-thread-per-token 的串行候选扫描；
- MoE kernel 尚无 async copy、TMA、WGMMA 或 profiler-driven tuning；
- staged MLA 已覆盖 FP32 correctness pipeline，但 prefill/decode 仍共用 correctness-first
  kernel；尚无 FP16/BF16、paged/per-slot cache、异步拷贝或 profiler-driven fusion；
- 代表性单 GPU shape matrix 已覆盖 regular/tail/decode/skew，但尚未接入 CUTLASS 或主流
  FlashAttention，也没有长上下文与更多低精度矩阵；
- chunk pipeline 只证明了软件异步协议，尚无 profiler 证据证明物理 overlap；
- one-sided symmetric memory 只有分析模型，没有 NVSHMEM/PGAS backend。

## 6. 下一步执行顺序

1. 注册并运行单 GPU self-hosted runner，让现有 354-test、9 组固定快照和 20-case matrix
   变成可持续 workflow artifact。
2. 在双 GPU runner 上验证 NCCL FP32、FP16 WMMA 与 chunked pipeline 的 forward/backward，
   保存每 rank 原始 latency 和通信量。
3. 用 Nsight Systems/Compute 检查 Attention、MLA、router、experts 的 kernel bottleneck，
   并验证 NCCL chunk pipeline 是否真正 overlap。
4. 在现有 20-case matrix 中加入可用的 CUTLASS 或主流 FlashAttention baseline，并按
   profiler 结果补充长上下文与低精度 case，而不是继续堆叠任意 shape。
5. 根据 profiler 把当前 staged MLA correctness backend 扩展为专用 prefill/decode、
   FP16/BF16 和 paged/per-slot cache，再考虑 FA2/FA3、TMA/WGMMA、router 优化和 one-sided EP。

前三步完成前，不应在 README 中加入“高性能”“快于某实现”等未经证实的结论。

## 7. 常用命令

CPU/reference 验证：

```bash
python -m pip install -e '.[test]'
ruff format --check src tests examples benchmarks setup.py
ruff check src tests examples benchmarks setup.py
pytest -ra --strict-markers -W error::UserWarning
```

CUDA 主机上的扩展构建：

```bash
python -m pip install '.[test,cuda-build]'
DS_FLASH_BUILD_CUDA=1 python -m pip install --no-build-isolation .
pytest -ra --strict-markers -W error::UserWarning
```

代表性单 GPU 成对矩阵：

```bash
python benchmarks/matrix.py --device cuda --profile representative \
  --warmup 5 --iterations 20 --seed 20260814 \
  --output benchmark-results/operator-matrix-representative.json
```

两 rank Gloo smoke test：

```bash
torchrun --master-addr=127.0.0.1 --master-port=29572 \
  --nproc-per-node=2 benchmarks/expert_parallel.py \
  --backend gloo --router-backend reference \
  --route-backend reference --expert-backend padded --dtype float64 \
  --tokens-per-rank 3 --token-skew 1 \
  --model-dim 4 --hidden-dim 5 --shared-experts 1 \
  --experts 4 --topk 1 --warmup 0 --iterations 1 --backward
```

更多参数和数值契约见 `README.md` 与 `docs/index.md`。

## 8. 分支与协作状态

- PR #1 已关闭且未 merge；其中的实现内容后来直接应用到 `main`。
- `main` 当前已经包含 staged end-to-end MLA CUDA、20-case operator matrix、两篇衍生面试文档和 14-operator native extension。
- `AI INFRA.ipynb` 是原始面试笔记；后续整理继续写入独立 Markdown，不覆盖原文件。
- 单 GPU JSON 是硬件相关证据快照；不要把它解释成跨实现性能领先结论。

## 9. 完成定义

本项目可以继续称为 correctness-first 学习仓库；若要称为“完善的高性能 MLA + MoE
实现”，至少还需要同时满足：

1. 默认分支 CPU、CUDA build、单 GPU CUDA tests 和双 GPU NCCL tests 全绿；
2. MLA、Attention、router、experts 和 EP 都有固定 shape/dtype 的可复现 benchmark；
3. 对照 PyTorch SDPA、cuBLAS/CUTLASS 或主流 FlashAttention 实现报告误差与性能；
4. Nsight 证据支持 kernel 利用率和通信计算 overlap 结论；
5. README 中每个性能声明都能追溯到环境信息、原始样本和生成命令。
