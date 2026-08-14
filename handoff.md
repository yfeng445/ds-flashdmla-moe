# Project Handoff

更新时间：2026-08-14

实现状态分析基线（不含本文档更新）：`main` / `eb7fea9` (`upd. infra面经`)

进行中分支：`agent/mla-cuda-interview-prep` / `d9118e1`

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

当前可视为 **v0.1 correctness milestone**。`main` 的 CPU/reference 路线完整且持续通过
CI；CUDA 源码已经接入，但默认分支尚未完成 GPU 实机闭环。Draft PR #1 已经打通 CUDA
wheel 构建并加入 absorbed MLA CUDA kernel，不过仍有一个 FakeTensor stride contract
失败，尚未达到可合并状态。

粗略进度：只计算 `main` 约为 **70%**；把 Draft PR #1 的未合并工作计算在内约为
**80%**。这里的百分比衡量的是学习/研究仓库的完成度，不代表生产可用性。

| 方向 | 状态 | 说明 |
| --- | --- | --- |
| 课程讲义 | 基本完成 | `docs/chapters/00`–`08`，覆盖 tiling、在线 Softmax、FlashAttention、MLA、MoE、EP、自定义算子、benchmark/roofline、对称内存 |
| PyTorch reference | 基本完成 | GEMM、Attention、MLA、grouped Top-K、SwiGLU MoE、路由、Expert Parallel |
| CPU/Gloo 验证 | 已完成 | 单元测试、梯度检查、非规则 shape、空 expert、空 rank、两 rank Gloo |
| CUDA 算子源码 | `main` 有 8 个算子 | Attention forward/backward、tiled GEMM、router、route pack/combine、expert-major pack、SwiGLU experts；PR #1 另加 absorbed MLA |
| FP16 Tensor Core | 已实现，有分支实机记录 | expert-major SwiGLU 使用 WMMA，FP32 accumulation，显式 FP16 hidden boundary；默认分支仍缺正式 GPU CI |
| NCCL Expert Parallel | 代码已实现，待双 GPU 验证 | 包括可微 All-to-All 和异步 chunk pipeline |
| MLA CUDA | Draft PR 中 | PR #1 新增 fused FP32 absorbed-attention core；尚未合并，且不是完整生产级 prefill/decode backend |
| One-sided/NVSHMEM | 未实现 | 当前只有 symmetric-buffer 成本与布局模型 |
| 性能结论 | 尚不可下结论 | 缺少真实 GPU benchmark、Nsight trace 和对照基线 |

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
- `src/ds_flash_mla_moe/*_benchmarking.py`：结构化 benchmark 与报告模型。
- `csrc/`：supported CUDA/C++ extension 源码。
- `csrc/experimental/`：未验证的课程时期原型，不属于 supported API。
- `benchmarks/`：GEMM、Attention、MLA、router、experts、Expert Parallel CLI。
- `tests/`：数值、梯度、dispatcher、benchmark schema 和 distributed contract 测试。
- `docs/`：讲义、练习、阅读顺序和参考资料。
- `AI INFRA.ipynb`：已经进入 `main` 的面试笔记，包含项目介绍、FA1/FA2、FlashMoE、
  token tile 生命周期和公司面试记录等主题。
- `handoff.md`：当前事实状态、阻塞项与下一步执行顺序；修改状态时应同步更新。

## 4. 已验证状态

在当前 `main@eb7fea9` 上于 2026-08-14 重新运行：

```text
pytest -ra --strict-markers -W error::UserWarning
271 passed, 52 skipped in 12.61s
```

另已完成：

- `ruff format --check`：通过；
- `ruff check`：通过；
- `git diff --check`：通过；
- Python `compileall`：通过；
- Markdown 本地链接检查：通过；
- sdist/wheel 构建和隔离安装 smoke test：通过；
- GitHub Actions 当前主线 `Reference tests`：通过：
  <https://github.com/yfeng445/ds-flashdmla-moe/actions/runs/31714103979>。

主线的 52 个 skip 都依赖本机没有的 CUDA/native extension。主线工作区与
`origin/main` 同步；仓库当前没有 release，也没有 open issue。

### 4.1 Draft PR #1 的额外验证

[Draft PR #1: Add fused MLA CUDA kernel and interview prep](https://github.com/yfeng445/ds-flashdmla-moe/pull/1)
尚未进入 `main`，但已经取得以下进展：

- `CUDA build / wheel` GitHub Actions 已成功，说明修复后的容器环境能够编译 native
  extension 和新增 MLA CUDA 源码；
- PR 描述记录了 Windows CPU `281 passed, 62 skipped`、WSL CUDA 12.8 / RTX 5090
  `343 passed`、MLA CUDA focused suite `36 passed`，以及多组 CUDA smoke benchmark；
- 上述本地 GPU 数据属于 PR 作者记录；GitHub 上当前直接可复核的 GPU 相关证据是
  CUDA wheel build 成功，不能替代完整 self-hosted GPU CI 或 profiler 结果。

## 5. 当前阻塞与已知缺口

### 5.1 Draft PR #1 尚未通过 CPU opcheck

PR #1 当前是 `OPEN`、`DRAFT`、`MERGEABLE`，但 check 状态为 `UNSTABLE`。CUDA wheel
已经构建成功；实际阻塞来自 Python 3.10 reference test：

```text
tests/test_mla.py::test_raw_absorbed_operator_passes_opcheck
FakeTensor stride: (36, 12, 3, 1)
concrete stride:   (9, 3, 18, 1)
```

新增 `ds_flash_mla_moe::mla_absorbed_attention` 的 FakeTensor/meta implementation 返回了
contiguous layout，而真实 reference 输出经过维度换位，保留了不同 stride。需要让 meta
kernel 精确复现真实输出的 shape、stride、dtype、device 和 `requires_grad` contract，再运行
完整 `torch.library.opcheck`。在此问题解决、Python 3.10/3.12 checks 全绿前，不应合并 PR。

旧的主线 CUDA workflow 曾因升级 Debian 系统 pip 而失败；PR #1 已改用隔离环境并让
CUDA wheel build 通过。因此“CUDA 尚未进入编译阶段”已经是过期状态。

### 5.2 GPU workflows 尚未运行

- `.github/workflows/cuda-tests.yml` 依赖标签为
  `[self-hosted, linux, x64, cuda]` 的 runner；
- `.github/workflows/nccl-expert-parallel.yml` 依赖额外带 `multi-gpu` 标签的双 GPU runner；
- 两者目前都是手动触发。虽然 PR 作者已记录单 GPU 本地结果，但仓库仍没有可持续的
  self-hosted 单 GPU/双 GPU CI 结果，也没有 NCCL benchmark artifact。

### 5.3 算法与性能缺口

- Attention CUDA 仍是 correctness-first FP32 路径，不是 FA2/FA3 级实现；
- grouped router 使用 one-thread-per-token 的串行候选扫描；
- MoE kernel 尚无 async copy、TMA、WGMMA 或 profiler-driven tuning；
- absorbed MLA CUDA core 尚在 Draft PR，且未覆盖完整生产级 MLA prefill/decode backend；
- chunk pipeline 只证明了软件异步协议，尚无 profiler 证据证明物理 overlap；
- one-sided symmetric memory 只有分析模型，没有 NVSHMEM/PGAS backend。

## 6. 下一步执行顺序

1. 在 PR #1 中修复 `mla_absorbed_attention` 的 FakeTensor/concrete stride mismatch。
2. 重跑完整 `torch.library.opcheck`，让 Python 3.10/3.12 Reference CI 全绿。
3. 复核 PR 的 9 个 CUDA dispatch kernels、CUDA wheel artifact 和 README 数值契约后合并。
4. 在单 GPU self-hosted runner 上运行完整 CUDA tests 与 GEMM、Attention、MLA、experts、
   router smoke benchmark。
5. 在双 GPU runner 上验证 NCCL FP32、FP16 WMMA 与 chunked pipeline 的 forward/backward。
6. 保存硬件、CUDA、PyTorch、shape、dtype、误差和原始 latency 样本。
7. 用 Nsight Systems/Compute 检查 kernel bottleneck 与通信计算 overlap。
8. 扩展完整 MLA prefill/decode CUDA backend，再考虑 FA2/FA3、TMA/WGMMA 和 one-sided EP。

前四步完成前，不应在 README 中加入“高性能”“快于某实现”等未经证实的结论。

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

- 状态分析时 `main@eb7fea9` 与 `origin/main` 同步，工作区是干净的；
- `AI INFRA.ipynb` 已由提交 `eb7fea9` 纳入 `main`，不再是未跟踪文件；
- 当前唯一 open PR 是 Draft PR #1，目标为 `main`；
- 当前没有 open issue 和 GitHub release；
- 不要把 PR #1 中的 fused MLA CUDA、两篇衍生面试文档或 9-operator 状态描述成
  `main` 已有能力，直至 PR 真正合并。

## 9. 完成定义

本项目可以继续称为 correctness-first 学习仓库；若要称为“完善的高性能 MLA + MoE
实现”，至少还需要同时满足：

1. 默认分支 CPU、CUDA build、单 GPU CUDA tests 和双 GPU NCCL tests 全绿；
2. MLA、Attention、router、experts 和 EP 都有固定 shape/dtype 的可复现 benchmark；
3. 对照 PyTorch SDPA、cuBLAS/CUTLASS 或主流 FlashAttention 实现报告误差与性能；
4. Nsight 证据支持 kernel 利用率和通信计算 overlap 结论；
5. README 中每个性能声明都能追溯到环境信息、原始样本和生成命令。
