# Project Handoff

更新时间：2026-08-13

当前基线：`main` / `e3f5932` (`feat: build MLA and MoE study stack`)

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

当前可视为 **v0.1 correctness milestone**：CPU/reference 路线完整，CUDA 源码已接入，
但真实 GPU 编译、运行与性能验证尚未闭环。

| 方向 | 状态 | 说明 |
| --- | --- | --- |
| 课程讲义 | 基本完成 | `docs/chapters/00`–`08`，覆盖 tiling、在线 Softmax、FlashAttention、MLA、MoE、EP、自定义算子、benchmark/roofline、对称内存 |
| PyTorch reference | 基本完成 | GEMM、Attention、MLA、grouped Top-K、SwiGLU MoE、路由、Expert Parallel |
| CPU/Gloo 验证 | 已完成 | 单元测试、梯度检查、非规则 shape、空 expert、空 rank、两 rank Gloo |
| CUDA 算子源码 | 已实现，待实机闭环 | Attention forward/backward、tiled GEMM、router、route pack/combine、expert-major pack、SwiGLU experts |
| FP16 Tensor Core | 已实现，待实机闭环 | expert-major SwiGLU 使用 WMMA，FP32 accumulation，显式 FP16 hidden boundary |
| NCCL Expert Parallel | 代码已实现，待双 GPU 验证 | 包括可微 All-to-All 和异步 chunk pipeline |
| MLA CUDA | 未实现 | 当前只有 PyTorch naive/absorbed prefill/decode reference |
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

## 4. 已验证状态

本地在提交 `e3f5932` 前完成：

```text
pytest -ra --strict-markers -W error::UserWarning
271 passed, 52 skipped
```

跳过项均依赖本机没有的 CUDA/native extension。另已完成：

- `ruff format --check`：通过；
- `ruff check`：通过；
- `git diff --check`：通过；
- Python `compileall`：通过；
- Markdown 本地链接检查：通过；
- sdist/wheel 构建和隔离安装 smoke test：通过；
- GitHub Actions `Reference tests`：通过：
  <https://github.com/yfeng445/ds-flashdmla-moe/actions/runs/31711852219>。

## 5. 当前阻塞与已知缺口

### 5.1 CUDA build CI 尚未进入编译阶段

`CUDA build` workflow 当前失败：
<https://github.com/yfeng445/ds-flashdmla-moe/actions/runs/31711852093>。

失败发生在 CUDA 容器中执行以下命令时：

```bash
python3 -m pip install --break-system-packages --upgrade pip
```

Ubuntu/Debian 自带的 `pip 24.0` 没有可供卸载的 `RECORD`，因此 upgrade 失败。这个结果
**不能说明 CUDA 源码编译失败**；编译步骤根本尚未执行。

建议首先把 `.github/workflows/cuda-build.yml` 改为在虚拟环境中安装依赖，例如创建
`.venv` 后将其 `bin` 加入 `PATH`，不要修改发行版管理的系统 pip。修复后重新触发 build，
再根据真正的 nvcc/compiler 输出处理源码问题。

### 5.2 GPU workflows 尚未运行

- `.github/workflows/cuda-tests.yml` 依赖标签为
  `[self-hosted, linux, x64, cuda]` 的 runner；
- `.github/workflows/nccl-expert-parallel.yml` 依赖额外带 `multi-gpu` 标签的双 GPU runner；
- 两者目前都是手动触发，因此没有 GPU 数值、梯度、NCCL 或 benchmark 结果。

### 5.3 算法与性能缺口

- Attention CUDA 仍是 correctness-first FP32 路径，不是 FA2/FA3 级实现；
- grouped router 使用 one-thread-per-token 的串行候选扫描；
- MoE kernel 尚无 async copy、TMA、WGMMA 或 profiler-driven tuning；
- MLA 没有原生 CUDA prefill/decode kernel；
- chunk pipeline 只证明了软件异步协议，尚无 profiler 证据证明物理 overlap；
- one-sided symmetric memory 只有分析模型，没有 NVSHMEM/PGAS backend。

## 6. 下一步执行顺序

1. 修复 `cuda-build.yml` 的 Python/pip 环境，重新运行 CUDA wheel build。
2. 处理实际 CUDA 编译错误，确认所有 8 个注册算子存在 CUDA dispatch kernel。
3. 在单 GPU runner 上运行完整 CUDA 测试与 4 组 smoke benchmark。
4. 在双 GPU runner 上验证 NCCL FP32、FP16 WMMA 与 chunked pipeline 的 forward/backward。
5. 保存硬件、CUDA、PyTorch、shape、dtype、误差和原始 latency 样本。
6. 用 Nsight Systems/Compute 检查 kernel bottleneck 与通信计算 overlap。
7. 实现 MLA prefill/decode CUDA，再考虑 FA2/FA3、TMA/WGMMA 和 one-sided EP。

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

## 8. 工作区注意事项

创建本文件时，工作区中已有一个未跟踪文件：

```text
AI INFRA.ipynb
```

它没有包含在提交 `e3f5932` 中，归属和用途尚未确认。后续操作不要擅自删除、修改或
提交该文件；在下一次 commit 前应由仓库所有者明确是否纳入项目。
