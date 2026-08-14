# Project Handoff

更新时间：2026-08-14

当前基线：`main`，已包含 FP16/BF16/FP32 staged end-to-end MLA CUDA、小维度
absorbed-attention warp-partition 调度、paged/per-slot latent cache、直接页表 attention、稳定输出
布局契约、单 GPU 成对基线快照、可复现的单 case Kineto/NVTX profiling 入口，以及 FP16/BF16
native Attention 与可选的 FlashAttention-4 同 dtype 成对矩阵。

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
CPU/reference 路线持续通过 Python 3.10/3.12 CI，CUDA wheel 能编译并注册 16 个 native
算子；RTX 5090 本地环境也已跑通完整 CUDA 测试、固定 shape smoke benchmark、20-case
代表性 shape matrix、四组低精度 native/FA4 paired matrix、四组 staged MLA 低精度
native/PyTorch paired matrix、两组 paged MLA decode paired matrix、MLA PyTorch/Kineto profiler
triage 和首轮 CUDA kernel 专项优化。
尚未完成持续 self-hosted GPU CI、双 GPU NCCL、原生 Nsight 取证、CUTLASS 对照和 1K/4K 以上
生产长上下文矩阵，因此仍不具备生产性能结论。

粗略进度约为 **94%**。这里的百分比衡量的是学习/研究仓库的完成度，不代表生产可用性。

| 方向 | 状态 | 说明 |
| --- | --- | --- |
| 课程讲义 | 基本完成 | `docs/chapters/00`–`08`，覆盖 tiling、在线 Softmax、FlashAttention、MLA、MoE、EP、自定义算子、benchmark/roofline、对称内存 |
| PyTorch reference | 基本完成 | GEMM、Attention、MLA、grouped Top-K、SwiGLU MoE、路由、Expert Parallel |
| CPU/Gloo 验证 | 已完成 | 单元测试、梯度检查、非规则 shape、空 expert、空 rank、两 rank Gloo |
| CUDA 算子源码 | `main` 有 16 个算子 | Attention forward/backward、tiled GEMM、router、route pack/combine、expert-major pack、SwiGLU experts，以及 8 个 staged/paged MLA 算子 |
| Attention CUDA | FP16/BF16/FP32 correctness 已完成本地单卡验证 | forward 使用 FP32 dot/online-softmax/numerator；backward 使用 FP32 gradient workspace 后 cast，仍是 row-wise scalar kernel |
| FP16 Tensor Core | 已实现并完成本地单卡 smoke | expert-major SwiGLU 使用 WMMA、FP32 accumulation 和显式 FP16 hidden boundary；仍缺持续 GPU CI |
| NCCL Expert Parallel | 代码已实现，待双 GPU 验证 | 包括可微 All-to-All 和异步 chunk pipeline |
| MLA CUDA | FP16/BF16/FP32 end-to-end + paged correctness 已完成本地单卡验证 | direct/LoRA query、KV projection、static/per-slot write、连续与 paged absorbed attention、output projection 使用统一 storage dtype，内部使用 FP32；连续小维度 attention 使用四 warp key partition，paged path 仍是 one-CTA correctness kernel，生产级调优与 serving 生命周期尚未实现 |
| One-sided/NVSHMEM | 未实现 | 当前只有 symmetric-buffer 成本与布局模型 |
| 性能结论 | 尚不可下结论 | 已有单卡固定 shape、20-case matrix、Kineto 聚合、四组同 dtype FA4、四组 staged MLA 低精度与两组 paged MLA 对照，但仍缺持续 runner、Nsight trace、CUTLASS 和生产长上下文；原始样本存在明显 WSL 波动 |

## 3. 代码地图

- `src/ds_flash_mla_moe/attention.py`：materialized 与 blockwise online attention reference。
- `src/ds_flash_mla_moe/gemm.py`：GEMM 数值契约与 tiled teaching reference。
- `src/ds_flash_mla_moe/mla.py`：MLA prefill、decode、compressed static/paged cache 与 absorbed 路径。
- `src/ds_flash_mla_moe/moe.py`：DeepSeek-style grouped routing 和 MoE reference。
- `src/ds_flash_mla_moe/router_ops.py`：group-limited Top-K API 与 backend selection。
- `src/ds_flash_mla_moe/route_ops.py`：route pack/combine 的 reference/native 接口。
- `src/ds_flash_mla_moe/expert_ops.py`：active-row expert-major SwiGLU 接口。
- `src/ds_flash_mla_moe/expert_parallel.py`：Gloo/NCCL Expert Parallel 协议与 autograd。
- `src/ds_flash_mla_moe/ops.py`：PyTorch dispatcher、FakeTensor、autograd 和 CUDA 注册。
- `src/ds_flash_mla_moe/*_benchmarking.py`：结构化 benchmark、成对 shape matrix 与报告模型。
- `src/ds_flash_mla_moe/profiling.py`：精确 matrix side 的 Kineto 聚合与 NVTX 包装。
- `csrc/`：supported CUDA/C++ extension 源码。
- `csrc/experimental/`：未验证的课程时期原型，不属于 supported API。
- `benchmarks/`：GEMM、Attention、MLA、router、experts、成对 matrix、operator profiler 和 Expert Parallel CLI。
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
351 passed, 169 skipped
```

另已完成：

- `ruff format --check`：通过；
- `ruff check`：通过；
- `git diff --check`：通过；
- GitHub Actions `Reference tests` 的 Python 3.10/3.12 matrix：paged MLA 批次通过：
  <https://github.com/yfeng445/ds-flashdmla-moe/actions/runs/31792335038>；
- GitHub Actions `CUDA build / wheel`：paged MLA 批次通过，并检查 16 个 CUDA dispatch kernel；wheel
  同时携带 forward-compatible PTX：
  <https://github.com/yfeng445/ds-flashdmla-moe/actions/runs/31792335105>。

### 4.1 已解决的 MLA output-layout contract

此前 `mla_absorbed_attention` 的 Python composite 继承了 `einsum` 的非连续 stride，而
FakeTensor 和原生 CUDA kernel 返回 contiguous `[B,S,H,Dv]`。当前 composite 已显式
`.contiguous()`，并新增 concrete-layout 回归断言；完整 `torch.library.opcheck` 以及
Python 3.10/3.12 CI 均已通过。

### 4.2 本地单 GPU 验证

WSL2 / RTX 5090 / PyTorch 2.10 + CUDA 12.8 环境最新运行：

```text
pytest -ra --strict-markers -W error::UserWarning
520 passed
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

默认 representative manifest 仍保持 20 组，避免可选 beta 依赖或新增低精度验证扩大普通
CI。另有 `--profile flash-attn-4` 的四组 Attention-only paired matrix，以及
`--profile mla-low-precision` 的四组 staged MLA paired matrix；二者都覆盖 BF16/FP16、
prefill/decode 与 regular/tail shape。`--profile mla-paged` 另有 BF16 `S=257` 与 FP16 `S=129`
两组 per-slot write + direct paged attention paired case。

原固定 shape 快照仍保留完整 MLA 的 1.155536/2.636864 ms prefill 和
1.028400/2.058480 ms decode native/baseline 数据。应用 Python-side position validation
优化并加入小维度 warp-partition kernel 后，最新 representative matrix 中对应 regular case 为
0.272960/1.835232 ms 和 0.258352/4.389200 ms。各轮都只是本机诊断样本；系统状态和测量轮次
不同，不应把差值全部归因于一次代码修改，也不能外推到其他 shape、dtype 或硬件。

### 4.3 MLA profiler-driven 同步与 kernel 优化

新增 `benchmarks/operator_profile.py`，可对 20-case matrix 中任意一个 `native`/`baseline` side 运行
PyTorch/Kineto 聚合或 NVTX 标记。runner 在 capture 外先完整预热一次，再捕获 fresh setup、
一次输出调用、配置的 warmup 和正式迭代；JSON 分开保存 custom operator、self-device 热点和
常见同步事件。`--mode nvtx` 已在本机 smoke 通过，但本机没有 `nsys`/`ncu`，因此它只是为
后续原生 profiler 准备稳定 range，不是 Nsight 结果。

对 `mla_prefill_regular`/`mla_decode_regular` 的 26-call capture，重复 position 校验优化使
`aten::_local_scalar_dense` 从 212/162 降到 28/29，`cudaStreamSynchronize` 从 220/170
降到 36/37。成功校验只在 Tensor identity/version 未变化时复用；latent cache、query
positions 和 static position storage 原地修改都会使缓存失效并重新校验。

该次 capture 将 absorbed-attention 定位为首要 device 热点后，`latent_dim`、`rope_dim` 和
`value_dim` 均不超过 32 的常见 DeepSeek-style shape 改用四 warp key partition：每个 warp
维护独立 online-softmax 状态，最后做一次稳定合并；任一维度超过 32 时仍走原 generic kernel。
specialized 边界、tail、causal/non-causal、非连续 stride 和三种 generic fallback 均已有 CUDA
回归测试。相同 26-call Kineto 口径下，absorbed kernel self-device 总时间从 prefill/decode 的
2.632/2.699 ms 降至 0.429/0.491 ms，当前占 custom-op self-device 时间的 41.8%/44.3%。这是
本机 profiler 观察，不是端到端或跨硬件加速结论；结构化报告保存在单 GPU validation 目录。

### 4.4 Native 低精度 Attention 与 FlashAttention-4 paired matrix

普通 native Attention forward/backward 现支持相同 dtype 的 FP16、BF16 与 FP32 contiguous
BHSD tensor。forward 的 dot、在线 Softmax 与 numerator 使用 FP32；backward 将三类梯度先
累积到 FP32 workspace，再 cast 回输入 dtype。低精度 forward、raw backward、autograd、
causal/non-causal、regular/tail/decode、非连续 fallback 和 mixed-dtype rejection 均有 CUDA
回归；合并当前 paged MLA 批次后，全仓 520 tests 通过。

`benchmarks/matrix.py --profile flash-attn-4` 使用 `flash-attn-4==4.0.0b22` 运行四组完全
同 dtype/config 的 paired case。四组 native 与 FA4 side 都通过 FP32 materialized reference
校验；正式 20-sample 快照的 native/FA4 median 比值为 7.996、2.881、17.604 与 0.744，前三组
FA4 更低，tail decode 一组 native 更低。原始样本存在明显桌面 WSL 波动，因此这里保留逐样本
JSON 作为后续 profiler/tuning 基线，不把单次排序外推为跨机器或生产 workload 结论。当前
row-wise scalar kernel 仍缺二维 tiling/Tensor Core；FA4 继续保持可选依赖，避免 resolver 替换
项目 PyTorch/CUDA 栈。报告位于
`validation/single-gpu/2026-08-14-rtx5090-cu128/operator-matrix-fa4.json`。

### 4.5 Staged MLA 低精度 storage 与 FP32 accumulation

六个 staged MLA native op 现统一支持 FP16、BF16 与 FP32。所有浮点输入、权重、cache 与输出
必须位于同一设备并使用同一 storage dtype；direct/LoRA query projection、RMSNorm、RoPE、
cache projection/write、absorbed score/online-softmax/value projection 和 output projection 的
内部算术使用 FP32，公开 stage 边界再写回 storage dtype。Python reference、FakeTensor、CUDA
eligibility 与 native dispatch 共享这一契约，mixed-dtype 输入会在进入 kernel 前被拒绝。

CUDA 回归覆盖 direct/LoRA、prefill/decode、causal/tail、specialized/generic dispatch、static
cache write、非连续 fallback、逐 stage 严格对照和 end-to-end composed tolerance；合并 paged
回归后全仓为 520 tests。`benchmarks/matrix.py --profile mla-low-precision` 另提供四组完全
同 dtype/config 的 native/PyTorch absorbed paired case。本机四组均通过 staged reference 数值
校验；正式 20-sample 快照的 native/baseline median 比值为 0.336、0.184、0.152 与 0.265，
本轮四组 native median 均更低。它们使用小 shape 与未缩放随机权重，只用于正确性与后续
profiler 定位，不构成通用性能结论。报告位于
`validation/single-gpu/2026-08-14-rtx5090-cu128/operator-matrix-mla-low-precision.json`。

### 4.6 Paged/per-slot latent cache 与直接 CUDA attention

新增 `MLAPagedCache`，物理 payload 为 `[num_pages,page_size,latent_or_rope_dim]`，absolute
positions 使用同页布局。`write_mla_paged_cache` 通过 `[B,S]` global slot mapping 写入；同一次
调用拒绝重复或越界 slot，后续调用允许完整覆盖旧 slot。读取侧用
`block_table[B,max_logical_pages] + sequence_lengths[B]` 表示 ragged logical sequence；每行有效页
必须在范围内且不重复，未使用表项为 `-1`，引用 slot 必须已写入且 position 严格递增。不同 batch
行可以共享物理页。API inference-only，causal 判断使用 absolute position 而不是物理地址。

两个新增 CUDA op 分别完成 projection/RMSNorm/RoPE 后的 per-slot scatter，以及不 materialize
连续 K/V 的 direct paged absorbed attention。CUDA/reference 回归覆盖 FP16/BF16/FP32、ragged
batch、非连续页、重复/越界 slot、覆盖写、非法页表、未写入/非单调位置与
`S=257,page_size=16` 尾页。公开 API 对成功校验的 metadata 按 Tensor identity/version 复用；raw
operator 仍保留防御检查，原地修改会使缓存失效。正式 BF16/FP16 paired matrix 的
native/baseline median 比值为 `0.157/0.284`，两组均通过 alternate naive/absorbed reference；这
只是两个小 shape 的本地 clean-source 快照，不是 serving 性能结论。

BF16 paged native 的 26-call Kineto capture 记录 5 次 `_local_scalar_dense` 与 17 次 stream
synchronize；paged attention kernel 的 aggregate self-device 时间为 `6.501 ms`，是后续单卡优化
重点。报告位于同一 validation 目录的 `operator-matrix-mla-paged.json` 与
`torch-profiler-mla-paged.json`。

## 5. 当前阻塞与已知缺口

### 5.1 持续 GPU workflows 尚未运行

- `.github/workflows/cuda-tests.yml` 依赖标签为
  `[self-hosted, linux, x64, cuda]` 的 runner；
- `.github/workflows/nccl-expert-parallel.yml` 依赖额外带 `multi-gpu` 标签的双 GPU runner；
- 两者目前都是手动触发。2026-08-14 查询远端 runner 数量仍为 0；本地单 GPU 结果已经
  固化为验证快照，但仓库仍没有注册并持续运行的 self-hosted 单 GPU/双 GPU runner，也没有
  NCCL benchmark artifact。

### 5.2 双 GPU 通信验证缺失

当前 WSL 环境只有一张可见 RTX 5090，无法证明 NCCL FP32、FP16 WMMA 和 chunked pipeline 的
forward/backward 正确性，也无法用 profiler 证明通信计算发生了物理 overlap。

### 5.3 算法与性能缺口

- Attention CUDA 已覆盖 FP16/BF16/FP32 correctness，但仍是每 query row 一个 CTA、每 key
  block reduction 的标量路径，没有二维 score tiling、Tensor Core 或 FA2/FA4 级调度；
- grouped router 使用 one-thread-per-token 的串行候选扫描；
- MoE kernel 尚无 async copy、TMA、WGMMA 或 profiler-driven tuning；
- staged MLA 已覆盖 FP16/BF16/FP32 correctness pipeline 与 paged/per-slot cache；Python 重复
  metadata 同步和连续小维度 absorbed-attention 已按 Kineto 结果优化；但 paged attention 仍是
  one-CTA correctness kernel，尚无 page allocator/request 生命周期、prefix-sharing 写保护、
  长上下文专用分块、异步拷贝或 profiler-driven fusion；
- 代表性单 GPU shape matrix 已覆盖 regular/tail/decode/skew，FA4 和 staged MLA 各有四组、
  paged MLA 有两组同 dtype FP16/BF16 paired case；尚缺 CUTLASS 与 1K/4K 以上生产长上下文矩阵；
- chunk pipeline 只证明了软件异步协议，尚无 profiler 证据证明物理 overlap；
- one-sided symmetric memory 只有分析模型，没有 NVSHMEM/PGAS backend。

## 6. 下一步执行顺序

1. 注册并运行单 GPU self-hosted runner，让现有 520-test、9 组固定快照、20-case matrix、
   可选四组 FA4、四组 staged MLA 低精度和两组 paged MLA matrix 变成可持续 workflow artifact。
2. 在双 GPU runner 上验证 NCCL FP32、FP16 WMMA 与 chunked pipeline 的 forward/backward，
   保存每 rank 原始 latency 和通信量。
3. 在已有 Kineto/NVTX 入口上用 Nsight Systems/Compute 检查 Attention、MLA、router、experts
   的 kernel bottleneck；Attention 先解释当前 native/FA4 差距，MLA 验证 warp-partition
   occupancy/访存并检查 paged key-loop，再在双卡上验证 NCCL chunk pipeline 是否真正 overlap。
4. 当前单卡优先把 paged attention 从 one-CTA 标量路径扩展为 warp-partition/tiled 版本，并补
   `S=1K/4K`、page size `8/16/32`、ragged batch 与共享页只读场景；用现有 Kineto case 做消融，
   有 `ncu` 时再以 occupancy、barrier stall 和 memory traffic 决定优化。
5. 再实现最小 page allocator/request 生命周期与 prefix-sharing 写保护；随后根据 profiler 考虑
   专用 prefill/decode、TMA/WGMMA、router 优化和 one-sided EP；
   有工具链时补 CUTLASS 对照。

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

可选同 dtype FlashAttention-4 矩阵：

```bash
python benchmarks/matrix.py --device cuda --profile flash-attn-4 \
  --warmup 5 --iterations 20 --seed 20260814 \
  --output benchmark-results/operator-matrix-fa4.json
```

Staged MLA 低精度矩阵：

```bash
python benchmarks/matrix.py --device cuda --profile mla-low-precision \
  --warmup 5 --iterations 20 --seed 20260814 \
  --output benchmark-results/operator-matrix-mla-low-precision.json
```

Paged MLA decode 矩阵：

```bash
python benchmarks/matrix.py --device cuda --profile mla-paged \
  --warmup 5 --iterations 20 --seed 20260814 \
  --output benchmark-results/operator-matrix-mla-paged.json
```

单 case Kineto 聚合：

```bash
python benchmarks/operator_profile.py --case mla_prefill_regular \
  --side native --mode torch --warmup 5 --iterations 20 --seed 20260814 \
  --output benchmark-results/torch-profiler-mla-prefill.json
```

为 Nsight 标记同一 workload：

```bash
python benchmarks/operator_profile.py --case mla_prefill_regular \
  --side native --mode nvtx --warmup 5 --iterations 20 --seed 20260814
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
- `main` 当前已经包含 FP16/BF16/FP32 staged end-to-end MLA CUDA、paged/per-slot latent cache、
  FP16/BF16/FP32 native Attention、20-case operator matrix、可选四组 FA4、四组 staged MLA
  低精度与两组 paged MLA matrix、单 case profiler、两篇衍生面试文档和 16-operator native
  extension。
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
