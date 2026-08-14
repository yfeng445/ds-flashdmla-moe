# AI Infra 高压模拟面试：MLA、CUDA 与 MoE

这份模拟题用于把知识讲义转换成可经受追问的口头能力。默认岗位是 **AI Infra / CUDA
Kernel / 推理系统工程师**，基于当前仓库事实；不修改根目录原始 `AI INFRA.ipynb`。

## 使用方法

1. 每题先遮住“参考回答”和“追问”，限时作答并录音；
2. 项目题首答控制在 60—90 秒，原理题 30—60 秒；
3. 每次回答都按“动作 → 系统能力 → 价值 → 证据 → 边界”收口；
4. 只有能脱稿推导、定位源码并复现实验的工作才使用“我实现了”；
5. 第一轮低于 70 分时，不继续背新题，先修正被追问击穿的概念。

评分共 100 分：准确性 30、推导深度 20、实现细节 20、验证证据 20、事实边界 10。

## 开始前的 5 个待确认项

当前仓库材料足以生成技术初稿，但下面信息决定最终口径，模拟时先按保守版本处理：

1. **目标 JD【待补】**：更偏 CUDA kernel、推理 Serving、训练系统还是通用 AI Infra；
2. **个人贡献【待补】**：哪些源码、测试、环境和文档是本人能够独立复现与维护的；
3. **协作方式【待补】**：哪些部分由课程、开源参考、队友或 AI 辅助完成；
4. **可展示材料【待补】**：面试时能否打开源码、benchmark 报告或现场运行测试；
5. **目标面试形式【待补】**：项目深挖、八股、手写 CUDA/C++，还是系统设计占比更高。

在这些信息确认前，不使用“主导”“负责人”“生产级”“多卡优化专家”等强定位。

## 第一轮：项目开场与所有权

### Q1：请用一分钟介绍这个项目，重点说你解决了什么问题

暂停，在 90 秒内回答。

<details>
<summary>参考回答</summary>

这是一个 correctness-first 的 DeepSeek MLA + MoE 算子项目。项目先用 PyTorch reference
固定 attention、MLA、routing 和 expert parallel 的数值/梯度语义，再逐步接入 PyTorch
custom op 与 CUDA。当前最完整的链路是 MLA：我能解释并复现 naive 到 absorbed 的等价
推导、compressed static/paged cache，以及直接读取 latent cache 的 FP16/BF16/FP32 CUDA core。
现在 paged 路径可以按 slot mapping 写入物理页，再按 block table 直接读取，不先 materialize 连续
K/V。这个 core
融合了 causal mask、online softmax 和 latent value accumulation，不展开完整 per-head K/V，
也不写出完整 score matrix。staged pipeline 已将 projection、RoPE、attention 与 `W_O` 分别
接入 native CUDA，并在单张 RTX 5090、CUDA 12.8 环境完成 forward、reference-recompute
backward、非默认 stream 和全仓测试。MLA 当前边界是同 dtype 低精度 storage、FP32 accumulation、
无显式 mask；多卡 NCCL 与
NVSHMEM 也没有本地性能验证。普通 Attention 另有 FP16/BF16/FP32 native forward/backward，
不要把两条 dtype contract 混在一起。

如果这些代码并非全部由本人独立完成，应把“我”替换为“项目”，并明确自己的具体贡献。

</details>

**面试官连续追问**

1. 为什么你称它 correctness-first？列出三个不是普通 `allclose` 的验证点。
2. 你个人改动最深的一层是什么？现场指出一个你解决过的真实 bug。
3. 如果删掉 reference，只保留 CUDA 测试，你会失去什么保障？
4. 这个项目与官方 FlashMLA 的关系是什么？

**合格要点**：能说出 fake/opcheck 或 compile、autograd/梯度、stream、determinism、tail/empty
等边界中的至少三类；不把教学 kernel 说成生产级实现。

### Q2：你凭什么说自己适合 Kernel Engineer，而不只是会调 PyTorch API？

<details>
<summary>参考回答</summary>

我的证据不是调用一个现成 attention API，而是能把同一算子跨三层讲清：数学层能用维度推导
absorbed MLA；框架层能说明 schema、CUDA dispatch、FakeTensor 和 autograd registration；
CUDA 层能解释一个 block 对应 `(B,H,S_q)` 中一行、shared memory 里各缓冲区、在线 softmax
状态更新、绝对位置 causal mask、当前 stream 和 device guard。与此同时，我也会主动指出它
现在 paged MLA 每 key 仍是 correctness-first 标量 FP32 accumulation，且尚无 serving page
allocator；普通 Attention 同样尚无
二维 tiling/Tensor Core 调度。我的定位是已有完整 correctness kernel 闭环、正向生产级优化
能力推进，而不是已经具备成熟 FA3/FA4 优化经验。

</details>

**红旗回答**：“我用了 CUDA，所以我是 CUDA 工程师”“测试都过了，所以是高性能 kernel”。

## 第二轮：MLA 代数与 CUDA 实现

### Q3：不用背定义，从维度推导 naive MLA 为什么可以吸收 K/V up-projection

<details>
<summary>参考回答</summary>

对一个 head，`q_nope:[d_nope]`、latent `c_t:[r]`、`W_K:[d_nope,r]`。
naive content score 是 `q_nope · (W_K c_t)`，结合律得到 `(q_nope W_K) · c_t`，所以可以
先把 query 投到 latent 维再与缓存 latent 点积。对 value，`W_V:[d_v,r]`，有
`sum_t p_t(W_V c_t)=W_V(sum_t p_t c_t)`，所以先对 latent 做概率加权再上投影。位置相关的
RoPE key 不能照搬这个吸收，因为它随 position 变化，因此实现单独缓存 `pe` 并单独加入
position score。前提是这些投影对 token 共享且 softmax 所用 score 保持代数等价。

</details>

**追问**：

- 如果 `r_kv > d_nope+d_v`，MLA 仍一定省 cache 吗？
- 为什么 cache 中还需要 `absolute positions`？
- `W_O` 能否继续吸收？训练和推理的权衡是什么？
- 浮点数里结合律并不严格成立，如何设计容差？

### Q4：逐步讲你的 CUDA MLA kernel；每个 block、thread 和 shared memory 分别做什么

<details>
<summary>参考回答</summary>

grid 是 `B*H*S_q` 个 block，一个 block 负责一个 query row，当前固定 128 threads。shared
memory 放 `q_latent[r]`、未归一化 latent numerator `[r]`、block reduction `[128]` 和两个
softmax statistics。线程先并行计算 `q_nope @ key_up`；随后逐 key 扫描 compressed `kv/pe`，
各线程计算部分 content/position dot，再做 block reduction。线程 0 更新 running max 和
denominator，并广播 previous/current rescale；全 block 用它更新 latent numerator。扫完后各
线程负责若干 value 维，除以 denominator 后乘 `value_up` 写 head output。host wrapper 验证
shape/dtype/device/shared-memory/grid 边界，使用 `CUDAGuard` 和 PyTorch current stream，再做
launch check。

</details>

**追问**：

1. 为什么每个 key 后必须同步？少一个 `__syncthreads()` 会发生什么？
2. 全 mask row 为什么不会 NaN？
3. 非 contiguous query 为什么仍可工作？代价是什么？
4. shared memory 公式是什么？`r_kv` 很大时会怎样？
5. 为什么 decode 的 block 数可能太少？如何重新划分 work？

### Q5：online softmax 的旧 numerator 为什么要 rescale？请推公式

<details>
<summary>参考回答</summary>

当前状态以旧最大值 `m` 为指数参考，旧 numerator 是
`o=sum_j exp(s_j-m)v_j`。新 score 或 tile 使最大值变成 `m'` 后，同一批旧项在新尺度下应为
`sum_j exp(s_j-m')v_j = exp(m-m')o`。新项按 `exp(s_new-m')` 累加；denominator 同理更新。
如果忘记 rescale，来自不同最大值尺度的量被直接相加，概率与输出都会错误。全 mask 时保持
`m=-inf,l=0`，最终显式输出零而不是执行 `0/0`。

</details>

### Q6：为什么 CUDA backward 不是原生 kernel？这算支持训练吗？

<details>
<summary>参考回答</summary>

当前 MLA CUDA 只实现 fused forward core。custom op 的 autograd 在 backward 中重算可微的
absorbed PyTorch reference，再用 `torch.autograd.grad` 求请求的输入梯度。因此可以验证和使用
一阶训练梯度语义，但不应称为 fused native backward，性能与显存特性也不同于手写反向。
这个选择让 forward 优化先建立在可靠梯度 oracle 上；下一步若写 native backward，要继续
检查 causal/full-mask、确定性、高阶梯度策略和非默认 stream。

</details>

## 第三轮：性能证据与诊断

### Q7：paged BF16 case 的 native/baseline 比值是 0.157；你能把它说成 6.36× 吗？

<details>
<summary>参考回答</summary>

不能脱离限定直接说 6.36×。这是单张 RTX 5090、PyTorch 2.10.0+cu128、CUDA 12.8 上的
clean-source paired snapshot。shape 是 BF16 `B2/S257/H4/r_kv32`、page size 16，5 次 warmup、
20 次测量；native median `0.336480 ms`，项目内同 dtype PyTorch paged baseline
`2.139152 ms`，所以这个固定 harness/case 的 ratio 是 `0.157`。两边都通过 alternate
naive/absorbed verification。它只能说明当前 native storage/compute path 在这个 case 的 median
较低；baseline 不是 vLLM/FlashMLA，测试不含 page allocator、continuous batching 或并发请求，
也没有 Nsight counter，因此不能外推为生产吞吐或通用 6.36×。

</details>

**追问**：为什么只报 median 不够？为什么不能和另一次进程里的数字随意相除？如何减少
GPU boost、后台负载、Python cache mutation 的影响？如果 baseline materialize 连续 K/V，
而 native 直接读页表，这个算子边界是否仍然公平？

### Q8：没有 profiler 时，你认为当前 kernel 的三个潜在瓶颈是什么？怎么证伪？

<details>
<summary>参考回答</summary>

假设一是每 key 的 block reduction 与多次同步成本高；用 Nsight Compute 看 barrier stall，
并做 warp-reduction 版本的消融。假设二是 decode 时 `B*H*S_q` block 数太少，GPU occupancy
不足；看 achieved occupancy/SM active，并增加按 key tile 或 value tile 的并行版本对比。
假设三是标量 FP32 FMA 和非连续 stride load 没有充分利用内存/Tensor Core；看 global load
efficiency、memory throughput、instruction mix，再尝试 contiguous packing、vectorized load 或
BF16/FP16 tile。未测之前都只能叫假设。

</details>

### Q9：为什么 prefill 的绝对误差比小尺寸单测大？`5e-4` 是不是为了让测试通过？

<details>
<summary>参考回答</summary>

prefill 经过多级 projection、dot/reduction、online softmax、latent accumulation 和 `W_O`；
native serial FMA、CUDA block reduction 与 cuBLAS/einsum 的归约顺序不同，FP32 不满足严格
结合律。容差不能只看某次失败后随意增大，因此对同一配置还要与 float64 oracle 比较误差
量级，并记录最大绝对误差与 tolerance ratio。当前 smoke 使用 `rtol=atol=5e-4`，报告中的
ratio 小于 1。更成熟做法是覆盖多 seed/shape/scale、统计 ULP/相对误差，并对异常大值保留
失败。不能用宽容差掩盖 mask、索引或 softmax 状态 bug。

</details>

## 第四轮：MoE 与分布式边界

### Q10：为什么 dispatch 前不乘 routing weight？route identity 至少要保存什么？

<details>
<summary>参考回答</summary>

expert 是非线性函数，一般 `f(w*x) != w*f(x)`，所以 activation 应原样送入 expert，routing
weight 在 expert output 返回源 token 后再用于 combine。每条 route 至少要能恢复 source
rank、token index、top-k slot、expert id/owner 和 route weight；项目将 route 按目标
rank/expert 打包并保留恢复顺序所需索引。空 expert、空源 rank 和不均匀 splits 也必须参与
collective，否则会死锁或丢梯度。

</details>

### Q11：group-limited Top-K 的 correction bias 为什么只影响选择？

<details>
<summary>参考回答</summary>

correction/load bias 用来改变哪些 expert 更可能入选，从而调节负载；如果最终 combine 权重也
直接使用 biased score，就把系统负载控制混进了模型概率语义。项目按 biased score 选择，
再从 unbiased sigmoid score gather 并归一化 route weight。面试时应区分 selection policy 和
mixture coefficient。

</details>

### Q12：你实现 NCCL overlap 了吗？如何证明？

<details>
<summary>参考回答</summary>

准确说法是：代码实现了 NCCL process group 下的 variable All-to-All、`async_op=True` work
handle 和 chunk pipeline 软件调度；两 rank Gloo reference 验证了 forward/backward 与 route
identity。但本地只有单卡，因此没有完成真实多 GPU NCCL 性能回归，也没有 Nsight timeline
证明物理通信计算重叠。要证明 overlap，需要同一硬件/shape 下与 serialized baseline 对照，
在 Systems timeline 看到通信和 expert compute 时间交叠，并确认依赖、结果和梯度不变。

</details>

**一票否决说法**：“用了 async_op 就已经 overlap”“Gloo 两进程通过等于 NCCL 多卡性能通过”。

### Q13：NCCL 与 NVSHMEM 的编程模型差异是什么？仓库做到哪一步？

<details>
<summary>参考回答</summary>

NCCL 主要提供 collective/点对点通信语义，参与 rank 按协议共同进入操作；NVSHMEM 是 PGAS
风格 one-sided put/get/atomic/signal，发起端可直接访问对端 symmetric allocation，但数据可见
性和 signal 协议要自己设计。仓库当前有 NCCL 软件路径和 symmetric-buffer 容量/偏移分析
模型，没有 NVSHMEM runtime backend，所以只能讨论 actor/persistent-kernel 设计目标。

</details>

## 第五轮：框架与工程细节

### Q14：一个 PyTorch CUDA custom op 仅“能 import”为什么远远不够？

<details>
<summary>参考回答</summary>

还需要 schema 与 dispatcher 注册、CUDA/Composite dispatch 边界、FakeTensor/meta 行为以支持
compile/export、autograd registration、device/dtype/shape validation、当前 device/stream、
launch error check、fallback 或显式拒绝策略，以及 wheel 中源码/二进制构建。测试至少覆盖
opcheck、`torch.compile`、forward/backward reference、非默认 stream 和不支持输入的 loud
failure。

</details>

### Q15：为什么 atomic backward 与 deterministic algorithms 冲突？项目怎么处理？

<details>
<summary>参考回答</summary>

多个 block/线程对相同 `dK/dV` 或 combine destination 做 atomic add 时，加法到达顺序不固定；
浮点加法非结合，因此 bitwise 结果可能变化。项目在原生 attention/route 路径检测 deterministic
mode，并选择 reference/analytic backward 或拒绝 nondeterministic CUDA 路径。确定性不等于
数值更准确，它是执行顺序和可复现性的契约。

</details>

### Q16：为什么必须取 PyTorch current stream，而不是默认 stream？

<details>
<summary>参考回答</summary>

上层可能在非默认 stream 中排队写输入并立即调用 custom op。若 kernel 偷跑默认 stream 且无
事件依赖，会读到未完成数据或让调用方过早使用输出。host wrapper 在正确 device guard 下取
current stream 并在该 stream launch；测试在自建 stream 中先修改输入、调用 op、记录输出并
同步，以验证 stream 语义。

</details>

## 第六轮：系统设计加试

### Q17：普通 Attention 已扩展到 BF16，你如何证明不是只改了指针类型？下一步怎么优化？

合格回答应覆盖：API/dispatch 接受相同 dtype 的 Q/K/V；load/store 使用低精度但 dot、
exp/softmax、numerator 与 backward gradient workspace 保持 FP32；原子累加后再 cast；测试覆盖
forward、raw backward、autograd、causal/tail/stride、SM 架构编译，并与同 dtype FA4 成对验证。
下一步需用 profiler 解释 row-wise block reduction、barrier、occupancy 与布局 adapter 成本，再
决定二维 score tile、warp partition、vectorized load 或 Tensor Core；不能只说换成 BF16 指针。

### Q18：你已经实现了 paged latent cache；请讲清数据契约、错误语义与仍缺的 serving 层

合格回答应覆盖：physical storage 是 `[pages,page_size,latent_or_rope_dim]`；global slot 决定写入
位置，block table 与 per-row length 决定逻辑读取；同一次写的重复/越界 slot 被拒绝，跨调用允许
完整覆盖；有效页不能在同一行重复，未用表项为 `-1`，有效 slot 必须已写入且 absolute position
严格递增。kernel 在 key loop 直接映射物理页，causal 仍比较 absolute positions，PagedAttention
不替代 online softmax。当前缺口是 allocator、request/page 生命周期、eviction、prefix sharing、
continuous batching 和 Nsight 驱动调优。

**连续追问**：为什么跨 batch 行允许共享页？写时如何避免共享前缀被误覆盖？如果 Python
validation 每 token 都 D2H，会发生什么？你如何用 tensor version 缓存校验，又如何证明原地
修改不会误用旧结果？

### Q19：如果现在给你 8×H100，你的验证顺序是什么？

参考顺序：先单卡重建相同 wheel/数值测试；再两卡 NCCL 小 shape 验证 variable splits、空 rank、
forward/backward；扩到 8 卡检查 collective 一致性与错误处理；建立 serialized baseline；最后才
开 chunk/async overlap，用 Nsight Systems 验证 timeline，用 Compute 检查 expert kernel，记录
拓扑、NVLink/NVSwitch、dtype、shape、warmup、raw samples 和误差。不要一上来报吞吐。

## 失败诊断清单

模拟后若出现以下现象，按对应动作补强：

| 现象 | 暴露的问题 | 补强动作 |
| --- | --- | --- |
| 只会说“减少显存访问” | 缺少量化和数据布局 | 写出每 token cache 元素数及 kernel I/O |
| absorbed 推导不写维度 | 可能只是背结论 | 手推 Q3，并随机换维度检查矩阵方向 |
| 说不出 shared memory 内容 | 没真正掌握 kernel | 对照源码画 block 数据流，计算字节数 |
| 一问性能就报 1.21x | 缺少实验边界 | 强制先说 GPU/shape/dtype/baseline/samples |
| 把 async 当 overlap | 分布式证据越界 | 复述 Q12，并画依赖/timeline |
| 把 reference backward 说成 CUDA backward | 实现边界混淆 | 跟踪 dispatcher/autograd 调用链 |
| 每题都说“我们” | 个人贡献不清 | 为每项成果标注本人动作与可展示证据 |
| 每题都说“我主导” | 强主张缺证据 | 降级动词，补决策、交付和复现材料 |

## 面试后复盘模板

```text
题目：
我的首答（不润色）：
面试官追问：
卡住的位置：公式 / 维度 / CUDA / PyTorch / 分布式 / 实验 / 个人边界
正确答案与源码证据：
下一次 60 秒回答：
需要补的实验：
```

建议先完成 Q1、Q3、Q4、Q7、Q12、Q14 六题。这六题能同时检验项目叙事、数学、kernel、
性能、分布式边界和框架工程；全部能经受二问后，再扩展到系统设计题。
