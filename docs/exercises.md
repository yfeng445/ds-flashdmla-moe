# 练习

练习按“可手算 → 可测试 → 可 profile”递进。完成优化题前，先为它增加失败测试。

## 第零组：并行执行与 GEMM

1. 用 Amdahl 定律计算 `p=0.95` 时 16、64、无限多个执行单元的理论加速比，并解释实际
   pthreads 程序还会增加哪些成本。
2. 为 row-major 矩阵写出二维 CUDA thread 到 `(row,col)` 的映射；交换 x/y 映射后分析
   一个 warp 的地址跨度。
3. 对 `M=37,N=29,K=23,T_M=16,T_N=8,T_K=7`，计算三个 tile counts、最后一个 tile
   的尺寸和单 stage shared-memory 容量。
4. 推导朴素 one-output-thread 与 one-output-tile 两种输入流量模型，并说明它们为何都不
   等于 Nsight Compute 的实测 DRAM bytes。
5. 在 `tiled_gemm_reference` 中删掉 tail 截断，使用最小反例定位越界发生在哪一维。
6. 画出双缓冲 producer/consumer 时间线；分别标出“tile 可读”和“buffer 可覆盖”的同步点。
7. 比较 FP16 accumulation 与 FP32 accumulation 的长 dot product 误差；不能只报告均值。

## 第一组：在线 Softmax

1. 手算序列 `[1000, 999, -1000]` 的稳定 Softmax，说明直接指数为何溢出。
2. 将序列分为 `[1000]` 与 `[999, -1000]` 两块，用 `(m,l)` 递推合并。
3. 修改 `blockwise_attention`，额外返回每个 tile 后的 `(m,l)`，验证不同
   `block_size` 最终状态相同。
4. 构造全遮挡行，解释每个 `torch.where` 分支为何必要。

## 第二组：Attention kernel

1. 画出正确性 kernel 中 shared memory 的 `query/numerator/reduction/statistics`
   布局，给出 `D=64,D_v=64` 时占用的字节数。
2. 修改 CUDA 测试矩阵，覆盖 `S_q != S_k` 的单 token decode，并解释右下对齐坐标。
3. 将逐 key 遍历改为 `B_c` 大小的 K/V tile，证明在线更新前后的结果等价。
4. 用 Nsight Compute 比较逐行、二维 tiled 和 Tensor Core 实现的 HBM bytes。
5. 检查 `D=24`、`S=37` 等 tail shape，不允许静默返回或越界。
6. 推导 backward 中 `D_i = dO_i · O_i` 的来源，并让解析公式同时通过
   `gradcheck` 与 `gradgradcheck`。
7. 令 Q/K/V 引用同一张量，解释三个角色的偏导何时相加，避免重复累计 alias 梯度。
8. 解释为何 query-row 并行可以无原子地写 dQ，却需要原子累加 dK/dV；设计一个
   deterministic 替代工作划分。

## 第三组：MLA

1. 根据第三章张量维度写 naive PyTorch MLA prefill。
2. 比较展开 K/V cache 与 latent cache 的元素数量，给出一组 DeepSeek 风格参数的比值。
3. 实现逐 token decode，证明它与一次性 causal prefill 的最后一个 token 输出一致。
4. 用矩阵结合律推导 NoPE key weight absorption；标出 RoPE 部分为何不能被同样吸收。
5. 推导 expanded/latent cache payload 压缩比，并分别计入 int64 position metadata。
6. 对 `prefill_attention`、`prefill_with_cache`、`decode_attention`、
   `decode_with_append` 与 `decode_with_static_write` 运行同一 shape，解释为何五个 latency
   不能混为一个指标。
7. 计算函数式 append 在长度 `S` 时的最低拷贝字节；验证预分配 cache 的 storage 指针不变，
   并解释为何这只把 cache 写入降为 `O(1)`，没有把 attention 降为 `O(1)`。
8. 用 1/2/4 token chunk 和非连续绝对 positions 验证 chunked causal decode 与全量
   prefill 等价。

## 第四组：MoE

1. 构造 correction bias 改变选择、但不改变原始 routing score 的例子。
2. 为 grouped Top-K 添加随机 property test：所有 index 必须来自被保留组。
3. 写 CPU histogram + exclusive scan，并证明 offsets 最后一项等于 `T*K`。
4. 从 `(token, slot)` 打包再恢复，验证每个 contribution 恰好出现一次。
5. 比较逐 expert 循环与 grouped GEMM 的 latency 和临时显存。
6. 给定 per-expert counts，推导 padded batched GEMM 的有效槽位率；构造利用率低于 25%
   的路由分布，并与逐 expert loop 比较。
7. 将两个宽度为 `D_h` 的 shared SwiGLU 参数沿中间维拼接，证明一个宽度为 `2D_h` 的
   SwiGLU 输出等于两个独立 shared experts 的输出和，并核对三个权重矩阵的拼接方向。
8. 为 counts `[2,0,5]` 写出 expert offsets 和逐行 expert id；比较 padded batched GEMM 与
   active-row kernel 的 executed rows，并说明后者为什么仍不等于高性能 grouped GEMM。
9. 构造至少三个 sigmoid score 精确饱和为 `1.0` 的 FP32 输入，验证 group 与 expert tie 都
   按较小 id 选择；再说明为什么普通随机单元测试可能漏掉这个边界。
10. 分别对 `n_groups=1` 和 group-limited 配置运行 router benchmark，固定 `T,D,E,K`，比较
    projection FLOPs、retained candidates、latency 与最终 expert load，不能把全部差异归因
    于 GEMM。
11. 对 counts `[17,0,5,31]`、`D=33`、`D_h=65`，手算 grouped forward 的 row tiles、hidden
    tasks、down tasks、inactive tail row lanes 与 lane utilization；再与统一 padding 的 124
    个 allocated slots、71 个 padding rows 比较，解释两种无效工作为何不能混用。
12. 在 grouped tiled kernel 中删掉消费后的第二个 `__syncthreads()`，构造多个 K tiles 的
    输入并用 racecheck 观察 shared tile 被提前覆盖；修复后验证所有 tail shapes。
13. 对 FP16 WMMA expert 手算一个 `D=17,D_h=33,n_e=1` 的 zero-padding 范围；分别列出
    gate/up accumulator、materialized hidden 和 down output 的 dtype，再删除 hidden 的
    FP16 再量化，构造能让 reference 对比失败的输入。
14. 故意让半个 warp 跳过 `mma_sync`，根据 WMMA 文档解释为何不能把 hang/未定义结果当成
    普通数值误差；恢复一致控制流后，再检查 shared pointer 对齐和 `ldm=16`。

## 第五组：Expert Parallelism

1. 给定四个 rank 的 send-count 矩阵，手算每个 rank 的 recv counts 和 displacement。
2. 构造一个 rank 接收 0 行的 case，通信调用仍须在所有 rank 上匹配。
3. 为 metadata 设计 64-bit packed encoding，并计算可支持的最大 token/expert/rank 范围。
4. 对 host-staging MPI 与 NCCL 路径做逐元素等价测试。
5. 用 CUDA events 画出 chunked dispatch/compute/restore 时间线，量化 overlap efficiency。
6. 证明变长 All-to-All 的 backward 是交换 send/recv splits 的反向 All-to-All。
7. 构造 source-major 接收缓冲，说明仅有 per-expert counts 为何不足以直接调用 grouped
   GEMM，并实现 permutation 与 inverse permutation。
8. 让一个 rank 的本地 token 数为 0、另一个 rank 的接收 route 数为 0，验证所有 rank
   仍进入相同的 forward/backward collective 序列。
9. 给定 expert counts `[10,1,1,4]` 和 round-robin 两 rank ownership，计算两个 rank 的
   receive rows、local padded slots、peak-to-mean 与 CV；再令 capacity factor 为 1，计算
   keep、drop、padding 和利用率。
10. 已测得 `dispatch=2 ms, compute=5 ms, restore=1 ms`。分别计算串行核心时间、把两次通信
    归入同一资源时的无限 chunk 乐观下界和最大可隐藏比例，并列出为何实测达不到该下界。
11. 为 replicated shared expert 画出前向和反向通信图：说明它为何不进入 routed
    All-to-All，以及各 rank 的 shared 参数梯度为何仍需另一次归约。
12. 对 send splits `[7,0,5,2]` 和 `pipeline_chunks=3` 写出每个 peer 在三个 chunks 中的
    counts 与原 buffer ranges；证明把整张 buffer 直接 `tensor.chunk(3)` 会破坏哪项协议。
13. 对同一 NCCL 配置分别运行 `pipeline_chunks=1/2/4/8`，保存端到端 rank-max raw samples、
    `pipelined_core` 和 Nsight Systems 时间线；解释为什么“重叠更多”仍可能端到端更慢。

## 第六组：PyTorch 自定义算子

1. 根据 Q/K/V shape 手写 Attention 的 FakeTensor 输出推导，并构造三个错误 shape。
2. 删除 current-stream 获取逻辑、故意改用 default stream，设计一个能暴露竞态的测试。
3. 比较 `backend="auto"` 与 `backend="cuda"`：说明基准脚本为何必须使用后者。
4. 在原生 backward 完成前，用 `gradcheck` 验证 reference-recompute autograd 公式。
5. 列出一个 CUDA wheel 的 Python、PyTorch、CUDA toolkit 与 GPU SM 兼容性维度。
6. 为多输出 route-pack 算子写 FakeTensor shape 推导，解释为什么能知道 counts 的 shape、
   却不能知道 counts 的值。
7. 推导 route-pack 与 route-combine 的一阶 backward，并用 `gradgradcheck` 检查 combine 的
   二阶梯度。
8. 搜索原生调用热路径中的 `.item()`、`.tolist()` 与 host copy；区分算法必需同步和可移除
   的输入校验同步，并用 profiler 验证修改前后时间线。
9. 在自定义 expert backward 中把 offsets 转成 Python `int`，观察 `opcheck` 的 AOT dynamic
   failure；再改成 tensor `repeat_interleave` 并解释 FakeTensor 与数据依赖的边界。

## 第七组：性能实验

1. 推导 `D=D_v=64` 时 Attention 的 matrix FLOPs 与 compulsory tensor bytes。
2. 解释为什么逐 query kernel 的实际 K/V 流量远高于 compulsory lower bound。
3. 对同一配置运行三次 benchmark，比较 median 与 p99，并保留全部 raw samples。
4. 从 Nsight Compute 读取实测 DRAM bytes，和讲义公式的下界计算流量放大倍数。
5. 在 Roofline 图上标出正确性 kernel 与二维 tiled kernel，判断下一步应优化数据重用还是
   Tensor Core 利用率。
6. 给定一个 `4 x 4` EP counts 矩阵，分别计算跨 rank route rows、forward activation
   payload 和 forward+backward payload；解释为什么不能把它们称为实测 NCCL bytes。
7. 同时记录每个 rank 的 latency，比较 rank 0、rank mean 与 rank max 三种统计如何描述
   一个明显负载不均衡的 step。
8. 构造两个 rank、两个阶段且最慢 rank 互换的例子，证明 `max_r(sum_s t_rs)` 不等于
   `sum_s(max_r t_rs)`；解释 profiling 同步为什么还会改变 overlap。
9. 分别构造 expert skew 与 rank skew：让两组 case 拥有相同 `total_routes` 和相同
   per-expert peak-to-mean，却因 expert ownership 不同产生不同 rank critical path。

## 第八组：对称内存与 One-sided 通信

1. 对 `P=4,R=2,B=2,E_l=2,C=16,H=128,FP16`，计算每 PE 与全作业 payload bytes；
   分别说明固定 `E_l` 和固定全局 `E` 时，增加 `P` 的尺度关系。
2. 给定 source×expert counts，分别计算 per-expert capacity 与 source—expert cell capacity，
   构造两者相差最大的路由分布。
3. 手算 `symmetric_moe_buffer_offset` 的最后一个元素和一个中间 cell 起点，证明相邻 feature
   连续、相邻 row 相差 `H`。
4. 画出 payload put、fence、signal、wait、read 的 happens-before 图；删除 fence 后给出
   consumer 读到旧 payload 的合法执行顺序。
5. 解释 fence、quiet 与 barrier 的 ordering、completion、notification 差异；不能把三者
   都写成“同步”。
6. 用 generation counter 替换二值 ready flag，构造 buffer 快速复用导致 ABA 的反例。
7. 为 persistent kernel 分配 communication、polling 与 compute worker roles，构造所有
   CTAs 等待而 progress CTA 无法 resident 的资源型 deadlock。
8. 实现两 GPU 固定路由的最小 put-with-signal 原型，并与 NCCL reference 比较 payload、
   最终输出和 1000 轮 buffer reuse；没有正确性证据前不要测 fused kernel 加速比。
