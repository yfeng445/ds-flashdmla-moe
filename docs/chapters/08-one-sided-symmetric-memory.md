# 第八章：对称内存与 One-sided MoE 流水线

第五章的两次 All-to-All 是清晰的语义基线：所有 rank 先完成 dispatch，再开始 expert
compute，随后统一 restore。它容易验证，却把一层划成几个较粗的全局阶段。若希望一个
expert tile 到达后立即计算、完成后立即返回，就需要比 collective 更细的通信与同步粒度。

本章讨论一种设计工具：使用 PGAS（partitioned global address space）与对称内存组织
one-sided 通信。它不是“把远端显存当普通本地指针”，也不自动带来重叠；真正的关键是
可寻址布局、data/flag 发布协议、buffer 生命周期与可持续前进的调度。

## 8.1 Collective 与 one-sided 的控制权

collective 描述一组 rank 共同参与的操作。变长 All-to-All 中，每个 rank 都要按兼容顺序
进入 counts、dispatch 和 restore。one-sided put 则由发起方描述：

```text
put(local_payload, remote_symmetric_address, destination_pe)
```

目标 PE 不必同时调用一个配对 receive；它稍后通过 signal 或其他同步对象判断数据是否可
消费。这种控制权差异允许把全局阶段拆成 peer/expert/tile 级事件，但同时把以下责任交给
程序员：

- 哪块远端区域属于当前消息；
- 多个 producer 是否会写到同一位置；
- payload 何时先于 ready flag 对 consumer 可见；
- buffer 何时可以安全复用；
- 某些 peer 没有消息时，调度器是否仍能前进。

因此 one-sided 不是“无同步”，而是从隐含的 collective rendezvous 改成显式、细粒度的
依赖协议。

## 8.2 对称对象不是统一虚拟地址

每个 processing element（PE）各自拥有一份名称、类型和大小对应的 symmetric object。
动态对称分配要求所有 PEs 以相同参数参与；分配结果位于各自的 symmetric heap。一次远端
访问由“本地生成的 symmetric address + 目标 PE”共同确定。

要区分三件事：

1. 同一 symmetric address 在本 PE 上也是合法的本地地址；
2. 不能把另一个 PE 生成的地址拿来直接解引用或传给要求 symmetric address 的 API；
3. 某些拓扑上可以查询可直接访问的远端指针，但这是一项额外能力，不是 PGAS 的普遍定义。

所以“像一张大 GPU 一样编程”是地址组织的抽象，不代表整个集群共享一张任意指针都有效的
统一虚拟地址空间。

## 8.3 MoE 对称张量布局

令：

- `P`：EP group 中的 PEs 数；
- `R`：通信轮数，通常 dispatch 与 combine 两轮；
- `B`：每轮的 temporal buffer roles，例如 outgoing 与 incoming；
- `E_l`：每个 PE 预留的 local expert slots；
- `C`：每个 source—expert cell 的 route-row capacity；
- `H`：一个 token activation 或 contribution 的特征宽度。

每个 PE 可分配一份行主序张量：

```math
L\in\mathbb{R}^{P\times R\times B\times E_l\times C\times H}.
```

第一维不是说本 PE 拥有其他 PE 的显存，而是在本地 symmetric object 中为每个通信伙伴
预留互不冲突的区域。`B=2` 时可以把一个 role 用于 outgoing staging，另一个用于 incoming
delivery；具体读写方向必须由协议固定，不能只凭变量名猜测。

若每个元素占 `s` 字节，则每 PE 的 payload storage 为：

```math
M_{PE}=P R B E_l C H s,
```

全作业为：

```math
M_{job}=P M_{PE}.
```

“每 GPU 内存是 `O(P)`”只在弱扩展下成立：若每个 PE 固定拥有 `E_l` 个 experts，增加
PEs 会同时增加全局 expert 数，`M_PE` 随 `P` 线性增长。若全局 expert 总数 `E` 固定并且
均匀切分，`E_l=E/P`，则：

```math
M_{PE}=R B E C H s,
```

其中显式的 `P` 被 expert shard 抵消。讨论可扩展性时必须声明固定的是 `E_l` 还是 `E`。
此外，signal、cursor、alignment、allocator metadata 与 topology-specific staging 还会增加
公式外开销。

## 8.4 从路由矩阵得到 cell capacity

对 source rank `i` 和全局 expert `e`，记实际 route 数为 `N[i,e]`。每个 source—expert
组合对应一个 cell；完全容纳本次已观测路由所需的容量为：

```math
C_{observed}=\max_{i,e}N[i,e].
```

若配置固定容量 `C`，则该布局最多保留：

```math
R_{keep}=\sum_{i,e}\min(N[i,e],C),
```

并丢弃：

```math
R_{drop}=\sum_{i,e}N[i,e]-R_{keep}.
```

这与第四章按全局 expert 总 count 设置统一 capacity 的模型不同。这里每个 source 都为一个
expert 拥有独立 cell，因此 `max_{i,e} N[i,e]` 通常小于 `max_e sum_i N[i,e]`。两种公式
服务于不同物理布局，报告不能只写一个含糊的 `capacity` 字段。

当前 `symmetric_moe_buffer_model_from_routes` 接收 `[source_rank, expert]` 的整数矩阵，输出
张量 shape、每 PE 与全作业 bytes、显式容量的 overflow、空 expert slot 和 storage
utilization。`symmetric_moe_buffer_offset` 则把六维坐标变成行主序 element offset，可用于
在写 kernel 前检查 producer 与 consumer 是否计算了同一地址。

例如两 rank、每 rank 两个 local experts、两轮、两个 buffer roles、`C=4,H=4` 时：

```text
shape per PE = [2, 2, 2, 2, 4, 4]
row slots per PE = 2 * 2 * 2 * 2 * 4 = 64
```

若全作业 7 条 routes 都命中同一个 expert，每条 route 在 dispatch/combine 的 outgoing 与
incoming 区各占一次，则活跃 row placements 为 `7*2*2=28`，全作业共有 `2*64=128`
slots，模型利用率为 `28/128`。这个数字衡量静态布局占用，不是网络带宽利用率。

## 8.5 发布协议：payload、ordering、signal

最小 data/flag 协议是：

```text
producer: reserve cell → put payload → order payload before signal → publish generation
consumer: wait for generation → read payload → publish consumed generation
producer: wait until consumed → reuse cell
```

不能先写 flag 再写 payload，否则 consumer 可能观察到 ready 后读取旧数据。NVSHMEM 中：

- `fence` 为此前与之后、发往同一 PE 的相关操作提供投递顺序，但不保证此前操作已经完成；
- `quiet` 保证调用 PE 此前发出的受支持操作完成，它是本地、非 collective 操作，也不会
  自动通知 consumer；
- signal wait/test 让目标 PE 观察本地 symmetric signal；put-with-signal 把连续 payload
  copy 与随后的 signal update 组合为一个明确的发布原语。

只有一位线程调用 fence/quiet 时，其他 producer threads 必须先用 CTA/warp 同步保证它们
的通信操作已经发出。CPU fence 也不能替 GPU 发出的操作排序，反之亦然。

二值 `ready=1` 容易出现 ABA：buffer 重用后，consumer 可能把上一轮遗留的 1 当成本轮
完成。更稳妥的做法是使用单调 generation/epoch，并在每个 `(peer,round,buffer,expert)`
cell 上检查期望值。generation 回绕也必须有明确位宽与处理策略。

## 8.6 写冲突与容量领取

两个 PEs 或两个线程并发访问同一位置，且至少一个执行写入时，就需要协议定义顺序或原子
语义。常见的安全 cell 分配方式是：

1. 每个 source—expert 拥有独立 atomic cursor；
2. producer 用 fetch-add 领取 row；
3. `row<C` 才能写 payload；
4. 最后一个 producer 或显式 completion counter 发布 tile-ready signal。

cursor 只能保证地址唯一，不能自动保证 payload 已发布。把“领取槽位”“写完数据”“整个
tile 完成”压成同一个计数器，往往会让 consumer 在最后一次 reservation 发生后、最后一行
真正写完前开始计算。

## 8.7 Temporal buffering 与持续前进

双缓冲的目的不是复制两份数据本身，而是把生命周期错开：consumer 读取 buffer `b` 时，
producer 可以填充 `1-b`。每个 buffer 至少需要状态：

```text
EMPTY → WRITING → READY → READING → EMPTY(next generation)
```

若 persistent kernel 同时承担通信和 GEMM，还要预留足够的 GPU execution resources 给进度
任务。所有 CTAs 都阻塞等待 signal，而负责发出远端 put 的 CTA 尚未被调度，会造成资源型
死锁。可采用固定 worker roles、非阻塞 task queue 或限制 compute occupancy，确保通信、
signal polling 与计算都能持续前进。

跨节点与节点内链路的 latency/bandwidth 不同。tile swizzling 可以优先计算已经到达的本地
或节点内 tile，同时让跨节点 tile 在后台传输；但调度顺序必须由 ready generation 驱动，
不能因为某 tile “通常先到”就省略依赖。

## 8.8 从模型到真实 backend 的验证阶梯

本仓库当前只提供布局和存储代价模型，没有链接 NVSHMEM，也没有宣称 single-kernel MoE。
实现 one-sided backend 时建议逐级建立证据：

1. 用整数 counts 验证 shape、offset、capacity 与 bytes；
2. 用单 GPU 模拟 producer/consumer generation，覆盖复用和 wraparound；
3. 两 GPU 固定路由，逐行比较 payload 与 Gloo/NCCL reference；
4. 加入 0-count peer、热门 expert、满容量和 overflow；
5. 重复多轮并随机延迟 producer，使用 Compute Sanitizer 检查 race；
6. 再用 Nsight Systems 验证通信、expert GEMM 和 combine 是否真的重叠；
7. 最后比较 payload、buffer footprint、端到端 rank-max latency 与尾延迟。

single persistent kernel 减少 launch 与全局阶段间隙，但也增加寄存器、shared memory、任务
调度和 deadlock 风险。是否值得融合，应由上述可复核测量决定，而不是由 kernel 数量决定。
