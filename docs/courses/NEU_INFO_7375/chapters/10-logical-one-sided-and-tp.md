# 第十章：可在单卡验证的 One-sided 协议与 TP 语义

第八章建立了对称 buffer 的容量模型，但“地址算对”不等于“通信协议正确”。真实多卡实现
之前，还需要先固定三个不会随传输库改变的契约：rank 如何映射到 DP/EP/TP 坐标、route
如何跨乱序 dispatch/return 保持身份，以及 TP expert 如何切分权重。本章对应的 Python
实现只在单进程中执行这些契约，不进行远端传输，也不证明远端可见性或通信计算重叠。

## 10.1 TP-fastest 的 rank 双射

`ParallelMesh(dp_size, ep_size, tp_size)` 使用固定公式：

```text
rank = ((dp * ep_size) + ep) * tp_size + tp
```

因此 TP 是变化最快的维度。`rank(coordinate)` 与 `coordinate(rank)` 互为严格逆映射，
`dp_group`、`ep_group`、`tp_group` 都从同一公式生成，不再由调用方各自猜测 rank 顺序。

expert owner 表示 EP 坐标，而不是一个展平 rank。一个 expert 在某个 DP replica 中属于
owner 对应的整个 TP group。`ExpertPlacement` 按 global expert id 升序分配 owner-local
slot；即使 owner 表不是 round-robin，slot 也保持稳定。

## 10.2 Payload-before-signal 状态机

一个 cell 由下列字段唯一标识：

```text
(producer_pe, consumer_pe, round_id, buffer_slot, local_expert_slot)
```

cell 生命周期是：

```text
EMPTY
  -> WRITING(generation, count)
  -> READY
  -> READING
  -> CONSUMED
  -> EMPTY(next generation)
```

dispatch 与 return 共用一个持久 protocol registry，但使用互不相同的 `round_id`：当前分别
为 0 和 1。generation 只表示 scheduler iteration，不再同时充当 phase id。即使 source 与
owner 是同一个 PE，或两个方向的 route 恰好交换 producer/consumer，两个 phase 也不会
alias 到同一个 cell。完成 consumed acknowledgement 后，下一 generation 复用原 cell；本轮
没有 payload 的既有 cell 也执行 count=0 生命周期，从而保持 generation 同步。

producer 必须先预留 count、写完 `[0,count)` 的所有 payload rows，再发布同 generation
的 ready signal。consumer 读取后发布 consumed acknowledgement；只有 producer 观察到
同 generation 的 acknowledgement 才能 recycle。`count=0` 仍经过完整的 signal/read/ack
生命周期，避免某个 peer 没有 route 时破坏全局推进。

每个 payload row 携带：

```text
RouteIdentity(source_pe, generation, route_id) + row_index + payload
```

协议允许 payload rows 乱序到达，但要求 row index 完整且唯一、route identity 唯一。
stale/future generation、重复 payload/signal/ack、提前 signal、count 不一致、capacity
overflow、错误 actor PE、非法 owner 与 generation wrap 都在改变状态前被拒绝。入队
payload 会 clone，调用方后续修改源张量不会改写已发布内容。

## 10.3 单进程 logical-PE simulator

`FakeDistributedMoE` 按 `(source PE, owner PE, owner-local expert slot)` 分组 route，并分别
执行 dispatch 与 return 两套 cell 生命周期。测试可以故意反转 payload 到达顺序；返回值
不依赖到达位置，而是通过全局 route identity 恢复到 source。

模拟报告固定带有以下证据边界：

```json
{
  "simulated": true,
  "remote_visibility_verified": false,
  "transport_performed": false,
  "multi_gpu_verified": false
}
```

这些字段不是保守措辞，而是结果语义的一部分。单卡 simulator 能证明状态转换、容量错误、
route 身份与乱序恢复；它不能证明另一个 GPU 何时看到 payload、传输是否真的发生、多个
GPU 是否无死锁，或通信是否和 expert compute 重叠。

## 10.4 Forward-only logical TP SwiGLU

DeepSeek expert 的计算是：

```math
y=W_2\left(\operatorname{SiLU}(W_1x)\odot W_3x\right).
```

TP size 为 `p` 时，`W1/W3` 沿 hidden rows 分成 `p` 份，`W2` 沿对应 hidden columns
分成 `p` 份。每个逻辑 shard 得到一个 `[... , model_dim]` partial，最终按元素求和：

```math
y=\sum_{j=0}^{p-1}W_2^{(j)}
\left(\operatorname{SiLU}(W_1^{(j)}x)\odot W_3^{(j)}x\right).
```

`tensor_parallel_swiglu_forward` 支持 `tp_size=1/2/4`，要求 hidden 可整除。FP64 输入使用
FP64 partial accumulation，其余浮点输入使用 FP32。为匹配仓库既有 expert oracle 与原生
WMMA stage boundary，FP16 会在 W2 前把每个 shard 的 materialized hidden round 到 FP16，
再转回 FP32 做 down projection；最终输出只 cast 一次。它是明确的
forward-only functional oracle：任何输入带 `requires_grad` 都会报错，包括调用方处于
`no_grad` 上下文时。这里的 sum 是本设备上的 Python 运算，不代表执行了跨卡 reduction。

## 10.5 可复现实例与下一步硬件门槛

运行结构化实例：

```bash
python benchmarks/logical_distributed.py \
  --pes 2 --experts 4 --routes-per-pe 4 \
  --cell-capacity 4 --tp-size 2
```

输出保留 route identities、checksum、协议 cell 数和上述四个证据字段，不报告速度提升。
仓库保留了该命令的
[结构化输出](../../../../validation/logical/2026-08-22-ep-tp-reference.json)；其 checksum 只用于
固定这个 deterministic example，不是跨机或跨 GPU 运行证据。
进入真实多卡实现前，至少还要补齐：

1. 两卡与四卡上的 payload 可见性和 generation 顺序；
2. 0-count、热门 expert、满容量、随机延迟与长期 generation 压力；
3. return 按 route identity 与现有 collective EP reference 的逐行数值比较；
4. 真实 TP reduction 与完整 expert oracle 的对齐；
5. profiler 中传输、expert compute、return 是否实际重叠，以及 occupancy 下是否持续前进。
