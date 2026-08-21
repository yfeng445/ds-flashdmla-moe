# 第六章：把 CUDA kernel 接入 PyTorch

一个能被 `nvcc` 编译和启动的 kernel，还不是一个可靠的 PyTorch 算子。它还需要明确
回答：由谁校验输入、如何选择 CPU/CUDA 实现、输出 metadata 如何推导、autograd 在哪
一层注册，以及 kernel 应该在哪个 device 和 stream 上运行。

## 6.1 算子 schema

本仓库的 Attention 原生入口采用如下逻辑 schema：

```text
attention_forward(Tensor q, Tensor k, Tensor v,
                  bool causal, float scale) -> Tensor
```

schema 是调用方与后端实现共同遵守的契约。`Tensor` 输入没有被标记为可变，因此实现
必须返回新张量，不能把某个输入原地改写后冒充输出。Python 中的 `float` 在 C++ kernel
wrapper 中对应 `double`；launch 前再显式转换成 FP32 scale。

GEMM 的 schema 展示了可选张量：

```text
tiled_gemm(Tensor a, Tensor b, Tensor? c,
           float alpha, float beta) -> Tensor
```

`Tensor?` 表示 epilogue 矩阵可以是 `None`。但 `c=None` 时必须同时满足 `beta=0`，否则
`βC` 没有定义。Python、FakeTensor 与 CUDA wrapper 都执行同一个检查；不能只在 native
路径校验，否则 backend fallback 会改变错误行为。

会原地修改 cache 的 schema 还必须标出 alias/mutation：

```text
mla_cache_projection_write_slots(...,
    Tensor(a!) kv_storage, Tensor(b!) pe_storage,
    Tensor(c!) position_storage,
    bool metadata_validated, ...) -> ()
```

`a!`/`b!`/`c!` 告诉 dispatcher、FakeTensor 与编译器这些输入会被修改。paged attention 自身
是 out-of-place，但与 slot write 都带一个内部 `metadata_validated` 标记：高层 API 完整检查
slot/page/position 值后传 `true`，避免 CUDA wrapper 再把相同 metadata 读回 host；直接调用 raw
operator 的测试传 `false`，保留越界、重复和页表防御检查。这个标记不是跳过公开 API 契约的
用户选项；高层只在 Tensor identity/version 仍匹配已验证记录时复用结果。

`TORCH_LIBRARY` 只定义 schema，`TORCH_LIBRARY_IMPL(..., CUDA, ...)` 为 CUDA dispatch
key 注册实现。二者分开以后，同一算子可以拥有 CPU、CUDA、Meta 或其他 backend，调用
方不必手写设备分支去直接调用 pybind 函数。

### 6.1.1 Whole-layer MoE 的 output-only 边界

single-device、staged、correctness-first 的比较 schema 保留为：

```text
deepseek_moe_forward(Tensor x, Tensor gate_weight,
                     Tensor expert_w1, Tensor expert_w2, Tensor expert_w3,
                     int topk, int n_groups, int topk_groups,
                     Tensor? score_bias, float route_scale) -> Tensor
```

相同参数和 output-only 返回契约还定义了 `deepseek_moe_forward_fused` 与
`deepseek_moe_forward_persistent`。前者使用 private pack 与 fused weighted-combine
epilogue；后者只在 single-device expert core 内增加 bounded persistent task queue，并为
小工作量保留 fused fallback。

它只返回最终 `[T,D]` output，不把 route indices、weights、counts、offsets 或中间 expert
contributions 暴露为 public outputs。Python 层只为这个 raw op 注册 FakeTensor metadata：
Fake 实现检查 shape/device/dtype/layout 与 forward-only 条件，然后返回与 `x` 同 shape 的空
FakeTensor；它不读取数据，也不运行 route 或 expert 数学。

这三个 raw operators 都只有 Fake + CUDA dispatch，没有 CPU、backward、autograd registration、
`CompositeExplicitAutograd` 或 `CompositeImplicitAutograd` 实现。CUDA v1 还要求 contiguous
CUDA FP32、sigmoid scoring、所有浮点输入均无 `requires_grad`，并在 deterministic algorithms
启用时拒绝 atomic routing。可移植 reference 与 fallback 属于公开 facade
`deepseek_moe_forward` facade：`cuda_staged`、`cuda_fused` 和 `cuda_persistent` 精确选择一个
raw operator，`cuda` 是 fused 别名，`auto` 优先 fused、可退到 staged，但不自动选择
persistent。不能把 Fake 注册误写成 Composite 或 whole-layer 训练支持。

## 6.2 加载顺序与纯 Python 安装

下述 Composite 加载顺序描述的是已有、可分解到 PyTorch reference 的 stage operators；
上一节的 output-only whole-layer raw op 是刻意保留的例外。

普通 CPU wheel 不包含 `_C` 动态库，因此 Python 先定义同名 schema，并注册一个由
PyTorch 运算组成的 `CompositeExplicitAutograd` reference。这里使用 explicit，是因为
算子还要拥有独立的 FakeTensor 和 autograd 注册。CUDA wheel 导入时先用
`torch.ops.load_library` 加载 `_C`，触发 C++ 静态注册；Python 随后只补充 reference、
FakeTensor 与 autograd 规则，不能再次定义同名 schema。

这种顺序带来两个性质：

- 没有 CUDA toolkit 的环境仍能安装、学习和运行全部数学 reference；
- 安装原生 wheel 后，CUDA dispatch 自动命中 kernel，其他 device 仍可分解为 reference。

动态库搜索必须要求“恰好一个候选文件”。若同时残留两个 `_C` 版本，随便选第一个会让
加载结果依赖目录遍历顺序，调试 ABI 或旧 wheel 问题会非常困难。

当前实现使用标准 ATen API，因此原生 wheel 与构建它的 PyTorch/Python 平台相关；不能把
文件名伪装成 abi3 后跨版本分发。若以后迁移到 PyTorch stable ABI，应同时替换 Tensor、
stream 与 dispatcher API，并在多 PyTorch 版本矩阵中验证后再改变 wheel tag。

## 6.3 FakeTensor 与 `torch.compile`

编译器 tracing 阶段不应该真的分配 GPU 数据或读取 `data_ptr`。FakeTensor kernel 只根据
输入 metadata 推导输出：

```text
q: [B,H,S_q,D]
k: [B,H,S_k,D]
v: [B,H,S_k,D_v]
o: [B,H,S_q,D_v]
```

同时检查 batch/head、序列长度、head dimension、device 和 dtype 关系。纯 composite
安装可以让 FakeTensor 继续分解到底层 PyTorch op；加载原生 backend 后，则显式注册这条
metadata 规则。Fake 实现不能访问真实元素值。

`torch.library.opcheck` 检查 schema、autograd 注册、FakeTensor 和 AOT dispatch 的组合
是否一致，但它不证明 Attention 数学正确。数值输出、梯度、mask 与 tail shape 仍需要
独立测试。

多输出 route-pack 算子的 FakeTensor 规则还要区分“由 shape 决定”与“由数据决定”的
metadata。若输入是 `[T,D]`、路由是 `[T,K]`，则 packed rows 恒为 `R=T*K`，所以可推导：

```text
packed_x:          [R,D]
packed_alpha:      [R]
packed_route_id:   [R]
packed_expert_id:  [R]
expert_counts:     [E]
rank_counts:       [P]
```

FakeTensor 能确定这些 shape，却不能知道 counts 的元素值或具体 permutation。编译图若要
根据某个 count 做 Python 控制流，就会触发数据依赖和 graph break；应把变长调度留在
能消费 device metadata 的算子边界，或把同步显式计入设计。

GEMM 的 metadata 只依赖输入 shape：`[M,K]×[K,N] -> [M,N]`。FakeTensor 还应验证内维、
device、dtype 以及可选 `C` 的 `[M,N]` shape，但绝不能执行真实 matmul。对应测试同时覆盖
`opcheck` 与 `torch.compile(fullgraph=True)`，因为 eager 数值正确并不能证明 tracing 合同
完整。

expert-major SwiGLU 的输出 shape 也只依赖 metadata：输入 `[R,D]` 与 weights
`[E_l,D_h,D]` 产生 `[R,D]`。但每个 row 属于哪个 expert 由 offsets 的元素值决定。
FakeTensor 只能检查 offsets 是 `[E_l+1]`，不能把它转换成 Python list。traceable reference
用 `counts=offsets[1:]-offsets[:-1]` 和 tensor `repeat_interleave` 生成 row-to-expert mapping；
若 backward 用 `int(offsets[e])` 写 Python 循环，AOT dynamic dispatch 会因 data-dependent
guard 失败，即使 eager 和普通 gradcheck 都可能通过。

## 6.4 Autograd 的过渡实现

当前原生 Attention 实现包含正确性优先的 FP16/BF16/FP32 forward/backward，三种路径都用
FP32 计算 dot、在线 Softmax 与梯度 workspace。公开 forward 算子注册 autograd 公式：普通
一阶 CUDA 反向可调度到原生 backward；CPU、不支持的 dtype、deterministic 模式和高阶梯度
调度到解析 PyTorch backward。

这是正确性过渡方案，不是性能终点：

- forward/backward 可以分别对解析 reference 做数值验证；
- 高阶梯度仍可由解析公式组成；
- 原生 dK/dV 使用原子累加，因此 deterministic 模式必须回退。

把 backward 改成二维 tiled 版本时，应保留同一公开 schema；随后用 `gradcheck`、FP64
解析 reference、current-stream 测试和 deterministic 行为重新验证，而不是仅依赖
forward 误差。

route-pack 的输出 permutation 由离散 expert id 决定，不对 expert id 求导。对连续输入，
其 backward 是逆 permutation 后按 token 累加：

```text
dX[token] = sum_slot dPackedX[route(token, slot)]
```

`packed_alpha` 的梯度按 `packed_route_id` scatter 回 `[T,K]`。counts、expert ids 与 route
ids 必须用 `mark_non_differentiable` 标记；否则 autograd 可能为纯 metadata 建立无意义的
梯度边。

route-combine 的解析 backward 更直接。若：

```math
y_t=\sum_{r:\,token(r)=t}\alpha_r c_r,
```

则：

```math
\frac{\partial L}{\partial c_r}=\alpha_r\frac{\partial L}{\partial y_{token(r)}},
```

```math
\frac{\partial L}{\partial \alpha_r}=c_r^T
\frac{\partial L}{\partial y_{token(r)}}.
```

即使 forward 使用 atomic combine，高阶梯度仍可让这条 PyTorch 解析公式承担。

对 `D=αAB+βC`，GEMM 的解析 backward 为：

```math
dA=\alpha\,dD\,B^T,
```

```math
dB=\alpha\,A^T dD,
```

```math
dC=\beta\,dD.
```

native kernel 目前只实现 forward；dispatcher 用上式构造一阶 backward。公式本身仍由
PyTorch matmul 组成，因此 gradgradcheck 可以继续穿过它。`C=None` 时返回 `dC=None`，
而不是伪造一张零张量。

grouped tiled SwiGLU 同样先注册 forward-only CUDA kernel；FP32 dispatch 到 shared-memory
CUDA-core tiles，FP16 dispatch 到 WMMA tiles。backward 在启用 grad 的上下文中
重算张量化 segmented reference，再用 `torch.autograd.grad` 求 activation 与三组 weights
的梯度；offsets 是离散 metadata，返回 `None`。这种写法同时保留一阶 CUDA 训练语义与
二阶梯度检查，但 backward 性能仍是 reference 水平。FP16 reference 还必须复现 forward
在 SiLU 之后、W2 之前的 hidden 半精度量化，benchmark 必须据实说明这条数值边界。

grouped router 的离散 index 不可微，setup context 将其标记为 non-differentiable，并保存
forward 已选 indices。backward 在这些固定 indices 上重算无偏 sigmoid score、gather 和
归一化，只向 activation 与 gate weight 返回梯度。correction bias 只改变离散选择，因此
返回 `None`；在选择边界处整个路由函数本来就不连续。这样的局部导数约定可以通过
`opcheck`、`gradcheck`、`gradgradcheck` 和 `torch.compile`，但不能被误解成对 Top-K index
本身求导。

## 6.5 Device、stream 与异步错误

wrapper 必须使用输入张量所在 device 的 `CUDAGuard`，并从 PyTorch 取得当前 CUDA
stream。若 kernel 偷用 default stream，在常规同步测试里可能看似正确，但和前后算子运行
在其他 stream 时会发生读取未完成数据或提前消费输出的问题。

kernel launch 后使用异步 launch check 捕获参数、shared memory 或非法配置错误，但不应
为了“方便检查”每次调用都执行 `cudaDeviceSynchronize()`。强制同步会破坏 PyTorch 的
异步执行和通信计算重叠。运行期越界等错误由后续同步点或专门的 CUDA 调试测试发现。

同样要警惕看似普通的 Python/C++ 校验造成隐式同步。例如 CUDA tensor 上的
`torch.any(...).item()`、`aminmax(...).item()` 或把 counts 转成 Python list，都会要求
host 等待 device。reference 路径可以用这些检查提供清晰异常；原生热路径应在 kernel 中
做 device-side assertion，或由上游已验证的契约保证范围。若算法本身需要 host split
sizes，应把这次同步视为协议成本并在 benchmark 中明确，而不是让额外校验偷偷增加同步。

公开 GEMM API 采用三种 backend 语义：`reference` 总是运行可微规范；`auto` 只在输入是
contiguous FP32 CUDA 二维矩阵且 native schema 已注册时选择 kernel；`cuda` 对任何不满足
条件的输入直接报错。这个边界防止 benchmark 配置写着 CUDA，实际却悄悄测到 reference。
attention、MLA、router 与 expert wrapper 延续同一约定，并各自公开 capability flag；Attention
和 staged MLA 的 CUDA dtype 集合是 FP16/BF16/FP32，expert 是 FP16/FP32，而 router 与 route
原生路径仍只有 FP32。性能实验应显式选择 backend，而不是根据运行环境猜测实际命中的实现。

## 6.6 构建与验证分层

原生算子的证据应分成三层：

1. macOS/CPU：reference、dispatcher schema、autograd、FakeTensor、sdist/wheel；
2. Linux + CUDA toolkit：`nvcc` 编译并安装 CUDA wheel，不要求可见 GPU；
3. Linux + NVIDIA GPU：运行多 shape 数值矩阵、current-stream 测试和 reference backward。

第二层通过不代表 kernel 数值正确，第三层通过也不替代不同 SM、dtype 和极限 shape 的
覆盖。按 `csrc/ops.cpp` 与各 `TORCH_LIBRARY_IMPL(..., CUDA, ...)` 的源码清点，当前扩展注册
21 个 formal CUDA operators。MoE 清单包含 staged `deepseek_moe_forward`、
`deepseek_moe_forward_fused` 与 `deepseek_moe_forward_persistent`；其中两个 paged MLA operator 仍分别负责
per-slot write 与直接页表 attention。把证据拆开记录，才能准确说清“源码可编译”和“算子已在
GPU 验证”的区别。
