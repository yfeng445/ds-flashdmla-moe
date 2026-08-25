# Single-Device DeepSeek MoE Forward Backend Design

## Status

This design defines the first implementation milestone on the repository's
FlashDMoE path. It is a single-device, forward-only DeepSeek-style MoE operator
with one public `reference/cuda/auto` facade and one output-only native operator.

It is deliberately **not** described as a complete FlashMoE implementation.
The paper's defining properties are a persistent kernel, tile-level scheduling,
and device-initiated inter-GPU communication. The first milestone keeps the
existing staged CUDA launches behind a stable whole-layer operator boundary;
later milestones may remove those intermediate buffers and launches without
changing the public semantic contract.

Primary references:

- [FlashMoE paper](https://arxiv.org/abs/2506.04667)
- [FlashMoE official implementation](https://github.com/osayamenja/FlashMoE)

## Goals

- Expose a complete single-device routed-MoE forward call instead of requiring
  callers to assemble router, pack, expert, and combine stages themselves.
- Keep the Python/reference and CUDA implementations semantically
  interchangeable through `backend="reference"`, `backend="cuda"`, and
  `backend="auto"`.
- Reuse the verified grouped router, route pack, active-row SwiGLU, and route
  combine implementations rather than copying their mathematics.
- Select one backend for the complete request before execution. A request must
  never mix reference and CUDA stages implicitly.
- Freeze an output-only native schema so implementation work can reduce private
  metadata, intermediate tensors, and launches without changing callers.
- Keep native backward out of the supported surface. Existing backward
  experiments remain under `csrc/experimental/`.

## Non-goals

- Multi-GPU expert parallelism, NCCL, NVSHMEM, RDMA, or symmetric allocation.
- A persistent scheduler/processor/subscriber megakernel.
- Shared experts, capacity limits, token dropping, auxiliary balancing loss, or
  returned routing traces.
- Native backward or a new backward schema.
- CUDA support for softmax routing, FP16, BF16, FP8, or INT8 in this milestone.
- A claim that the staged CUDA implementation is faster than the reference.

## Approaches Considered

### 1. Python facade over public stage APIs only

This is the smallest change and proves the whole-layer contract, but it adds no
whole-layer dispatcher boundary. Profilers and compiled callers would still see
the implementation as unrelated public stages, and future fusion would require
another API migration.

### 2. Output-only native operator wrapping the staged CUDA path

This is the selected approach. The Python facade chooses the backend once. The
native operator initially launches the existing router, pack, active-expert,
and combine CUDA stages internally and returns only the final tensor. The raw
schema remains valid when those stages are later fused.

This approach is honest about current launch count while establishing the
correct abstraction boundary for subsequent FlashDMoE work.

### 3. Implement the persistent FlashMoE megakernel immediately

This would require an in-kernel scheduler, task descriptors, tile-level GEMMs,
capacity/storage protocols, and eventually one-sided multi-GPU communication.
Attempting all of that before freezing and testing the whole-layer semantics
would combine too many new failure modes. It is deferred.

## Public Python API

The new facade lives in `src/ds_flash_mla_moe/moe_ops.py` to avoid circular
imports between `moe.py`, `ops.py`, `router_ops.py`, and the stage wrappers.

```python
from typing import Literal

MoEBackend = Literal["auto", "cuda", "reference"]
MoEScoreFunction = Literal["sigmoid", "softmax"]


def deepseek_moe_forward(
    x: Tensor,
    gate_weight: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
    *,
    topk: int,
    n_groups: int = 1,
    topk_groups: int | None = None,
    score_func: MoEScoreFunction = "sigmoid",
    score_bias: Tensor | None = None,
    route_scale: float = 1.0,
    backend: MoEBackend = "auto",
) -> Tensor: ...
```

The name is `deepseek_moe_forward`, not `flashdmoe_forward`. The latter would
prematurely imply the distributed persistent-kernel properties that this phase
does not yet implement.

`src/ds_flash_mla_moe/__init__.py` exports `MoEBackend`,
`MoEScoreFunction`, `deepseek_moe_forward`, and `cuda_moe_available`.

## Tensor and Numerical Contract

| Value | Contract |
| --- | --- |
| `x` | floating `[*token_shape, D]`, rank at least 2, `D > 0`; the flattened token count may be zero |
| `gate_weight` | `[E, D]`, `E > 0` |
| `expert_w1`, `expert_w3` | `[E, H, D]`, `H > 0` |
| `expert_w2` | `[E, D, H]` |
| `score_bias` | optional `[E]`; affects selection only |
| floating inputs | identical device and dtype |
| output | contiguous, with the same shape, device, and dtype as `x` |

Routing follows the existing executable specification:

- `E` is divisible by `n_groups`;
- `1 <= topk_groups <= n_groups`;
- `1 <= topk <= topk_groups * (E / n_groups)`;
- smaller group and expert ids win exact ties;
- sigmoid scores are renormalized over selected experts before multiplying by
  `route_scale`;
- softmax scores follow the existing reference and are not renormalized after
  Top-K;
- route weights are applied after the nonlinear expert, never to expert inputs;
- every selected route is evaluated; this phase has no capacity drop.

## Backend Policy

### Reference

`backend="reference"` uses the staged packed reference and is valid on CPU or
CUDA for every dtype and score function already supported by that reference.
It retains PyTorch autograd behavior. The existing token-loop
`deepseek_moe_reference` remains an independent numerical oracle in tests.

### CUDA

`backend="cuda"` requires the complete request to satisfy all of these rules:

- every floating tensor is a contiguous CUDA FP32 tensor on one device;
- routing uses `score_func="sigmoid"`;
- no floating input has `requires_grad=True`;
- deterministic algorithms are disabled;
- the loaded extension registers the whole-layer CUDA operator.

An explicit CUDA request fails with a precise `RuntimeError` when any condition
is not met. Kernel and launch errors propagate; the facade never catches a CUDA
failure and retries the reference implementation.

### Auto

`backend="auto"` evaluates the same eligibility predicate once. It invokes the
complete CUDA operator only when every condition holds; otherwise it runs the
complete reference path. Individual stages never receive `backend="auto"`.

This prevents accidental mixed execution such as a reference FP16 router and
route path feeding the FP16 CUDA expert kernel.

## Native Operator Boundary

The dispatcher schema is:

```text
ds_flash_mla_moe::deepseek_moe_forward(
    Tensor x,
    Tensor gate_weight,
    Tensor expert_w1,
    Tensor expert_w2,
    Tensor expert_w3,
    int topk,
    int n_groups,
    int topk_groups,
    Tensor? score_bias,
    float route_scale
) -> Tensor
```

The raw operator accepts flattened `[T, D]` inputs, is registered for CUDA, and
returns `[T, D]`. It has a FakeTensor implementation for shape propagation but
no Composite or autograd registration. Both Python and C++ enforce the
forward-only rule so direct `torch.ops` calls cannot bypass it.

The first C++ implementation performs:

1. grouped sigmoid routing;
2. `route_pack` with `world_size=1` and one local owner for every expert;
3. `counts_per_expert` prefix sum to obtain active expert offsets;
4. active-row SwiGLU expert computation;
5. post-expert route weighting and accumulation by source token.

Single-device route packing already produces expert-major rows. Running
`expert_major_pack` again would be redundant and is prohibited by this design.

The implementation file is `csrc/moe/deepseek_moe_forward_cuda.cu`. A focused
`csrc/moe/moe_cuda_ops.h` exposes only the existing host entry points needed by
the whole-layer operator; CUDA kernels stay private to their current translation
units. `csrc/ops.cpp`, `setup.py`, `MANIFEST.in`, and the CUDA build workflow add
the schema/source/operator check.

## Error Handling

- Structural errors use `ValueError` or `TypeError` before backend selection.
- Explicit CUDA ineligibility uses `RuntimeError` prefixed with
  `CUDA DeepSeek MoE is unavailable:`.
- The raw CUDA operator repeats critical shape, dtype, device, contiguity, and
  forward-only checks at the dispatcher boundary.
- Empty token input returns an empty contiguous tensor with the input shape;
  empty experts are allowed as long as the global expert count is positive.
- CUDA atomic pack/combine order is nondeterministic. Deterministic mode uses
  the reference path under `auto` and rejects explicit CUDA.

## Test Strategy

Implementation follows red-green-refactor. Each behavior is introduced by a
test that first fails for the missing facade/schema/backend.

### Reference and facade tests

- FP64 staged facade output and gradients match `deepseek_moe_reference`.
- Sigmoid/softmax, bias/no-bias, grouped selection, route scale, exact ties,
  hot experts, empty experts, rank-2/rank-3 inputs, and zero tokens.
- Invalid backend, shapes, dtypes, devices, and routing configuration.

### Backend policy tests

- Explicit CUDA rejects CPU, non-FP32, softmax, non-contiguous,
  `requires_grad`, deterministic mode, and a missing whole-layer kernel.
- `auto` sends each of those requests through the complete reference path.
- Reference on a CUDA tensor never redispatches native stages.
- A failure after CUDA selection propagates and does not trigger fallback.

### Dispatcher tests

- The schema exists without a native extension.
- FakeTensor preserves shape, dtype, and device.
- No AutogradCUDA or Composite implementation is registered for the raw op.
- Detached CUDA inputs pass `torch.library.opcheck` and
  `torch.compile(fullgraph=True)`.

### CUDA correctness tests

- Zero tokens and scalar dimensions.
- Arbitrary non-tile-aligned `D` and `H` tails.
- Grouped Top-K with optional selection bias and exact ties.
- All routes selecting one expert and multiple inactive experts.
- Output shape, dtype, device, contiguity, finiteness, and current-stream use.
- Explicit CUDA and eligible auto outputs match an independent reference within
  a recorded FP32 tolerance; tests do not assert a speedup.

## Benchmark and Documentation Evidence

A focused whole-layer benchmark records token/expert/hidden dimensions, Top-K,
group configuration, route skew, dtype, hardware, latency samples, and numerical
error against the independent reference. Profiler output distinguishes the
whole-layer dispatcher event from its private staged kernels and records launch
count plus major intermediate sizes.

README and the MoE/custom-operator chapters must call the first CUDA backend a
single-device, staged, correctness-first implementation. They must explicitly
state that persistent scheduling, full fusion, one-sided communication, and
multi-GPU evidence remain future work.

## Delivery Stages

1. Add and verify the facade/reference policy.
2. Add the raw schema, FakeTensor contract, and forward-only dispatcher policy.
3. Add the staged CUDA whole-layer implementation by reusing host launchers.
4. Add CUDA correctness/current-stream tests and the focused benchmark.
5. Record single-GPU evidence through the existing remote-build/local-GPU
   validation workflow.
6. Only after profiler evidence, choose which private intermediates or launches
   to eliminate. The public API and raw schema remain unchanged.

## Repository Rename Boundary

Repository branding is an independent decision. A GitHub repository rename must
not be coupled to the native operator implementation or to the high-coupling
`ds_flash_mla_moe` import/dispatcher namespace. If approved separately, the
repository URL and README title may change while the distribution, import name,
extension name, and historical validation labels remain stable.
