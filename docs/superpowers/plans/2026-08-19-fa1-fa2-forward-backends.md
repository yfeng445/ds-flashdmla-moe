# FA1/FA2 Forward Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strict Python-selectable `reference`, `blockwise`, `cuda_rowwise`, `fa1`, and `fa2` attention forward backends, with separate formal FA1/FA2 CUDA kernels that visibly implement the paper-level algorithm and work-partition differences.

**Architecture:** Keep `flash_attention_forward` as the semantic facade and register FA1 and FA2 as separate `torch.library` operators. Both formal kernels use FP16 storage, FP32 dot/softmax/output accumulation, identical `4 x 16` query/key tiles, and the same causal-coordinate helpers; FA1 owns a whole batch-head and repeatedly stores normalized output state, while FA2 owns one query tile and retains an unnormalized accumulator on chip. `auto` preserves current behavior by preferring `cuda_rowwise` and otherwise using the blockwise PyTorch implementation; it never silently selects the teaching FA variants.

**Tech Stack:** Python 3.10+, PyTorch 2.4+ custom operators, C++17, CUDA C++17, pytest, ruff, setuptools `CUDAExtension`.

**Spec:** `docs/superpowers/specs/2026-08-19-explicit-cuda-backends-fa1-fa2-design.md`

## Global Constraints

- The public tensor layout is contiguous rank-four `BHSD`.
- FA1 and FA2 accept CUDA FP16 inputs and return CUDA FP16 output only.
- FA1 and FA2 use FP32 dot-product, online-softmax, and output accumulation.
- The default scale is exactly `1 / sqrt(head_dim)`.
- Causal attention is right aligned: key index `k` is visible when `k <= q + S_k - S_q`.
- Formal FA1/FA2 support `1 <= head_dim <= 128` and `0 <= value_dim <= 128`; key sequence length must be positive.
- Empty batch, head, query-length, or value-dimension outputs return without launching a kernel.
- Explicit backend selection never falls back; only `auto` may fall back.
- `auto` considers `cuda_rowwise` only, then falls back to `blockwise`; it does not select FA1 or FA2.
- `backend="cuda"` is a temporary warning-producing alias for `cuda_rowwise`.
- FA1 and FA2 reject any input tensor with `requires_grad=True`, even inside `torch.no_grad()`.
- Existing row-wise forward/backward behavior remains intact.
- Existing files under `csrc/experimental/attention/` are not edited, built, or registered.
- Do not add Tensor Core/WMMA code to either formal variant in this plan.
- Do not add FA1/FA2 backward, public LSE returns, explicit mask support, FA3, or FA4.
- GEMM, Experts, MLA, expert-parallel transport, and NVSHMEM backend migrations are follow-up sub-projects and are not implemented by this attention plan.

## File Map

| File | Responsibility after this plan |
|---|---|
| `src/ds_flash_mla_moe/ops.py` | Attention backend types, strict capability checks, schemas/fakes, facade dispatch, compatibility alias |
| `src/ds_flash_mla_moe/__init__.py` | Public capability-query export |
| `csrc/ops.cpp` | Native schemas for the two new operators |
| `csrc/attention/attention_common.cuh` | Shared tile constants, right-aligned causal helper, warp reductions, and common host validation |
| `csrc/attention/fa1_forward_cuda.cu` | Formal FA1 batch-head-owned forward and operator registration |
| `csrc/attention/fa2_forward_cuda.cu` | Formal FA2 query-tile-owned forward and operator registration |
| `setup.py` | Compile and presence-check the two formal sources |
| `tests/test_attention_backends.py` | Strict dispatch, forward-only constraints, FA1/FA2 correctness, tails, streams, and stability |
| `tests/test_ops.py` | Rename row-wise callers and retain one compatibility-alias test |
| `src/ds_flash_mla_moe/benchmarking.py` | Benchmark backend names and paired FA1/FA2 report helper |
| `benchmarks/attention.py` | CLI backend choices and paired comparison switch |
| `tests/test_benchmarking.py` | Benchmark validation and paired-report tests |
| `docs/chapters/02-flash-attention.md` | Runnable backend matrix and four-level FA1/FA2 comparison |

---

### Task 1: Strict Attention Backend Facade

**Files:**
- Modify: `src/ds_flash_mla_moe/ops.py:23,76-81,144-172,176-200,1101-1152,1631-1641,2220-2310`
- Modify: `src/ds_flash_mla_moe/__init__.py:48-65,71-112`
- Create: `tests/test_attention_backends.py`
- Modify: `tests/test_ops.py:1-110,250-435`

**Interfaces:**
- Consumes: existing `scaled_dot_product_attention_reference`, `blockwise_attention`, `torch.ops.ds_flash_mla_moe.attention_forward`, and `_operator_has_cuda_kernel(name: str) -> bool`.
- Produces: `AttentionBackend`, `NativeAttentionBackend`, `cuda_attention_backend_available()`, two operator schemas/fakes, and strict `flash_attention_forward` dispatch used by all later tasks.

- [ ] **Step 1: Write failing backend-dispatch tests**

Create `tests/test_attention_backends.py` with CPU-runnable tests that lock the semantic branches and strict errors:

```python
from __future__ import annotations

import pytest
import torch

import ds_flash_mla_moe.ops as attention_ops
from ds_flash_mla_moe import flash_attention_forward


def _cpu_inputs(*, requires_grad: bool = False):
    q = torch.randn(1, 2, 3, 5, requires_grad=requires_grad)
    k = torch.randn(1, 2, 7, 5, requires_grad=requires_grad)
    v = torch.randn(1, 2, 7, 4, requires_grad=requires_grad)
    return q, k, v


def test_reference_and_blockwise_are_distinct_explicit_branches(monkeypatch) -> None:
    q, k, v = _cpu_inputs()
    materialized = torch.full((1, 2, 3, 4), 1.0)
    blocked = torch.full((1, 2, 3, 4), 2.0)
    monkeypatch.setattr(
        attention_ops,
        "scaled_dot_product_attention_reference",
        lambda *args, **kwargs: materialized,
    )
    monkeypatch.setattr(
        attention_ops,
        "blockwise_attention",
        lambda *args, **kwargs: blocked,
    )

    assert flash_attention_forward(q, k, v, backend="reference") is materialized
    assert flash_attention_forward(q, k, v, backend="blockwise") is blocked


@pytest.mark.parametrize("backend", ["fa1", "fa2"])
def test_formal_fa_backends_reject_autograd_before_device_dispatch(backend: str) -> None:
    q, k, v = _cpu_inputs(requires_grad=True)
    with pytest.raises(RuntimeError, match="forward-only"):
        flash_attention_forward(q, k, v, backend=backend)  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["fa1", "fa2"])
def test_formal_fa_backends_never_fall_back_on_cpu(backend: str) -> None:
    q, k, v = _cpu_inputs()
    with pytest.raises(RuntimeError, match=rf"{backend} attention is unavailable"):
        flash_attention_forward(q, k, v, backend=backend)  # type: ignore[arg-type]


def test_cuda_alias_warns_and_uses_rowwise_contract() -> None:
    q, k, v = _cpu_inputs()
    with pytest.warns(FutureWarning, match="cuda_rowwise"):
        with pytest.raises(RuntimeError, match="cuda_rowwise attention is unavailable"):
            flash_attention_forward(q, k, v, backend="cuda")


def test_formal_operator_schemas_exist_without_native_extension() -> None:
    assert attention_ops._operator_is_defined("attention_fa1_forward")
    assert attention_ops._operator_is_defined("attention_fa2_forward")
```

Also change existing explicit native calls in `tests/test_ops.py` from
`backend="cuda"` to `backend="cuda_rowwise"`. Keep exactly one alias test in
the new file so the suite does not emit repeated deprecation warnings.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```powershell
python -m pytest tests/test_attention_backends.py tests/test_ops.py -q
```

Expected: new tests fail because the backend literal, schemas, dispatch paths,
and capability query do not exist; existing row-wise tests reject
`cuda_rowwise`.

- [ ] **Step 3: Add backend types, schemas, and fake implementations**

In `ops.py`, import `warnings` and
`scaled_dot_product_attention_reference`, then define:

```python
AttentionBackend = Literal[
    "auto",
    "cuda",
    "cuda_rowwise",
    "reference",
    "blockwise",
    "fa1",
    "fa2",
]
NativeAttentionBackend = Literal["cuda_rowwise", "fa1", "fa2"]

_FA1_FORWARD_SCHEMA = (
    "attention_fa1_forward(Tensor q, Tensor k, Tensor v, bool causal, float scale) -> Tensor"
)
_FA2_FORWARD_SCHEMA = (
    "attention_fa2_forward(Tensor q, Tensor k, Tensor v, bool causal, float scale) -> Tensor"
)
```

Add both schemas to `_SCHEMAS`. Define `_fake_formal_attention_forward` by
reusing the shape checks from `_fake_attention_forward` and additionally
checking rank four, FP16 dtype, equal input dtypes, and final dimensions no
larger than 128. Register it for both formal operators. Do not register
CompositeExplicitAutograd or autograd kernels for them. Direct CPU execution
must therefore remain unavailable even though FakeTensor shape propagation
works.

- [ ] **Step 4: Implement strict capability and dispatch logic**

Replace `_cuda_ineligibility_reason` with an operator-aware helper:

```python
_ATTENTION_OPERATOR = {
    "cuda_rowwise": "attention_forward",
    "fa1": "attention_fa1_forward",
    "fa2": "attention_fa2_forward",
}


def _attention_backend_ineligibility_reason(
    backend: NativeAttentionBackend,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    attn_mask: Tensor | None,
) -> str | None:
    if backend in {"fa1", "fa2"} and any(t.requires_grad for t in (q, k, v)):
        return f"{backend} is forward-only and does not accept requires_grad tensors"
    if q.ndim != 4:
        return "the CUDA kernel requires [batch, heads, sequence, dimension] tensors"
    supported = (
        {torch.float16}
        if backend in {"fa1", "fa2"}
        else {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }
    )
    if q.dtype not in supported:
        rendered = "float16" if backend in {"fa1", "fa2"} else "float16, bfloat16, or float32"
        return f"{backend} supports {rendered}"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return "q, k, and v must have the same dtype"
    if not all(t.is_contiguous() for t in (q, k, v)):
        return "the CUDA kernel requires contiguous tensors"
    if attn_mask is not None:
        return "the CUDA kernel does not support an explicit attention mask"
    if backend in {"fa1", "fa2"} and (q.shape[-1] > 128 or v.shape[-1] > 128):
        return "formal FA1/FA2 currently require head_dim <= 128 and value_dim <= 128"
    if not _NATIVE_EXTENSION_LOADED:
        return "the native extension is not installed"
    if any(t.device.type != "cuda" for t in (q, k, v)):
        return "q, k, and v must be CUDA tensors"
    operator = _ATTENTION_OPERATOR[backend]
    if not _operator_has_cuda_kernel(operator):
        return f"the loaded native extension does not register {operator}"
    return None
```

Define the public capability query and preserve the old convenience flag:

```python
def cuda_attention_backend_available(
    backend: NativeAttentionBackend = "cuda_rowwise",
) -> bool:
    if backend not in _ATTENTION_OPERATOR:
        raise ValueError("native attention backend must be cuda_rowwise, fa1, or fa2")
    return (
        _NATIVE_EXTENSION_LOADED
        and torch.cuda.is_available()
        and _operator_has_cuda_kernel(_ATTENTION_OPERATOR[backend])
    )


def cuda_kernel_available() -> bool:
    return cuda_attention_backend_available("cuda_rowwise")
```

Export `cuda_attention_backend_available` from `__init__.py`.

Implement facade selection in this exact order:

```python
valid = {"auto", "cuda", "cuda_rowwise", "reference", "blockwise", "fa1", "fa2"}
if backend not in valid:
    raise ValueError("backend must be auto, cuda_rowwise, reference, blockwise, fa1, or fa2")
if backend == "cuda":
    warnings.warn(
        "backend='cuda' is deprecated; use backend='cuda_rowwise'",
        FutureWarning,
        stacklevel=2,
    )
    backend = "cuda_rowwise"

if backend == "reference":
    return scaled_dot_product_attention_reference(
        q,
        k,
        v,
        causal=causal,
        scale=effective_scale,
        attn_mask=attn_mask,
    )
if backend == "blockwise":
    return blockwise_attention(
        q,
        k,
        v,
        causal=causal,
        scale=effective_scale,
        attn_mask=attn_mask,
        block_size=reference_block_size,
    )

selected = "cuda_rowwise" if backend == "auto" else backend
reason = _attention_backend_ineligibility_reason(selected, q, k, v, attn_mask=attn_mask)
if reason is None:
    operator = getattr(torch.ops.ds_flash_mla_moe, _ATTENTION_OPERATOR[selected]).default
    return operator(q, k, v, causal, effective_scale)
if backend != "auto":
    raise RuntimeError(f"{selected} attention is unavailable: {reason}")
return blockwise_attention(
    q,
    k,
    v,
    causal=causal,
    scale=effective_scale,
    attn_mask=attn_mask,
    block_size=reference_block_size,
)
```

Do not change `_attention_setup_context` or the autograd registration for the
existing row-wise `attention_forward` operator.

- [ ] **Step 5: Run CPU tests and lint**

Run:

```powershell
python -m pytest tests/test_attention_backends.py tests/test_ops.py -q
python -m ruff check src/ds_flash_mla_moe/ops.py src/ds_flash_mla_moe/__init__.py tests/test_attention_backends.py tests/test_ops.py
```

Expected: PASS. The formal backends are visible but explicitly unavailable
until their native CUDA implementations are built.

- [ ] **Step 6: Commit the strict facade**

```powershell
git add src/ds_flash_mla_moe/ops.py src/ds_flash_mla_moe/__init__.py tests/test_attention_backends.py tests/test_ops.py
git commit -m "Add strict attention backend dispatch"
```

---

### Task 2: Formal FA1 Forward Operator

**Files:**
- Create: `csrc/attention/attention_common.cuh`
- Create: `csrc/attention/fa1_forward_cuda.cu`
- Modify: `csrc/ops.cpp:14-21`
- Modify: `setup.py:20-60`
- Modify: `tests/test_attention_backends.py`

**Interfaces:**
- Consumes: `attention_fa1_forward(q, k, v, causal, scale) -> Tensor` schema and `_ATTENTION_OPERATOR["fa1"]` from Task 1.
- Produces: a CUDA implementation registered as `ds_flash_mla_moe::attention_fa1_forward` and common constants/helpers reused unchanged by FA2.

- [ ] **Step 1: Add failing FA1 CUDA tests**

Append the following structure to `tests/test_attention_backends.py`:

```python
from ds_flash_mla_moe import blockwise_attention, cuda_attention_backend_available


def _fa_tolerances() -> tuple[float, float]:
    return 1e-2, 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    ("query_length", "key_length", "head_dim", "value_dim"),
    [(1, 7, 8, 5), (4, 4, 32, 32), (7, 11, 65, 33), (9, 17, 128, 127)],
)
def test_fa1_forward_matches_reference(
    causal: bool,
    query_length: int,
    key_length: int,
    head_dim: int,
    value_dim: int,
) -> None:
    torch.manual_seed(101)
    q = torch.randn(2, 3, query_length, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 3, key_length, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 3, key_length, value_dim, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        actual = flash_attention_forward(q, k, v, causal=causal, backend="fa1")
        expected = blockwise_attention(q, k, v, causal=causal, block_size=3)
    rtol, atol = _fa_tolerances()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert actual.shape == (2, 3, query_length, value_dim)
    assert actual.dtype == torch.float16
    assert actual.is_contiguous()
```

Add a current-stream test that fills tensors on a non-default stream, invokes
`backend="fa1"`, records the output on that stream, synchronizes it, and compares
against `blockwise_attention`.

- [ ] **Step 2: Run the CUDA test and confirm native dispatch fails**

Run on the current single-GPU environment:

```powershell
python -m pytest tests/test_attention_backends.py -k "fa1_forward" -q
```

Expected: FAIL with `fa1 attention is unavailable` or an absent CUDA kernel.
It must not pass through the blockwise fallback.

- [ ] **Step 3: Create the common header**

Create `attention_common.cuh` with a unique include guard and these constants:

```cpp
namespace ds_flash_mla_moe::attention {

constexpr int kWarpSize = 32;
constexpr int kWarps = 4;
constexpr int kThreads = kWarpSize * kWarps;
constexpr int kQueryTile = 4;
constexpr int kKeyTile = 16;
constexpr int kMaxHeadDim = 128;
constexpr int kMaxValueDim = 128;

__device__ __forceinline__ bool key_is_visible(
    int64_t query_position,
    int64_t key_position,
    int64_t query_length,
    int64_t key_length,
    bool causal) {
  return !causal || key_position <= query_position + key_length - query_length;
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

}  // namespace ds_flash_mla_moe::attention
```

Also add one inline host validator shared by FA1/FA2. It must repeat all
safety-critical checks from the facade: same CUDA device, rank four, FP16 same
dtype, contiguous storage, matching batch/head dimensions, matching Q/K head
dimension, matching K/V sequence length, positive head dimension and key
length, `head_dim <= 128`, `value_dim <= 128`, right-aligned causal
`query_length <= key_length`, and finite scale.

Do not place a query/KV loop or output recurrence in the header.

- [ ] **Step 4: Implement the FA1 kernel with visible FA1 ownership**

In `fa1_forward_cuda.cu`, include the FA1 paper URL in the file comment and use
this top-level signature:

```cpp
__global__ void fa1_forward_kernel(
    const at::Half* q,
    const at::Half* k,
    const at::Half* v,
    float* normalized_output,
    float* row_max,
    float* row_sum,
    at::Half* output,
    int64_t query_length,
    int64_t key_length,
    int64_t head_dim,
    int64_t value_dim,
    float scale,
    bool causal);
```

Use one flattened block per `(batch, head)`. Keep the two paper-defining loops
directly in this function:

```cpp
for (int64_t key_block = 0; key_block < key_length; key_block += kKeyTile) {
  // Cooperative FP32 K/V load.
  for (int64_t query_block = 0; query_block < query_length; query_block += kQueryTile) {
    // Skip this pair when the whole K/V tile lies above the causal boundary.
    // Cooperative FP32 Q and normalized O/m/l reload.
    // Each warp loops key_in_tile = warp_id; key_in_tile < valid_keys;
    // key_in_tile += kWarps.
    // Merge warp-local m/l/numerators through shared memory.
    // Write normalized O and m/l to global FP32 workspaces.
  }
}
// Cast the final normalized FP32 workspace to FP16 output once.
```

Within every warp, lanes reduce each FP32 dot product with `warp_sum`. Each warp
stores its local maximum, local exponential sum, and local unnormalized value
numerator in shared memory. After `__syncthreads()`, merge the four warp states
for each query row with:

```text
next_m = max(old_m, local_m[0], local_m[1], local_m[2], local_m[3])
next_l = old_l * exp(old_m - next_m)
       + sum_w local_l[w] * exp(local_m[w] - next_m)
next_O[d] = (
    old_O[d] * old_l * exp(old_m - next_m)
    + sum_w local_numerator[w, d] * exp(local_m[w] - next_m)
) / next_l
```

This normalized recurrence and the global workspace reload/write must remain
plainly visible. Initialize `normalized_output` and `row_sum` to zero and
`row_max` to `-inf` in the C++ wrapper with ATen FP32 tensors.

The wrapper must:

- return an empty correctly shaped FP16 tensor for zero batch/head/query/value;
- guard the input device with `c10::cuda::CUDAGuard`;
- check the flattened batch-head grid against `maxGridSize[0]`;
- compute dynamic shared-memory bytes for Q, K, V, four-warp partial output,
  and merge state, then compare with `sharedMemPerBlock`;
- launch on `at::cuda::getCurrentCUDAStream()`;
- call `C10_CUDA_KERNEL_LAUNCH_CHECK()`;
- register only `attention_fa1_forward` under the CUDA dispatch key.

- [ ] **Step 5: Add native schema and build integration**

In `csrc/ops.cpp`, add exactly:

```cpp
m.def(
    "attention_fa1_forward(Tensor q, Tensor k, Tensor v, bool causal, float scale) -> Tensor");
```

Add `csrc/attention/fa1_forward_cuda.cu` to both the `CUDAExtension.sources`
list and `assert_native_sources_present()` tuple in `setup.py`. The header is an
include dependency and does not belong in the source list.

- [ ] **Step 6: Rebuild and run FA1 tests**

Run:

```powershell
$env:DS_FLASH_BUILD_CUDA='1'
python -m pip install -e '.[test,cuda-build]' --no-build-isolation
python -m pytest tests/test_attention_backends.py -k "fa1 or backend" -q
Remove-Item Env:DS_FLASH_BUILD_CUDA
```

Expected: all FA1 correctness, strict dispatch, dtype, layout, and current-
stream tests pass on the single GPU.

- [ ] **Step 7: Commit FA1**

```powershell
git add csrc/attention/attention_common.cuh csrc/attention/fa1_forward_cuda.cu csrc/ops.cpp setup.py tests/test_attention_backends.py
git commit -m "Add formal FA1 forward backend"
```

---

### Task 3: Formal FA2 Forward Operator

**Files:**
- Create: `csrc/attention/fa2_forward_cuda.cu`
- Modify: `csrc/ops.cpp:14-25`
- Modify: `setup.py:20-65`
- Modify: `tests/test_attention_backends.py`

**Interfaces:**
- Consumes: the Task 1 `attention_fa2_forward` schema and Task 2 constants/helpers in `attention_common.cuh`.
- Produces: a separate CUDA implementation registered as `ds_flash_mla_moe::attention_fa2_forward` with no FA1 workspace or implementation sharing.

- [ ] **Step 1: Add failing FA2 correctness and ownership tests**

Add `test_fa2_forward_matches_reference` using the exact same seed, parameter
matrix, tolerance, expected reference, and output assertions as FA1, changing
only `backend="fa2"`. Add a current-stream test for FA2.

Add a direct cross-backend test so both kernels see byte-identical tensors:

```python
@pytest.mark.skipif(
    not (cuda_attention_backend_available("fa1") and cuda_attention_backend_available("fa2")),
    reason="requires built FA1 and FA2 CUDA kernels",
)
@pytest.mark.cuda
def test_fa1_and_fa2_match_the_same_reference_on_identical_inputs() -> None:
    torch.manual_seed(103)
    q = torch.randn(1, 2, 9, 65, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 17, 65, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 17, 33, device="cuda", dtype=torch.float16)
    expected = blockwise_attention(q, k, v, causal=True, block_size=5)
    rtol, atol = _fa_tolerances()
    for backend in ("fa1", "fa2"):
        actual = flash_attention_forward(q, k, v, causal=True, backend=backend)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
```

- [ ] **Step 2: Run the FA2 tests and confirm explicit failure**

```powershell
python -m pytest tests/test_attention_backends.py -k "fa2" -q
```

Expected: FAIL because the loaded extension has the schema but no FA2 CUDA
kernel. It must not execute FA1 or row-wise attention.

- [ ] **Step 3: Implement the FA2 query-tile-owned kernel**

In `fa2_forward_cuda.cu`, include the FA2 paper URL in the file comment and use:

```cpp
__global__ void fa2_forward_kernel(
    const at::Half* q,
    const at::Half* k,
    const at::Half* v,
    at::Half* output,
    int64_t query_length,
    int64_t key_length,
    int64_t head_dim,
    int64_t value_dim,
    int64_t query_blocks,
    float scale,
    bool causal);
```

Flatten `batch * heads * query_blocks` into `blockIdx.x`. Decode a unique
`query_block`, and make each of the four warps own one query row. Keep the main
loop and recurrence visible:

```cpp
// Load this CTA's four Q rows once and initialize FP32 m/l/O on chip.
for (int64_t key_block = 0; key_block < key_length; key_block += kKeyTile) {
  // Skip the tile when it is fully masked for all valid Q rows in this CTA.
  // Cooperatively load FP32 K/V tile.
  // warp_id exclusively updates query row query_block + warp_id.
  for (int key_in_tile = 0; key_in_tile < valid_keys; ++key_in_tile) {
    // FP32 dot reduction, right-aligned causal check, unnormalized recurrence.
  }
}
// Divide each owned O row by l once and store FP16 output once.
```

For every visible score, update the unnormalized accumulator exactly as:

```text
next_m = max(m, score)
alpha = exp(m - next_m), with alpha = 0 when m is -inf
beta = exp(score - next_m)
O[d] = alpha * O[d] + beta * V[d]
l = alpha * l + beta
m = next_m
```

Q, O, m, and l remain in registers or shared memory for the full K/V traversal.
No global FP32 O/m/l workspace is allocated. No warp writes an output row owned
by another warp, and there is no cross-warp partial-output merge.

The host wrapper repeats Task 2's validation, empty-output behavior, device
guard, grid-limit check, dynamic shared-memory check, current-stream launch, and
launch-error check. Register only `attention_fa2_forward`.

- [ ] **Step 4: Add FA2 native schema and build source**

Add to `csrc/ops.cpp`:

```cpp
m.def(
    "attention_fa2_forward(Tensor q, Tensor k, Tensor v, bool causal, float scale) -> Tensor");
```

Add `csrc/attention/fa2_forward_cuda.cu` to both native source lists in
`setup.py`.

- [ ] **Step 5: Rebuild and run both formal kernels**

```powershell
$env:DS_FLASH_BUILD_CUDA='1'
python -m pip install -e '.[test,cuda-build]' --no-build-isolation
python -m pytest tests/test_attention_backends.py -q
python -m pytest tests/test_ops.py -q
Remove-Item Env:DS_FLASH_BUILD_CUDA
```

Expected: PASS, including the identical-input comparison and preservation of
existing row-wise forward/backward tests.

- [ ] **Step 6: Commit FA2**

```powershell
git add csrc/attention/fa2_forward_cuda.cu csrc/ops.cpp setup.py tests/test_attention_backends.py
git commit -m "Add formal FA2 forward backend"
```

---

### Task 4: Formal Backend Edge and Stability Matrix

**Files:**
- Modify: `tests/test_attention_backends.py`
- Modify if a test exposes a defect: `src/ds_flash_mla_moe/ops.py`, `csrc/attention/attention_common.cuh`, `csrc/attention/fa1_forward_cuda.cu`, `csrc/attention/fa2_forward_cuda.cu`

**Interfaces:**
- Consumes: completed `fa1` and `fa2` explicit backends.
- Produces: one shared acceptance matrix proving strict capability, empty/tail handling, right-aligned causality, numerical stability, stream correctness, and lack of autograd registration.

- [ ] **Step 1: Add the shared explicit-error matrix**

Parameterize both formal backends and assert each property separately:

```python
@pytest.mark.parametrize("backend", ["fa1", "fa2"])
def test_formal_backends_reject_non_fp16_before_dispatch(backend: str) -> None:
    q = torch.randn(1, 1, 3, 8)
    with pytest.raises(RuntimeError, match="supports float16"):
        flash_attention_forward(q, q, q, backend=backend)  # type: ignore[arg-type]
```

For CUDA tensors, add cases for BF16/FP32, mixed dtype, non-contiguous storage,
`head_dim=129`, `value_dim=129`, explicit boolean mask, and `requires_grad`.
Check that the message includes the selected backend and the rejected property.
Do not accept output from another backend as evidence of correct rejection.

- [ ] **Step 2: Add empty and tail behavior tests**

For each formal backend on CUDA, test empty batch, heads, query length, and value
dimension. Assert the exact expected shape/dtype/device and zero elements where
appropriate. Separately assert key length zero raises `key sequence length must
be positive` before dispatch.

Retain non-multiple dimensions from Tasks 2/3 so `S_q`, `S_k`, `D`, and `D_v`
all exercise tile tails.

- [ ] **Step 3: Add right-aligned causal and stable-softmax stress tests**

Use `S_q=1, S_k=17` with causal attention and verify the single query can see
the full history. Use `S_q=7, S_k=11` to cover a partial causal boundary tile.

For stability, multiply independently sampled FP16 Q and K by `20` and compare
both outputs to `blockwise_attention`; assert every output is finite before
`assert_close`. Use the shared FP16 tolerance and a fixed seed.

- [ ] **Step 4: Run the complete acceptance matrix**

```powershell
python -m pytest tests/test_attention_backends.py tests/test_ops.py -q
python -m ruff check src tests
```

Expected: PASS. If a case fails, make the smallest change in the files listed
above, rerun the single failing node, then rerun both full files.

- [ ] **Step 5: Commit the acceptance matrix**

```powershell
git add tests/test_attention_backends.py src/ds_flash_mla_moe/ops.py csrc/attention/attention_common.cuh csrc/attention/fa1_forward_cuda.cu csrc/attention/fa2_forward_cuda.cu
git commit -m "Test formal attention backend contracts"
```

---

### Task 5: Paired FA1/FA2 Benchmark Reporting

**Files:**
- Modify: `src/ds_flash_mla_moe/benchmarking.py:28-78,337-480`
- Modify: `benchmarks/attention.py:20-70`
- Modify: `tests/test_benchmarking.py:1-220`

**Interfaces:**
- Consumes: `flash_attention_forward(q, k, v, backend="fa1" | "fa2")` and existing `benchmark_attention` reports.
- Produces: expanded benchmark backend names and `benchmark_attention_backends(config, backends) -> dict[str, Any]` for identical-seed paired reports.

- [ ] **Step 1: Write failing benchmark configuration tests**

Extend `AttentionBenchmarkBackend` expectations with `cuda_rowwise`,
`blockwise`, `fa1`, and `fa2`. Add:

```python
def test_formal_fa_benchmark_requires_cuda_float16() -> None:
    with pytest.raises(ValueError, match="CUDA float16"):
        AttentionBenchmarkConfig(backend="fa1", device="cpu", dtype="float16").validate()
    with pytest.raises(ValueError, match="CUDA float16"):
        AttentionBenchmarkConfig(backend="fa2", device="cuda", dtype="float32").validate()


def test_paired_benchmark_uses_the_same_configuration(monkeypatch) -> None:
    seen = []

    def fake_benchmark(config):
        seen.append(config)
        return {"configuration": {"backend": config.backend}, "raw_samples_ms": [1.0]}

    monkeypatch.setattr(benchmarking, "benchmark_attention", fake_benchmark)
    base = AttentionBenchmarkConfig(backend="fa1", device="cuda", dtype="float16", seed=17)
    report = benchmarking.benchmark_attention_backends(base, ("fa1", "fa2"))

    assert [config.backend for config in seen] == ["fa1", "fa2"]
    assert all(config.seed == 17 for config in seen)
    assert report["comparison_backends"] == ["fa1", "fa2"]
```

- [ ] **Step 2: Run tests and confirm the new names/helper are absent**

```powershell
python -m pytest tests/test_benchmarking.py -q
```

Expected: FAIL on validation and missing `benchmark_attention_backends`.

- [ ] **Step 3: Expand benchmark validation and paired reports**

Define:

```python
AttentionBenchmarkBackend = Literal[
    "auto",
    "cuda",
    "cuda_rowwise",
    "reference",
    "blockwise",
    "fa1",
    "fa2",
    "sdpa",
    "flash-attn-4",
]
```

Require `device.type == "cuda"` and `dtype == "float16"` for FA1/FA2. Retain
the `cuda` compatibility choice only while the facade alias exists. Add notes to
single-backend reports that FA1/FA2 are repository teaching kernels using FP32
accumulation and no Tensor Cores.

Implement paired reports with `dataclasses.replace`:

```python
def benchmark_attention_backends(
    config: AttentionBenchmarkConfig,
    backends: tuple[AttentionBenchmarkBackend, ...],
) -> dict[str, Any]:
    if not backends or len(set(backends)) != len(backends):
        raise ValueError("comparison backends must be non-empty and unique")
    reports = {
        backend: benchmark_attention(replace(config, backend=backend)) for backend in backends
    }
    return {
        "schema_version": 1,
        "comparison_backends": list(backends),
        "shared_seed": config.seed,
        "reports": reports,
    }
```

Because every child config retains the same seed and dimensions,
`benchmark_attention` generates identical Q/K/V values for the two runs.

- [ ] **Step 4: Extend the CLI**

Add the new backend choices and `--compare-fa1-fa2`. When the flag is present,
write `benchmark_attention_backends(config, ("fa1", "fa2"))`; otherwise retain
the current one-report behavior. Reject combining the flag with a backend other
than the parser default to avoid ambiguous output.

- [ ] **Step 5: Run benchmark unit tests and a single-GPU smoke comparison**

```powershell
python -m pytest tests/test_benchmarking.py -q
python benchmarks/attention.py --device cuda --dtype float16 --batch 1 --heads 2 --query-length 128 --key-length 128 --head-dim 64 --value-dim 64 --warmup 2 --iterations 5 --compare-fa1-fa2
```

Expected: unit tests pass; the JSON comparison contains both raw-sample arrays,
verification results, environment metadata, and no asserted speed ordering.

- [ ] **Step 6: Commit benchmark support**

```powershell
git add src/ds_flash_mla_moe/benchmarking.py benchmarks/attention.py tests/test_benchmarking.py
git commit -m "Benchmark FA1 and FA2 backends"
```

---

### Task 6: Documentation and Full Verification

**Files:**
- Modify: `docs/chapters/02-flash-attention.md`
- Verify only: `csrc/experimental/attention/*`
- Verify only: all files changed in Tasks 1-5

**Interfaces:**
- Consumes: measured backend behavior, capability limits, and source structure from Tasks 1-5.
- Produces: reader-facing FA1/FA2 explanation and final evidence that the attention sub-project is independently complete.

- [ ] **Step 1: Update the backend matrix and runnable examples**

In Chapter 2, replace the single `auto/cuda/reference` description with:

| Backend | Implementation | Dtype/device | Gradient behavior |
|---|---|---|---|
| `reference` | materialized PyTorch specification | floating CPU/CUDA | differentiable |
| `blockwise` | online-softmax PyTorch specification | floating CPU/CUDA | differentiable |
| `cuda_rowwise` | one query row per CTA | FP16/BF16/FP32 CUDA | existing native/reference backward policy |
| `fa1` | formal KV-outer teaching kernel | FP16 CUDA, `D,D_v <= 128` | forward-only |
| `fa2` | formal Q-tile-owned teaching kernel | FP16 CUDA, `D,D_v <= 128` | forward-only |

Add runnable examples calling the same `flash_attention_forward` arguments with
`backend="fa1"` and `backend="fa2"`. State that `auto` does not select either
formal variant.

- [ ] **Step 2: Document the four implementation-level differences**

Add separate subsections for:

1. loop order: FA1 K/V outer then Q, FA2 query-tile ownership then K/V;
2. grid: FA1 batch-head, FA2 batch-head-query-tile;
3. warp partition: FA1 splits keys and merges partial output, FA2 splits Q rows;
4. recurrence: FA1 stores normalized O/m/l per KV tile, FA2 keeps unnormalized O
   on chip and divides once.

Link directly to the FA1 and FA2 papers and to the three production source
files. Explain that causal block skipping is common to both and is not the
version-defining difference. Keep the existing backward material clearly
scoped to `cuda_rowwise` and experiments, not the new formal backends.

- [ ] **Step 3: Verify experimental sources were untouched and unregistered**

Run:

```powershell
git status --short -- csrc/experimental/attention
rg -n "flash_attention_v[12]|experimental/attention" setup.py csrc/ops.cpp src/ds_flash_mla_moe/ops.py
if ($LASTEXITCODE -eq 1) { $global:LASTEXITCODE = 0 }
```

Expected: the first command has no output; the second finds no build or
registration reference.

- [ ] **Step 4: Run static and CPU validation**

```powershell
python -m ruff check .
python -m pytest -m "not cuda" -q
git diff --check
```

Expected: PASS with no whitespace errors.

- [ ] **Step 5: Rebuild from the final source set and run CUDA validation**

```powershell
$env:DS_FLASH_BUILD_CUDA='1'
python -m pip install -e '.[test,cuda-build]' --no-build-isolation
python -m pytest -m cuda -q
Remove-Item Env:DS_FLASH_BUILD_CUDA
```

Expected: PASS on the available single GPU. Record any device-specific skipped
tests separately; do not convert missing multi-GPU coverage into a success
claim.

- [ ] **Step 6: Run final paired smoke benchmark**

```powershell
python benchmarks/attention.py --device cuda --dtype float16 --batch 1 --heads 4 --query-length 256 --key-length 256 --head-dim 64 --value-dim 64 --causal --warmup 5 --iterations 20 --compare-fa1-fa2
```

Expected: both backends verify against the independent reference and report raw
latency samples. Preserve the observed ordering; do not edit results to imply
FA2 always wins.

- [ ] **Step 7: Commit documentation**

```powershell
git add docs/chapters/02-flash-attention.md
git commit -m "Document formal FA1 and FA2 backends"
```

## Follow-up Plans

After this attention sub-project passes independently, write separate plans in
this order so each family remains reviewable:

1. GEMM: `reference`, `tiled_reference`, `cuda_tiled`;
2. Experts: `reference`, `padded_reference`, `cuda_core`, `cuda_wmma`;
3. MLA: `reference_naive`, `reference_absorbed`, `cuda_staged`, `cuda_paged`;
4. expert-parallel transport: `gloo`, `nccl`, then a separately designed
   `nvshmem` implementation when its runtime and hardware requirements exist.

Each follow-up plan must reuse the strict explicit/`auto` selection rule from
the design spec and must not relabel an experimental helper as a production
backend.
