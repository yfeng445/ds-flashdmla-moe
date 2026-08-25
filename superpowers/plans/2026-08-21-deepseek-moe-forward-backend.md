# DeepSeek MoE Forward Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete single-device, forward-only DeepSeek routed-MoE call with strict `reference`, `cuda`, and `auto` backends, backed by one output-only native operator that initially reuses the existing staged CUDA kernels.

**Architecture:** Put backend selection in a new cycle-free Python facade. The reference branch calls the existing packed PyTorch implementation; the CUDA branch calls one dispatcher operator. That operator privately executes grouped sigmoid routing, single-rank route packing, expert-major active-row SwiGLU, and route combination. Existing kernels remain translation-unit private; only narrow host entry points become linkable. The raw schema returns the final output only, so future launch/intermediate fusion does not change callers.

**Tech Stack:** Python 3.10+, PyTorch 2.4+ custom operators, C++17, CUDA C++17, pytest, ruff, setuptools `CUDAExtension`, GitHub Actions CUDA 12.8, local WSL/RTX 5090 validation.

**Spec:** `superpowers/specs/2026-08-21-single-device-deepseek-moe-forward-design.md`

## Global Constraints

- The public API is named `deepseek_moe_forward`; do not call this milestone `flashdmoe_forward` or claim that it is full FlashMoE.
- Public backends are exactly `auto`, `cuda`, and `reference`.
- The public facade validates the entire request and chooses one backend before any stage runs. It never mixes stage backends.
- `reference` calls `deepseek_moe_packed_reference`; `deepseek_moe_reference` remains the independent numerical oracle.
- The public input is floating `[*token_shape, D]` with rank at least two. Output has the same shape, dtype, device, and contiguous layout.
- Routed weights are `gate_weight=[E,D]`, `expert_w1/expert_w3=[E,H,D]`, and `expert_w2=[E,D,H]`, with `E,H,D > 0`.
- Every floating input, including optional `score_bias=[E]`, shares one dtype and device.
- Routing accepts `sigmoid` and `softmax` on the reference path, with current group-limited Top-K, tie-breaking, selection-bias, and route-scale semantics unchanged.
- CUDA v1 accepts only contiguous CUDA FP32 tensors, sigmoid routing, disabled deterministic algorithms, and tensors with `requires_grad=False`.
- Explicit CUDA rejects every unsupported request with `RuntimeError` prefixed by `CUDA DeepSeek MoE is unavailable:`.
- `auto` falls back only before dispatch. Exceptions from the selected native operator propagate and are never retried through reference.
- The raw operator accepts flattened contiguous `[T,D]`, fixed sigmoid routing, and returns `[T,D]`.
- The raw operator has a CUDA implementation and FakeTensor implementation only. Do not register Composite or autograd implementations.
- Python and C++ both enforce forward-only behavior, including inside `torch.no_grad()`.
- Single-rank `route_pack` already emits expert-major rows. Do not call `expert_major_pack` in the whole-layer path.
- Derive packed token ids as `floor(packed_route_index / topk)` before route combine.
- Preserve all existing stage APIs, backward behavior, and files under `csrc/experimental/`.
- Do not add shared experts, capacity/drop policies, routing return values, FP16/BF16/FP8/INT8 CUDA, native backward, multi-GPU transport, persistent scheduling, or a performance claim.
- Never stage or restore the user-owned deletions of root `AI INFRA.ipynb` and `handoff.md`.

## File Map

| File | Responsibility after this plan |
| --- | --- |
| `src/ds_flash_mla_moe/moe_ops.py` | Whole-layer validation, eligibility, backend selection, facade, capability query |
| `src/ds_flash_mla_moe/ops.py` | Raw schema and FakeTensor shape/forward-only contract |
| `src/ds_flash_mla_moe/__init__.py` | Public type, facade, and capability exports |
| `tests/test_moe_backends.py` | Reference, policy, dispatcher, CUDA correctness, stream, compile, and opcheck tests |
| `csrc/moe/moe_cuda_ops.h` | Linkable declarations for four existing host entry points |
| `csrc/moe/grouped_topk_cuda.cu` | Existing private router kernel plus external host wrapper |
| `csrc/moe/route_ops_cuda.cu` | Existing private pack/combine kernels plus external host wrappers |
| `csrc/moe/swiglu_experts_cuda.cu` | Existing private expert kernels plus external host wrapper |
| `csrc/moe/deepseek_moe_forward_cuda.cu` | Whole-layer checks, staged orchestration, and CUDA registration |
| `csrc/ops.cpp` | Native schema declaration |
| `setup.py`, `MANIFEST.in` | Wheel source/build inclusion |
| `src/ds_flash_mla_moe/moe_benchmarking.py` | Whole-layer benchmark configuration, work/intermediate model, verification, timing |
| `benchmarks/moe.py` | Reproducible CLI for the whole-layer benchmark |
| `tests/test_moe_benchmarking.py` | CPU-runnable benchmark/report tests |
| `README.md`, `docs/chapters/04-deepseek-moe.md`, `docs/chapters/06-pytorch-custom-operators.md`, `docs/chapters/07-benchmarking-and-roofline.md` | Honest status, usage, operator boundary, and benchmark documentation |
| `.github/workflows/cuda-build.yml` | Compile/package gate and complete CUDA-operator presence check |

---

### Task 1: Whole-Layer Reference Facade and Strict Backend Policy

**Files:**
- Create: `src/ds_flash_mla_moe/moe_ops.py`
- Modify: `src/ds_flash_mla_moe/__init__.py`
- Create: `tests/test_moe_backends.py`

**Interfaces:**
- Consumes: `deepseek_moe_packed_reference`, `_operator_has_cuda_kernel`, and `torch.ops.ds_flash_mla_moe.deepseek_moe_forward.default`.
- Produces: `MoEBackend`, `MoEScoreFunction`, `deepseek_moe_forward`, and `cuda_moe_available`.

- [ ] **Step 1: Write failing reference and policy tests**

Create deterministic helpers in `tests/test_moe_backends.py` for shapes such as `T=7,D=5,H=9,E=4,K=2,G=2,Gk=1`. Add CPU tests that:

- compare FP64 reference-facade output and gradients against fresh inputs passed to `deepseek_moe_reference`;
- cover sigmoid and softmax, optional bias, grouped selection, route scale, rank-2/rank-3 input, exact ties, one hot expert, inactive experts, and zero tokens;
- reject an invalid backend, rank, shape, dtype/device mismatch, nonfloating input, invalid `topk/n_groups/topk_groups`, nonfinite `route_scale`, and malformed bias;
- assert explicit CUDA rejects a CPU request with the required prefix;
- assert `cuda_moe_available()` returns `bool`;
- monkeypatch the raw op/capability predicate to prove eligible `auto` calls native once, ineligible `auto` calls reference once, and a native exception propagates without fallback;
- prove `backend="reference"` never invokes the native operator, even when tensors happen to be CUDA tensors.

The central policy test should use module-level monkeypatching, not mock implementation details of the four stages:

```python
def test_selected_cuda_failure_is_not_retried(monkeypatch) -> None:
    inputs = _moe_inputs(dtype=torch.float32)
    monkeypatch.setattr(moe_ops, "_cuda_moe_ineligibility_reason", lambda *a, **k: None)
    monkeypatch.setattr(
        moe_ops,
        "_call_cuda_moe",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("launch failed")),
    )
    monkeypatch.setattr(
        moe_ops, "deepseek_moe_packed_reference", lambda *a, **k: pytest.fail("fallback")
    )
    with pytest.raises(RuntimeError, match="launch failed"):
        deepseek_moe_forward(*inputs, topk=2, backend="auto")
```

- [ ] **Step 2: Run the focused test and confirm RED**

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m pytest tests/test_moe_backends.py -q
```

Expected: collection fails because `moe_ops.py` and the public exports do not exist.

- [ ] **Step 3: Implement public validation and reference dispatch**

Define exactly:

```python
MoEBackend = Literal["auto", "cuda", "reference"]
MoEScoreFunction = Literal["sigmoid", "softmax"]
```

Implement the public signature from the spec. `_validate_moe_inputs(...) -> int` returns the effective `topk_groups`. Perform structural validation before backend selection. Use the current routing constraints and require all floating tensors to share dtype/device.

For the reference branch, call:

```python
deepseek_moe_packed_reference(
    x,
    gate_weight,
    expert_w1,
    expert_w2,
    expert_w3,
    topk=topk,
    n_groups=n_groups,
    topk_groups=effective_topk_groups,
    score_func=score_func,
    score_bias=score_bias,
    route_scale=route_scale,
)
```

Return `.contiguous()` without detaching so reference gradients remain valid.

- [ ] **Step 4: Implement one-shot CUDA eligibility and dispatch**

`_cuda_moe_ineligibility_reason(...) -> str | None` checks, in this order:

1. any floating input has `requires_grad=True`;
2. tensors are not CUDA;
3. dtype is not FP32;
4. score function is not sigmoid;
5. tensors are not contiguous;
6. deterministic algorithms are enabled;
7. the raw operator has no CUDA kernel.

Keep `_call_cuda_moe(...)` as a small separately testable function that flattens `x` with `reshape(-1, D)`, invokes the raw operator, restores the original shape, and returns contiguous output. It does not catch exceptions.

`cuda_moe_available()` reports only installation/device/operator availability; it does not promise that an arbitrary request is eligible.

- [ ] **Step 5: Export and verify GREEN**

Export the two types, facade, and capability query from `__init__.py`. Run:

```powershell
python -m pytest tests/test_moe_backends.py tests/test_moe.py tests/test_router_ops.py tests/test_route_ops.py tests/test_expert_ops.py -q
python -m ruff check src/ds_flash_mla_moe/moe_ops.py tests/test_moe_backends.py src/ds_flash_mla_moe/__init__.py
```

- [ ] **Step 6: Commit**

```powershell
git add src/ds_flash_mla_moe/moe_ops.py src/ds_flash_mla_moe/__init__.py tests/test_moe_backends.py
git commit -m "feat: add DeepSeek MoE forward facade"
```

---

### Task 2: Raw Schema, FakeTensor Contract, and Forward-Only Dispatcher Policy

**Files:**
- Modify: `src/ds_flash_mla_moe/ops.py`
- Modify: `tests/test_moe_backends.py`

**Interfaces:**
- Consumes: the `torch.library` schema/fake registration helpers already used by formal FA1/FA2.
- Produces: an always-defined `ds_flash_mla_moe::deepseek_moe_forward` schema with FakeTensor shape propagation and no CPU/Composite/autograd implementation.

- [ ] **Step 1: Add failing dispatcher tests**

Add CPU/FakeTensor tests that require:

```python
assert moe_ops._operator_is_defined("deepseek_moe_forward")
```

Use `FakeTensorMode` on flattened inputs and assert `[T,D]`, dtype, device, and contiguous stride. Add invalid-fake tests for rank, shapes, dtype equality, bias, routing configuration, and `requires_grad=True`.

Assert these dispatch keys are absent:

```python
for key in ("CPU", "AutogradCUDA", "CompositeExplicitAutograd", "CompositeImplicitAutograd"):
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(
        "ds_flash_mla_moe::deepseek_moe_forward", key
    )
```

- [ ] **Step 2: Confirm RED**

```powershell
python -m pytest tests/test_moe_backends.py -q
```

Expected: schema-definition and FakeTensor tests fail because the raw operator is not registered.

- [ ] **Step 3: Add schema and fake implementation**

In `ops.py`, add this exact schema to `_SCHEMAS`:

```python
_DEEPSEEK_MOE_FORWARD_SCHEMA = (
    "deepseek_moe_forward(Tensor x, Tensor gate_weight, Tensor expert_w1, "
    "Tensor expert_w2, Tensor expert_w3, int topk, int n_groups, "
    "int topk_groups, Tensor? score_bias, float route_scale) -> Tensor"
)
```

The fake implementation checks the raw flattened contract, all dimension relationships, shared floating dtype/device, contiguous layout, `E/H/D > 0`, routing bounds, finite route scale, optional bias shape, and no `requires_grad`. It returns `x.new_empty(x.shape)`.

Register only the fake implementation. Do not add Composite or autograd registration.

- [ ] **Step 4: Verify and commit**

```powershell
python -m pytest tests/test_moe_backends.py tests/test_ops.py -q
python -m ruff check src/ds_flash_mla_moe/ops.py tests/test_moe_backends.py
git add src/ds_flash_mla_moe/ops.py tests/test_moe_backends.py
git commit -m "feat: define forward-only MoE operator contract"
```

---

### Task 3: Staged Whole-Layer CUDA Operator

**Files:**
- Create: `csrc/moe/moe_cuda_ops.h`
- Modify: `csrc/moe/grouped_topk_cuda.cu`
- Modify: `csrc/moe/route_ops_cuda.cu`
- Modify: `csrc/moe/swiglu_experts_cuda.cu`
- Create: `csrc/moe/deepseek_moe_forward_cuda.cu`
- Modify: `csrc/ops.cpp`
- Modify: `setup.py`
- Modify: `MANIFEST.in`

**Interfaces:**
- Consumes: the existing CUDA stage implementations without changing their mathematics or public schemas.
- Produces: external host wrappers in `ds_flash_mla_moe::moe` and a CUDA implementation for the whole-layer schema.

- [ ] **Step 1: Add a source/build presence test and confirm RED**

Add a CPU-runnable repository test in `tests/test_moe_backends.py` that reads the tracked source/build files and requires:

- `csrc/moe/deepseek_moe_forward_cuda.cu` and `csrc/moe/moe_cuda_ops.h` exist;
- `setup.py` contains the new `.cu` source;
- `csrc/ops.cpp` declares the exact schema;
- `MANIFEST.in` recursively includes `*.h`, `*.cu`, and `*.cpp` under `csrc`.

Run it once and observe failure for the missing files/registration.

- [ ] **Step 2: Declare narrow host entry points**

`moe_cuda_ops.h` declares these signatures inside `namespace ds_flash_mla_moe::moe`:

```cpp
std::tuple<at::Tensor, at::Tensor> grouped_topk_cuda_entry(...);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
route_pack_cuda_entry(...);
at::Tensor swiglu_experts_cuda_entry(...);
at::Tensor route_combine_cuda_entry(...);
```

Use the exact arguments of the existing internal functions. Include ATen, optional, and tuple declarations. Do not expose device kernels or generic helper functions.

- [ ] **Step 3: Add wrappers without moving private kernels**

In each existing `.cu`, retain kernels and validation helpers in the anonymous namespace. After that namespace, define the external wrapper, which calls the existing internal host function. Register the existing public stage operator through the wrapper so both the whole-layer operator and dispatcher use the same implementation.

- [ ] **Step 4: Implement whole-layer checks and orchestration**

In `deepseek_moe_forward_cuda.cu`:

1. validate raw ranks, shapes, same CUDA device, FP32 dtype, contiguous layout, no `requires_grad`, finite scale, routing bounds, optional bias, and disabled deterministic algorithms;
2. create `expert_owner=zeros([E], int64, same CUDA device)`;
3. call `grouped_topk_cuda_entry`;
4. call `route_pack_cuda_entry(..., expert_owner, 1)`;
5. build `expert_offsets = cat([zeros([1], int64), counts_per_expert.cumsum(0)])`;
6. call `swiglu_experts_cuda_entry(packed_activations, expert_offsets, weights...)`;
7. derive `token_indices = floor_divide(packed_route_indices, topk)`;
8. call `route_combine_cuda_entry(contributions, packed_weights, token_indices, T)`;
9. return `[T,D]`.

Do not inspect packed expert ids on the host, synchronize the device, call `expert_major_pack`, or expose intermediates.

Register only:

```cpp
TORCH_LIBRARY_IMPL(ds_flash_mla_moe, CUDA, m) {
  m.impl("deepseek_moe_forward", TORCH_FN(ds_flash_mla_moe::moe::deepseek_moe_forward_cuda));
}
```

- [ ] **Step 5: Wire source and schema into the wheel**

Add the raw schema to `csrc/ops.cpp`, add the new source to the CUDA source list in `setup.py`, and ensure `MANIFEST.in` packages the header/source patterns.

- [ ] **Step 6: Run CPU/static verification and commit**

```powershell
python -m pytest tests/test_moe_backends.py -q
python -m ruff check tests/test_moe_backends.py
git diff --check
git add csrc/moe/moe_cuda_ops.h csrc/moe/grouped_topk_cuda.cu csrc/moe/route_ops_cuda.cu csrc/moe/swiglu_experts_cuda.cu csrc/moe/deepseek_moe_forward_cuda.cu csrc/ops.cpp setup.py MANIFEST.in tests/test_moe_backends.py
git commit -m "feat: add staged CUDA DeepSeek MoE forward"
```

The local Windows checkout lacks `nvcc`; do not claim compilation until Task 7 completes remote build validation.

---

### Task 4: Native CUDA Correctness, Stream, Compile, and Opcheck Coverage

**Files:**
- Modify: `tests/test_moe_backends.py`

**Interfaces:**
- Consumes: completed public facade and raw CUDA operator.
- Produces: GPU-only evidence for semantic parity and dispatcher integration.

- [ ] **Step 1: Add skipped-when-unavailable CUDA tests before the remote build**

Parameterize at least these `(T,D,H,E,K,G,Gk)` shapes:

```python
(
    (0, 5, 7, 4, 2, 2, 1),
    (1, 1, 1, 1, 1, 1, 1),
    (7, 15, 17, 4, 2, 2, 1),
    (17, 33, 65, 8, 3, 4, 2),
    (31, 65, 33, 9, 4, 3, 2),
)
```

For detached contiguous CUDA FP32 inputs, compare explicit `cuda` and eligible `auto` against `deepseek_moe_reference` with initial `rtol=atol=1e-3`, then tighten only if all adversarial cases support it. Cover bias/no-bias, exact ties, all routes to one expert, inactive experts, finite/contiguous output, and rank-3 facade restoration.

Add strict CUDA rejection tests for FP16/BF16, softmax, noncontiguous tensors, `requires_grad=True`, deterministic mode, and missing native kernel. Confirm `auto` uses a complete reference path for each.

- [ ] **Step 2: Add current-stream, raw-op, opcheck, and compile tests**

- Execute input mutation and whole-layer call on a non-default `torch.cuda.Stream`, record the output on that stream, synchronize that stream only, and compare with reference.
- Run `torch.library.opcheck` on detached raw inputs.
- Wrap the raw op in `torch.compile(fullgraph=True, backend="eager")` and compare output.
- Assert the raw op rejects `requires_grad=True` both normally and inside `torch.no_grad()`.

- [ ] **Step 3: Run CPU collection/static checks and commit**

```powershell
python -m pytest tests/test_moe_backends.py -q
python -m ruff check tests/test_moe_backends.py
git add tests/test_moe_backends.py
git commit -m "test: cover native DeepSeek MoE forward"
```

Expected locally: CPU policy/fake tests pass and GPU tests skip until the built wheel is installed in WSL.

---

### Task 5: Reproducible Whole-Layer Benchmark

**Files:**
- Create: `src/ds_flash_mla_moe/moe_benchmarking.py`
- Create: `benchmarks/moe.py`
- Create: `tests/test_moe_benchmarking.py`

**Interfaces:**
- Consumes: `deepseek_moe_forward`, `deepseek_moe_reference`, and existing benchmarking timing/environment helpers.
- Produces: a JSON-reporting forward-only benchmark with route-skew and intermediate-size evidence.

- [ ] **Step 1: Write failing benchmark tests**

Define a small CPU reference configuration and require the report to contain:

- schema version and benchmark name `deepseek_moe_forward`;
- full config including `T,D,H,E,K,G,Gk`, dtype, backend, seed, warmup, iterations;
- initialization scales;
- output metadata;
- numerical error versus `deepseek_moe_reference`;
- raw latency samples and summary;
- routed-row distribution including active/empty experts and peak-to-mean skew;
- analytical bytes for dense scores, packed activations, packed weights/indices, expert hidden state, contributions, and total major intermediates;
- `implementation="single_device_staged"` and `performance_claim=false`.

Test invalid dimensions/routing/backend combinations and `--no-verify` behavior.

- [ ] **Step 2: Confirm RED**

```powershell
python -m pytest tests/test_moe_benchmarking.py -q
```

- [ ] **Step 3: Implement config, report, and CLI**

Use a frozen `MoEForwardBenchmarkConfig`, deterministic fan-in-scaled inputs, existing CPU/CUDA measurement helpers, and output verification. Time only the facade call, not input creation, routing analysis, or reference verification. Do not assert that CUDA is faster.

The CLI exposes dimensions, grouping, dtype, device, backend, seed, warmup, iterations, route scale, bias toggle, verification toggle, and JSON output path.

- [ ] **Step 4: Verify and commit**

```powershell
python -m pytest tests/test_moe_benchmarking.py -q
python -m ruff check src/ds_flash_mla_moe/moe_benchmarking.py benchmarks/moe.py tests/test_moe_benchmarking.py
python benchmarks/moe.py --device cpu --dtype float64 --backend reference --tokens 7 --model-dim 5 --hidden-dim 9 --experts 4 --topk 2 --n-groups 2 --topk-groups 1 --warmup 0 --iterations 1
git add src/ds_flash_mla_moe/moe_benchmarking.py benchmarks/moe.py tests/test_moe_benchmarking.py
git commit -m "bench: add whole-layer MoE forward report"
```

---

### Task 6: Status, Teaching Notes, Packaging Gate, and Operator Inventory

**Files:**
- Modify: `README.md`
- Modify: `docs/chapters/04-deepseek-moe.md`
- Modify: `docs/chapters/06-pytorch-custom-operators.md`
- Modify: `docs/chapters/07-benchmarking-and-roofline.md`
- Modify: `.github/workflows/cuda-build.yml`
- Modify: `tests/test_moe_backends.py`

**Interfaces:**
- Consumes: the tested API, native schema, benchmark, and current formal FA1/FA2 operators.
- Produces: an accurate reader-facing status and a CI build gate for every formal CUDA operator.

- [ ] **Step 1: Add failing documentation/CI assertions**

Add a small source assertion that the workflow operator tuple contains:

```text
attention_fa1_forward
attention_fa2_forward
deepseek_moe_forward
```

Also require the reader docs to contain `deepseek_moe_forward` and the phrases `single-device`, `staged`, and `correctness-first` in the relevant section.

- [ ] **Step 2: Confirm RED**

```powershell
python -m pytest tests/test_moe_backends.py -q
```

- [ ] **Step 3: Update documentation honestly**

- README status table gains a complete single-device MoE forward entry and a minimal `reference/cuda/auto` usage example.
- Chapter 4 shows route → pack → offsets → expert → combine, links the new `.cu`, and contrasts this milestone with true FlashMoE persistent/tile-scheduled/multi-GPU behavior.
- Chapter 6 records the output-only raw schema, FakeTensor-only support, forward-only policy, and updated CUDA operator count.
- Chapter 7 adds the whole-layer benchmark command and explains launch/intermediate evidence without speed claims.
- Keep backward material in the experimental-history context; do not present it as supported for the new operator.

- [ ] **Step 4: Update the CUDA build gate**

Add the two already-built formal FA operators and the new whole-layer MoE operator to the workflow's CUDA-dispatch assertions. Keep the wheel artifact upload unchanged.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest tests/test_moe_backends.py tests/test_moe_benchmarking.py -q
python -m ruff check .
git diff --check
git add README.md docs/chapters/04-deepseek-moe.md docs/chapters/06-pytorch-custom-operators.md docs/chapters/07-benchmarking-and-roofline.md .github/workflows/cuda-build.yml tests/test_moe_backends.py
git commit -m "docs: document staged DeepSeek MoE backend"
```

---

### Task 7: Full Verification, Remote CUDA Build, Local RTX 5090 Run, Merge, and Push

**Files:**
- Modify only if verification exposes a defect: files already owned by Tasks 1-6.
- Optional evidence output: `validation/deepseek-moe-forward-rtx5090.json` only when produced by the committed benchmark CLI with complete environment metadata.

**Interfaces:**
- Consumes: feature branch, GitHub Actions wheel artifact, WSL Python 3.12, RTX 5090.
- Produces: fresh test/build/GPU evidence and an integrated `main` branch.

- [ ] **Step 1: Run complete local CPU verification**

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m pytest -q
python -m ruff check .
git diff --check
```

Record exact pass/skip counts. Do not substitute an earlier run.

- [ ] **Step 2: Push the feature branch and wait for the CUDA wheel build**

```powershell
git push -u origin feature/deepseek-moe-forward
gh run list --workflow "CUDA build" --branch feature/deepseek-moe-forward --limit 1
gh run watch <run-id> --exit-status
gh run download <run-id> --name cuda-wheel --dir wheelhouse/deepseek-moe-forward
```

If the compiler fails, use systematic debugging, add a focused regression/static test where possible, fix on the feature branch, and repeat until the workflow succeeds.

- [ ] **Step 3: Install and validate the wheel inside WSL**

Create a repo-scoped WSL virtual environment with `/usr/bin/python3.12`, install PyTorch 2.10.0 CUDA 12.8 and the downloaded wheel, then run:

```bash
python -c 'import torch, ds_flash_mla_moe as m; print(torch.cuda.get_device_name()); assert m.cuda_moe_available()'
python -m pytest tests/test_moe_backends.py -m cuda -q
python benchmarks/moe.py --device cuda --dtype float32 --backend cuda --tokens 31 --model-dim 65 --hidden-dim 33 --experts 9 --topk 4 --n-groups 3 --topk-groups 2 --warmup 3 --iterations 10 --output validation/deepseek-moe-forward-rtx5090.json
```

Ensure tests and benchmark import the installed wheel rather than the Windows editable source. The Ampere cubin plus PTX wheel may JIT for Blackwell; record the actual result, not an assumption.

- [ ] **Step 4: Run final branch review and repair any blocking findings**

Use `superpowers:requesting-code-review` over the merge-base-to-HEAD diff. Resolve Critical/Important findings, rerun their focused tests, then rerun the complete verification commands.

- [ ] **Step 5: Merge to main without touching user-owned deletions**

From the original checkout, verify the only unstaged main changes remain the two user-owned deletions, merge the feature branch, and run the complete CPU suite again with `PYTHONPATH` pointing at main.

```powershell
git merge --no-ff feature/deepseek-moe-forward
python -m pytest -q
python -m ruff check .
git diff --check
git push origin main
```

Do not stage, restore, or include `AI INFRA.ipynb` or `handoff.md` in the merge commit.

- [ ] **Step 6: Clean owned worktree and report**

After confirming `origin/main` contains the merge and the owned worktree is clean, remove only `.worktrees/deepseek-moe-forward`. Report:

- facade and raw operator behavior;
- CPU, remote compile, and RTX 5090 evidence;
- remaining FlashMoE gaps (persistent scheduler, fusion, multi-GPU one-sided communication);
- repository rename recommendation separately from the package/import namespace.
