# AI Infra Kernel Lab Next-Phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan.

**Goal:** Deliver measurable single-device MoE fusion, a distinct FA3 teaching forward, CUDA Graph/quantization/continuous-batching building blocks, and single-card-verifiable EP/TP protocol foundations.

**Architecture:** Existing reference implementations remain the semantic oracles. New native paths are strict explicit backends, while `auto` only selects verified eligible implementations. Distributed work is split into a vendor-neutral protocol/simulator now and real transport adapters only on suitable multi-GPU hardware.

**Tech Stack:** Python 3.10+, PyTorch 2.10 custom operators/FakeTensor, CUDA 12.8 C++17, pytest, Ruff, Kineto/NVTX, GitHub Actions CUDA build and self-hosted RTX 5090 validation.

**Spec:** `superpowers/specs/2026-08-22-ai-infra-kernel-next-phase-design.md`

## Global Constraints

- Forward-only native scope; do not add supported backward operators.
- Preserve `reference` backends and existing public route/EP schemas.
- Every `csrc` operator must have an explicit Python-selectable backend.
- Explicit native backends fail loudly; only `auto` may fall back.
- Do not claim complete FlashMoE, production FA3, NVSHMEM, or multi-GPU overlap without matching hardware evidence.
- Do not edit `AI INFRA.ipynb` or `handoff.md`.
- Tests assert correctness and evidence structure, never fixed speedups.

---

## Task 1: Profile and Inventory Whole-Layer MoE

- [x] Extend `src/ds_flash_mla_moe/moe_benchmarking.py` with named staged/fused intermediate inventories and complete metadata byte accounting.
- [x] Extend `src/ds_flash_mla_moe/profiling.py` and `benchmarks/moe.py` with Kineto/NVTX modes that report device activities, custom kernels, synchronization, and peak allocated memory.
- [x] Add CPU unit tests in `tests/test_moe_benchmarking.py` and profiler aggregation tests in `tests/test_profiling.py`; watch each new behavior fail before implementation.
- [x] Run `python -m pytest tests/test_moe_benchmarking.py tests/test_profiling.py` and Ruff on touched files.
- [x] Commit as `feat: add MoE profiling and intermediate baselines`.

## Task 2: Add Single-Device Fused and Persistent MoE Backends

- [x] Add failing dispatch/policy/FakeTensor tests to `tests/test_moe_backends.py` for `cuda_staged`, `cuda_fused`, and `cuda_persistent`.
- [x] Add private single-device pack and fused expert-combine entries in `csrc/moe/route_ops_cuda.cu`, `csrc/moe/swiglu_experts_cuda.cu`, and `csrc/moe/moe_cuda_ops.h` without changing public route operators.
- [x] Add raw fused/persistent schemas and registrations in `csrc/ops.cpp`, `src/ds_flash_mla_moe/ops.py`, `setup.py`, `MANIFEST.in`, and `.github/workflows/cuda-build.yml`.
- [x] Update `src/ds_flash_mla_moe/moe_ops.py` so explicit backends select exactly one whole-layer raw operator; keep `cuda` as the fused compatibility alias and preserve staged comparison.
- [x] Remove whole-layer `expert_owner`, rank metadata, duplicate expert counts/offsets, `cat`, `cumsum`, `floor_divide`, materialized contributions, and standalone combine from fused paths.
- [x] Implement a bounded persistent expert task loop with a small-work fallback; document it as a single-device core rather than a distributed megakernel.
- [x] Run focused CPU tests, CUDA build CI, RTX 5090 correctness/current-stream/skew stress, and a structured profiler comparison.
- [x] Commit as `feat: add fused and persistent MoE forwards`.

## Task 3: Add a Distinct FA3 Teaching Forward Backend

- [x] Add failing tests in `tests/test_attention_backends.py` for backend validation, schema/FakeTensor, forward-only dispatch, shape/dtype restrictions, and explicit no-fallback policy.
- [x] Add `csrc/attention/fa3_forward_cuda.cu` with double-buffered asynchronous K/V staging, one-CTA-per-query-tile ownership, and FP32 online-softmax/output accumulation.
- [x] Register `attention_fa3_forward` in `csrc/ops.cpp`, `src/ds_flash_mla_moe/ops.py`, `setup.py`, `MANIFEST.in`, and CUDA build checks; expose `backend="fa3"` from Python.
- [x] Add benchmark selection and document the concrete FA1/FA2/FA3 implementation differences without a speed claim.
- [x] Run CPU dispatch tests, CUDA build CI, and RTX 5090 FP16 causal/non-causal/tail/decode/current-stream comparisons.
- [x] Commit as `feat: add FA3 teaching forward backend`.

## Task 4: Add CUDA Graph and Continuous-Batching Control Plane

- [x] Add failing tests for static CUDA graph input contracts and FIFO transactional scheduler behavior.
- [x] Create `src/ds_flash_mla_moe/cuda_graph.py` with a reusable single-output graph runner and an MLA paged-decode capture wrapper using stable buffers and prevalidated raw operators.
- [x] Create `src/ds_flash_mla_moe/scheduler.py` with `FixedPageAllocator`, request/sequence state, `ScheduledBatch`, and `submit/schedule/complete/abort/cancel`.
- [x] Ensure `schedule` reserves pages/lengths atomically and `abort` restores them; decode uses one token per request and admits replacements at iteration boundaries.
- [x] Export supported APIs, add `benchmarks/cuda_graph.py` and `benchmarks/continuous_batching.py`, and document graph buckets and scheduler limits.
- [x] Run CPU scheduler tests plus CUDA graph replay tests on the available RTX 5090.
- [x] Commit as `feat: add graph replay and continuous batching`.

## Task 5: Add FP8 and INT8 Quantization Experiments

- [x] Add failing tests for per-row/per-channel scales, zeros, clamps, non-finite rejection, dequantized linear correctness, strict backend selection, and forward-only behavior.
- [x] Create `src/ds_flash_mla_moe/quantization.py` with explicit quantized matrix metadata and FP8 E4M3/INT8 reference paths.
- [x] Add native quantize/scaled-linear schemas only for formats implemented in `csrc/quantization/`; unavailable explicit CUDA formats must fail rather than use an internal PyTorch fallback.
- [x] Add `src/ds_flash_mla_moe/quantized_benchmarking.py`, `benchmarks/quantized_gemm.py`, tests, exports, and reader documentation.
- [x] Run CPU numerical tests, CUDA build CI, RTX 5090 current-stream/graph checks, and paired dequantized reference validation.
- [x] Commit as `feat: add FP8 and INT8 forward experiments`.

## Task 6: Add Single-Card-Verifiable EP/TP Foundations

- [x] Add failing tests for DP/EP/TP rank mapping, owner-local expert slots, protocol generation/state transitions, out-of-order logical-PE delivery, zero-count cells, overflow, and forward-only TP equivalence/rejection. The earlier TP-gradient wording is superseded by the global forward-only constraint.
- [x] Create `src/ds_flash_mla_moe/parallel_topology.py`, `one_sided_protocol.py`, `fake_distributed.py`, and `tensor_parallel.py`.
- [x] Carry explicit route IDs through simulated dispatch/return and model payload-before-signal plus consumed-generation acknowledgement.
- [x] Report simulator evidence as `simulated=true` and `remote_visibility_verified=false`; do not add an `nvshmem` backend literal.
- [x] Implement logical TP SwiGLU sharding for TP sizes 1/2/4 and compare forward output with the full expert oracle. Backward remains outside the supported milestone.
- [x] Run focused tests and the full CPU/reference suite.
- [x] Commit as `feat: add one-sided protocol and TP references`.

## Task 7: Integrate Evidence and Documentation

- [x] Update README and relevant chapters for backend tables, activity/intermediate deltas, FA version distinctions, graph/scheduler use, quantization scope, and distributed validation boundary.
- [x] Add reproducible benchmark commands and checked-in structured single-GPU evidence only for measurements actually run.
- [x] Run the full pytest suite, Ruff check/format check, `git diff --check`, hosted CUDA build workflow, and direct installed-wheel RTX CUDA tests. No matching self-hosted workflow runner was registered, so the native matrix ran directly on the same RTX 5090 instead.
- [x] Request whole-branch code review, resolve blocking findings, merge the feature branch into `main`, and push both branch and updated `main`.
- [x] Commit final integration as `docs: record next-phase kernel evidence`.

## Deferred hardware evidence

- [ ] Collect Nsight Systems/Compute timelines and counters for physical launch count, occupancy, DRAM traffic, instruction mix, and overlap. Kineto aggregate activities are not a substitute.
- [ ] Implement and validate real two-/four-GPU one-sided EP/TP transport, remote visibility, deadlock freedom, and communication/compute overlap. The checked-in logical simulator does not satisfy this item.
