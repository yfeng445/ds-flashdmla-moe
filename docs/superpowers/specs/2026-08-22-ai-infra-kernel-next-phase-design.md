# AI Infra Kernel Lab Next-Phase Design

## Status

This design turns the approved next-stage roadmap into forward-only, selectable,
and measurable operator paths. It preserves every existing reference backend and
keeps native backward experiments under `csrc/experimental/`.

## Goals

- Make whole-layer MoE launch count and materialized intermediates observable with
  Kineto/NVTX and Nsight-ready ranges.
- Keep the current staged MoE backend as a comparison point while adding explicit
  fused and persistent single-device backends behind the same mathematical API.
- Remove single-device routing metadata that exists only because the public EP
  pack operator returns a distributed six-tensor protocol.
- Add a visibly distinct, forward-only FA3 teaching backend after FA1 and FA2.
- Add reusable CUDA Graph capture, FP8/INT8 quantization experiments, and a
  transactional continuous-batching control plane.
- Freeze vendor-neutral one-sided protocol and EP/TP topology semantics that are
  fully testable with logical PEs on one device, without claiming remote-memory
  correctness or communication overlap.

## Non-goals

- Native backward for FA3, MoE, quantized kernels, or paged decode graphs.
- Claiming that a single-device persistent expert core is the complete FlashMoE
  scheduler/subscriber/processor megakernel.
- Claiming NVSHMEM, NCCL device API, remote visibility, or physical overlap from a
  one-GPU simulator.
- Prefix sharing, eviction, speculative decoding, networking, or a production LLM
  serving runtime.
- Hard latency assertions in unit tests.

## Public Backend Contracts

### MoE

`deepseek_moe_forward` accepts these forward paths:

- `reference`: existing differentiable PyTorch oracle.
- `cuda_staged`: the current route-pack-expert-combine implementation.
- `cuda_fused`: single-device pack metadata specialization plus down-projection
  weighted combine epilogue.
- `cuda_persistent`: the fused path with an explicitly bounded persistent expert
  task scheduler.
- `cuda`: compatibility alias for `cuda_fused`.
- `auto`: selects the best eligible verified native path, otherwise the complete
  reference path.

The existing raw `deepseek_moe_forward` schema remains the staged comparison
operator. New raw fused/persistent operators return only `[T,D]`. Public
`route_pack`, `route_combine`, and EP behavior do not change.

### Attention

FA1 and FA2 remain unchanged. `fa3` is a separate forward-only CUDA operator with
an asynchronous double-buffered K/V pipeline and on-chip FP32 online-softmax
state. The code and docs call it a teaching implementation: it demonstrates the
FA3 pipeline direction but does not claim Hopper/Blackwell production parity,
TMA/WGMMA peak utilization, or FA3 backward.

Explicit FA backends never silently fall back. `auto` remains conservative and
does not select FA3 until hardware evidence exists.

### Quantization

Quantization is introduced first as an independent linear/GEMM experiment rather
than being hidden inside MLA or MoE. Per-row activations and per-output-channel
weights carry explicit scales. FP8 E4M3 and symmetric INT8 both dequantize to a
stable FP32 accumulation oracle; native paths are selectable only for formats the
loaded extension actually implements.

### CUDA Graph and serving

CUDA Graph capture owns static input buffers, validates replay metadata before
copying, and returns a stable output buffer. The minimal scheduler uses FIFO
admission, homogeneous prefill/decode microbatches, fixed pages, and transactional
schedule/complete/abort semantics. Requests enter and leave at iteration
boundaries, which is the defining behavior being tested.

### Distributed research layer

`ParallelMesh` freezes DP/EP/TP rank mapping. The one-sided protocol carries
route identity, generation, count, buffer slot, and explicit cell state. A
deterministic logical-PE simulator may reorder payload and signal events, but all
reports must state `simulated=true` and `remote_visibility_verified=false`.
Tensor-parallel SwiGLU shards W1/W3 across hidden output and W2 across hidden
input, then sums partial outputs on one device as a functional oracle.

## MoE Implementation Stages

1. Record structured profiler evidence: kernel/activity count, synchronization,
   peak allocated memory, and a complete analytical intermediate inventory.
2. Add a private single-device pack returning packed activations, packed weights,
   token indices, and the scan-produced expert offsets. This removes owner/rank
   metadata, duplicate counts, `cat`, `cumsum`, and `floor_divide`.
3. Fuse the down-projection epilogue with weighted atomic combine, removing the
   materialized contributions tensor and standalone combine launch.
4. Expose a bounded persistent task variant. It must retain a fused fallback for
   small route counts and remain described as a single-device expert core.

The route GEMM remains a library GEMM in this phase. The materialized hidden state
remains because W2 depends on the complete SwiGLU hidden vector.

## Verification Boundary

- CPU/reference tests cover every public policy and simulator state transition.
- FakeTensor and dispatch tests prove native schemas are forward-only and do not
  accidentally acquire Composite or autograd kernels.
- CUDA build CI proves every native source compiles and registers.
- The RTX 5090 single-card runner checks numerical equality, current-stream use,
  repeated skewed routing, graph replay, and profiler evidence.
- Real NCCL/NVSHMEM transport, remote ordering, deadlock freedom under occupancy,
  and EP/TP communication require two or four GPUs and remain a later hardware
  gate, not a single-card claim.

## Documentation Contract

Reader-facing chapters must preserve the distinction between educational
mechanisms and production implementations. Benchmark JSON contains raw samples
and facts, not speedup claims. The tracked source notebook and handoff are not
rewritten as part of this work.
