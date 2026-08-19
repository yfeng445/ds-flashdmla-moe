# Explicit CUDA Backends and FA1/FA2 Forward Design

## Goal

Make every production-oriented CUDA implementation in `csrc` reachable through
a stable Python semantic API, so selecting an implementation is approximately a
backend change rather than an application rewrite. Introduce separate, readable,
and runnable FlashAttention-1 and FlashAttention-2 forward backends whose source
structure makes their algorithmic differences visible.

The first FA milestone is intentionally educational as well as functional. The
implementation should be useful for answering “What changed from FA1 to FA2?”
by reading and running the two kernels under the same public contract.

## Source Grounding

The formal implementations are based on the published algorithms rather than
promoting the existing experimental prototypes:

- FlashAttention-1: <https://arxiv.org/pdf/2205.14135>
- FlashAttention-2: <https://arxiv.org/pdf/2307.08691>

The version names in this repository refer to the algorithmic and work-
partitioning ideas described by those papers. They do not imply parity with a
particular release of an external FlashAttention library.

## Core Semantic Contract

Python owns the operation's meaning; a backend chooses its implementation.
Within one operation family, changing the backend must preserve:

- input and output tensor meaning;
- shape, layout, and causal-coordinate conventions;
- documented dtype behavior;
- error behavior for unsupported inputs;
- numerical agreement within a backend-appropriate tolerance.

For example, application code should be able to compare attention
implementations without changing tensor preparation or output handling:

```python
out_reference = flash_attention_forward(q, k, v, backend="reference")
out_fa1 = flash_attention_forward(q, k, v, backend="fa1")
out_fa2 = flash_attention_forward(q, k, v, backend="fa2")
```

Backend equivalence is semantic, not a claim that every implementation supports
every dtype or device. Each explicit backend validates its own capability and
raises a precise error when the request is unsupported.

## Backend Selection Rules

Explicit backend selection is strict:

- an explicit backend never silently runs another implementation;
- unsupported device, dtype, rank, layout, mask, gradient mode, or GPU
  capability raises an actionable exception;
- `auto` is the only selection mode allowed to choose another backend;
- `auto` chooses from a documented preference order and may fall back to a
  reference implementation;
- backend names are scoped by operation family rather than placed in one global
  enum.

The existing attention name `cuda` remains temporarily as a deprecated alias
for `cuda_rowwise`. It must not become an alias for FA1 or FA2 because doing so
would hide which algorithm was actually executed.

## Backend Families

The target public inventory is:

| Family | Python-visible backends | Meaning |
|---|---|---|
| Attention | `auto`, `reference`, `blockwise`, `cuda_rowwise`, `fa1`, `fa2` | Dense scaled dot-product attention |
| GEMM | `auto`, `reference`, `tiled_reference`, `cuda_tiled` | Matrix multiplication |
| Experts | `auto`, `reference`, `padded_reference`, `cuda_core`, `cuda_wmma` | Grouped/padded expert computation |
| MLA | `auto`, `reference_naive`, `reference_absorbed`, `cuda_staged`, `cuda_paged` | MLA decode/projection paths |
| Expert parallelism | `auto`, `gloo`, `nccl` | Distributed expert exchange |
| Future one-sided communication | `nvshmem` | Explicit future backend after a real implementation exists |

This table is a migration target, not a claim that every listed backend already
exists. Implementation work must expose and test a backend before documentation
marks it available. No placeholder FA3, FA4, or NVSHMEM implementation is added.

Low-level stage operators may remain public for focused testing and teaching.
The normal Python facade, however, selects a complete semantic path such as
`cuda_staged` or `cuda_paged`; users should not have to assemble an equivalent
MLA operation manually just to choose CUDA.

## Attention Python API

The stable forward facade remains conceptually:

```python
def flash_attention_forward(
    query,
    key,
    value,
    *,
    causal=False,
    scale=None,
    backend="auto",
):
    ...
```

The first formal FA1/FA2 milestone supports:

- rank-4 `BHSD` query, key, and value tensors;
- contiguous CUDA tensors;
- FP16 storage and FP32 score, softmax-state, and output accumulation;
- self-attention and cross-attention where query and key sequence lengths may
  differ;
- non-causal attention and right-aligned causal attention;
- forward output only.

The facade returns one output tensor with the same shape as query except that
its final dimension is the value dimension. It returns FP16 for FA1 and FA2.
Internal online-softmax state, including any log-sum-exp value, is not part of
the public return contract in this milestone.

Explicit FA1 and FA2 reject:

- CPU tensors;
- non-FP16 inputs;
- non-contiguous inputs;
- unsupported rank or inconsistent batch/head/sequence dimensions;
- explicit attention masks;
- tensors participating in an autograd request.

Forward-only means the Python entry point must detect `requires_grad` and raise
instead of returning a detached result that appears differentiable. Existing
experimental backward sources remain available for study but are not registered
as production operators.

The current row-wise CUDA attention retains its independently documented dtype
support. Adding the narrower FA1/FA2 capability must not reduce it.

## Operator Boundary

FA1 and FA2 are registered as distinct extension operators, for example:

```text
ds_flash_mla_moe::attention_fa1_forward(
    Tensor query,
    Tensor key,
    Tensor value,
    bool causal,
    float? scale
) -> Tensor

ds_flash_mla_moe::attention_fa2_forward(
    Tensor query,
    Tensor key,
    Tensor value,
    bool causal,
    float? scale
) -> Tensor
```

Separate schemas make dispatch observable, keep capability checks local, and
prevent a hidden internal switch from making both Python backends execute the
same kernel. Operator names may be adjusted to the repository's established
registration convention, but they must remain separately callable.

## Shared Numerical Basis

The comparison is meaningful only when both kernels use comparable arithmetic.
FA1 and FA2 therefore share:

- FP16 input and output conversion;
- FP32 dot-product accumulation for this teaching milestone;
- FP32 online-softmax state;
- FP32 output accumulation;
- the same scale default, `1 / sqrt(head_dim)`;
- the same right-aligned causal-coordinate rule;
- equivalent tile-size selection where hardware resources permit;
- common validation and numerical tolerance definitions.

Tensor Core acceleration may be added later as another controlled dimension.
It must not be enabled in only one of the initial variants and then presented as
evidence of an algorithm-only FA1/FA2 difference.

`attention_common.cuh` may contain conversion, coordinate, masking, and stable
online-softmax primitives. It must not hide either kernel's main loop order,
thread-block grid, warp partition, or output-update policy.

## FA1 Forward Kernel

The formal FA1 teaching kernel follows the FlashAttention-1 dataflow:

```text
grid = batch x heads
for each K/V column block:
    load the K/V block into on-chip storage
    for each Q row block:
        load Q, O, m, and l for that row block
        partition the K/V work across warps
        form and merge partial output contributions through shared memory
        update the normalized output and online-softmax state
        write O, m, and l back to global memory
```

Its defining, visible properties are:

- the K/V-block loop is outside the Q-block loop;
- the launch parallelizes primarily over batch and head;
- K/V work is split across warps;
- partial output contributions require a cross-warp merge;
- output and online state are read and written as K/V blocks advance;
- the output is maintained in normalized form during the blockwise recurrence.

This organization deliberately exposes the additional global-memory traffic
and lower sequence-dimension parallelism associated with the original
partitioning. The goal is fidelity and comparison, not forcing FA1 to win a
benchmark.

## FA2 Forward Kernel

The formal FA2 kernel implements the main forward changes described by the
FlashAttention-2 paper:

```text
grid = batch x heads x query_blocks
keep one Q block and its O, m, and l state on chip
for each K/V column block:
    load the K/V block
    partition Q rows across warps
    update an unnormalized O accumulator and online-softmax state
normalize O once after all K/V blocks
write O once
```

Its defining, visible properties are:

- the Q-block assignment is part of the thread-block grid;
- K/V blocks are traversed inside one independently scheduled Q block;
- Q rows, rather than K/V columns, are partitioned across warps;
- warps own distinct output rows, avoiding a cross-warp output reduction;
- the output accumulator stays unnormalized across K/V blocks;
- old output is rescaled when the running maximum changes, and final division
  by the normalizer occurs once;
- sequence-level thread-block parallelism improves occupancy when batch and
  head counts alone expose too little parallel work.

The implementation may compute log-sum-exp internally for diagnostics or later
backward work, but the initial public facade still returns only the output.

## FA1/FA2 Comparison Surface

| Dimension | FA1 backend | FA2 backend |
|---|---|---|
| Thread-block grid | batch x heads | batch x heads x Q blocks |
| Outer ownership | K/V traversal for a head | one Q block |
| Main loop order | K/V blocks, then Q blocks | Q block ownership, then K/V blocks |
| Warp partition | split K/V work | split Q rows |
| Cross-warp output merge | required | avoided by row ownership |
| Output recurrence | normalized each step | unnormalized until final division |
| Global O/state traffic | repeated as K/V advances | retained on chip, final write |
| Expected parallelism | limited for small batch/head count | adds sequence dimension |

Causal block skipping is an optimization available to both variants and is not
used as a version-defining distinction. Both should skip completely masked
tiles and apply elementwise causal masking only where a tile crosses the causal
boundary.

## Source Layout

Production attention sources become visibly versioned:

```text
csrc/attention/
|-- attention_common.cuh
|-- attention_forward_cuda.cu
|-- fa1_forward_cuda.cu
`-- fa2_forward_cuda.cu
```

The existing row-wise implementation remains `attention_forward_cuda.cu` and
maps to `cuda_rowwise`. The two new files must contain their own plainly named
launch functions and top-level kernel loops.

Existing files under `csrc/experimental/attention/` stay in place. They are not
renamed to the production paths, built into formal backends, or rewritten as
part of the first milestone. Their mixed or incomplete algorithmic choices are
useful historical material but do not satisfy the formal contracts above.

## Current Prototype Treatment

The experimental `flash_attention_v1.cu` already assigns Q blocks through the
grid and loops over K/V blocks inside a Q block. That mixes FA2-style sequence
parallelism into a file named v1, so promoting it would make the requested
comparison misleading.

The experimental `flash_attention_v2_fwd.cu` has a FA2-like loop orientation
but still normalizes the output during each K/V update rather than preserving
the unnormalized accumulator through the full traversal. Its row loop also
allows multiple lanes to traverse and write the same rows. It remains an
experiment; the formal FA2 implementation is written from the paper algorithm
with exclusive output ownership.

No experimental backward file is deleted or modified. It is simply omitted
from production registration until backward has its own reviewed design.

## Exposing the Rest of `csrc`

The repository-wide rule is that a semantically complete CUDA path should be
selectable from Python. This does not mean every helper kernel becomes a top-
level backend. A helper is exposed through the complete operation that owns it,
while independently meaningful stages may also retain low-level bindings for
testing.

Migration proceeds by family:

1. inventory existing registered operators and their Python call paths;
2. add family-specific backend type aliases and capability descriptions;
3. give ambiguous CUDA paths explicit names such as `cuda_rowwise`,
   `cuda_tiled`, `cuda_core`, and `cuda_wmma`;
4. ensure expert core and WMMA paths are explicitly selectable rather than
   selected only as an undocumented dtype side effect;
5. expose complete staged and paged MLA paths while keeping useful stage-level
   operators callable;
6. document and test the selected distributed transport (`gloo` or `nccl`);
7. leave incomplete experiments out of the production backend registry.

This migration can be delivered incrementally. The FA1/FA2 implementation must
not be blocked on completing every other family, but the facade and naming
introduced for it must follow the same repository-wide convention.

## Validation and Error Flow

Python performs fast semantic validation and resolves `auto`. The C++/CUDA
operator repeats safety-critical checks so direct operator calls cannot bypass
the contract. CUDA launch errors are surfaced at the operator boundary using
the repository's established checking mechanism.

Error messages should name:

- the requested backend;
- the unsupported property and observed value;
- the supported alternatives where concise.

For example, requesting `fa2` with BF16 should report that formal FA2 currently
requires FP16. It must not quietly execute `cuda_rowwise`.

## Correctness Tests

Tests are layered so meaningful validation is possible even on a CPU-only
machine.

CPU/schema tests cover:

- all backend strings are accepted or rejected by the correct family;
- explicit backends never take the `auto` fallback path;
- deprecated `cuda` resolves only to `cuda_rowwise` with a warning;
- FA1/FA2 reject CPU, wrong dtype, wrong rank, non-contiguous tensors,
  inconsistent dimensions, explicit masks, and autograd requests;
- extension schemas and capability metadata include both formal operators;
- source/build manifests include both implementation files;
- experimental backward sources are not registered.

CUDA correctness tests run FA1 and FA2 on identical FP16 inputs and compare to
a higher-precision reference for:

- causal and non-causal attention;
- equal query/key sequence lengths;
- shorter query than key with right-aligned causal coordinates;
- sequence, head-dimension, and value-dimension tails;
- multiple batch and head counts;
- inputs that stress stable softmax with large positive and negative scores;
- output shape, dtype, contiguity, and device.

Tolerance is chosen from measured FP16-storage/FP32-accumulation error and
recorded in one shared test helper. FA1 and FA2 are compared to the reference,
not merely to each other, so identical bugs cannot pass by agreement.

Supported zero-length behavior must be decided consistently at the facade. If
the native kernels do not support an empty sequence in this milestone, both
explicit backends reject it with the same clear contract rather than relying on
an invalid launch.

## Performance Comparison

Benchmarks compare FA1 and FA2 using:

- identical tensors, shapes, scale, and causal mode;
- the same arithmetic basis and tile-size policy;
- explicit warm-up and synchronized timing;
- multiple iterations with raw samples plus median or percentile summaries;
- shapes that vary batch, head count, and sequence length.

Results report environment, dtype, dimensions, and backend name. Tests do not
assert that FA2 is always faster: small shapes, tile choices, a teaching-first
FA1 implementation, and a single available GPU can all affect ordering. The
benchmark's purpose is to connect observed behavior to launch parallelism,
memory traffic, and work partition without manufacturing a performance claim.

## Documentation

The attention chapter should be updated after implementation to include:

- the common semantic contract;
- a four-level FA1/FA2 explanation: loop order, thread-block grid, warp work
  partition, and online-softmax/output recurrence;
- runnable backend-selection examples;
- capability and forward-only limitations;
- benchmark methodology and interpretation;
- links from the formal source files to the two papers.

The repository backend matrix must distinguish implemented, experimental, and
planned paths. Experimental names are never presented as validated backends.

## Compatibility and Rollout

The change is delivered in small, reviewable phases:

1. add strict family-scoped backend dispatch and compatibility aliases;
2. add common attention validation and formal FA1 forward;
3. add formal FA2 forward on the same numerical basis;
4. add correctness and benchmark coverage;
5. expose and rename the remaining production `csrc` paths family by family;
6. update the course/repository documentation to match measured behavior.

Existing default callers continue to use `auto`. Existing explicit
`backend="cuda"` attention callers receive a deprecation path to
`cuda_rowwise`; no existing caller is silently redirected to FA1 or FA2.

## Future FA3 and FA4 Work

Future variants should extend the same semantic facade with explicit backend
names and capability metadata. Their production files should preserve the main
algorithmic distinctions in readable source just as FA1 and FA2 do.

FA3 and FA4 are not added as empty enums, aliases, or documentation claims.
Their contracts will be designed from primary sources and actual hardware
requirements when implementation begins.

## Non-goals

- Do not implement or register attention backward in this milestone.
- Do not modify or delete the original experimental attention files.
- Do not claim drop-in support for arbitrary PyTorch attention masks or layouts.
- Do not add Tensor Core acceleration to only one formal variant during the
  initial algorithm comparison.
- Do not present single-GPU measurements as distributed scalability evidence.
- Do not expose incomplete helpers as if they were complete semantic backends.
- Do not create placeholder FA3, FA4, or NVSHMEM implementations.
