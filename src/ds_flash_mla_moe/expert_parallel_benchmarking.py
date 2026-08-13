"""Reproducible multi-process benchmarks for Expert Parallel MoE."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
from torch import Tensor

from .benchmarking import (
    _dtype_from_name,
    _environment_metadata,
    _verification_tolerances,
    summarize_latencies,
)
from .expert_benchmarking import (
    expert_grouped_tile_model,
    expert_initialization_model,
    expert_native_numeric_model,
)
from .expert_parallel import (
    ExpertComputeBackend,
    ExpertParallelTrace,
    _gather_rank_major_chunk,
    _rank_major_chunk_plan,
    deepseek_moe_expert_parallel_reference,
)
from .moe import deepseek_moe_reference
from .route_ops import RouteBackend
from .router_ops import RouterBackend
from .symmetric_memory import symmetric_moe_buffer_model_from_routes


@dataclass(frozen=True)
class ExpertParallelBenchmarkConfig:
    """Shape, routing, measurement, and verification settings for one EP run."""

    tokens_per_rank: int = 32
    token_skew: int = 0
    model_dim: int = 64
    hidden_dim: int = 128
    shared_experts: int = 0
    experts: int = 4
    topk: int = 2
    n_groups: int = 1
    topk_groups: int | None = None
    hot_expert_bias: float = 0.0
    capacity_factor: float = 1.0
    symmetric_cell_capacity: int | None = None
    dtype: str = "float32"
    backend: str = "gloo"
    router_backend: RouterBackend = "auto"
    route_backend: RouteBackend = "auto"
    expert_backend: ExpertComputeBackend = "loop"
    pipeline_chunks: int = 1
    warmup: int = 2
    iterations: int = 5
    seed: int = 0
    backward: bool = False
    verify: bool = True

    def validate(self, world_size: int) -> None:
        if not isinstance(self.pipeline_chunks, int) or isinstance(self.pipeline_chunks, bool):
            raise TypeError("pipeline_chunks must be an integer")
        positive = (
            self.model_dim,
            self.hidden_dim,
            self.experts,
            self.topk,
            self.n_groups,
            self.pipeline_chunks,
            self.iterations,
            world_size,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("EP dimensions, iterations, and world_size must be positive")
        if self.tokens_per_rank < 0:
            raise ValueError("tokens_per_rank must be non-negative")
        if self.token_skew < 0:
            raise ValueError("token_skew must be non-negative")
        if self.shared_experts < 0:
            raise ValueError("shared_experts must be non-negative")
        total_tokens = sum(
            self.tokens_per_rank + rank * self.token_skew for rank in range(world_size)
        )
        if total_tokens <= 0:
            raise ValueError("the EP benchmark requires at least one token across all ranks")
        if not math.isfinite(self.hot_expert_bias) or self.hot_expert_bias < 0:
            raise ValueError("hot_expert_bias must be finite and non-negative")
        if not math.isfinite(self.capacity_factor) or self.capacity_factor <= 0:
            raise ValueError("capacity_factor must be finite and positive")
        if self.symmetric_cell_capacity is not None and self.symmetric_cell_capacity <= 0:
            raise ValueError("symmetric_cell_capacity must be positive when provided")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.experts % self.n_groups != 0:
            raise ValueError("experts must be divisible by n_groups")
        topk_groups = self.n_groups if self.topk_groups is None else self.topk_groups
        if not 1 <= topk_groups <= self.n_groups:
            raise ValueError("topk_groups must be in [1, n_groups]")
        if not 1 <= self.topk <= self.experts:
            raise ValueError("topk must be in [1, experts]")
        if self.topk > topk_groups * (self.experts // self.n_groups):
            raise ValueError("topk exceeds the experts retained by group selection")
        if self.dtype not in {"float16", "bfloat16", "float32", "float64"}:
            raise ValueError("unsupported EP benchmark dtype")
        if self.backend not in {"gloo", "nccl"}:
            raise ValueError("backend must be gloo or nccl")
        if self.route_backend not in {"auto", "cuda", "reference"}:
            raise ValueError("route_backend must be auto, cuda, or reference")
        if self.router_backend not in {"auto", "cuda", "reference"}:
            raise ValueError("router_backend must be auto, cuda, or reference")
        if self.expert_backend not in {"loop", "padded", "cuda"}:
            raise ValueError("expert_backend must be loop, padded, or cuda")
        if self.backend == "gloo" and self.dtype in {"float16", "bfloat16"}:
            raise ValueError("the Gloo benchmark supports float32 and float64")
        if self.route_backend == "cuda" and (self.backend != "nccl" or self.dtype != "float32"):
            raise ValueError("CUDA route primitives require NCCL/CUDA float32 benchmark tensors")
        if self.router_backend == "cuda" and (self.backend != "nccl" or self.dtype != "float32"):
            raise ValueError("CUDA grouped router requires NCCL/CUDA float32 benchmark tensors")
        if self.expert_backend == "cuda" and (
            self.backend != "nccl" or self.dtype not in {"float16", "float32"}
        ):
            raise ValueError("CUDA expert compute requires NCCL/CUDA float16 or float32 tensors")
        if self.pipeline_chunks > 1 and self.backend != "nccl":
            raise ValueError("pipeline_chunks > 1 requires the NCCL backend")


class _ExpertParallelWorkload(NamedTuple):
    x: Tensor
    gate: Tensor
    score_bias: Tensor | None
    local_w1: Tensor
    local_w2: Tensor
    local_w3: Tensor
    shared_w1: Tensor | None
    shared_w2: Tensor | None
    shared_w3: Tensor | None
    expert_owner: Tensor
    local_expert_ids: Tensor
    global_x: Tensor
    global_gate: Tensor
    global_w1: Tensor
    global_w2: Tensor
    global_w3: Tensor
    token_counts: tuple[int, ...]
    local_token_slice: slice


def expert_parallel_work_estimate(
    config: ExpertParallelBenchmarkConfig,
    *,
    world_size: int,
    counts_matrix: list[list[int]],
) -> dict[str, int]:
    """Estimate matrix FLOPs and protocol payload from measured route counts."""

    config.validate(world_size)
    if len(counts_matrix) != world_size or any(len(row) != world_size for row in counts_matrix):
        raise ValueError("counts_matrix must have shape [world_size, world_size]")
    if any(value < 0 for row in counts_matrix for value in row):
        raise ValueError("counts_matrix cannot contain negative counts")

    expected_row_counts = [
        (config.tokens_per_rank + rank * config.token_skew) * config.topk
        for rank in range(world_size)
    ]
    actual_row_counts = [sum(row) for row in counts_matrix]
    if actual_row_counts != expected_row_counts:
        raise ValueError("counts_matrix row sums do not match the configured routes per rank")

    total_tokens = sum(
        config.tokens_per_rank + rank * config.token_skew for rank in range(world_size)
    )
    total_routes = sum(actual_row_counts)
    router_matrix_flops = 2 * total_tokens * config.model_dim * config.experts
    routed_expert_matrix_flops = 6 * total_routes * config.model_dim * config.hidden_dim
    shared_expert_matrix_flops = (
        6 * total_tokens * config.model_dim * config.hidden_dim * config.shared_experts
    )
    matrix_flops = router_matrix_flops + routed_expert_matrix_flops + shared_expert_matrix_flops
    cross_rank_rows = sum(
        counts_matrix[source][destination]
        for source in range(world_size)
        for destination in range(world_size)
        if source != destination
    )
    element_size = torch.empty((), dtype=_dtype_from_name(config.dtype)).element_size()
    forward_activation_bytes = 2 * cross_rank_rows * config.model_dim * element_size
    measured_activation_bytes = forward_activation_bytes * (2 if config.backward else 1)
    return {
        "forward_matrix_flops": matrix_flops,
        "forward_router_matrix_flops": router_matrix_flops,
        "forward_routed_expert_matrix_flops": routed_expert_matrix_flops,
        "forward_shared_expert_matrix_flops": shared_expert_matrix_flops,
        "total_tokens": total_tokens,
        "total_routes": total_routes,
        "cross_rank_route_rows": cross_rank_rows,
        "forward_cross_rank_activation_bytes": forward_activation_bytes,
        "measured_step_cross_rank_activation_bytes": measured_activation_bytes,
        "forward_cross_rank_expert_id_bytes": cross_rank_rows
        * torch.tensor([], dtype=torch.long).element_size(),
    }


def _load_distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        raise ValueError("a load distribution must contain at least one value")
    if any(value < 0 for value in values):
        raise ValueError("load values cannot be negative")
    total = sum(values)
    mean = total / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "values": values,
        "total": total,
        "mean": mean,
        "minimum": min(values),
        "maximum": max(values),
        "peak_to_mean": max(values) / mean if mean else 1.0,
        "coefficient_of_variation": math.sqrt(variance) / mean if mean else 0.0,
        "zero_load_count": sum(value == 0 for value in values),
    }


def summarize_rank_latency_samples(
    rank_samples_ms: list[list[float]],
) -> dict[str, Any]:
    """Summarize a rectangular ``[iteration, rank]`` latency sample matrix."""

    if not rank_samples_ms or not rank_samples_ms[0]:
        raise ValueError("rank latency samples must contain an iteration and a rank")
    rank_count = len(rank_samples_ms[0])
    if any(len(iteration) != rank_count for iteration in rank_samples_ms):
        raise ValueError("rank latency samples must be rectangular")
    normalized = [[float(value) for value in iteration] for iteration in rank_samples_ms]
    if any(
        not math.isfinite(value) or value < 0 for iteration in normalized for value in iteration
    ):
        raise ValueError("rank latency samples must be finite and non-negative")

    per_rank = []
    for rank in range(rank_count):
        values = [iteration[rank] for iteration in normalized]
        per_rank.append(
            {
                "rank": rank,
                "latency": summarize_latencies(values),
                "raw_samples_ms": values,
            }
        )
    rank_mean_samples = [sum(iteration) / rank_count for iteration in normalized]
    rank_max_samples = [max(iteration) for iteration in normalized]
    return {
        "rank_count": rank_count,
        "iteration_count": len(normalized),
        "per_iteration_rank_samples_ms": normalized,
        "per_rank": per_rank,
        "rank_mean": {
            "latency": summarize_latencies(rank_mean_samples),
            "raw_samples_ms": rank_mean_samples,
        },
        "rank_max": {
            "latency": summarize_latencies(rank_max_samples),
            "raw_samples_ms": rank_max_samples,
        },
    }


def expert_parallel_load_analysis(
    config: ExpertParallelBenchmarkConfig,
    *,
    world_size: int,
    counts_matrix: list[list[int]],
    expert_counts: list[int],
    expert_owner: list[int] | None = None,
) -> dict[str, Any]:
    """Quantify rank/expert skew and model route-row capacity outcomes.

    The capacity model is analytical: it does not change routing or drop rows
    from the benchmarked operation.
    """

    work = expert_parallel_work_estimate(
        config,
        world_size=world_size,
        counts_matrix=counts_matrix,
    )
    if len(expert_counts) != config.experts:
        raise ValueError("expert_counts must contain one value per expert")
    if any(value < 0 for value in expert_counts):
        raise ValueError("expert_counts cannot contain negative counts")
    if sum(expert_counts) != work["total_routes"]:
        raise ValueError("expert_counts must account for every routed row")

    if expert_owner is None:
        expert_owner = [expert_id % world_size for expert_id in range(config.experts)]
    if len(expert_owner) != config.experts:
        raise ValueError("expert_owner must contain one rank per expert")
    if any(owner < 0 or owner >= world_size for owner in expert_owner):
        raise ValueError("expert_owner contains a rank outside world_size")

    rank_send_rows = [sum(row) for row in counts_matrix]
    rank_receive_rows = [
        sum(counts_matrix[source][destination] for source in range(world_size))
        for destination in range(world_size)
    ]
    rank_cross_send_rows = [
        sum(value for destination, value in enumerate(row) if destination != source)
        for source, row in enumerate(counts_matrix)
    ]
    rank_cross_receive_rows = [
        sum(
            counts_matrix[source][destination]
            for source in range(world_size)
            if source != destination
        )
        for destination in range(world_size)
    ]
    owner_rank_expert_rows = [0 for _ in range(world_size)]
    for count, owner in zip(expert_counts, expert_owner, strict=True):
        owner_rank_expert_rows[owner] += count
    if owner_rank_expert_rows != rank_receive_rows:
        raise ValueError("expert counts do not match destination-rank receive counts")

    local_capacities: list[int] = []
    local_allocated_slots: list[int] = []
    for rank in range(world_size):
        owned_counts = [
            count for count, owner in zip(expert_counts, expert_owner, strict=True) if owner == rank
        ]
        capacity = max(owned_counts, default=0)
        local_capacities.append(capacity)
        local_allocated_slots.append(capacity * len(owned_counts))
    padded_slots = sum(local_allocated_slots)

    mean_expert_rows = work["total_routes"] / config.experts
    uniform_capacity = math.ceil(mean_expert_rows * config.capacity_factor)
    accepted_rows = sum(min(count, uniform_capacity) for count in expert_counts)
    dropped_rows = work["total_routes"] - accepted_rows
    allocated_capacity_slots = uniform_capacity * config.experts
    padding_rows = allocated_capacity_slots - accepted_rows
    maximum_expert_rows = max(expert_counts)
    return {
        "rank_send_route_rows": _load_distribution(rank_send_rows),
        "rank_receive_route_rows": _load_distribution(rank_receive_rows),
        "rank_cross_send_route_rows": _load_distribution(rank_cross_send_rows),
        "rank_cross_receive_route_rows": _load_distribution(rank_cross_receive_rows),
        "expert_route_rows": _load_distribution(expert_counts),
        "owner_local_padding_model": {
            "capacity_per_rank": local_capacities,
            "allocated_expert_slots_per_rank": local_allocated_slots,
            "allocated_expert_slots": padded_slots,
            "padding_rows": padded_slots - work["total_routes"],
            "utilization": work["total_routes"] / padded_slots if padded_slots else 1.0,
        },
        "uniform_capacity_model": {
            "capacity_factor": config.capacity_factor,
            "mean_route_rows_per_expert": mean_expert_rows,
            "capacity_per_expert": uniform_capacity,
            "minimum_capacity_per_expert_without_drop": maximum_expert_rows,
            "capacity_factor_sufficient_without_drop": (maximum_expert_rows / mean_expert_rows),
            "accepted_route_rows": accepted_rows,
            "dropped_route_rows": dropped_rows,
            "dropped_route_fraction": dropped_rows / work["total_routes"],
            "allocated_expert_slots": allocated_capacity_slots,
            "padding_rows": padding_rows,
            "utilization": accepted_rows / allocated_capacity_slots,
            "analytical_only": True,
        },
    }


def expert_parallel_overlap_model(
    stage_latency: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build an optimistic communication/compute overlap model from stage profiles."""

    stage_names = ("dispatch", "expert_compute", "restore")
    rank_samples: dict[str, list[list[float]]] = {}
    for stage in stage_names:
        try:
            values = [
                [float(value) for value in iteration]
                for iteration in stage_latency[stage]["rank_raw_samples_ms"]
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"missing valid rank samples for stage {stage}") from error
        if (
            not values
            or not values[0]
            or any(len(iteration) != len(values[0]) for iteration in values)
            or any(
                not math.isfinite(value) or value < 0 for iteration in values for value in iteration
            )
        ):
            raise ValueError(f"stage {stage} rank samples must be rectangular and non-negative")
        rank_samples[stage] = values
    sample_count = len(rank_samples[stage_names[0]])
    rank_count = len(rank_samples[stage_names[0]][0])
    if any(len(rank_samples[stage]) != sample_count for stage in stage_names):
        raise ValueError("overlap-model stage sample counts must match")
    if any(
        len(iteration) != rank_count for stage in stage_names for iteration in rank_samples[stage]
    ):
        raise ValueError("overlap-model rank counts must match")

    serialized = [
        max(
            sum(rank_samples[stage][sample][rank] for stage in stage_names)
            for rank in range(rank_count)
        )
        for sample in range(sample_count)
    ]
    communication = [
        max(
            rank_samples["dispatch"][sample][rank] + rank_samples["restore"][sample][rank]
            for rank in range(rank_count)
        )
        for sample in range(sample_count)
    ]
    compute = [max(rank_samples["expert_compute"][sample]) for sample in range(sample_count)]
    optimistic_lower_bound = [
        max(
            max(
                rank_samples["dispatch"][sample][rank] + rank_samples["restore"][sample][rank],
                rank_samples["expert_compute"][sample][rank],
            )
            for rank in range(rank_count)
        )
        for sample in range(sample_count)
    ]
    overlap_opportunity = [
        serial - lower for serial, lower in zip(serialized, optimistic_lower_bound, strict=True)
    ]
    median_serialized = float(summarize_latencies(serialized)["median_ms"])
    median_lower_bound = float(summarize_latencies(optimistic_lower_bound)["median_ms"])
    return {
        "model": "infinite-chunk communication/compute steady-state lower bound",
        "stages": list(stage_names),
        "rank_count": rank_count,
        "resource_groups": {
            "communication": ["dispatch", "restore"],
            "compute": ["expert_compute"],
        },
        "communication_stage_max_latency": {
            "latency": summarize_latencies(communication),
            "raw_samples_ms": communication,
        },
        "expert_compute_stage_max_latency": {
            "latency": summarize_latencies(compute),
            "raw_samples_ms": compute,
        },
        "serialized_stage_max_core_latency": {
            "latency": summarize_latencies(serialized),
            "raw_samples_ms": serialized,
        },
        "optimistic_steady_state_lower_bound": {
            "latency": summarize_latencies(optimistic_lower_bound),
            "raw_samples_ms": optimistic_lower_bound,
        },
        "maximum_overlap_opportunity": {
            "latency": summarize_latencies(overlap_opportunity),
            "raw_samples_ms": overlap_opportunity,
        },
        "derived_at_median": {
            "optimistic_pipeline_speedup": (
                median_serialized / median_lower_bound if median_lower_bound else 1.0
            ),
            "maximum_overlap_fraction": (
                (median_serialized - median_lower_bound) / median_serialized
                if median_serialized
                else 0.0
            ),
        },
        "notes": [
            "resource times are combined per rank before taking each sample's rank maximum",
            "dispatch and restore are conservatively assigned to one communication resource",
            "the lower bound excludes chunk fill/drain, dependencies, and scheduling overhead",
            "the model is not a measured overlapped execution time",
        ],
    }


def expert_parallel_chunked_tile_model(
    rank_chunk_expert_counts: list[list[list[int]]],
    *,
    model_dim: int,
    hidden_dim: int,
) -> dict[str, Any]:
    """Aggregate the exact grouped-GEMM tasks created by EP chunking."""

    if not rank_chunk_expert_counts or not rank_chunk_expert_counts[0]:
        raise ValueError("chunked tile counts require a rank and a chunk")
    if model_dim <= 0 or hidden_dim <= 0:
        raise ValueError("model_dim and hidden_dim must be positive")
    pipeline_chunks = len(rank_chunk_expert_counts[0])
    if any(len(rank_chunks) != pipeline_chunks for rank_chunks in rank_chunk_expert_counts):
        raise ValueError("every rank must report the same number of pipeline chunks")

    aggregate_keys = (
        "active_route_rows",
        "active_expert_row_tiles",
        "hidden_projection_tasks",
        "down_projection_tasks",
        "allocated_row_lanes",
        "inactive_tail_row_lanes",
    )
    aggregate = {key: 0 for key in aggregate_keys}
    per_rank_chunks = []
    for rank, raw_rank_chunks in enumerate(rank_chunk_expert_counts):
        expert_width = len(raw_rank_chunks[0])
        if any(len(raw_counts) != expert_width for raw_counts in raw_rank_chunks):
            raise ValueError("a rank's pipeline chunks must cover the same local experts")
        rank_models = []
        for raw_counts in raw_rank_chunks:
            if any(not isinstance(value, int) or isinstance(value, bool) for value in raw_counts):
                raise TypeError("pipeline chunk expert counts must be integers")
            chunk_counts = tuple(raw_counts)
            if any(value < 0 for value in chunk_counts):
                raise ValueError("pipeline chunk expert counts cannot be negative")
            active_routes = sum(chunk_counts)
            if active_routes == 0:
                model = {
                    "expert_counts": list(chunk_counts),
                    "active_route_rows": 0,
                    "analytical_only": True,
                    "tile_shape": [16, 16, 16],
                    "hidden_output_tiles": math.ceil(hidden_dim / 16),
                    "model_output_tiles": math.ceil(model_dim / 16),
                    **{key: 0 for key in aggregate_keys if key != "active_route_rows"},
                    "row_lane_utilization": 1.0,
                }
            else:
                model = {
                    "expert_counts": list(chunk_counts),
                    "active_route_rows": active_routes,
                    **expert_grouped_tile_model(
                        chunk_counts,
                        model_dim=model_dim,
                        hidden_dim=hidden_dim,
                    ),
                }
            rank_models.append(model)
            for key in aggregate_keys:
                aggregate[key] += int(model[key])
        per_rank_chunks.append({"rank": rank, "chunks": rank_models})

    if (
        aggregate["allocated_row_lanes"] - aggregate["inactive_tail_row_lanes"]
        != (aggregate["active_route_rows"])
    ):
        raise RuntimeError("chunked tile lanes do not account for every active route")
    return {
        "pipeline_chunks": pipeline_chunks,
        "chunked_task_counts_are_reported": True,
        "chunked_aggregate": {
            **aggregate,
            "row_lane_utilization": (
                aggregate["active_route_rows"] / aggregate["allocated_row_lanes"]
                if aggregate["allocated_row_lanes"]
                else 1.0
            ),
        },
        "per_rank_chunks": per_rank_chunks,
    }


def _make_workload(
    config: ExpertParallelBenchmarkConfig,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> _ExpertParallelWorkload:
    dtype = _dtype_from_name(config.dtype)
    token_counts = tuple(
        config.tokens_per_rank + source_rank * config.token_skew
        for source_rank in range(world_size)
    )
    token_offsets = [0]
    for count in token_counts:
        token_offsets.append(token_offsets[-1] + count)
    local_token_slice = slice(token_offsets[rank], token_offsets[rank + 1])

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    initialization = expert_initialization_model(
        model_dim=config.model_dim,
        hidden_dim=config.hidden_dim,
    )
    global_x_cpu = (
        torch.randn(token_offsets[-1], config.model_dim, dtype=dtype, generator=generator)
        * initialization["activation_standard_deviation"]
    )
    global_gate_cpu = (
        torch.randn(config.experts, config.model_dim, dtype=dtype, generator=generator)
        * initialization["gate_up_weight_standard_deviation"]
    )
    score_bias_cpu = None
    if config.hot_expert_bias:
        score_bias_cpu = torch.zeros(config.experts, dtype=dtype)
        score_bias_cpu[0] = config.hot_expert_bias
    global_w1_cpu = (
        torch.randn(
            config.experts,
            config.hidden_dim,
            config.model_dim,
            dtype=dtype,
            generator=generator,
        )
        * initialization["gate_up_weight_standard_deviation"]
    )
    global_w2_cpu = (
        torch.randn(
            config.experts,
            config.model_dim,
            config.hidden_dim,
            dtype=dtype,
            generator=generator,
        )
        * initialization["down_weight_standard_deviation"]
    )
    global_w3_cpu = (
        torch.randn(
            config.experts,
            config.hidden_dim,
            config.model_dim,
            dtype=dtype,
            generator=generator,
        )
        * initialization["gate_up_weight_standard_deviation"]
    )
    shared_hidden_dim = config.shared_experts * config.hidden_dim
    shared_w1_cpu = (
        torch.randn(
            shared_hidden_dim,
            config.model_dim,
            dtype=dtype,
            generator=generator,
        )
        * initialization["gate_up_weight_standard_deviation"]
        if config.shared_experts
        else None
    )
    shared_w2_cpu = (
        torch.randn(
            config.model_dim,
            shared_hidden_dim,
            dtype=dtype,
            generator=generator,
        )
        / math.sqrt(shared_hidden_dim)
        if config.shared_experts
        else None
    )
    shared_w3_cpu = (
        torch.randn(
            shared_hidden_dim,
            config.model_dim,
            dtype=dtype,
            generator=generator,
        )
        * initialization["gate_up_weight_standard_deviation"]
        if config.shared_experts
        else None
    )

    global_x = global_x_cpu.to(device)
    global_gate = global_gate_cpu.to(device)
    score_bias = None if score_bias_cpu is None else score_bias_cpu.to(device)
    global_w1 = global_w1_cpu.to(device)
    global_w2 = global_w2_cpu.to(device)
    global_w3 = global_w3_cpu.to(device)
    shared_w1 = None if shared_w1_cpu is None else shared_w1_cpu.to(device)
    shared_w2 = None if shared_w2_cpu is None else shared_w2_cpu.to(device)
    shared_w3 = None if shared_w3_cpu is None else shared_w3_cpu.to(device)
    expert_owner = (torch.arange(config.experts, device=device) % world_size).to(torch.long)
    local_expert_ids = torch.where(expert_owner == rank)[0]
    requires_grad = config.backward
    return _ExpertParallelWorkload(
        x=global_x[local_token_slice].clone().requires_grad_(requires_grad),
        gate=global_gate.clone().requires_grad_(requires_grad),
        score_bias=score_bias,
        local_w1=global_w1[local_expert_ids].clone().requires_grad_(requires_grad),
        local_w2=global_w2[local_expert_ids].clone().requires_grad_(requires_grad),
        local_w3=global_w3[local_expert_ids].clone().requires_grad_(requires_grad),
        shared_w1=(None if shared_w1 is None else shared_w1.clone().requires_grad_(requires_grad)),
        shared_w2=(None if shared_w2 is None else shared_w2.clone().requires_grad_(requires_grad)),
        shared_w3=(None if shared_w3 is None else shared_w3.clone().requires_grad_(requires_grad)),
        expert_owner=expert_owner,
        local_expert_ids=local_expert_ids,
        global_x=global_x,
        global_gate=global_gate,
        global_w1=global_w1,
        global_w2=global_w2,
        global_w3=global_w3,
        token_counts=token_counts,
        local_token_slice=local_token_slice,
    )


def _loss(output: Tensor) -> Tensor:
    compute_dtype = torch.float64 if output.dtype == torch.float64 else torch.float32
    return output.to(compute_dtype).square().sum()


def _zero_grad(workload: _ExpertParallelWorkload) -> None:
    for tensor in (
        workload.x,
        workload.gate,
        workload.local_w1,
        workload.local_w2,
        workload.local_w3,
        workload.shared_w1,
        workload.shared_w2,
        workload.shared_w3,
    ):
        if tensor is not None:
            tensor.grad = None


def _run_operation(
    config: ExpertParallelBenchmarkConfig,
    workload: _ExpertParallelWorkload,
    group: dist.ProcessGroup | None,
    *,
    return_trace: bool = False,
    stage_observer: Callable[[str], None] | None = None,
) -> Tensor | tuple[Tensor, ExpertParallelTrace]:
    if config.backward:
        _zero_grad(workload)
    result = deepseek_moe_expert_parallel_reference(
        workload.x,
        workload.gate,
        workload.local_w1,
        workload.local_w2,
        workload.local_w3,
        expert_owner=workload.expert_owner,
        topk=config.topk,
        n_groups=config.n_groups,
        topk_groups=config.topk_groups,
        score_bias=workload.score_bias,
        router_backend=config.router_backend,
        shared_w1=workload.shared_w1,
        shared_w2=workload.shared_w2,
        shared_w3=workload.shared_w3,
        group=group,
        route_backend=config.route_backend,
        expert_backend=config.expert_backend,
        pipeline_chunks=config.pipeline_chunks,
        return_trace=return_trace,
        stage_observer=stage_observer,
    )
    output = result[0] if return_trace else result
    if config.backward:
        _loss(output).backward()
        if stage_observer is not None:
            stage_observer("backward")
    return result


def _normalized_error(actual: Tensor, expected: Tensor, *, rtol: float, atol: float) -> Tensor:
    if actual.shape != expected.shape:
        raise ValueError(f"cannot compare shapes {tuple(actual.shape)} and {tuple(expected.shape)}")
    if actual.numel() == 0:
        return torch.zeros(3, device=actual.device, dtype=torch.float64)
    actual64 = actual.detach().to(torch.float64)
    expected64 = expected.detach().to(torch.float64)
    finite = torch.isfinite(actual64) & torch.isfinite(expected64)
    infinite = torch.full_like(actual64, torch.inf)
    difference = torch.where(finite, (actual64 - expected64).abs(), infinite)
    denominator = expected64.abs().clamp_min(torch.finfo(torch.float64).tiny)
    tolerance = atol + rtol * expected64.abs()
    return torch.stack(
        (
            difference.max(),
            torch.where(finite, difference / denominator, infinite).max(),
            torch.where(finite, difference / tolerance, infinite).max(),
        )
    )


def _gradient_or_zeros(tensor: Tensor) -> Tensor:
    return torch.zeros_like(tensor) if tensor.grad is None else tensor.grad


def _verify(
    config: ExpertParallelBenchmarkConfig,
    workload: _ExpertParallelWorkload,
    actual: Tensor,
    *,
    group: dist.ProcessGroup | None,
) -> dict[str, Any]:
    expected_x = workload.global_x.detach().clone().requires_grad_(config.backward)
    expected_gate = workload.global_gate.detach().clone().requires_grad_(config.backward)
    expected_w1 = workload.global_w1.detach().clone().requires_grad_(config.backward)
    expected_w2 = workload.global_w2.detach().clone().requires_grad_(config.backward)
    expected_w3 = workload.global_w3.detach().clone().requires_grad_(config.backward)
    expected_shared_w1 = (
        None
        if workload.shared_w1 is None
        else workload.shared_w1.detach().clone().requires_grad_(config.backward)
    )
    expected_shared_w2 = (
        None
        if workload.shared_w2 is None
        else workload.shared_w2.detach().clone().requires_grad_(config.backward)
    )
    expected_shared_w3 = (
        None
        if workload.shared_w3 is None
        else workload.shared_w3.detach().clone().requires_grad_(config.backward)
    )
    expected = deepseek_moe_reference(
        expected_x,
        expected_gate,
        expected_w1,
        expected_w2,
        expected_w3,
        topk=config.topk,
        n_groups=config.n_groups,
        topk_groups=config.topk_groups,
        score_bias=workload.score_bias,
        shared_w1=expected_shared_w1,
        shared_w2=expected_shared_w2,
        shared_w3=expected_shared_w3,
    )
    if config.backward:
        _loss(expected).backward()

    rtol, atol = _verification_tolerances(actual.dtype)
    if config.expert_backend == "cuda" and actual.dtype == torch.float16:
        rtol = max(rtol, 3e-2)
        atol = max(atol, 3e-2)
    output_error = _normalized_error(
        actual,
        expected[workload.local_token_slice],
        rtol=rtol,
        atol=atol,
    )
    dist.all_reduce(output_error, op=dist.ReduceOp.MAX, group=group)

    gradient_error = torch.zeros_like(output_error)
    if config.backward:
        aggregate_gate_grad = _gradient_or_zeros(workload.gate).clone()
        dist.all_reduce(aggregate_gate_grad, group=group)
        comparisons = [
            (_gradient_or_zeros(workload.x), expected_x.grad[workload.local_token_slice]),
            (aggregate_gate_grad, expected_gate.grad),
            (_gradient_or_zeros(workload.local_w1), expected_w1.grad[workload.local_expert_ids]),
            (_gradient_or_zeros(workload.local_w2), expected_w2.grad[workload.local_expert_ids]),
            (_gradient_or_zeros(workload.local_w3), expected_w3.grad[workload.local_expert_ids]),
        ]
        shared_pairs = (
            (workload.shared_w1, expected_shared_w1),
            (workload.shared_w2, expected_shared_w2),
            (workload.shared_w3, expected_shared_w3),
        )
        for replicated, expected_replicated in shared_pairs:
            if replicated is None or expected_replicated is None:
                continue
            aggregate_shared_grad = _gradient_or_zeros(replicated).clone()
            dist.all_reduce(aggregate_shared_grad, group=group)
            comparisons.append((aggregate_shared_grad, expected_replicated.grad))
        gradient_error = torch.stack(
            [
                _normalized_error(actual_grad, expected_grad, rtol=rtol, atol=atol)
                for actual_grad, expected_grad in comparisons
            ]
        ).amax(dim=0)
        dist.all_reduce(gradient_error, op=dist.ReduceOp.MAX, group=group)

    if output_error[2].item() > 1.0:
        raise AssertionError(f"distributed EP output verification failed: {output_error.tolist()}")
    if config.backward and gradient_error[2].item() > 1.0:
        raise AssertionError(
            f"distributed EP gradient verification failed: {gradient_error.tolist()}"
        )
    return {
        "performed": True,
        "reference": "deepseek_moe_reference",
        "rtol": rtol,
        "atol": atol,
        "output": {
            "max_absolute_error": output_error[0].item(),
            "max_relative_error": output_error[1].item(),
            "max_tolerance_ratio": output_error[2].item(),
        },
        "gradients": {
            "performed": config.backward,
            "max_absolute_error": gradient_error[0].item(),
            "max_relative_error": gradient_error[1].item(),
            "max_tolerance_ratio": gradient_error[2].item(),
        },
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    config: ExpertParallelBenchmarkConfig,
    workload: _ExpertParallelWorkload,
    *,
    device: torch.device,
    group: dist.ProcessGroup | None,
) -> tuple[list[float], list[list[float]]]:
    world_size = dist.get_world_size(group=group)
    for _ in range(config.warmup):
        dist.barrier(group=group)
        _run_operation(config, workload, group)
        _synchronize(device)

    samples: list[float] = []
    rank_samples: list[list[float]] = []
    for _ in range(config.iterations):
        _synchronize(device)
        dist.barrier(group=group)
        start = time.perf_counter_ns()
        _run_operation(config, workload, group)
        _synchronize(device)
        local_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        local_sample = torch.tensor([local_ms], device=device, dtype=torch.float64)
        gathered_samples = [torch.empty_like(local_sample) for _ in range(world_size)]
        dist.all_gather(gathered_samples, local_sample, group=group)
        rank_values = [sample.item() for sample in gathered_samples]
        rank_samples.append(rank_values)
        samples.append(max(rank_values))
    return samples, rank_samples


def _measure_stages(
    config: ExpertParallelBenchmarkConfig,
    workload: _ExpertParallelWorkload,
    *,
    device: torch.device,
    group: dist.ProcessGroup | None,
) -> dict[str, dict[str, Any]]:
    world_size = dist.get_world_size(group=group)
    stage_names = ["route_and_pack", "exchange_counts"]
    if config.pipeline_chunks == 1:
        stage_names.extend(("dispatch", "expert_compute", "restore"))
    else:
        stage_names.append("pipelined_core")
    stage_names.append("combine")
    if config.shared_experts:
        stage_names.append("shared_expert")
    if config.backward:
        stage_names.append("backward")
    samples: dict[str, list[float]] = {stage: [] for stage in stage_names}
    rank_samples: dict[str, list[list[float]]] = {stage: [] for stage in stage_names}
    for _ in range(config.iterations):
        _synchronize(device)
        dist.barrier(group=group)
        timestamps: dict[str, int] = {}
        start = time.perf_counter_ns()

        def observe(stage: str, current_timestamps: dict[str, int] = timestamps) -> None:
            _synchronize(device)
            current_timestamps[stage] = time.perf_counter_ns()

        _run_operation(
            config,
            workload,
            group,
            stage_observer=observe,
        )
        previous = start
        for stage in stage_names:
            if stage not in timestamps:
                raise RuntimeError(f"the EP operation did not report stage {stage}")
            local_ms = (timestamps[stage] - previous) / 1_000_000.0
            local_sample = torch.tensor([local_ms], device=device, dtype=torch.float64)
            gathered_samples = [torch.empty_like(local_sample) for _ in range(world_size)]
            dist.all_gather(gathered_samples, local_sample, group=group)
            rank_values = [sample.item() for sample in gathered_samples]
            rank_samples[stage].append(rank_values)
            samples[stage].append(max(rank_values))
            previous = timestamps[stage]
    return {
        stage: {
            "latency": summarize_latencies(values),
            "raw_samples_ms": values,
            "rank_raw_samples_ms": rank_samples[stage],
        }
        for stage, values in samples.items()
    }


def benchmark_expert_parallel(
    config: ExpertParallelBenchmarkConfig,
    *,
    device: torch.device,
    local_rank: int,
    group: dist.ProcessGroup | None = None,
) -> dict[str, Any] | None:
    """Benchmark an initialized process group and return a report on rank zero."""

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before the EP benchmark")
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    config.validate(world_size)
    if str(dist.get_backend(group=group)).lower() != config.backend:
        raise ValueError("configuration backend does not match the initialized process group")
    if config.backend == "gloo" and device.type != "cpu":
        raise ValueError("Gloo benchmark device must be CPU")
    if config.backend == "nccl" and device.type != "cuda":
        raise ValueError("NCCL benchmark device must be CUDA")

    workload = _make_workload(
        config,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    trace_result = _run_operation(config, workload, group, return_trace=True)
    actual, trace = trace_result
    verification = (
        _verify(config, workload, actual, group=group) if config.verify else {"performed": False}
    )

    gathered_send_counts = [torch.empty_like(trace.packed.rank_counts) for _ in range(world_size)]
    dist.all_gather(gathered_send_counts, trace.packed.rank_counts, group=group)
    counts_matrix = [counts.detach().cpu().tolist() for counts in gathered_send_counts]
    local_source_expert_counts = torch.bincount(
        trace.packed.expert_indices,
        minlength=config.experts,
    )
    gathered_source_expert_counts = [
        torch.empty_like(local_source_expert_counts) for _ in range(world_size)
    ]
    dist.all_gather(gathered_source_expert_counts, local_source_expert_counts, group=group)
    source_expert_counts_matrix = [
        counts.detach().cpu().tolist() for counts in gathered_source_expert_counts
    ]
    if [sum(row) for row in source_expert_counts_matrix] != [sum(row) for row in counts_matrix]:
        raise RuntimeError("source-expert counts disagree with destination-rank counts")
    samples, rank_samples = _measure(config, workload, device=device, group=group)
    stage_latency = _measure_stages(config, workload, device=device, group=group)
    latency = summarize_latencies(samples)
    rank_latency = summarize_rank_latency_samples(rank_samples)
    if rank_latency["rank_max"]["raw_samples_ms"] != samples:
        raise RuntimeError("rank latency summary disagrees with global maximum samples")
    work = expert_parallel_work_estimate(
        config,
        world_size=world_size,
        counts_matrix=counts_matrix,
    )

    local_details = {
        "rank": rank,
        "local_rank": local_rank,
        "device": str(device),
        "tokens": workload.token_counts[rank],
        "local_expert_ids": workload.local_expert_ids.detach().cpu().tolist(),
        "send_counts": trace.packed.rank_counts.detach().cpu().tolist(),
        "recv_counts": trace.recv_counts.detach().cpu().tolist(),
        "received_routes": trace.received_expert_indices.numel(),
        "environment": _environment_metadata(device),
    }
    local_expert_counts = torch.bincount(
        trace.received_expert_indices,
        minlength=config.experts,
    )[workload.local_expert_ids]
    local_capacity = int(local_expert_counts.max()) if local_expert_counts.numel() else 0
    local_details["expert_counts"] = local_expert_counts.detach().cpu().tolist()
    local_details["padded_capacity_per_expert"] = local_capacity
    local_details["padded_expert_slots"] = local_capacity * workload.local_expert_ids.numel()
    if config.pipeline_chunks > 1:
        local_recv_splits = tuple(int(value) for value in trace.recv_counts.detach().cpu().tolist())
        receive_plan = _rank_major_chunk_plan(local_recv_splits, config.pipeline_chunks)
        local_details["chunk_expert_counts"] = [
            torch.bincount(
                _gather_rank_major_chunk(trace.received_expert_indices, ranges),
                minlength=config.experts,
            )[workload.local_expert_ids]
            .detach()
            .cpu()
            .tolist()
            for ranges in receive_plan.ranges
        ]
    rank_details: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(rank_details, local_details, group=group)
    if rank != 0:
        return None

    received_routes = sum(int(details["received_routes"]) for details in rank_details if details)
    padded_expert_slots = sum(
        int(details["padded_expert_slots"]) for details in rank_details if details
    )
    if received_routes != work["total_routes"]:
        raise RuntimeError("received route totals do not match the global routing contract")
    global_expert_counts = [0 for _ in range(config.experts)]
    seen_experts: set[int] = set()
    for details in rank_details:
        if details is None:
            raise RuntimeError("rank metadata gather returned an empty entry")
        expert_ids = [int(value) for value in details["local_expert_ids"]]
        counts = [int(value) for value in details["expert_counts"]]
        if len(expert_ids) != len(counts):
            raise RuntimeError("local expert ids and counts have different lengths")
        for expert_id, count in zip(expert_ids, counts, strict=True):
            if expert_id in seen_experts:
                raise RuntimeError("an expert appears on more than one owner rank")
            seen_experts.add(expert_id)
            global_expert_counts[expert_id] = count
    if seen_experts != set(range(config.experts)):
        raise RuntimeError("rank metadata does not cover every expert")
    load_balance = expert_parallel_load_analysis(
        config,
        world_size=world_size,
        counts_matrix=counts_matrix,
        expert_counts=global_expert_counts,
    )
    modeled_padded_slots = load_balance["owner_local_padding_model"]["allocated_expert_slots"]
    if modeled_padded_slots != padded_expert_slots:
        raise RuntimeError("padded-slot model disagrees with gathered rank metadata")
    element_size = torch.empty((), dtype=_dtype_from_name(config.dtype)).element_size()
    symmetric_buffer_model = symmetric_moe_buffer_model_from_routes(
        source_expert_counts_matrix,
        expert_owner=workload.expert_owner.detach().cpu().tolist(),
        model_dim=config.model_dim,
        element_size_bytes=element_size,
        cell_capacity=config.symmetric_cell_capacity,
    )
    executed_expert_slots = (
        padded_expert_slots if config.expert_backend == "padded" else received_routes
    )
    grouped_tile_model = expert_grouped_tile_model(
        global_expert_counts,
        model_dim=config.model_dim,
        hidden_dim=config.hidden_dim,
    )
    grouped_tile_model["applies_to_backend"] = "cuda"
    if config.pipeline_chunks > 1:
        rank_chunk_expert_counts = []
        for details in rank_details:
            if details is None:
                raise RuntimeError("rank metadata gather returned an empty entry")
            chunk_expert_counts = details.get("chunk_expert_counts")
            if not isinstance(chunk_expert_counts, list) or len(chunk_expert_counts) != (
                config.pipeline_chunks
            ):
                raise RuntimeError("rank metadata does not contain every pipeline chunk")
            expected_rank_counts = tuple(int(value) for value in details["expert_counts"])
            if any(
                len(raw_counts) != len(expected_rank_counts) for raw_counts in chunk_expert_counts
            ):
                raise RuntimeError("pipeline chunk expert-count vectors have inconsistent widths")
            reconstructed_rank_counts = tuple(
                sum(int(chunk[expert]) for chunk in chunk_expert_counts)
                for expert in range(len(expected_rank_counts))
            )
            if reconstructed_rank_counts != expected_rank_counts:
                raise RuntimeError("pipeline chunks do not reconstruct the rank's expert counts")
            rank_chunk_expert_counts.append(
                [[int(value) for value in counts] for counts in chunk_expert_counts]
            )
        chunked_tile_model = expert_parallel_chunked_tile_model(
            rank_chunk_expert_counts,
            model_dim=config.model_dim,
            hidden_dim=config.hidden_dim,
        )
        if chunked_tile_model["chunked_aggregate"]["active_route_rows"] != received_routes:
            raise RuntimeError("chunked tile model does not account for every received route")
        grouped_tile_model.update(chunked_tile_model)
    expert_compute = {
        "active_route_rows": received_routes,
        "theoretical_padded_expert_slots": padded_expert_slots,
        "theoretical_padding_rows": padded_expert_slots - received_routes,
        "theoretical_padding_utilization": (
            received_routes / padded_expert_slots if padded_expert_slots else 1.0
        ),
        "backend_executed_expert_slots": executed_expert_slots,
        "forward_ideal_expert_matrix_flops": (
            6 * received_routes * config.model_dim * config.hidden_dim
        ),
        "forward_backend_executed_expert_matrix_flops": (
            6 * executed_expert_slots * config.model_dim * config.hidden_dim
        ),
        "native_grouped_tile_model": grouped_tile_model,
        "native_numeric_model": expert_native_numeric_model(config.dtype),
    }
    shared_expert_compute = {
        "enabled": bool(config.shared_experts),
        "shared_expert_count": config.shared_experts,
        "effective_hidden_dim": config.shared_experts * config.hidden_dim,
        "active_token_rows": work["total_tokens"],
        "forward_matrix_flops": work["forward_shared_expert_matrix_flops"],
        "replicated_across_ranks": True,
        "gradient_reduction_included_in_measured_step": False,
    }

    median_seconds = float(latency["median_ms"]) / 1000.0
    derived = {
        "forward_matrix_tflops_equivalent_at_median": (
            work["forward_matrix_flops"] / median_seconds / 1e12
        ),
        "aggregate_cross_rank_activation_gb_s_at_median": (
            work["measured_step_cross_rank_activation_bytes"] / median_seconds / 1e9
        ),
    }
    overlap_model = (
        expert_parallel_overlap_model(stage_latency)
        if config.pipeline_chunks == 1
        else {
            "model": "measured chunked asynchronous execution",
            "pipeline_chunks": config.pipeline_chunks,
            "async_pipeline_executed": True,
            "hardware_overlap_verified": False,
            "pipelined_core_latency": stage_latency["pipelined_core"],
            "notes": [
                "dispatch, expert compute, and restore share one asynchronously scheduled stage",
                "stream scheduling permits overlap; a profiler must verify physical concurrency",
                "a serialized baseline must be measured separately with pipeline_chunks=1",
            ],
        }
    )
    return {
        "schema_version": 2,
        "benchmark": "deepseek_moe_expert_parallel_reference",
        "configuration": asdict(config),
        "initialization": {
            **expert_initialization_model(
                model_dim=config.model_dim,
                hidden_dim=config.hidden_dim,
            ),
            "router_weight_standard_deviation": 1.0 / math.sqrt(config.model_dim),
            "shared_down_weight_standard_deviation": (
                1.0 / math.sqrt(config.shared_experts * config.hidden_dim)
                if config.shared_experts
                else None
            ),
        },
        "distributed": {
            "backend": config.backend,
            "world_size": world_size,
            "counts_matrix": counts_matrix,
            "ranks": rank_details,
        },
        "verification": verification,
        "work_estimate": work,
        "load_balance": load_balance,
        "symmetric_buffer_model": symmetric_buffer_model,
        "expert_compute": expert_compute,
        "shared_expert_compute": shared_expert_compute,
        "latency": latency,
        "rank_latency": rank_latency,
        "stage_latency": stage_latency,
        "overlap_model": overlap_model,
        "derived": derived,
        "raw_samples_ms": samples,
        "notes": [
            "each sample is the maximum synchronized operation time across ranks",
            "rank latency samples exclude the all-gather used to collect those samples",
            "stage profiling is a separate measured pass and is not additive across rank maxima",
            "capacity-factor results are analytical and do not drop benchmark routes",
            "the symmetric-buffer report is analytical and does not provide an NVSHMEM backend",
            "matrix FLOPs count the router plus routed/shared three-projection SwiGLU matrices",
            "expert_compute FLOPs describe routed forward matrices even for forward+backward timing",
            "shared expert gradient reduction is verified separately and excluded from timing",
            "cross-rank bytes exclude self-routes, counts, barriers, and collective metadata",
            "the implementation is a correctness reference, not a fused performance kernel",
        ],
    }
