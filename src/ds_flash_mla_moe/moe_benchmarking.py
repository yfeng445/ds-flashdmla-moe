"""Reproducible forward-only benchmarks for the whole DeepSeek-style MoE layer."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import torch
from torch import Tensor

from .benchmarking import (
    _dtype_from_name,
    _environment_metadata,
    _measure_cpu,
    _measure_cuda,
    _verification_tolerances,
    summarize_latencies,
)
from .moe import deepseek_grouped_topk, deepseek_moe_reference
from .moe_ops import MoEBackend, deepseek_moe_forward
from .ops import _operator_has_cuda_kernel

ExecutedMoEBackend = Literal[
    "reference",
    "cuda_staged",
    "cuda_fused",
    "cuda_persistent",
]
_PERSISTENT_SMALL_WORK_ROUTES = 8


@dataclass(frozen=True)
class MoEForwardBenchmarkConfig:
    """Whole-layer shape, routing policy, and forward measurement settings."""

    tokens: int = 128
    model_dim: int = 64
    hidden_dim: int = 128
    experts: int = 8
    topk: int = 2
    n_groups: int = 1
    topk_groups: int | None = None
    dtype: str = "float32"
    device: str = "cpu"
    backend: MoEBackend = "reference"
    seed: int = 0
    warmup: int = 2
    iterations: int = 5
    route_scale: float = 1.0
    score_bias: bool = False
    verify: bool = True

    def validate(self) -> None:
        dimensions = (
            self.tokens,
            self.model_dim,
            self.hidden_dim,
            self.experts,
            self.topk,
            self.n_groups,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("MoE dimensions and routing counts must be positive")
        if self.experts % self.n_groups:
            raise ValueError("experts must be divisible by n_groups")
        effective_topk_groups = self.n_groups if self.topk_groups is None else self.topk_groups
        if not 1 <= effective_topk_groups <= self.n_groups:
            raise ValueError("topk_groups must be in [1, n_groups]")
        retained_experts = effective_topk_groups * (self.experts // self.n_groups)
        if self.topk > retained_experts:
            raise ValueError("topk exceeds the experts retained by group selection")
        if not math.isfinite(self.route_scale):
            raise ValueError("route_scale must be finite")
        if self.iterations <= 0 or self.warmup < 0:
            raise ValueError("iterations must be positive and warmup must be non-negative")
        if self.dtype not in {"float16", "bfloat16", "float32", "float64"}:
            raise ValueError("unsupported whole-layer MoE benchmark dtype")
        if self.backend not in {
            "auto",
            "cuda",
            "cuda_staged",
            "cuda_fused",
            "cuda_persistent",
            "reference",
        }:
            raise ValueError(
                "backend must be auto, cuda, cuda_staged, cuda_fused, cuda_persistent, or reference"
            )
        try:
            device_type = torch.device(self.device).type
        except RuntimeError as error:
            raise ValueError("MoE benchmark device must be a valid torch device") from error
        if device_type not in {"cpu", "cuda"}:
            raise ValueError("MoE benchmark device must be CPU or CUDA")
        if device_type == "cpu" and self.dtype == "float16":
            raise ValueError("float16 CPU MoE compute is not a supported benchmark")
        if self.backend in {"cuda", "cuda_staged", "cuda_fused", "cuda_persistent"} and (
            device_type != "cuda" or self.dtype != "float32"
        ):
            raise ValueError("native CUDA backends require device=cuda and dtype=float32")


def moe_initialization_model(config: MoEForwardBenchmarkConfig) -> dict[str, Any]:
    """Describe the deterministic fan-in-scaled normal input initialization."""

    config.validate()
    return {
        "distribution": "normal",
        "activation_standard_deviation": 1.0,
        "gate_weight_standard_deviation": 1.0 / math.sqrt(config.model_dim),
        "expert_gate_up_weight_standard_deviation": 1.0 / math.sqrt(config.model_dim),
        "expert_down_weight_standard_deviation": 1.0 / math.sqrt(config.hidden_dim),
        "score_bias_standard_deviation": (
            1.0 / math.sqrt(config.experts) if config.score_bias else None
        ),
    }


def _inventory_buffer(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    kind: str,
) -> dict[str, Any]:
    element_size = torch.empty((), dtype=dtype).element_size()
    return {
        "shape": list(shape),
        "dtype": str(dtype).removeprefix("torch."),
        "kind": kind,
        "bytes": math.prod(shape) * element_size,
    }


def _summarize_inventory(
    implementation: str,
    modeled_backend: str,
    executed_backend: ExecutedMoEBackend,
    buffers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "implementation": implementation,
        "modeled_backend": modeled_backend,
        "executed_backend": executed_backend,
        "evidence_kind": "analytical_model",
        "materialized_by_execution": modeled_backend == executed_backend,
        "buffers": buffers,
        "metadata_bytes": sum(
            int(buffer["bytes"]) for buffer in buffers.values() if buffer["kind"] == "metadata"
        ),
        "total_bytes": sum(int(buffer["bytes"]) for buffer in buffers.values()),
    }


def _moe_intermediate_inventories(
    config: MoEForwardBenchmarkConfig,
    *,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    executed_backend: ExecutedMoEBackend,
) -> dict[str, Any]:
    route_rows = config.tokens * config.topk
    experts_plus_one = config.experts + 1
    staged_buffers = {
        "gate_logits": _inventory_buffer(
            (config.tokens, config.experts), compute_dtype, kind="scratch"
        ),
        "dense_scores": _inventory_buffer(
            (config.tokens, config.experts), compute_dtype, kind="scratch"
        ),
        "selected_route_weights": _inventory_buffer(
            (config.tokens, config.topk), storage_dtype, kind="metadata"
        ),
        "selected_expert_indices": _inventory_buffer(
            (config.tokens, config.topk), torch.long, kind="metadata"
        ),
        "expert_owner": _inventory_buffer((config.experts,), torch.long, kind="metadata"),
        "route_key_counts": _inventory_buffer((config.experts,), torch.long, kind="metadata"),
        "route_key_offsets": _inventory_buffer((experts_plus_one,), torch.long, kind="metadata"),
        "route_key_cursors": _inventory_buffer((config.experts,), torch.long, kind="metadata"),
        "counts_per_expert": _inventory_buffer((config.experts,), torch.long, kind="metadata"),
        "rank_counts": _inventory_buffer((1,), torch.long, kind="metadata"),
        "packed_activations": _inventory_buffer(
            (route_rows, config.model_dim), storage_dtype, kind="activation"
        ),
        "packed_route_weights": _inventory_buffer((route_rows,), storage_dtype, kind="metadata"),
        "packed_route_indices": _inventory_buffer((route_rows,), torch.long, kind="metadata"),
        "packed_expert_indices": _inventory_buffer((route_rows,), torch.long, kind="metadata"),
        "expert_count_prefix_sum": _inventory_buffer(
            (config.experts,), torch.long, kind="metadata"
        ),
        "expert_offset_seed": _inventory_buffer((1,), torch.long, kind="metadata"),
        "expert_offsets": _inventory_buffer((experts_plus_one,), torch.long, kind="metadata"),
        "packed_token_indices": _inventory_buffer((route_rows,), torch.long, kind="metadata"),
        "hidden_task_offsets": _inventory_buffer((experts_plus_one,), torch.long, kind="metadata"),
        "down_task_offsets": _inventory_buffer((experts_plus_one,), torch.long, kind="metadata"),
        "expert_hidden_state": _inventory_buffer(
            (route_rows, config.hidden_dim), compute_dtype, kind="activation"
        ),
        "contributions": _inventory_buffer(
            (route_rows, config.model_dim), storage_dtype, kind="activation"
        ),
    }
    fused_buffers = {
        name: staged_buffers[name]
        for name in (
            "gate_logits",
            "dense_scores",
            "selected_route_weights",
            "selected_expert_indices",
            "packed_activations",
            "packed_route_weights",
            "packed_token_indices",
            "expert_offsets",
            "expert_hidden_state",
        )
    }
    fused_buffers["route_pack_cursors"] = _inventory_buffer(
        (config.experts,), torch.long, kind="metadata"
    )
    persistent_queue_materialized = route_rows > _PERSISTENT_SMALL_WORK_ROUTES
    persistent_buffers = dict(fused_buffers)
    if persistent_queue_materialized:
        persistent_buffers["persistent_task_queue"] = _inventory_buffer(
            (1,), torch.long, kind="metadata"
        )
    persistent_inventory = _summarize_inventory(
        "single_device_cuda_persistent",
        "cuda_persistent",
        executed_backend,
        persistent_buffers,
    )
    persistent_inventory["scheduler_policy"] = {
        "small_work_fallback_max_routes": _PERSISTENT_SMALL_WORK_ROUTES,
        "persistent_queue_materialized": persistent_queue_materialized,
    }
    return {
        "cuda_staged": _summarize_inventory(
            "single_device_cuda_staged",
            "cuda_staged",
            executed_backend,
            staged_buffers,
        ),
        "cuda_fused": _summarize_inventory(
            "single_device_cuda_fused",
            "cuda_fused",
            executed_backend,
            fused_buffers,
        ),
        "cuda_persistent": persistent_inventory,
    }


def _resolved_benchmark_backend(config: MoEForwardBenchmarkConfig) -> ExecutedMoEBackend:
    """Resolve the backend used by benchmark-generated contiguous inference inputs."""

    if config.backend == "reference":
        return "reference"
    if config.backend == "cuda_staged":
        return "cuda_staged"
    if config.backend in {"cuda", "cuda_fused"}:
        return "cuda_fused"
    if config.backend == "cuda_persistent":
        return "cuda_persistent"
    cuda_inputs_are_eligible = (
        torch.cuda.is_available()
        and torch.device(config.device).type == "cuda"
        and config.dtype == "float32"
        and not torch.are_deterministic_algorithms_enabled()
    )
    if not cuda_inputs_are_eligible:
        return "reference"
    for operator, executed_backend in (
        ("deepseek_moe_forward_fused", "cuda_fused"),
        ("deepseek_moe_forward", "cuda_staged"),
    ):
        if _operator_has_cuda_kernel(operator):
            return cast(ExecutedMoEBackend, executed_backend)
    return "reference"


def moe_intermediate_bytes(
    config: MoEForwardBenchmarkConfig,
    *,
    executed_backend: ExecutedMoEBackend | None = None,
) -> dict[str, Any]:
    """Model backend-qualified CUDA staged, fused, and persistent buffers."""

    config.validate()
    if executed_backend is None:
        executed_backend = _resolved_benchmark_backend(config)
    storage_dtype = _dtype_from_name(config.dtype)
    compute_dtype = torch.float64 if storage_dtype == torch.float64 else torch.float32
    storage_element_size = torch.empty((), dtype=storage_dtype).element_size()
    compute_element_size = torch.empty((), dtype=compute_dtype).element_size()
    index_element_size = torch.empty((), dtype=torch.long).element_size()
    route_rows = config.tokens * config.topk
    dense_scores = config.tokens * config.experts * compute_element_size
    packed_activations = route_rows * config.model_dim * storage_element_size
    packed_weights = route_rows * storage_element_size
    packed_indices = 2 * route_rows * index_element_size
    expert_hidden_state = route_rows * config.hidden_dim * compute_element_size
    contributions = route_rows * config.model_dim * storage_element_size
    return {
        "analytical_only": True,
        "executed_backend": executed_backend,
        "floating_dtype": config.dtype,
        "floating_element_size": storage_element_size,
        "index_dtype": "int64",
        "index_element_size": index_element_size,
        "route_rows": route_rows,
        "dense_scores": dense_scores,
        "packed_activations": packed_activations,
        "packed_weights": packed_weights,
        "packed_indices": packed_indices,
        "expert_hidden_state": expert_hidden_state,
        "contributions": contributions,
        "total_major_intermediates": (
            dense_scores
            + packed_activations
            + packed_weights
            + packed_indices
            + expert_hidden_state
            + contributions
        ),
        "legacy_flat_fields": {
            "historical_compatibility": True,
            "modeled_backend": "cuda_staged",
            "executed_backend": executed_backend,
            "materialized_by_execution": executed_backend == "cuda_staged",
        },
        "inventories": _moe_intermediate_inventories(
            config,
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
            executed_backend=executed_backend,
        ),
    }


def _make_inputs(
    config: MoEForwardBenchmarkConfig,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    dtype = _dtype_from_name(config.dtype)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    initialization = moe_initialization_model(config)

    def normal(*shape: int, standard_deviation: float) -> Tensor:
        sample = torch.randn(*shape, dtype=dtype, generator=generator)
        return (sample * standard_deviation).to(device).contiguous()

    activation_scale = cast(float, initialization["activation_standard_deviation"])
    gate_scale = cast(float, initialization["gate_weight_standard_deviation"])
    gate_up_scale = cast(
        float,
        initialization["expert_gate_up_weight_standard_deviation"],
    )
    down_scale = cast(float, initialization["expert_down_weight_standard_deviation"])
    score_bias_scale = cast(float | None, initialization["score_bias_standard_deviation"])
    score_bias = (
        normal(config.experts, standard_deviation=score_bias_scale)
        if score_bias_scale is not None
        else None
    )
    return (
        normal(config.tokens, config.model_dim, standard_deviation=activation_scale),
        normal(config.experts, config.model_dim, standard_deviation=gate_scale),
        normal(
            config.experts,
            config.hidden_dim,
            config.model_dim,
            standard_deviation=gate_up_scale,
        ),
        normal(
            config.experts,
            config.model_dim,
            config.hidden_dim,
            standard_deviation=down_scale,
        ),
        normal(
            config.experts,
            config.hidden_dim,
            config.model_dim,
            standard_deviation=gate_up_scale,
        ),
        score_bias,
    )


def _run_operation(
    config: MoEForwardBenchmarkConfig,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None],
) -> Tensor:
    x, gate_weight, expert_w1, expert_w2, expert_w3, score_bias = inputs
    return deepseek_moe_forward(
        x,
        gate_weight,
        expert_w1,
        expert_w2,
        expert_w3,
        topk=config.topk,
        n_groups=config.n_groups,
        topk_groups=config.topk_groups,
        score_bias=score_bias,
        route_scale=config.route_scale,
        backend=config.backend,
    )


def _normalized_error(actual: Tensor, expected: Tensor, rtol: float, atol: float) -> Tensor:
    if actual.numel() == 0:
        return torch.zeros(3, dtype=torch.float64, device=actual.device)
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


def _verify(
    config: MoEForwardBenchmarkConfig,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None],
    actual: Tensor,
) -> dict[str, Any]:
    x, gate_weight, expert_w1, expert_w2, expert_w3, score_bias = inputs
    expected = cast(
        Tensor,
        deepseek_moe_reference(
            x,
            gate_weight,
            expert_w1,
            expert_w2,
            expert_w3,
            topk=config.topk,
            n_groups=config.n_groups,
            topk_groups=config.topk_groups,
            score_bias=score_bias,
            route_scale=config.route_scale,
        ),
    )
    rtol, atol = _verification_tolerances(actual.dtype)
    if actual.device.type == "cuda" and config.backend != "reference":
        rtol = max(rtol, 1e-3)
        atol = max(atol, 1e-3)
    error = _normalized_error(actual, expected, rtol, atol)
    if error[2].item() > 1.0:
        raise AssertionError(f"whole-layer MoE output verification failed: {error.tolist()}")
    return {
        "performed": True,
        "reference": "deepseek_moe_reference",
        "rtol": rtol,
        "atol": atol,
        "output": {
            "max_absolute_error": error[0].item(),
            "max_relative_error": error[1].item(),
            "max_tolerance_ratio": error[2].item(),
        },
    }


def _route_distribution(indices: Tensor, experts: int) -> dict[str, Any]:
    values = torch.bincount(indices.reshape(-1), minlength=experts).cpu().tolist()
    total = sum(values)
    mean = total / experts
    variance = sum((value - mean) ** 2 for value in values) / experts
    active_experts = sum(value > 0 for value in values)
    return {
        "values": values,
        "total": total,
        "mean": mean,
        "minimum": min(values),
        "maximum": max(values),
        "active_experts": active_experts,
        "empty_experts": experts - active_experts,
        "peak_to_mean": max(values) / mean,
        "coefficient_of_variation": math.sqrt(variance) / mean,
    }


def benchmark_moe_forward(config: MoEForwardBenchmarkConfig) -> dict[str, Any]:
    """Benchmark only the whole-layer facade call and return a JSON-ready report."""

    config.validate()
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA benchmark device is not available")
    inputs = _make_inputs(config, device=device)
    x, gate_weight, _, _, _, score_bias = inputs

    with torch.inference_mode():
        routing = deepseek_grouped_topk(
            x,
            gate_weight,
            topk=config.topk,
            n_groups=config.n_groups,
            topk_groups=config.topk_groups,
            score_bias=score_bias,
            route_scale=config.route_scale,
        )
        output = _run_operation(config, inputs)
        verification = _verify(config, inputs, output) if config.verify else {"performed": False}
        operation = lambda: _run_operation(config, inputs)
        samples = (
            _measure_cuda(operation, config.warmup, config.iterations, device)
            if device.type == "cuda"
            else _measure_cpu(operation, config.warmup, config.iterations)
        )

    executed_backend = _resolved_benchmark_backend(config)
    implementation = {
        "reference": "reference",
        "cuda_staged": "single_device_cuda_staged",
        "cuda_fused": "single_device_cuda_fused",
        "cuda_persistent": "single_device_cuda_persistent",
    }[executed_backend]

    return {
        "schema_version": 1,
        "benchmark": "deepseek_moe_forward",
        "implementation": implementation,
        "executed_backend": executed_backend,
        "performance_claim": False,
        "configuration": asdict(config),
        "initialization": moe_initialization_model(config),
        "environment": _environment_metadata(device),
        "output": {
            "shape": list(output.shape),
            "dtype": str(output.dtype).removeprefix("torch."),
            "device": str(output.device),
            "contiguous": output.is_contiguous(),
        },
        "verification": verification,
        "route_distribution": _route_distribution(routing.indices, config.experts),
        "intermediate_bytes": moe_intermediate_bytes(
            config,
            executed_backend=executed_backend,
        ),
        "latency": summarize_latencies(samples),
        "raw_samples_ms": samples,
        "notes": [
            "latency samples time deepseek_moe_forward only",
            "input creation, route analysis, and independent reference verification are untimed",
            "intermediate bytes are backend-qualified analytical models, not observations or memory traffic",
            "legacy flat intermediate fields are retained as historical CUDA-staged estimates",
            "packed indices count packed route and expert int64 arrays",
            "the fused and persistent inventories describe implemented single-device paths",
            "persistent scheduling is an expert-core task queue, not a distributed megakernel",
            "inventory totals sum materialized sizes and are not simultaneous peak memory",
            "this backend-explicit implementation report makes no performance claim",
        ],
    }
