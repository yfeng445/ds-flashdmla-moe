"""Structured benchmarks for expert-major SwiGLU compute."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

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
from .expert_ops import ExpertBackend, swiglu_experts_expert_major
from .moe import swiglu_experts_padded_reference

_NATIVE_EXPERT_TILE = 16


@dataclass(frozen=True)
class ExpertBenchmarkConfig:
    """Expert row distribution, tensor shape, and measurement settings."""

    expert_counts: tuple[int, ...] = (32, 32, 32, 32)
    model_dim: int = 64
    hidden_dim: int = 128
    dtype: str = "float32"
    device: str = "cpu"
    backend: ExpertBackend = "reference"
    warmup: int = 2
    iterations: int = 5
    seed: int = 0
    backward: bool = False
    verify: bool = True

    def validate(self) -> None:
        if not self.expert_counts:
            raise ValueError("expert_counts must contain at least one expert")
        if any(count < 0 for count in self.expert_counts):
            raise ValueError("expert_counts cannot contain negative rows")
        if sum(self.expert_counts) <= 0:
            raise ValueError("the expert benchmark requires at least one active row")
        if self.model_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("model_dim and hidden_dim must be positive")
        if self.iterations <= 0 or self.warmup < 0:
            raise ValueError("iterations must be positive and warmup must be non-negative")
        if self.dtype not in {"float16", "bfloat16", "float32", "float64"}:
            raise ValueError("unsupported expert benchmark dtype")
        if self.backend not in {"auto", "cuda", "reference"}:
            raise ValueError("backend must be auto, cuda, or reference")
        try:
            device_type = torch.device(self.device).type
        except RuntimeError as error:
            raise ValueError("expert benchmark device must be a valid torch device") from error
        if self.backend == "cuda" and (
            device_type != "cuda" or self.dtype not in {"float16", "float32"}
        ):
            raise ValueError("backend=cuda requires CUDA float16 or float32 tensors")


def _count_distribution(counts: tuple[int, ...]) -> dict[str, Any]:
    total = sum(counts)
    mean = total / len(counts)
    variance = sum((count - mean) ** 2 for count in counts) / len(counts)
    return {
        "values": list(counts),
        "total": total,
        "mean": mean,
        "minimum": min(counts),
        "maximum": max(counts),
        "peak_to_mean": max(counts) / mean,
        "coefficient_of_variation": math.sqrt(variance) / mean,
        "zero_load_count": sum(count == 0 for count in counts),
    }


def expert_grouped_tile_model(
    expert_counts: tuple[int, ...] | list[int],
    *,
    model_dim: int,
    hidden_dim: int,
) -> dict[str, Any]:
    """Model expert-segmented 16x16x16 forward tasks without capacity padding."""

    if not expert_counts or any(count < 0 for count in expert_counts):
        raise ValueError("expert_counts must be non-empty and non-negative")
    active_rows = sum(expert_counts)
    if active_rows <= 0:
        raise ValueError("the grouped tile model requires at least one active row")
    if model_dim <= 0 or hidden_dim <= 0:
        raise ValueError("model_dim and hidden_dim must be positive")
    grouped_row_tiles = sum(math.ceil(count / _NATIVE_EXPERT_TILE) for count in expert_counts)
    hidden_output_tiles = math.ceil(hidden_dim / _NATIVE_EXPERT_TILE)
    model_output_tiles = math.ceil(model_dim / _NATIVE_EXPERT_TILE)
    grouped_allocated_rows = grouped_row_tiles * _NATIVE_EXPERT_TILE
    return {
        "analytical_only": True,
        "tile_shape": [_NATIVE_EXPERT_TILE, _NATIVE_EXPERT_TILE, _NATIVE_EXPERT_TILE],
        "active_expert_row_tiles": grouped_row_tiles,
        "hidden_output_tiles": hidden_output_tiles,
        "model_output_tiles": model_output_tiles,
        "hidden_projection_tasks": grouped_row_tiles * hidden_output_tiles,
        "down_projection_tasks": grouped_row_tiles * model_output_tiles,
        "allocated_row_lanes": grouped_allocated_rows,
        "inactive_tail_row_lanes": grouped_allocated_rows - active_rows,
        "row_lane_utilization": active_rows / grouped_allocated_rows,
    }


def expert_native_numeric_model(dtype: str) -> dict[str, Any]:
    """Describe the native CUDA forward arithmetic selected by ``dtype``."""

    if dtype not in {"float16", "bfloat16", "float32", "float64"}:
        raise ValueError("unsupported expert dtype")
    supported = dtype in {"float16", "float32"}
    return {
        "applies_to_backend": "cuda",
        "configuration_dtype": dtype,
        "supported": supported,
        "forward_engine": (
            "wmma_tensor_cores"
            if dtype == "float16"
            else "shared_memory_cuda_cores"
            if dtype == "float32"
            else None
        ),
        "multiplicand_dtype": dtype if supported else None,
        "accumulator_dtype": "float32" if supported else None,
        "materialized_hidden_dtype": dtype if supported else None,
        "minimum_compute_capability": "7.0" if dtype == "float16" else None,
    }


def expert_initialization_model(*, model_dim: int, hidden_dim: int) -> dict[str, Any]:
    """Return the fan-in-scaled normal initialization used by expert benchmarks."""

    if model_dim <= 0 or hidden_dim <= 0:
        raise ValueError("model_dim and hidden_dim must be positive")
    return {
        "distribution": "normal",
        "activation_standard_deviation": 1.0,
        "gate_up_weight_standard_deviation": 1.0 / math.sqrt(model_dim),
        "down_weight_standard_deviation": 1.0 / math.sqrt(hidden_dim),
    }


def expert_work_estimate(config: ExpertBenchmarkConfig) -> dict[str, Any]:
    """Count active-row and padded-baseline forward matrix work."""

    config.validate()
    active_rows = sum(config.expert_counts)
    experts = len(config.expert_counts)
    active_experts = sum(count > 0 for count in config.expert_counts)
    padded_capacity = max(config.expert_counts)
    padded_slots = experts * padded_capacity
    element_size = torch.empty((), dtype=_dtype_from_name(config.dtype)).element_size()
    active_flops = 6 * active_rows * config.model_dim * config.hidden_dim
    padded_flops = 6 * padded_slots * config.model_dim * config.hidden_dim
    compulsory_elements = (
        2 * active_rows * config.model_dim
        + 3 * active_experts * config.model_dim * config.hidden_dim
    )
    return {
        "expert_route_rows": _count_distribution(config.expert_counts),
        "expert_count": experts,
        "active_expert_count": active_experts,
        "active_route_rows": active_rows,
        "padded_capacity_per_expert": padded_capacity,
        "padded_expert_slots": padded_slots,
        "padding_rows": padded_slots - active_rows,
        "padding_utilization": active_rows / padded_slots,
        "forward_active_row_matrix_flops": active_flops,
        "forward_padded_matrix_flops": padded_flops,
        "native_grouped_tile_model": expert_grouped_tile_model(
            config.expert_counts,
            model_dim=config.model_dim,
            hidden_dim=config.hidden_dim,
        ),
        "native_numeric_model": expert_native_numeric_model(config.dtype),
        "active_row_compulsory_tensor_bytes_lower_bound": compulsory_elements * element_size,
    }


def _make_inputs(
    config: ExpertBenchmarkConfig,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    dtype = _dtype_from_name(config.dtype)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    initialization = expert_initialization_model(
        model_dim=config.model_dim,
        hidden_dim=config.hidden_dim,
    )

    def normal(*shape: int, standard_deviation: float) -> Tensor:
        sample = torch.randn(*shape, dtype=dtype, generator=generator)
        return (sample * standard_deviation).to(device)

    offsets = torch.tensor(
        [0, *torch.tensor(config.expert_counts, dtype=torch.long).cumsum(0).tolist()],
        dtype=torch.long,
        device=device,
    )
    requires_grad = config.backward
    return (
        normal(
            sum(config.expert_counts),
            config.model_dim,
            standard_deviation=initialization["activation_standard_deviation"],
        ).requires_grad_(requires_grad),
        offsets,
        normal(
            len(config.expert_counts),
            config.hidden_dim,
            config.model_dim,
            standard_deviation=initialization["gate_up_weight_standard_deviation"],
        ).requires_grad_(requires_grad),
        normal(
            len(config.expert_counts),
            config.model_dim,
            config.hidden_dim,
            standard_deviation=initialization["down_weight_standard_deviation"],
        ).requires_grad_(requires_grad),
        normal(
            len(config.expert_counts),
            config.hidden_dim,
            config.model_dim,
            standard_deviation=initialization["gate_up_weight_standard_deviation"],
        ).requires_grad_(requires_grad),
    )


def _loss(output: Tensor) -> Tensor:
    compute_dtype = torch.float64 if output.dtype == torch.float64 else torch.float32
    return output.to(compute_dtype).square().sum()


def _run_operation(
    config: ExpertBenchmarkConfig,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
) -> Tensor:
    if config.backward:
        for tensor in (inputs[0], *inputs[2:]):
            tensor.grad = None
    output = swiglu_experts_expert_major(*inputs, backend=config.backend)
    if config.backward:
        _loss(output).backward()
    return output


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
    config: ExpertBenchmarkConfig,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    actual: Tensor,
) -> dict[str, Any]:
    expected_inputs = tuple(
        tensor.detach().clone().requires_grad_(config.backward)
        if index != 1
        else tensor.detach().clone()
        for index, tensor in enumerate(inputs)
    )
    expected = swiglu_experts_padded_reference(*expected_inputs)
    if config.backward:
        _loss(expected).backward()

    rtol, atol = _verification_tolerances(actual.dtype)
    if config.backend == "cuda" and actual.dtype == torch.float16:
        rtol = max(rtol, 3e-2)
        atol = max(atol, 3e-2)
    output_error = _normalized_error(actual, expected, rtol, atol)
    gradient_error = torch.zeros_like(output_error)
    if config.backward:
        gradient_error = torch.stack(
            [
                _normalized_error(inputs[index].grad, expected_inputs[index].grad, rtol, atol)
                for index in (0, 2, 3, 4)
            ]
        ).amax(dim=0)
    if output_error[2].item() > 1.0:
        raise AssertionError(f"expert output verification failed: {output_error.tolist()}")
    if config.backward and gradient_error[2].item() > 1.0:
        raise AssertionError(f"expert gradient verification failed: {gradient_error.tolist()}")
    return {
        "performed": True,
        "reference": "swiglu_experts_padded_reference",
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


def benchmark_experts(config: ExpertBenchmarkConfig) -> dict[str, Any]:
    """Benchmark expert-major SwiGLU and return a JSON-serializable report."""

    config.validate()
    device = torch.device(config.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("expert benchmark device must be CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA benchmark device is not available")
    if device.type == "cpu" and config.dtype == "float16":
        raise ValueError("float16 CPU expert compute is not a supported benchmark")
    inputs = _make_inputs(config, device=device)

    output = _run_operation(config, inputs)
    verification = _verify(config, inputs, output) if config.verify else {"performed": False}
    samples = (
        _measure_cuda(
            lambda: _run_operation(config, inputs),
            config.warmup,
            config.iterations,
            device,
        )
        if device.type == "cuda"
        else _measure_cpu(
            lambda: _run_operation(config, inputs),
            config.warmup,
            config.iterations,
        )
    )
    latency = summarize_latencies(samples)
    work = expert_work_estimate(config)
    median_seconds = float(latency["median_ms"]) / 1000.0
    if median_seconds <= 0:
        raise RuntimeError("measured median latency must be positive")
    return {
        "schema_version": 1,
        "benchmark": "swiglu_experts_expert_major",
        "configuration": asdict(config),
        "initialization": expert_initialization_model(
            model_dim=config.model_dim,
            hidden_dim=config.hidden_dim,
        ),
        "environment": _environment_metadata(device),
        "output": {
            "shape": list(output.shape),
            "dtype": str(output.dtype).removeprefix("torch."),
            "device": str(output.device),
        },
        "verification": verification,
        "work_estimate": work,
        "latency": latency,
        "derived": {
            "forward_active_row_matrix_tflops_equivalent_at_median": (
                work["forward_active_row_matrix_flops"] / median_seconds / 1e12
            ),
            "active_row_compulsory_bandwidth_gb_s_at_median": (
                work["active_row_compulsory_tensor_bytes_lower_bound"] / median_seconds / 1e9
            ),
        },
        "raw_samples_ms": samples,
        "notes": [
            "forward matrix FLOPs count the three SwiGLU projections only",
            "the compulsory byte count assumes each active expert weight is read once",
            "compulsory bytes are an analytical lower bound, not measured DRAM traffic",
            "forward-equivalent FLOPs remain forward-only when backward timing is enabled",
            "the native CUDA path uses expert-segmented shared-memory FP32 GEMM tiles",
            "float16 uses WMMA with FP32 accumulation and an FP16 materialized hidden state",
            "native grouped tile counts are analytical launch-work models, not profiler counters",
            "native CUDA backward recomputes the traceable PyTorch reference",
        ],
    }
