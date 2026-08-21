"""Single-GPU Kineto and NVTX profiling for deterministic benchmark workloads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .matrix_benchmarking import (
    BenchmarkMatrixCase,
    BenchmarkMatrixConfig,
    MatrixSide,
    benchmark_matrix_case_side,
    build_benchmark_matrix_cases,
)
from .moe_benchmarking import MoEForwardBenchmarkConfig, benchmark_moe_forward

_SYNCHRONIZATION_EVENTS = {
    "aten::_local_scalar_dense": "device_to_host_scalar",
    "Memcpy DtoH (Device -> Pinned)": "device_to_host_copy",
    "cudaStreamSynchronize": "stream_synchronize",
    "cudaDeviceSynchronize": "device_synchronize",
}

# Kernel symbols compiled by the production CUDA extension in setup.py. Kineto
# commonly reports these anonymous-namespace names without a project namespace.
_PROJECT_CUDA_KERNEL_NAMES = frozenset(
    {
        # attention
        "attention_backward_kernel",
        "attention_forward_kernel",
        "fa1_forward_kernel",
        "fa2_forward_kernel",
        # GEMM
        "tiled_gemm_float_kernel",
        # MLA
        "cache_rope_kernel",
        "copy_positions_kernel",
        "copy_query_nope_kernel",
        "linear_weight_kernel",
        "mla_absorbed_attention_generic_kernel",
        "mla_absorbed_attention_warp_partition_kernel",
        "mla_paged_absorbed_attention_kernel",
        "query_rope_kernel",
        "rms_norm_prefix_kernel",
        "scatter_cache_slots_kernel",
        # MoE
        "build_grouped_tile_offsets_kernel",
        "combine_routes_float_kernel",
        "count_single_device_experts_kernel",
        "count_local_experts_kernel",
        "count_route_keys_kernel",
        "exclusive_scan_kernel",
        "fused_down_atomic_float_kernel",
        "fused_hidden_float_kernel",
        "grouped_topk_select_kernel",
        "pack_expert_major_kernel",
        "pack_routes_float_kernel",
        "pack_single_device_routes_float_kernel",
        "persistent_down_atomic_float_kernel",
        "persistent_hidden_float_kernel",
        "scan_single_device_expert_offsets_kernel",
        "summarize_counts_kernel",
        "swiglu_down_grouped_tiled_float_kernel",
        "swiglu_down_grouped_wmma_half_kernel",
        "swiglu_hidden_grouped_tiled_float_kernel",
        "swiglu_hidden_grouped_wmma_half_kernel",
        "validate_offsets_kernel",
    }
)


@dataclass(frozen=True)
class OperatorProfileConfig:
    """Select one matrix side and the amount of work captured by a profiler."""

    case: str
    side: MatrixSide = "native"
    device: str = "cuda"
    warmup: int = 2
    iterations: int = 5
    seed: int = 0
    verify: bool = False
    record_shapes: bool = True
    row_limit: int = 50

    def validate(self) -> None:
        try:
            device = torch.device(self.device)
        except RuntimeError as error:
            raise ValueError("profile device must be a valid torch device") from error
        if device.type != "cuda":
            raise ValueError("operator profiling requires a CUDA device")
        if not self.case:
            raise ValueError("an exact matrix case name is required")
        if self.side not in {"native", "baseline"}:
            raise ValueError("profile side must be native or baseline")
        if self.warmup < 0 or self.iterations <= 0:
            raise ValueError("warmup must be non-negative and iterations must be positive")
        if self.row_limit <= 0:
            raise ValueError("row_limit must be positive")


def _selected_case(config: OperatorProfileConfig) -> BenchmarkMatrixCase:
    config.validate()
    if config.case.startswith("attention_fa4_"):
        profile = "flash-attn-4"
    elif config.case.startswith("mla_low_precision_"):
        profile = "mla-low-precision"
    elif config.case.startswith("mla_paged_"):
        profile = "mla-paged"
    else:
        profile = "representative"
    selected = build_benchmark_matrix_cases(
        BenchmarkMatrixConfig(
            device=config.device,
            profile=profile,
            cases=(config.case,),
            warmup=config.warmup,
            iterations=config.iterations,
            seed=config.seed,
            verify=config.verify,
            fail_fast=True,
        )
    )
    if len(selected) != 1:
        raise AssertionError("an exact profile case must resolve to one matrix entry")
    return selected[0]


def _event_time(event: Any, attribute: str, legacy_attribute: str | None = None) -> float:
    value = getattr(event, attribute, None)
    if value is None and legacy_attribute is not None:
        value = getattr(event, legacy_attribute, 0.0)
    return float(value or 0.0)


def _event_record(event: Any) -> dict[str, Any]:
    return {
        "name": str(event.key),
        "count": int(event.count),
        "self_cpu_ms": _event_time(event, "self_cpu_time_total") / 1000.0,
        "cpu_total_ms": _event_time(event, "cpu_time_total") / 1000.0,
        "self_device_ms": _event_time(
            event,
            "self_device_time_total",
            "self_cuda_time_total",
        )
        / 1000.0,
        "device_total_ms": _event_time(
            event,
            "device_time_total",
            "cuda_time_total",
        )
        / 1000.0,
    }


def _is_project_cuda_kernel(name: str) -> bool:
    return any(kernel_name in name for kernel_name in _PROJECT_CUDA_KERNEL_NAMES)


def _summarize_events(events: list[Any], row_limit: int) -> dict[str, Any]:
    records = [_event_record(event) for event in events]
    device_records = sorted(
        (record for record in records if record["self_device_ms"] > 0),
        key=lambda record: record["self_device_ms"],
        reverse=True,
    )
    custom_kernels = [
        record for record in device_records if _is_project_cuda_kernel(record["name"])
    ]
    custom_operators = sorted(
        (record for record in records if record["name"].startswith("ds_flash_mla_moe::")),
        key=lambda record: record["device_total_ms"],
        reverse=True,
    )
    synchronization = []
    for record in records:
        category = _SYNCHRONIZATION_EVENTS.get(record["name"])
        if category is not None:
            synchronization.append({"category": category, **record})
    return {
        "observed_device_activity_count": sum(record["count"] for record in device_records),
        "observed_custom_kernel_count": sum(record["count"] for record in custom_kernels),
        "top_self_device_events": device_records[:row_limit],
        "custom_kernel_events": custom_kernels,
        "custom_operator_events": custom_operators,
        "synchronization_events": synchronization,
    }


def _begin_memory_capture(device: torch.device) -> int:
    allocated_before = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    return allocated_before


def _finish_memory_capture(device: torch.device, allocated_before: int) -> dict[str, int]:
    allocated_after = int(torch.cuda.memory_allocated(device))
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    return {
        "allocated_before_bytes": allocated_before,
        "allocated_after_bytes": allocated_after,
        "allocated_delta_bytes": allocated_after - allocated_before,
        "peak_allocated_bytes": peak_allocated,
        "peak_delta_bytes": max(0, peak_allocated - allocated_before),
    }


def _external_profiler_events() -> dict[str, Any]:
    return {
        "observation": "external_profiler_required",
        "observed_device_activity_count": None,
        "observed_custom_kernel_count": None,
        "custom_kernel_events": [],
        "synchronization_events": None,
    }


def _require_cuda_profile_device(device_name: str, workload: str) -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError(f"{workload} profiling requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA profiling device is not available")
    return device


def _capture_kineto_workload(
    operation: Callable[[], dict[str, Any]],
    *,
    device: torch.device,
    record_shapes: bool,
    row_limit: int,
    trace_path: str | Path | None,
) -> dict[str, Any]:
    if row_limit <= 0:
        raise ValueError("row_limit must be positive")
    operation()
    allocated_before = _begin_memory_capture(device)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=record_shapes,
        acc_events=True,
    ) as profiler:
        benchmark_report = operation()
    events = list(profiler.key_averages())
    memory = _finish_memory_capture(device, allocated_before)

    rendered_trace_path = None
    if trace_path is not None:
        destination = Path(trace_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(destination))
        rendered_trace_path = str(destination)

    return {
        "activities": ["cpu", "cuda"],
        "events": _summarize_events(events, row_limit),
        "memory": memory,
        "trace_path": rendered_trace_path,
        "benchmark_report": benchmark_report,
    }


def _capture_nvtx_workload(
    operation: Callable[[], dict[str, Any]],
    *,
    device: torch.device,
    label: str,
    record_shapes: bool,
) -> dict[str, Any]:
    operation()
    allocated_before = _begin_memory_capture(device)
    with (
        torch.cuda.nvtx.range(label),
        torch.autograd.profiler.emit_nvtx(record_shapes=record_shapes),
    ):
        benchmark_report = operation()
    return {
        "activities": ["external_profiler"],
        "events": _external_profiler_events(),
        "memory": _finish_memory_capture(device, allocated_before),
        "nvtx_range": label,
        "benchmark_report": benchmark_report,
    }


def _case_metadata(case: BenchmarkMatrixCase, side: MatrixSide) -> dict[str, Any]:
    return {
        "name": case.name,
        "family": case.family,
        "shape_class": case.shape_class,
        "description": case.description,
        "side": side,
        "baseline_label": case.baseline_label,
    }


def profile_operator_case(
    config: OperatorProfileConfig,
    *,
    trace_path: str | Path | None = None,
) -> dict[str, Any]:
    """Capture one matrix side with Kineto and return structured event aggregates."""

    case = _selected_case(config)
    device = _require_cuda_profile_device(config.device, "operator")
    capture = _capture_kineto_workload(
        lambda: benchmark_matrix_case_side(case, config.side),
        device=device,
        record_shapes=config.record_shapes,
        row_limit=config.row_limit,
        trace_path=trace_path,
    )
    benchmark_report = capture["benchmark_report"]

    return {
        "schema_version": 1,
        "profile": "single_gpu_operator_torch_profiler",
        "profile_configuration": asdict(config),
        "case": _case_metadata(case, config.side),
        "environment": benchmark_report.get("environment", {}),
        "activities": capture["activities"],
        "events": capture["events"],
        "memory": capture["memory"],
        "trace_path": capture["trace_path"],
        "benchmark_report": benchmark_report,
        "notes": [
            "torch.profiler/Kineto evidence is a local precursor to Nsight, not an Nsight report",
            "one complete preflight run occurs before capture to load kernels and warm runtime state",
            "the capture includes fresh setup, one output call, configured warmup, and timed iterations",
            "self-device rows can describe correlated operator and kernel views and are not additive",
            "observed activity counts are Kineto occurrences, not physical kernel launch counts",
            "peak allocated memory is allocator evidence for the capture, not memory traffic",
            "verification is disabled by default so an alternate implementation is not mixed into the trace",
            "physical communication-compute overlap still requires a multi-GPU Nsight timeline",
        ],
    }


def run_nvtx_operator_case(config: OperatorProfileConfig) -> dict[str, Any]:
    """Execute one case with an outer NVTX range and PyTorch operator ranges."""

    case = _selected_case(config)
    device = _require_cuda_profile_device(config.device, "operator")
    label = f"ds_flash_mla_moe::{case.name}::{config.side}"
    capture = _capture_nvtx_workload(
        lambda: benchmark_matrix_case_side(case, config.side),
        device=device,
        label=label,
        record_shapes=config.record_shapes,
    )
    benchmark_report = capture["benchmark_report"]
    return {
        "schema_version": 1,
        "profile": "single_gpu_operator_nvtx_run",
        "profile_configuration": asdict(config),
        "case": _case_metadata(case, config.side),
        "environment": benchmark_report.get("environment", {}),
        "activities": capture["activities"],
        "events": capture["events"],
        "memory": capture["memory"],
        "nvtx_range": capture["nvtx_range"],
        "benchmark_report": benchmark_report,
        "notes": [
            "run this mode under Nsight Systems or Nsight Compute to persist a native profiler report",
            "unobserved activity and custom-kernel counts remain null rather than being zero",
            "peak allocated memory is allocator evidence for this run, not memory traffic",
            "NVTX execution alone does not prove utilization, traffic, or overlap",
        ],
    }


def profile_moe_forward(
    config: MoEForwardBenchmarkConfig,
    *,
    trace_path: str | Path | None = None,
    record_shapes: bool = True,
    row_limit: int = 50,
) -> dict[str, Any]:
    """Capture a whole-layer MoE benchmark with Kineto event aggregation."""

    config.validate()
    device = _require_cuda_profile_device(config.device, "MoE")
    capture = _capture_kineto_workload(
        lambda: benchmark_moe_forward(config),
        device=device,
        record_shapes=record_shapes,
        row_limit=row_limit,
        trace_path=trace_path,
    )
    benchmark_report = capture["benchmark_report"]
    return {
        "schema_version": 1,
        "profile": "single_gpu_moe_kineto",
        "profile_configuration": {
            "benchmark": asdict(config),
            "record_shapes": record_shapes,
            "row_limit": row_limit,
        },
        "workload": {
            "name": "deepseek_moe_forward",
            "backend": config.backend,
        },
        "environment": benchmark_report.get("environment", {}),
        "activities": capture["activities"],
        "events": capture["events"],
        "memory": capture["memory"],
        "trace_path": capture["trace_path"],
        "benchmark_report": benchmark_report,
        "performance_claim": False,
        "notes": [
            "Kineto counts are observed aggregated device activities, not physical launch counts",
            "custom kernel rows are matched against supported production csrc CUDA kernel names",
            "peak allocated memory is allocator evidence for this capture, not memory traffic",
            "one complete preflight benchmark occurs before capture to warm runtime state",
            (
                "the capture wraps the complete benchmark harness (input setup, route analysis, "
                "optional verification, configured warmups, and timed calls), not one isolated forward"
            ),
        ],
    }


def run_nvtx_moe_forward(
    config: MoEForwardBenchmarkConfig,
    *,
    record_shapes: bool = True,
) -> dict[str, Any]:
    """Run a whole-layer MoE benchmark inside an Nsight-ready NVTX range."""

    config.validate()
    device = _require_cuda_profile_device(config.device, "MoE")
    label = f"ds_flash_mla_moe::deepseek_moe_forward::{config.backend}"
    capture = _capture_nvtx_workload(
        lambda: benchmark_moe_forward(config),
        device=device,
        label=label,
        record_shapes=record_shapes,
    )
    benchmark_report = capture["benchmark_report"]
    return {
        "schema_version": 1,
        "profile": "single_gpu_moe_nvtx_run",
        "profile_configuration": {
            "benchmark": asdict(config),
            "record_shapes": record_shapes,
        },
        "workload": {
            "name": "deepseek_moe_forward",
            "backend": config.backend,
        },
        "environment": benchmark_report.get("environment", {}),
        "activities": capture["activities"],
        "events": capture["events"],
        "memory": capture["memory"],
        "nvtx_range": capture["nvtx_range"],
        "benchmark_report": benchmark_report,
        "performance_claim": False,
        "notes": [
            "run this mode under Nsight Systems or Nsight Compute to observe activities and kernels",
            "unobserved event counts remain null rather than being reported as zero",
            "peak allocated memory is allocator evidence for this run, not memory traffic",
            (
                "the capture wraps the complete benchmark harness (input setup, route analysis, "
                "optional verification, configured warmups, and timed calls), not one isolated forward"
            ),
            "NVTX execution alone does not prove utilization, traffic, or overlap",
        ],
    }
