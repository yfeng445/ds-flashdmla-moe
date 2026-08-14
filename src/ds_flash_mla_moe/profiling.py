"""Single-GPU operator profiling over the deterministic benchmark matrix."""

from __future__ import annotations

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

_SYNCHRONIZATION_EVENTS = {
    "aten::_local_scalar_dense": "device_to_host_scalar",
    "Memcpy DtoH (Device -> Pinned)": "device_to_host_copy",
    "cudaStreamSynchronize": "stream_synchronize",
    "cudaDeviceSynchronize": "device_synchronize",
}


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


def _summarize_events(events: list[Any], row_limit: int) -> dict[str, Any]:
    records = [_event_record(event) for event in events]
    cuda_records = sorted(
        (record for record in records if record["self_device_ms"] > 0),
        key=lambda record: record["self_device_ms"],
        reverse=True,
    )
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
        "top_self_device_events": cuda_records[:row_limit],
        "custom_operator_events": custom_operators,
        "synchronization_events": synchronization,
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
    if not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA profiling device is not available")
    benchmark_matrix_case_side(case, config.side)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=config.record_shapes,
        acc_events=True,
    ) as profiler:
        benchmark_report = benchmark_matrix_case_side(case, config.side)
    events = list(profiler.key_averages())

    rendered_trace_path = None
    if trace_path is not None:
        destination = Path(trace_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(destination))
        rendered_trace_path = str(destination)

    return {
        "schema_version": 1,
        "profile": "single_gpu_operator_torch_profiler",
        "profile_configuration": asdict(config),
        "case": _case_metadata(case, config.side),
        "environment": benchmark_report.get("environment", {}),
        "events": _summarize_events(events, config.row_limit),
        "trace_path": rendered_trace_path,
        "benchmark_report": benchmark_report,
        "notes": [
            "torch.profiler/Kineto evidence is a local precursor to Nsight, not an Nsight report",
            "one complete preflight run occurs before capture to load kernels and warm runtime state",
            "the capture includes fresh setup, one output call, configured warmup, and timed iterations",
            "self-device rows can describe correlated operator and kernel views and are not additive",
            "verification is disabled by default so an alternate implementation is not mixed into the trace",
            "physical communication-compute overlap still requires a multi-GPU Nsight timeline",
        ],
    }


def run_nvtx_operator_case(config: OperatorProfileConfig) -> dict[str, Any]:
    """Execute one case with an outer NVTX range and PyTorch operator ranges."""

    case = _selected_case(config)
    if not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA profiling device is not available")
    benchmark_matrix_case_side(case, config.side)
    label = f"ds_flash_mla_moe::{case.name}::{config.side}"
    with (
        torch.cuda.nvtx.range(label),
        torch.autograd.profiler.emit_nvtx(record_shapes=config.record_shapes),
    ):
        benchmark_report = benchmark_matrix_case_side(case, config.side)
    return {
        "schema_version": 1,
        "profile": "single_gpu_operator_nvtx_run",
        "profile_configuration": asdict(config),
        "case": _case_metadata(case, config.side),
        "environment": benchmark_report.get("environment", {}),
        "nvtx_range": label,
        "benchmark_report": benchmark_report,
        "notes": [
            "run this mode under Nsight Systems or Nsight Compute to persist a native profiler report",
            "NVTX execution alone does not prove utilization, traffic, or overlap",
        ],
    }
