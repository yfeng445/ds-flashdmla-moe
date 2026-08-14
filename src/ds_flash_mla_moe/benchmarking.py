"""Shared helpers and reports for reproducible operator benchmarks."""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .attention import scaled_dot_product_attention_reference
from .ops import AttentionBackend, flash_attention_forward, native_extension_loaded
from .version import __version__


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    batch: int = 1
    heads: int = 4
    query_length: int = 128
    key_length: int = 128
    head_dim: int = 64
    value_dim: int = 64
    dtype: str = "float32"
    device: str = "cpu"
    causal: bool = False
    backend: AttentionBackend = "auto"
    warmup: int = 5
    iterations: int = 20
    seed: int = 0
    reference_block_size: int = 64
    verify: bool = True

    def validate(self) -> None:
        positive = (
            self.batch,
            self.heads,
            self.query_length,
            self.key_length,
            self.head_dim,
            self.value_dim,
            self.iterations,
            self.reference_block_size,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("attention dimensions, iterations, and block size must be positive")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.causal and self.query_length > self.key_length:
            raise ValueError("causal benchmark requires query_length <= key_length")
        if self.dtype not in {"float16", "bfloat16", "float32", "float64"}:
            raise ValueError("dtype must be float16, bfloat16, float32, or float64")
        if self.backend not in {"auto", "cuda", "reference"}:
            raise ValueError("backend must be auto, cuda, or reference")


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }[name]


def _percentile(sorted_samples: list[float], probability: float) -> float:
    if not sorted_samples:
        raise ValueError("cannot compute a percentile of an empty sample")
    position = probability * (len(sorted_samples) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_samples[lower]
    fraction = position - lower
    return sorted_samples[lower] * (1.0 - fraction) + sorted_samples[upper] * fraction


def summarize_latencies(samples_ms: list[float]) -> dict[str, float | int]:
    """Summarize raw per-iteration latency samples without dropping outliers."""

    if not samples_ms:
        raise ValueError("at least one latency sample is required")
    if any(not math.isfinite(sample) or sample < 0 for sample in samples_ms):
        raise ValueError("latency samples must be finite and non-negative")
    ordered = sorted(float(sample) for sample in samples_ms)
    return {
        "count": len(ordered),
        "min_ms": ordered[0],
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p90_ms": _percentile(ordered, 0.90),
        "p99_ms": _percentile(ordered, 0.99),
        "max_ms": ordered[-1],
    }


def attention_work_estimate(config: AttentionBenchmarkConfig) -> dict[str, int]:
    """Return algorithmic matrix FLOPs and a compulsory tensor-byte lower bound."""

    config.validate()
    dtype = _dtype_from_name(config.dtype)
    element_size = torch.empty((), dtype=dtype).element_size()
    batch_heads = config.batch * config.heads
    score_flops = 2 * batch_heads * config.query_length * config.key_length * config.head_dim
    value_flops = 2 * batch_heads * config.query_length * config.key_length * config.value_dim
    compulsory_elements = batch_heads * (
        config.query_length * config.head_dim
        + config.key_length * config.head_dim
        + config.key_length * config.value_dim
        + config.query_length * config.value_dim
    )
    return {
        "matrix_flops": score_flops + value_flops,
        "compulsory_tensor_bytes_lower_bound": compulsory_elements * element_size,
    }


def _environment_metadata(device: torch.device) -> dict[str, Any]:
    source_revision, source_dirty = _source_state()
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "package": __version__,
        "native_extension_loaded": native_extension_loaded(),
        "torch_cuda_version": torch.version.cuda,
        "torch_git_version": torch.version.git_version,
        "source_revision": source_revision,
        "source_dirty": source_dirty,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata["gpu"] = {
            "index": index,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "multiprocessors": properties.multi_processor_count,
        }
    else:
        metadata["cpu"] = platform.processor() or platform.machine()
    return metadata


def _source_state() -> tuple[str | None, bool | None]:
    github_revision = os.environ.get("GITHUB_SHA")
    try:
        if github_revision:
            revision = github_revision
        else:
            revision_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            revision = revision_result.stdout.strip()
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return github_revision or None, None
    return revision or None, bool(status_result.stdout.strip())


def _measure_cpu(operation: Callable[[], Tensor], warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        operation()
        end = time.perf_counter_ns()
        samples.append((end - start) / 1_000_000.0)
    return samples


def _measure_cuda(
    operation: Callable[[], Tensor],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[float]:
    with torch.cuda.device(device):
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize(device)

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for index in range(iterations):
            starts[index].record()
            operation()
            ends[index].record()
        torch.cuda.synchronize(device)
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def _verification_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float64:
        return 1e-9, 1e-9
    if dtype == torch.float32:
        return 5e-5, 5e-5
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 5e-3, 5e-3


def _verify_output(
    output: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool,
) -> dict[str, Any]:
    expected = scaled_dot_product_attention_reference(q, k, v, causal=causal)
    rtol, atol = _verification_tolerances(output.dtype)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    difference = (output.to(torch.float64) - expected.to(torch.float64)).abs()
    denominator = expected.to(torch.float64).abs().clamp_min(torch.finfo(torch.float64).tiny)
    return {
        "performed": True,
        "reference": "scaled_dot_product_attention_reference",
        "rtol": rtol,
        "atol": atol,
        "max_absolute_error": difference.max().item() if difference.numel() else 0.0,
        "max_relative_error": (difference / denominator).max().item()
        if difference.numel()
        else 0.0,
    }


def benchmark_attention(config: AttentionBenchmarkConfig) -> dict[str, Any]:
    """Benchmark one attention configuration and return a JSON-serializable report."""

    config.validate()
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA benchmark device is not available")
    dtype = _dtype_from_name(config.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU attention is not a supported benchmark configuration")

    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device).manual_seed(config.seed)
    q = torch.randn(
        config.batch,
        config.heads,
        config.query_length,
        config.head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    k = torch.randn(
        config.batch,
        config.heads,
        config.key_length,
        config.head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    v = torch.randn(
        config.batch,
        config.heads,
        config.key_length,
        config.value_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )

    def operation() -> Tensor:
        return flash_attention_forward(
            q,
            k,
            v,
            causal=config.causal,
            backend=config.backend,
            reference_block_size=config.reference_block_size,
        )

    with torch.inference_mode():
        output = operation()
        verification = (
            _verify_output(output, q, k, v, causal=config.causal)
            if config.verify
            else {"performed": False}
        )
        samples = (
            _measure_cuda(operation, config.warmup, config.iterations, device)
            if device.type == "cuda"
            else _measure_cpu(operation, config.warmup, config.iterations)
        )

    latency = summarize_latencies(samples)
    work = attention_work_estimate(config)
    median_seconds = float(latency["median_ms"]) / 1000.0
    if median_seconds <= 0:
        raise RuntimeError("measured median latency must be positive")
    derived = {
        "matrix_tflops_at_median": work["matrix_flops"] / median_seconds / 1e12,
        "compulsory_bandwidth_gb_s_at_median": (
            work["compulsory_tensor_bytes_lower_bound"] / median_seconds / 1e9
        ),
    }
    return {
        "schema_version": 1,
        "configuration": asdict(config),
        "environment": _environment_metadata(device),
        "output": {
            "shape": list(output.shape),
            "dtype": str(output.dtype).removeprefix("torch."),
            "device": str(output.device),
        },
        "verification": verification,
        "work_estimate": work,
        "latency": latency,
        "derived": derived,
        "raw_samples_ms": samples,
        "notes": [
            "matrix_flops counts QK^T and PV multiply-adds only",
            "compulsory bytes are a tensor-I/O lower bound, not measured DRAM traffic",
        ],
    }


def write_benchmark_report(report: dict[str, Any], path: str | Path | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path is None:
        sys.stdout.write(rendered)
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
