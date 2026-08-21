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
from dataclasses import asdict, dataclass, replace
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from .attention import scaled_dot_product_attention_reference
from .ops import flash_attention_forward, native_extension_loaded
from .version import __version__

AttentionBenchmarkBackend = Literal[
    "auto",
    "cuda",
    "cuda_rowwise",
    "reference",
    "blockwise",
    "fa1",
    "fa2",
    "sdpa",
    "flash-attn-4",
]


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
    backend: AttentionBenchmarkBackend = "auto"
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
        if self.backend not in {
            "auto",
            "cuda",
            "cuda_rowwise",
            "reference",
            "blockwise",
            "fa1",
            "fa2",
            "sdpa",
            "flash-attn-4",
        }:
            raise ValueError(
                "backend must be auto, cuda, cuda_rowwise, reference, blockwise, fa1, fa2, "
                "sdpa, or flash-attn-4"
            )
        if self.backend in {"fa1", "fa2"}:
            try:
                device_type = torch.device(self.device).type
            except RuntimeError as error:
                raise ValueError("device must be a valid torch device") from error
            if device_type != "cuda" or self.dtype != "float16":
                raise ValueError("formal FA benchmarks require CUDA float16")
        if self.backend == "flash-attn-4":
            try:
                device_type = torch.device(self.device).type
            except RuntimeError as error:
                raise ValueError("device must be a valid torch device") from error
            if device_type != "cuda":
                raise ValueError("flash-attn-4 benchmark requires a CUDA device")
            if self.dtype not in {"float16", "bfloat16"}:
                raise ValueError("flash-attn-4 benchmark requires float16 or bfloat16")


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


def _right_aligned_causal_mask(q: Tensor, k: Tensor) -> Tensor:
    query_length = q.shape[-2]
    key_length = k.shape[-2]
    query_positions = torch.arange(query_length, device=q.device) + (key_length - query_length)
    key_positions = torch.arange(key_length, device=q.device)
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


def _sdpa_attention_baseline(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool,
    attention_mask: Tensor | None,
) -> Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=causal and attention_mask is None,
    )


def _load_flash_attn_4() -> Callable[..., Any]:
    try:
        module = import_module("flash_attn.cute")
        implementation = module.flash_attn_func
    except (AttributeError, ImportError, OSError) as error:
        raise RuntimeError(
            "backend=flash-attn-4 requires a working optional flash-attn-4 installation "
            "compatible with the active PyTorch, CUDA toolkit, and GPU"
        ) from error
    return implementation


def _flash_attn_4_version() -> str | None:
    try:
        return version("flash-attn-4")
    except PackageNotFoundError:
        return None


def _flash_attn_4_attention_baseline(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool,
    implementation: Callable[..., Any] | None = None,
) -> Tensor:
    flash_attn_func = _load_flash_attn_4() if implementation is None else implementation
    result = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        causal=causal,
    )
    output = result[0] if isinstance(result, tuple) else result
    if not isinstance(output, Tensor):
        raise TypeError("flash-attn-4 returned an unsupported output type")
    return output.transpose(1, 2).contiguous()


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
    sdpa_attention_mask = (
        _right_aligned_causal_mask(q, k)
        if config.backend == "sdpa" and config.causal and config.query_length != config.key_length
        else None
    )
    flash_attn_4_implementation = _load_flash_attn_4() if config.backend == "flash-attn-4" else None

    def operation() -> Tensor:
        if config.backend == "sdpa":
            return _sdpa_attention_baseline(
                q,
                k,
                v,
                causal=config.causal,
                attention_mask=sdpa_attention_mask,
            )
        if config.backend == "flash-attn-4":
            if flash_attn_4_implementation is None:
                raise RuntimeError("flash-attn-4 implementation was not initialized")
            return _flash_attn_4_attention_baseline(
                q,
                k,
                v,
                causal=config.causal,
                implementation=flash_attn_4_implementation,
            )
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
    backend_metadata = (
        {
            "provider": "Dao-AILab/flash-attention",
            "distribution": "flash-attn-4",
            "version": _flash_attn_4_version(),
        }
        if config.backend == "flash-attn-4"
        else None
    )
    notes = [
        "matrix_flops counts QK^T and PV multiply-adds only",
        "compulsory bytes are a tensor-I/O lower bound, not measured DRAM traffic",
    ]
    if config.backend == "sdpa":
        notes.append(
            "backend=sdpa delegates kernel selection to PyTorch scaled_dot_product_attention"
        )
    elif config.backend == "flash-attn-4":
        notes.append(
            "backend=flash-attn-4 uses the optional CuTeDSL implementation and supports "
            "float16/bfloat16 CUDA inputs only"
        )
        notes.append(
            "latency includes the BHSD-to-BSHD layout adapter and contiguous BHSD output copy"
        )
    elif config.backend in {"fa1", "fa2"}:
        notes.append(
            f"backend={config.backend} is a repository teaching kernel using FP32 accumulation "
            "and no Tensor Cores"
        )
    report = {
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
        "notes": notes,
    }
    if backend_metadata is not None:
        report["external_backend"] = backend_metadata
    return report


def benchmark_attention_backends(
    config: AttentionBenchmarkConfig,
    backends: tuple[AttentionBenchmarkBackend, ...],
) -> dict[str, Any]:
    """Benchmark multiple backends with identical seeds and dimensions."""

    if not backends or len(set(backends)) != len(backends):
        raise ValueError("comparison backends must be non-empty and unique")
    reports = {
        backend: benchmark_attention(replace(config, backend=backend)) for backend in backends
    }
    return {
        "schema_version": 1,
        "comparison_backends": list(backends),
        "shared_seed": config.seed,
        "reports": reports,
    }


def write_benchmark_report(report: dict[str, Any], path: str | Path | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path is None:
        sys.stdout.write(rendered)
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
