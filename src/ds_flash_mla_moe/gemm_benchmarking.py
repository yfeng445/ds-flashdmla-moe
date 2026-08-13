"""Structured experiments for the tiled GEMM teaching specification."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

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
from .gemm import gemm_reference, tiled_gemm_reference
from .ops import tiled_gemm

GEMMImplementation = Literal["torch", "tiled", "cuda"]


@dataclass(frozen=True)
class GEMMBenchmarkConfig:
    m: int = 128
    n: int = 128
    k: int = 128
    tile_m: int = 32
    tile_n: int = 32
    tile_k: int = 32
    alpha: float = 1.0
    beta: float = 0.0
    dtype: str = "float32"
    device: str = "cpu"
    implementation: GEMMImplementation = "tiled"
    warmup: int = 2
    iterations: int = 5
    seed: int = 0
    verify: bool = True

    def validate(self) -> None:
        if min(self.m, self.n, self.k, self.tile_m, self.tile_n, self.tile_k) <= 0:
            raise ValueError("GEMM dimensions and tile dimensions must be positive")
        if self.iterations <= 0 or self.warmup < 0:
            raise ValueError("iterations must be positive and warmup must be non-negative")
        if self.dtype not in {"float16", "bfloat16", "float32", "float64"}:
            raise ValueError("unsupported GEMM benchmark dtype")
        if self.implementation not in {"torch", "tiled", "cuda"}:
            raise ValueError("implementation must be torch, tiled, or cuda")
        if self.implementation == "cuda" and (self.tile_m, self.tile_n, self.tile_k) != (
            16,
            16,
            16,
        ):
            raise ValueError("the native CUDA GEMM uses fixed 16x16x16 tiles")
        if not math.isfinite(self.alpha) or not math.isfinite(self.beta):
            raise ValueError("alpha and beta must be finite")


def gemm_work_estimate(config: GEMMBenchmarkConfig) -> dict[str, int | float]:
    """Return FLOPs plus explicit lower-bound and pedagogical traffic models."""

    config.validate()
    element_size = torch.empty((), dtype=_dtype_from_name(config.dtype)).element_size()
    output_elements = config.m * config.n
    epilogue_read_elements = output_elements if config.beta != 0.0 else 0
    matrix_flops = 2 * config.m * config.n * config.k
    compulsory_elements = (
        config.m * config.k + config.k * config.n + output_elements + epilogue_read_elements
    )

    tile_count_m = math.ceil(config.m / config.tile_m)
    tile_count_n = math.ceil(config.n / config.tile_n)
    tile_count_k = math.ceil(config.k / config.tile_k)
    one_thread_input_elements = 2 * config.m * config.n * config.k
    tiled_input_elements = tile_count_n * config.m * config.k + tile_count_m * config.k * config.n
    one_thread_elements = one_thread_input_elements + output_elements + epilogue_read_elements
    tiled_elements = tiled_input_elements + output_elements + epilogue_read_elements
    shared_elements_per_stage = config.tile_m * config.tile_k + config.tile_k * config.tile_n

    return {
        "matrix_flops": matrix_flops,
        "compulsory_tensor_bytes_lower_bound": compulsory_elements * element_size,
        "one_output_thread_global_bytes_model": one_thread_elements * element_size,
        "one_output_tile_global_bytes_model": tiled_elements * element_size,
        "modeled_input_reuse_ratio": one_thread_input_elements / tiled_input_elements,
        "tile_count_m": tile_count_m,
        "tile_count_n": tile_count_n,
        "tile_count_k": tile_count_k,
        "last_tile_m": config.m - (tile_count_m - 1) * config.tile_m,
        "last_tile_n": config.n - (tile_count_n - 1) * config.tile_n,
        "last_tile_k": config.k - (tile_count_k - 1) * config.tile_k,
        "shared_memory_bytes_per_stage_model": shared_elements_per_stage * element_size,
        "ideal_arithmetic_intensity_flops_per_byte": (
            matrix_flops / (compulsory_elements * element_size)
        ),
        "tiled_model_arithmetic_intensity_flops_per_byte": (
            matrix_flops / (tiled_elements * element_size)
        ),
    }


def _error_report(actual: Tensor, expected: Tensor) -> dict[str, float | bool | str]:
    rtol, atol = _verification_tolerances(actual.dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    difference = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
    tolerance = atol + rtol * expected.to(torch.float64).abs()
    return {
        "performed": True,
        "reference": "gemm_reference",
        "rtol": rtol,
        "atol": atol,
        "max_absolute_error": difference.max().item() if difference.numel() else 0.0,
        "max_tolerance_ratio": (difference / tolerance).max().item() if difference.numel() else 0.0,
    }


def benchmark_gemm(config: GEMMBenchmarkConfig) -> dict[str, Any]:
    """Benchmark one GEMM configuration and return a JSON-serializable report."""

    config.validate()
    device = torch.device(config.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("GEMM benchmark device must be CPU or CUDA")
    if config.implementation == "cuda" and device.type != "cuda":
        raise ValueError("implementation=cuda requires a CUDA benchmark device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA benchmark device is not available")
    dtype = _dtype_from_name(config.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU GEMM is not a supported benchmark configuration")

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    a = torch.randn(config.m, config.k, dtype=dtype, generator=generator).to(device)
    b = torch.randn(config.k, config.n, dtype=dtype, generator=generator).to(device)
    c = (
        torch.randn(config.m, config.n, dtype=dtype, generator=generator).to(device)
        if config.beta != 0.0
        else None
    )

    def operation() -> Tensor:
        if config.implementation == "torch":
            return gemm_reference(a, b, c, alpha=config.alpha, beta=config.beta)
        if config.implementation == "cuda":
            return tiled_gemm(
                a,
                b,
                c,
                alpha=config.alpha,
                beta=config.beta,
                backend="cuda",
            )
        return tiled_gemm_reference(
            a,
            b,
            c,
            alpha=config.alpha,
            beta=config.beta,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            tile_k=config.tile_k,
        )

    with torch.inference_mode():
        output = operation()
        verification = (
            _error_report(
                output,
                gemm_reference(a, b, c, alpha=config.alpha, beta=config.beta),
            )
            if config.verify
            else {"performed": False}
        )
        samples = (
            _measure_cuda(operation, config.warmup, config.iterations, device)
            if device.type == "cuda"
            else _measure_cpu(operation, config.warmup, config.iterations)
        )

    latency = summarize_latencies(samples)
    work = gemm_work_estimate(config)
    median_seconds = float(latency["median_ms"]) / 1000.0
    if median_seconds <= 0:
        raise RuntimeError("measured median latency must be positive")
    return {
        "schema_version": 1,
        "benchmark": "general_matrix_multiplication",
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
        "derived": {
            "matrix_tflops_equivalent_at_median": (work["matrix_flops"] / median_seconds / 1e12),
            "compulsory_bandwidth_gb_s_at_median": (
                work["compulsory_tensor_bytes_lower_bound"] / median_seconds / 1e9
            ),
        },
        "raw_samples_ms": samples,
        "notes": [
            "matrix FLOPs use the conventional 2mnk count and omit the epilogue",
            "traffic values are analytical models, not profiler measurements",
            "implementation=tiled exposes PyTorch loop semantics and is not a CUDA kernel",
            "implementation=cuda selects the fixed 16x16x16 native CUDA teaching kernel",
            "shared-memory bytes model one stage and omits padding and multistage buffering",
        ],
    }
