"""Structured forward-only benchmarks for explicit FP8/INT8 matrix experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from .benchmarking import (
    _environment_metadata,
    _measure_cpu,
    _measure_cuda,
    summarize_latencies,
)
from .quantization import (
    QuantizationBackend,
    QuantizationFormat,
    cuda_quantization_available,
    dequantized_linear,
    quantize_activations,
    quantize_weights,
)


@dataclass(frozen=True)
class QuantizedGEMMBenchmarkConfig:
    m: int = 128
    n: int = 128
    k: int = 128
    format: QuantizationFormat = "int8"
    backend: QuantizationBackend = "reference"
    device: str = "cpu"
    warmup: int = 2
    iterations: int = 5
    seed: int = 0
    verify: bool = True

    def validate(self) -> None:
        if min(self.m, self.n, self.k) <= 0:
            raise ValueError("quantized GEMM dimensions must be positive")
        if self.format not in {"int8", "fp8_e4m3fn"}:
            raise ValueError("format must be int8 or fp8_e4m3fn")
        if self.backend not in {"auto", "cuda", "reference"}:
            raise ValueError("backend must be auto, cuda, or reference")
        if self.iterations <= 0 or self.warmup < 0:
            raise ValueError("iterations must be positive and warmup must be non-negative")
        device = torch.device(self.device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("quantized GEMM benchmark device must be CPU or CUDA")
        if self.backend == "cuda" and device.type != "cuda":
            raise ValueError("backend=cuda requires a CUDA benchmark device")


def quantized_gemm_work_estimate(
    config: QuantizedGEMMBenchmarkConfig,
) -> dict[str, int | bool]:
    """Return analytical payload and matrix-work counts without a speed claim."""

    config.validate()
    return {
        "matrix_flops": 2 * config.m * config.n * config.k,
        "quantized_payload_bytes": config.m * config.k + config.n * config.k,
        "scale_bytes": (config.m + config.n) * 4,
        "fp32_output_bytes": config.m * config.n * 4,
        "analytical_only": True,
    }


def _paired_error_report(actual: Tensor, expected: Tensor) -> dict[str, float | bool]:
    rtol = 2e-5
    atol = 2e-5
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    difference = (actual - expected).abs().to(torch.float64)
    tolerance = atol + rtol * expected.abs().to(torch.float64)
    return {
        "performed": True,
        "rtol": rtol,
        "atol": atol,
        "max_absolute_error": difference.max().item(),
        "max_tolerance_ratio": (difference / tolerance).max().item(),
    }


def _quantization_error_report(actual: Tensor, original: Tensor) -> dict[str, float | bool]:
    difference = (actual - original).abs().to(torch.float64)
    denominator = original.abs().to(torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
    return {
        "performed": True,
        "max_absolute_error": difference.max().item(),
        "max_relative_error_where_reference_nonzero": (difference / denominator).max().item(),
    }


def benchmark_quantized_gemm(config: QuantizedGEMMBenchmarkConfig) -> dict[str, Any]:
    """Benchmark dequantized linear with quantization outside the timed boundary."""

    config.validate()
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA benchmark device is not available")

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    activations = torch.randn(config.m, config.k, generator=generator).to(device)
    weight = torch.randn(config.n, config.k, generator=generator).to(device)
    quantized_activations = quantize_activations(
        activations,
        format=config.format,
        backend=config.backend,
    )
    quantized_weight = quantize_weights(
        weight,
        format=config.format,
        backend=config.backend,
    )
    executed_backend = (
        "cuda"
        if device.type == "cuda"
        and config.backend != "reference"
        and cuda_quantization_available(config.format)
        else "reference"
    )

    def operation() -> Tensor:
        return dequantized_linear(
            quantized_activations,
            quantized_weight,
            backend=config.backend,
        )

    with torch.inference_mode():
        output = operation()
        if config.verify:
            dequantized_reference = dequantized_linear(
                quantized_activations,
                quantized_weight,
                backend="reference",
            )
            original_reference = activations @ weight.T
            verification: dict[str, Any] = {
                "paired_dequantized_reference": _paired_error_report(output, dequantized_reference),
                "original_fp32_linear": _quantization_error_report(output, original_reference),
            }
        else:
            verification = {"performed": False}
        samples = (
            _measure_cuda(operation, config.warmup, config.iterations, device)
            if device.type == "cuda"
            else _measure_cpu(operation, config.warmup, config.iterations)
        )

    return {
        "schema_version": 1,
        "benchmark": "dequantized_quantized_linear",
        "configuration": asdict(config),
        "executed_backend": executed_backend,
        "environment": _environment_metadata(device),
        "output": {
            "shape": list(output.shape),
            "dtype": str(output.dtype).removeprefix("torch."),
            "device": str(output.device),
        },
        "verification": verification,
        "work_estimate": quantized_gemm_work_estimate(config),
        "latency": summarize_latencies(samples),
        "raw_samples_ms": samples,
        "performance_claim": False,
        "notes": [
            "activation and weight quantization execute outside the timed linear call",
            "FP8 E4M3FN uses uint8 payload bits; INT8 uses symmetric [-127, 127] codes",
            "activation scales are per row and weight scales are per output channel",
            "the linear oracle dequantizes both operands and accumulates in float32",
            "the native kernels are scalar teaching kernels, not a Tensor Core claim",
        ],
    }
