"""Structured benchmarks for DeepSeek-style grouped routing."""

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
from .moe import RoutingResult, deepseek_grouped_topk
from .router_ops import RouterBackend, grouped_topk


@dataclass(frozen=True)
class RouterBenchmarkConfig:
    """Router tensor shape, selection policy, and measurement settings."""

    tokens: int = 128
    model_dim: int = 64
    experts: int = 8
    topk: int = 2
    n_groups: int = 1
    topk_groups: int | None = None
    hot_expert_bias: float = 0.0
    route_scale: float = 1.0
    dtype: str = "float32"
    device: str = "cpu"
    backend: RouterBackend = "reference"
    warmup: int = 2
    iterations: int = 5
    seed: int = 0
    backward: bool = False
    verify: bool = True

    def validate(self) -> None:
        if min(self.tokens, self.model_dim, self.experts, self.topk, self.n_groups) <= 0:
            raise ValueError("router dimensions must be positive")
        if self.experts % self.n_groups:
            raise ValueError("experts must be divisible by n_groups")
        effective_topk_groups = self.n_groups if self.topk_groups is None else self.topk_groups
        if not 1 <= effective_topk_groups <= self.n_groups:
            raise ValueError("topk_groups must be in [1, n_groups]")
        if self.topk > effective_topk_groups * (self.experts // self.n_groups):
            raise ValueError("topk exceeds the experts retained by group selection")
        if not math.isfinite(self.hot_expert_bias) or self.hot_expert_bias < 0:
            raise ValueError("hot_expert_bias must be finite and non-negative")
        if not math.isfinite(self.route_scale):
            raise ValueError("route_scale must be finite")
        if self.iterations <= 0 or self.warmup < 0:
            raise ValueError("iterations must be positive and warmup must be non-negative")
        if self.dtype not in {"float16", "bfloat16", "float32", "float64"}:
            raise ValueError("unsupported router benchmark dtype")
        if self.backend not in {"auto", "cuda", "reference"}:
            raise ValueError("backend must be auto, cuda, or reference")
        try:
            device_type = torch.device(self.device).type
        except RuntimeError as error:
            raise ValueError("router benchmark device must be a valid torch device") from error
        if self.backend == "cuda" and (device_type != "cuda" or self.dtype != "float32"):
            raise ValueError("backend=cuda requires device=cuda and dtype=float32")


def router_work_estimate(config: RouterBenchmarkConfig) -> dict[str, int]:
    """Count the router projection and selection candidates."""

    config.validate()
    topk_groups = config.n_groups if config.topk_groups is None else config.topk_groups
    retained_candidates = topk_groups * (config.experts // config.n_groups)
    return {
        "router_projection_matrix_flops": 2 * config.tokens * config.model_dim * config.experts,
        "logit_elements": config.tokens * config.experts,
        "group_score_candidates": config.tokens * config.n_groups,
        "retained_expert_candidates": config.tokens * retained_candidates,
        "selected_routes": config.tokens * config.topk,
    }


def _make_inputs(
    config: RouterBenchmarkConfig,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor | None]:
    dtype = _dtype_from_name(config.dtype)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    x = torch.randn(
        config.tokens,
        config.model_dim,
        dtype=dtype,
        generator=generator,
    ).to(device)
    gate = torch.randn(
        config.experts,
        config.model_dim,
        dtype=dtype,
        generator=generator,
    ).to(device)
    bias = None
    if config.hot_expert_bias:
        bias = torch.zeros(config.experts, dtype=dtype, device=device)
        bias[0] = config.hot_expert_bias
    if config.backward:
        x.requires_grad_()
        gate.requires_grad_()
    return x, gate, bias


def _run_operation(
    config: RouterBenchmarkConfig,
    inputs: tuple[Tensor, Tensor, Tensor | None],
) -> RoutingResult:
    x, gate, bias = inputs
    if config.backward:
        x.grad = None
        gate.grad = None
    routing = grouped_topk(
        x,
        gate,
        topk=config.topk,
        n_groups=config.n_groups,
        topk_groups=config.topk_groups,
        score_bias=bias,
        route_scale=config.route_scale,
        backend=config.backend,
    )
    if config.backward:
        routing.weights.to(torch.float64).square().sum().backward()
    return routing


def _normalized_error(actual: Tensor, expected: Tensor, rtol: float, atol: float) -> Tensor:
    if actual.numel() == 0:
        return torch.zeros(3, dtype=torch.float64, device=actual.device)
    actual64 = actual.detach().to(torch.float64)
    expected64 = expected.detach().to(torch.float64)
    difference = (actual64 - expected64).abs()
    denominator = expected64.abs().clamp_min(torch.finfo(torch.float64).tiny)
    tolerance = atol + rtol * expected64.abs()
    return torch.stack(
        (difference.max(), (difference / denominator).max(), (difference / tolerance).max())
    )


def _verify(
    config: RouterBenchmarkConfig,
    inputs: tuple[Tensor, Tensor, Tensor | None],
    actual: RoutingResult,
) -> dict[str, Any]:
    x, gate, bias = inputs
    expected_x = x.detach().clone().requires_grad_(config.backward)
    expected_gate = gate.detach().clone().requires_grad_(config.backward)
    expected = deepseek_grouped_topk(
        expected_x,
        expected_gate,
        topk=config.topk,
        n_groups=config.n_groups,
        topk_groups=config.topk_groups,
        score_bias=bias,
        route_scale=config.route_scale,
    )
    if not torch.equal(actual.indices, expected.indices):
        mismatch = torch.count_nonzero(actual.indices != expected.indices).item()
        raise AssertionError(f"router selected {mismatch} expert slots differently")
    if config.backward:
        expected.weights.to(torch.float64).square().sum().backward()
    rtol, atol = _verification_tolerances(actual.weights.dtype)
    output_error = _normalized_error(actual.weights, expected.weights, rtol, atol)
    gradient_error = torch.zeros_like(output_error)
    if config.backward:
        gradient_error = torch.stack(
            [
                _normalized_error(x.grad, expected_x.grad, rtol, atol),
                _normalized_error(gate.grad, expected_gate.grad, rtol, atol),
            ]
        ).amax(dim=0)
    if output_error[2].item() > 1.0:
        raise AssertionError(f"router weight verification failed: {output_error.tolist()}")
    if config.backward and gradient_error[2].item() > 1.0:
        raise AssertionError(f"router gradient verification failed: {gradient_error.tolist()}")
    return {
        "performed": True,
        "reference": "deepseek_grouped_topk",
        "indices_exact": True,
        "rtol": rtol,
        "atol": atol,
        "weights": {
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


def _expert_load(indices: Tensor, experts: int) -> dict[str, Any]:
    values = torch.bincount(indices.reshape(-1), minlength=experts).cpu().tolist()
    total = sum(values)
    mean = total / experts
    variance = sum((value - mean) ** 2 for value in values) / experts
    return {
        "values": values,
        "total": total,
        "mean": mean,
        "minimum": min(values),
        "maximum": max(values),
        "peak_to_mean": max(values) / mean,
        "coefficient_of_variation": math.sqrt(variance) / mean,
        "zero_load_count": sum(value == 0 for value in values),
    }


def benchmark_router(config: RouterBenchmarkConfig) -> dict[str, Any]:
    """Benchmark grouped routing and return a JSON-serializable report."""

    config.validate()
    device = torch.device(config.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("router benchmark device must be CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA benchmark device is not available")
    if device.type == "cpu" and config.dtype == "float16":
        raise ValueError("float16 CPU routing is not a supported benchmark")
    inputs = _make_inputs(config, device=device)
    routing = _run_operation(config, inputs)
    verification = _verify(config, inputs, routing) if config.verify else {"performed": False}
    samples = (
        _measure_cuda(
            lambda: _run_operation(config, inputs).weights,
            config.warmup,
            config.iterations,
            device,
        )
        if device.type == "cuda"
        else _measure_cpu(
            lambda: _run_operation(config, inputs).weights,
            config.warmup,
            config.iterations,
        )
    )
    latency = summarize_latencies(samples)
    work = router_work_estimate(config)
    median_seconds = float(latency["median_ms"]) / 1000.0
    if median_seconds <= 0:
        raise RuntimeError("measured median latency must be positive")
    return {
        "schema_version": 1,
        "benchmark": "deepseek_grouped_topk",
        "configuration": asdict(config),
        "environment": _environment_metadata(device),
        "output": {
            "weights_shape": list(routing.weights.shape),
            "indices_shape": list(routing.indices.shape),
            "dtype": str(routing.weights.dtype).removeprefix("torch."),
            "device": str(routing.weights.device),
        },
        "verification": verification,
        "expert_load": _expert_load(routing.indices, config.experts),
        "work_estimate": work,
        "latency": latency,
        "derived": {
            "forward_router_projection_tflops_equivalent_at_median": (
                work["router_projection_matrix_flops"] / median_seconds / 1e12
            )
        },
        "raw_samples_ms": samples,
        "notes": [
            "router matrix FLOPs count the dense gate projection only",
            "sigmoid, group scoring, Top-K, gather, and normalization are excluded from FLOPs",
            "correction bias affects selection but not gathered route-weight values",
            "forward-equivalent FLOPs remain forward-only when backward timing is enabled",
            "the CUDA selector is a correctness kernel with serial candidate scans per token",
        ],
    }
