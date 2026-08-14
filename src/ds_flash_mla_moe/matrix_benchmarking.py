"""Paired single-GPU benchmark matrices across representative operator shapes."""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import torch

from .benchmarking import AttentionBenchmarkConfig, benchmark_attention
from .expert_benchmarking import ExpertBenchmarkConfig, benchmark_experts
from .gemm_benchmarking import GEMMBenchmarkConfig, benchmark_gemm
from .mla_benchmarking import MLABenchmarkConfig, benchmark_mla
from .router_benchmarking import RouterBenchmarkConfig, benchmark_router

MatrixFamily = Literal["gemm", "attention", "mla", "experts", "router"]
MatrixProfile = Literal["smoke", "representative"]
ShapeClass = Literal["regular", "tail", "decode", "skew"]
BenchmarkConfig = (
    AttentionBenchmarkConfig
    | ExpertBenchmarkConfig
    | GEMMBenchmarkConfig
    | MLABenchmarkConfig
    | RouterBenchmarkConfig
)

_ALL_FAMILIES: tuple[MatrixFamily, ...] = (
    "gemm",
    "attention",
    "mla",
    "experts",
    "router",
)
_SMOKE_CASES = {
    "gemm_tail_127x95x63",
    "attention_decode_regular",
    "mla_prefill_tail",
    "experts_skew_fp16",
    "router_skew_hot_expert",
}


@dataclass(frozen=True)
class BenchmarkMatrixConfig:
    """Execution controls shared by every paired matrix case."""

    device: str = "cuda"
    profile: MatrixProfile = "representative"
    families: tuple[MatrixFamily, ...] = _ALL_FAMILIES
    cases: tuple[str, ...] = ()
    warmup: int = 5
    iterations: int = 20
    seed: int = 0
    verify: bool = True
    fail_fast: bool = False

    def validate(self) -> None:
        try:
            device = torch.device(self.device)
        except RuntimeError as error:
            raise ValueError("matrix benchmark device must be a valid torch device") from error
        if device.type != "cuda":
            raise ValueError("the paired native matrix requires a CUDA device")
        if self.profile not in {"smoke", "representative"}:
            raise ValueError("profile must be smoke or representative")
        if self.warmup < 0 or self.iterations <= 0:
            raise ValueError("warmup must be non-negative and iterations must be positive")
        if not self.families:
            raise ValueError("at least one benchmark family must be selected")
        if len(set(self.families)) != len(self.families):
            raise ValueError("matrix benchmark families must be unique")
        if any(family not in _ALL_FAMILIES for family in self.families):
            raise ValueError("unsupported matrix benchmark family")
        if len(set(self.cases)) != len(self.cases):
            raise ValueError("explicit matrix case names must be unique")


@dataclass(frozen=True)
class BenchmarkMatrixCase:
    """One native/baseline pair with identical inputs and measurement controls."""

    name: str
    family: MatrixFamily
    shape_class: ShapeClass
    description: str
    baseline_label: str
    native_config: BenchmarkConfig
    baseline_config: BenchmarkConfig

    def validate(self) -> None:
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("matrix case names must be non-empty slugs")
        if type(self.native_config) is not type(self.baseline_config):
            raise ValueError("paired matrix configurations must use the same config type")
        if _paired_configuration(self.native_config) != _paired_configuration(self.baseline_config):
            raise ValueError("paired matrix configurations differ beyond backend selection")
        expected_native, expected_baseline = {
            "gemm": ("cuda", "torch"),
            "attention": ("cuda", "sdpa"),
            "mla": ("cuda", "absorbed"),
            "experts": ("cuda", "reference"),
            "router": ("cuda", "reference"),
        }[self.family]
        if _backend_selector(self.native_config) != expected_native:
            raise ValueError(f"{self.name} does not select its native CUDA backend")
        if _backend_selector(self.baseline_config) != expected_baseline:
            raise ValueError(f"{self.name} does not select its expected baseline")


def _backend_selector(config: BenchmarkConfig) -> str:
    if isinstance(config, (GEMMBenchmarkConfig, MLABenchmarkConfig)):
        return config.implementation
    return config.backend


def _paired_configuration(config: BenchmarkConfig) -> dict[str, Any]:
    paired = asdict(config)
    paired.pop("implementation", None)
    paired.pop("backend", None)
    return paired


def _gemm_cases(config: BenchmarkMatrixConfig, start_seed: int) -> list[BenchmarkMatrixCase]:
    shapes = (
        (
            "gemm_regular_128",
            "regular",
            "tile-aligned square GEMM",
            {"m": 128, "n": 128, "k": 128, "alpha": 1.0, "beta": 0.0},
        ),
        (
            "gemm_tail_127x95x63",
            "tail",
            "all three dimensions end in partial tiles",
            {"m": 127, "n": 95, "k": 63, "alpha": 1.0, "beta": 0.0},
        ),
        (
            "gemm_decode_1x256x128",
            "decode",
            "single-row decode-like projection",
            {"m": 1, "n": 256, "k": 128, "alpha": 1.0, "beta": 0.0},
        ),
        (
            "gemm_tail_epilogue_65x97x33",
            "tail",
            "partial tiles with alpha/beta epilogue",
            {"m": 65, "n": 97, "k": 33, "alpha": 0.75, "beta": 0.25},
        ),
    )
    cases = []
    for offset, (name, shape_class, description, shape) in enumerate(shapes):
        native = GEMMBenchmarkConfig(
            **shape,
            tile_m=16,
            tile_n=16,
            tile_k=16,
            dtype="float32",
            device=config.device,
            implementation="cuda",
            warmup=config.warmup,
            iterations=config.iterations,
            seed=start_seed + offset,
            verify=config.verify,
        )
        cases.append(
            BenchmarkMatrixCase(
                name=name,
                family="gemm",
                shape_class=shape_class,
                description=description,
                baseline_label="torch_cublas",
                native_config=native,
                baseline_config=replace(native, implementation="torch"),
            )
        )
    return cases


def _attention_cases(
    config: BenchmarkMatrixConfig,
    start_seed: int,
) -> list[BenchmarkMatrixCase]:
    shapes = (
        (
            "attention_prefill_regular",
            "regular",
            "square causal prefill with conventional head width",
            {
                "batch": 1,
                "heads": 4,
                "query_length": 128,
                "key_length": 128,
                "head_dim": 64,
                "value_dim": 64,
            },
        ),
        (
            "attention_prefill_tail",
            "tail",
            "non-power-of-two causal prefill and unequal QK/V widths",
            {
                "batch": 2,
                "heads": 3,
                "query_length": 127,
                "key_length": 127,
                "head_dim": 40,
                "value_dim": 48,
            },
        ),
        (
            "attention_decode_regular",
            "decode",
            "single-query right-aligned causal decode",
            {
                "batch": 2,
                "heads": 4,
                "query_length": 1,
                "key_length": 128,
                "head_dim": 64,
                "value_dim": 64,
            },
        ),
        (
            "attention_decode_tail",
            "tail",
            "short-query decode with tail sequence and head dimensions",
            {
                "batch": 1,
                "heads": 3,
                "query_length": 7,
                "key_length": 129,
                "head_dim": 40,
                "value_dim": 24,
            },
        ),
    )
    cases = []
    for offset, (name, shape_class, description, shape) in enumerate(shapes):
        native = AttentionBenchmarkConfig(
            **shape,
            dtype="float32",
            device=config.device,
            causal=True,
            backend="cuda",
            warmup=config.warmup,
            iterations=config.iterations,
            seed=start_seed + offset,
            verify=config.verify,
        )
        cases.append(
            BenchmarkMatrixCase(
                name=name,
                family="attention",
                shape_class=shape_class,
                description=description,
                baseline_label="torch_sdpa",
                native_config=native,
                baseline_config=replace(native, backend="sdpa"),
            )
        )
    return cases


def _mla_cases(config: BenchmarkMatrixConfig, start_seed: int) -> list[BenchmarkMatrixCase]:
    regular = {
        "batch": 1,
        "sequence_length": 128,
        "model_dim": 128,
        "n_heads": 4,
        "q_lora_rank": 32,
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 16,
        "v_head_dim": 32,
    }
    tail = {
        "batch": 1,
        "sequence_length": 127,
        "model_dim": 96,
        "n_heads": 3,
        "q_lora_rank": 17,
        "kv_lora_rank": 19,
        "qk_nope_head_dim": 24,
        "qk_rope_head_dim": 10,
        "v_head_dim": 20,
    }
    shapes = (
        (
            "mla_prefill_regular",
            "regular",
            "full prefill including latent-cache projection",
            regular,
            "prefill_with_cache",
        ),
        (
            "mla_prefill_tail",
            "tail",
            "full prefill with tail ranks, dimensions, and sequence length",
            tail,
            "prefill_with_cache",
        ),
        (
            "mla_decode_regular",
            "decode",
            "single-token projection, static-cache write, and decode attention",
            regular,
            "decode_with_static_write",
        ),
        (
            "mla_decode_tail",
            "tail",
            "static decode with tail projection and cache dimensions",
            {**tail, "sequence_length": 129},
            "decode_with_static_write",
        ),
        (
            "mla_decode_direct_query",
            "decode",
            "direct-query path without query LoRA",
            {
                "batch": 2,
                "sequence_length": 65,
                "model_dim": 80,
                "n_heads": 5,
                "q_lora_rank": 0,
                "kv_lora_rank": 13,
                "qk_nope_head_dim": 18,
                "qk_rope_head_dim": 6,
                "v_head_dim": 17,
            },
            "decode_with_static_write",
        ),
    )
    cases = []
    for offset, (name, shape_class, description, shape, workload) in enumerate(shapes):
        native = MLABenchmarkConfig(
            **shape,
            dtype="float32",
            device=config.device,
            implementation="cuda",
            workload=workload,
            warmup=config.warmup,
            iterations=config.iterations,
            seed=start_seed + offset,
            verify=config.verify,
        )
        cases.append(
            BenchmarkMatrixCase(
                name=name,
                family="mla",
                shape_class=shape_class,
                description=description,
                baseline_label="pytorch_absorbed",
                native_config=native,
                baseline_config=replace(native, implementation="absorbed"),
            )
        )
    return cases


def _expert_cases(config: BenchmarkMatrixConfig, start_seed: int) -> list[BenchmarkMatrixCase]:
    shapes = (
        (
            "experts_regular_fp32",
            "regular",
            "balanced tile-aligned FP32 experts",
            {
                "expert_counts": (32, 32, 32, 32),
                "model_dim": 64,
                "hidden_dim": 128,
                "dtype": "float32",
            },
        ),
        (
            "experts_tail_fp32",
            "tail",
            "balanced active rows with tail model and hidden widths",
            {
                "expert_counts": (17, 17, 17, 17),
                "model_dim": 65,
                "hidden_dim": 97,
                "dtype": "float32",
            },
        ),
        (
            "experts_skew_fp32",
            "skew",
            "skewed FP32 load with empty experts",
            {
                "expert_counts": (61, 3, 0, 1, 27, 0, 5, 11),
                "model_dim": 64,
                "hidden_dim": 96,
                "dtype": "float32",
            },
        ),
        (
            "experts_skew_fp16",
            "skew",
            "skewed FP16 WMMA load with dimension and row tails",
            {"expert_counts": (61, 3, 0, 1), "model_dim": 65, "hidden_dim": 97, "dtype": "float16"},
        ),
    )
    cases = []
    for offset, (name, shape_class, description, shape) in enumerate(shapes):
        native = ExpertBenchmarkConfig(
            **shape,
            device=config.device,
            backend="cuda",
            warmup=config.warmup,
            iterations=config.iterations,
            seed=start_seed + offset,
            backward=False,
            verify=config.verify,
        )
        cases.append(
            BenchmarkMatrixCase(
                name=name,
                family="experts",
                shape_class=shape_class,
                description=description,
                baseline_label="pytorch_padded",
                native_config=native,
                baseline_config=replace(native, backend="reference"),
            )
        )
    return cases


def _router_cases(config: BenchmarkMatrixConfig, start_seed: int) -> list[BenchmarkMatrixCase]:
    shapes = (
        (
            "router_regular",
            "regular",
            "regular token and expert counts without correction bias",
            {
                "tokens": 256,
                "model_dim": 64,
                "experts": 8,
                "topk": 2,
                "n_groups": 4,
                "topk_groups": 2,
                "hot_expert_bias": 0.0,
            },
        ),
        (
            "router_tail",
            "tail",
            "tail token/model dimensions and three expert groups",
            {
                "tokens": 257,
                "model_dim": 65,
                "experts": 12,
                "topk": 3,
                "n_groups": 3,
                "topk_groups": 2,
                "hot_expert_bias": 0.0,
            },
        ),
        (
            "router_skew_hot_expert",
            "skew",
            "selection skew induced by a correction bias",
            {
                "tokens": 256,
                "model_dim": 64,
                "experts": 8,
                "topk": 3,
                "n_groups": 4,
                "topk_groups": 2,
                "hot_expert_bias": 0.4,
            },
        ),
    )
    cases = []
    for offset, (name, shape_class, description, shape) in enumerate(shapes):
        native = RouterBenchmarkConfig(
            **shape,
            dtype="float32",
            device=config.device,
            backend="cuda",
            warmup=config.warmup,
            iterations=config.iterations,
            seed=start_seed + offset,
            backward=False,
            verify=config.verify,
        )
        cases.append(
            BenchmarkMatrixCase(
                name=name,
                family="router",
                shape_class=shape_class,
                description=description,
                baseline_label="pytorch_reference",
                native_config=native,
                baseline_config=replace(native, backend="reference"),
            )
        )
    return cases


def build_benchmark_matrix_cases(
    config: BenchmarkMatrixConfig,
) -> tuple[BenchmarkMatrixCase, ...]:
    """Build and validate the deterministic case manifest for one matrix profile."""

    config.validate()
    builders = (_gemm_cases, _attention_cases, _mla_cases, _expert_cases, _router_cases)
    all_cases: list[BenchmarkMatrixCase] = []
    next_seed = config.seed
    for builder in builders:
        family_cases = builder(config, next_seed)
        all_cases.extend(family_cases)
        next_seed += len(family_cases)

    names = {case.name for case in all_cases}
    missing = sorted(set(config.cases) - names)
    if missing:
        raise ValueError(f"unknown matrix case names: {', '.join(missing)}")
    selected = [case for case in all_cases if case.family in config.families]
    if config.profile == "smoke":
        selected = [case for case in selected if case.name in _SMOKE_CASES]
    if config.cases:
        selected = [case for case in selected if case.name in config.cases]
    if not selected:
        raise ValueError("matrix filters selected no benchmark cases")
    for case in selected:
        case.validate()
    return tuple(selected)


def benchmark_matrix_manifest(config: BenchmarkMatrixConfig) -> dict[str, Any]:
    """Return the selected cases without allocating tensors or requiring a GPU."""

    cases = build_benchmark_matrix_cases(config)
    return {
        "schema_version": 1,
        "benchmark": "single_gpu_operator_matrix_manifest",
        "matrix_configuration": asdict(config),
        "case_count": len(cases),
        "cases": [
            {
                "name": case.name,
                "family": case.family,
                "shape_class": case.shape_class,
                "description": case.description,
                "baseline_label": case.baseline_label,
                "native_selector": _backend_selector(case.native_config),
                "baseline_selector": _backend_selector(case.baseline_config),
                "paired_configuration": _paired_configuration(case.native_config),
            }
            for case in cases
        ],
    }


def _execute_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    if isinstance(config, GEMMBenchmarkConfig):
        return benchmark_gemm(config)
    if isinstance(config, AttentionBenchmarkConfig):
        return benchmark_attention(config)
    if isinstance(config, MLABenchmarkConfig):
        return benchmark_mla(config)
    if isinstance(config, ExpertBenchmarkConfig):
        return benchmark_experts(config)
    if isinstance(config, RouterBenchmarkConfig):
        return benchmark_router(config)
    raise TypeError(f"unsupported benchmark configuration: {type(config).__name__}")


def _checked_median(report: dict[str, Any], label: str) -> float:
    try:
        median = float(report["latency"]["median_ms"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} report does not contain a numeric median_ms") from error
    if not math.isfinite(median) or median <= 0:
        raise ValueError(f"{label} report median_ms must be finite and positive")
    return median


def _run_matrix_case(case: BenchmarkMatrixCase, index: int, verify: bool) -> dict[str, Any]:
    order = ("native", "baseline") if index % 2 == 0 else ("baseline", "native")
    reports: dict[str, dict[str, Any]] = {}
    for label in order:
        selected_config = case.native_config if label == "native" else case.baseline_config
        reports[label] = _execute_benchmark(selected_config)
    native_report = reports["native"]
    baseline_report = reports["baseline"]
    if native_report.get("output") != baseline_report.get("output"):
        raise AssertionError("paired benchmark output metadata differs")
    if verify:
        for label, report in reports.items():
            if report.get("verification", {}).get("performed") is not True:
                raise AssertionError(f"{label} report did not perform numerical verification")

    native_median = _checked_median(native_report, "native")
    baseline_median = _checked_median(baseline_report, "baseline")
    ratio = native_median / baseline_median
    if math.isclose(native_median, baseline_median, rel_tol=1e-12, abs_tol=0.0):
        lower_median = "tie"
    else:
        lower_median = "native" if native_median < baseline_median else "baseline"
    return {
        "name": case.name,
        "family": case.family,
        "shape_class": case.shape_class,
        "description": case.description,
        "baseline_label": case.baseline_label,
        "execution_order": list(order),
        "paired_configuration": _paired_configuration(case.native_config),
        "comparison": {
            "native_median_ms": native_median,
            "baseline_median_ms": baseline_median,
            "native_over_baseline": ratio,
            "lower_median": lower_median,
        },
        "native_report": native_report,
        "baseline_report": baseline_report,
    }


def _matrix_summary(
    selected: tuple[BenchmarkMatrixCase, ...],
    completed: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    ratios = [float(case["comparison"]["native_over_baseline"]) for case in completed]
    lower_counts = Counter(case["comparison"]["lower_median"] for case in completed)
    family_selected = Counter(case.family for case in selected)
    family_completed = Counter(case["family"] for case in completed)
    shape_selected = Counter(case.shape_class for case in selected)
    ratio_summary = None
    if ratios:
        ratio_summary = {
            "count": len(ratios),
            "minimum": min(ratios),
            "median": statistics.median(ratios),
            "geometric_mean": math.exp(statistics.fmean(math.log(ratio) for ratio in ratios)),
            "maximum": max(ratios),
        }
    return {
        "status": "success" if not failures else "partial" if completed else "failed",
        "selected_case_count": len(selected),
        "completed_case_count": len(completed),
        "failed_case_count": len(failures),
        "unattempted_case_count": len(selected) - len(completed) - len(failures),
        "selected_by_family": dict(sorted(family_selected.items())),
        "completed_by_family": dict(sorted(family_completed.items())),
        "selected_by_shape_class": dict(sorted(shape_selected.items())),
        "lower_median_counts": {
            "native": lower_counts["native"],
            "baseline": lower_counts["baseline"],
            "tie": lower_counts["tie"],
        },
        "native_over_baseline": ratio_summary,
    }


def benchmark_operator_matrix(config: BenchmarkMatrixConfig) -> dict[str, Any]:
    """Execute paired cases, retaining raw reports and isolating case failures."""

    selected = build_benchmark_matrix_cases(config)
    if not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA matrix benchmark device is not available")
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, case in enumerate(selected):
        try:
            completed.append(_run_matrix_case(case, index, config.verify))
        except Exception as error:  # noqa: BLE001 - matrix reports must preserve later cases
            failures.append(
                {
                    "name": case.name,
                    "family": case.family,
                    "shape_class": case.shape_class,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            torch.cuda.empty_cache()
            if config.fail_fast:
                break
    elapsed = time.perf_counter() - started
    environment = completed[0]["native_report"].get("environment", {}) if completed else {}
    return {
        "schema_version": 1,
        "benchmark": "single_gpu_operator_matrix",
        "matrix_configuration": asdict(config),
        "environment": environment,
        "summary": _matrix_summary(selected, completed, failures),
        "elapsed_wall_seconds": elapsed,
        "cases": completed,
        "failures": failures,
        "notes": [
            "each side independently verifies against its operator reference before comparison",
            "native/baseline execution order alternates between adjacent cases",
            "each ratio compares medians for one fixed case and environment",
            (
                "aggregate ratio statistics are unweighted descriptors across heterogeneous "
                "cases and baselines, not an overall speedup"
            ),
            "nested reports retain every raw post-warmup latency sample",
            "traffic, FLOPs, and tile counts remain analytical unless a profiler report says otherwise",
        ],
    }
