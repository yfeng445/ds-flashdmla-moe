from dataclasses import asdict

import pytest

from ds_flash_mla_moe import matrix_benchmarking
from ds_flash_mla_moe.matrix_benchmarking import (
    BenchmarkMatrixConfig,
    benchmark_matrix_case_side,
    benchmark_matrix_manifest,
    benchmark_operator_matrix,
    build_benchmark_matrix_cases,
)


def test_representative_matrix_covers_families_and_shape_classes() -> None:
    config = BenchmarkMatrixConfig(warmup=0, iterations=1)
    cases = build_benchmark_matrix_cases(config)

    assert len(cases) == 20
    assert len({case.name for case in cases}) == len(cases)
    assert {case.family for case in cases} == {
        "gemm",
        "attention",
        "mla",
        "experts",
        "router",
    }
    assert {case.shape_class for case in cases} == {"regular", "tail", "decode", "skew"}
    for case in cases:
        case.validate()
        native = asdict(case.native_config)
        baseline = asdict(case.baseline_config)
        native.pop("implementation", None)
        native.pop("backend", None)
        baseline.pop("implementation", None)
        baseline.pop("backend", None)
        assert native == baseline


def test_smoke_matrix_selects_one_case_per_family() -> None:
    report = benchmark_matrix_manifest(
        BenchmarkMatrixConfig(profile="smoke", warmup=0, iterations=1)
    )

    assert report["case_count"] == 5
    assert {case["family"] for case in report["cases"]} == {
        "gemm",
        "attention",
        "mla",
        "experts",
        "router",
    }
    assert all(case["native_selector"] == "cuda" for case in report["cases"])


def test_flash_attn_4_profile_contains_only_same_dtype_attention_pairs() -> None:
    config = BenchmarkMatrixConfig(profile="flash-attn-4", warmup=0, iterations=1)
    cases = build_benchmark_matrix_cases(config)

    assert len(cases) == 4
    assert {case.name for case in cases} == {
        "attention_fa4_prefill_bfloat16",
        "attention_fa4_prefill_tail_float16",
        "attention_fa4_decode_bfloat16",
        "attention_fa4_decode_tail_float16",
    }
    assert {case.shape_class for case in cases} == {"regular", "tail", "decode"}
    for case in cases:
        case.validate()
        assert case.family == "attention"
        assert case.baseline_label == "flash_attention_4"
        assert case.native_config.backend == "cuda"
        assert case.baseline_config.backend == "flash-attn-4"
        assert case.native_config.dtype in {"float16", "bfloat16"}
        assert case.native_config.dtype == case.baseline_config.dtype
        assert asdict(case.native_config) | {"backend": "flash-attn-4"} == asdict(
            case.baseline_config
        )


def test_mla_low_precision_profile_contains_only_same_dtype_staged_pairs() -> None:
    config = BenchmarkMatrixConfig(profile="mla-low-precision", warmup=0, iterations=1)
    cases = build_benchmark_matrix_cases(config)

    assert len(cases) == 4
    assert {case.name for case in cases} == {
        "mla_low_precision_prefill_bfloat16",
        "mla_low_precision_prefill_tail_float16",
        "mla_low_precision_decode_bfloat16",
        "mla_low_precision_decode_tail_float16",
    }
    assert {case.shape_class for case in cases} == {"regular", "tail", "decode"}
    for case in cases:
        case.validate()
        assert case.family == "mla"
        assert case.baseline_label == "pytorch_absorbed"
        assert case.native_config.implementation == "cuda"
        assert case.baseline_config.implementation == "absorbed"
        assert case.native_config.dtype in {"float16", "bfloat16"}
        assert case.native_config.dtype == case.baseline_config.dtype
        assert asdict(case.native_config) | {"implementation": "absorbed"} == asdict(
            case.baseline_config
        )


def test_mla_paged_profile_contains_long_and_tail_decode_pairs() -> None:
    config = BenchmarkMatrixConfig(profile="mla-paged", warmup=0, iterations=1)
    cases = build_benchmark_matrix_cases(config)

    assert {case.name for case in cases} == {
        "mla_paged_decode_bfloat16_long",
        "mla_paged_decode_tail_float16",
    }
    assert {case.shape_class for case in cases} == {"decode", "tail"}
    for case in cases:
        case.validate()
        assert case.family == "mla"
        assert case.baseline_label == "pytorch_paged_absorbed"
        assert case.native_config.implementation == "cuda"
        assert case.baseline_config.implementation == "absorbed"
        assert case.native_config.workload == "decode_with_paged_write"
        assert case.native_config.sequence_length % case.native_config.page_size == 1
        assert asdict(case.native_config) | {"implementation": "absorbed"} == asdict(
            case.baseline_config
        )


def test_matrix_filters_family_and_exact_case() -> None:
    config = BenchmarkMatrixConfig(
        families=("attention", "mla"),
        cases=("attention_decode_tail", "mla_decode_direct_query"),
        warmup=0,
        iterations=1,
    )

    assert [case.name for case in build_benchmark_matrix_cases(config)] == [
        "attention_decode_tail",
        "mla_decode_direct_query",
    ]


@pytest.mark.parametrize(
    "config,match",
    [
        (BenchmarkMatrixConfig(device="cpu"), "CUDA"),
        (BenchmarkMatrixConfig(iterations=0), "iterations"),
        (BenchmarkMatrixConfig(families=()), "family"),
        (BenchmarkMatrixConfig(cases=("missing",)), "unknown"),
        (
            BenchmarkMatrixConfig(
                profile="smoke",
                families=("gemm",),
                cases=("gemm_regular_128",),
            ),
            "selected no",
        ),
    ],
)
def test_invalid_matrix_configuration_is_rejected(
    config: BenchmarkMatrixConfig,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_benchmark_matrix_cases(config)


def _fake_report(config, *, median: float) -> dict:
    return {
        "configuration": asdict(config),
        "environment": {"device": str(config.device)},
        "output": {"shape": [1], "dtype": "float32", "device": str(config.device)},
        "verification": {"performed": True},
        "latency": {"median_ms": median},
        "raw_samples_ms": [median],
    }


def test_matrix_case_side_executes_the_requested_configuration(monkeypatch) -> None:
    case = build_benchmark_matrix_cases(
        BenchmarkMatrixConfig(
            cases=("gemm_regular_128",),
            warmup=0,
            iterations=1,
        )
    )[0]
    executed = []

    def execute(config):
        executed.append(config)
        return {"selector": config.implementation}

    monkeypatch.setattr(matrix_benchmarking, "_execute_benchmark", execute)

    assert benchmark_matrix_case_side(case, "native") == {"selector": "cuda"}
    assert benchmark_matrix_case_side(case, "baseline") == {"selector": "torch"}
    assert executed == [case.native_config, case.baseline_config]

    with pytest.raises(ValueError, match="side"):
        benchmark_matrix_case_side(case, "unknown")  # type: ignore[arg-type]


def test_matrix_report_pairs_results_and_summarizes_ratios(monkeypatch) -> None:
    monkeypatch.setattr(matrix_benchmarking.torch.cuda, "is_available", lambda: True)

    def execute(config):
        median = 1.0 if getattr(config, "implementation", None) == "cuda" else 2.0
        return _fake_report(config, median=median)

    monkeypatch.setattr(matrix_benchmarking, "_execute_benchmark", execute)
    report = benchmark_operator_matrix(
        BenchmarkMatrixConfig(
            profile="smoke",
            families=("gemm",),
            warmup=0,
            iterations=1,
        )
    )

    assert report["summary"]["status"] == "success"
    assert report["summary"]["completed_case_count"] == 1
    assert report["summary"]["native_over_baseline"]["median"] == 0.5
    assert report["cases"][0]["comparison"]["lower_median"] == "native"
    assert report["cases"][0]["execution_order"] == ["native", "baseline"]
    assert any("not an overall speedup" in note for note in report["notes"])


def test_matrix_report_isolates_failure_and_continues(monkeypatch) -> None:
    monkeypatch.setattr(matrix_benchmarking.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(matrix_benchmarking.torch.cuda, "empty_cache", lambda: None)

    def execute(config):
        if config.__class__.__name__ == "GEMMBenchmarkConfig":
            raise RuntimeError("synthetic failure")
        selector = getattr(config, "implementation", getattr(config, "backend", ""))
        return _fake_report(config, median=1.0 if selector == "cuda" else 2.0)

    monkeypatch.setattr(matrix_benchmarking, "_execute_benchmark", execute)
    report = benchmark_operator_matrix(
        BenchmarkMatrixConfig(
            profile="smoke",
            families=("gemm", "attention"),
            warmup=0,
            iterations=1,
        )
    )

    assert report["summary"]["status"] == "partial"
    assert report["summary"]["failed_case_count"] == 1
    assert report["summary"]["completed_case_count"] == 1
    assert report["failures"][0]["name"] == "gemm_tail_127x95x63"
    assert report["failures"][0]["message"] == "synthetic failure"
