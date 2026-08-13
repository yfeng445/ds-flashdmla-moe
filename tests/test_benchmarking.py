import json

import pytest

from ds_flash_mla_moe.benchmarking import (
    AttentionBenchmarkConfig,
    attention_work_estimate,
    benchmark_attention,
    summarize_latencies,
    write_benchmark_report,
)


def test_latency_summary_uses_interpolated_percentiles() -> None:
    summary = summarize_latencies([5.0, 1.0, 4.0, 2.0, 3.0])

    assert summary == {
        "count": 5,
        "min_ms": 1.0,
        "mean_ms": 3.0,
        "median_ms": 3.0,
        "p90_ms": pytest.approx(4.6),
        "p99_ms": pytest.approx(4.96),
        "max_ms": 5.0,
    }


def test_work_estimate_counts_matrix_flops_and_tensor_bytes() -> None:
    config = AttentionBenchmarkConfig(
        batch=2,
        heads=3,
        query_length=5,
        key_length=7,
        head_dim=11,
        value_dim=13,
        dtype="float32",
        iterations=1,
    )

    estimate = attention_work_estimate(config)

    assert estimate["matrix_flops"] == 2 * 2 * 3 * 5 * 7 * (11 + 13)
    expected_elements = 2 * 3 * (5 * 11 + 7 * 11 + 7 * 13 + 5 * 13)
    assert estimate["compulsory_tensor_bytes_lower_bound"] == expected_elements * 4


def test_small_cpu_benchmark_produces_self_describing_report() -> None:
    config = AttentionBenchmarkConfig(
        batch=1,
        heads=1,
        query_length=3,
        key_length=5,
        head_dim=4,
        value_dim=2,
        device="cpu",
        dtype="float64",
        causal=True,
        backend="reference",
        warmup=0,
        iterations=2,
        seed=17,
        reference_block_size=2,
    )

    report = benchmark_attention(config)

    assert report["schema_version"] == 1
    assert report["configuration"]["seed"] == 17
    assert report["output"] == {
        "shape": [1, 1, 3, 2],
        "dtype": "float64",
        "device": "cpu",
    }
    assert report["latency"]["count"] == 2
    assert len(report["raw_samples_ms"]) == 2
    assert report["environment"]["package"] == "0.1.0.dev0"
    assert report["environment"]["source_revision"]
    assert isinstance(report["environment"]["source_dirty"], bool)
    assert report["verification"]["performed"] is True
    assert report["verification"]["max_absolute_error"] < 1e-9
    assert report["derived"]["matrix_tflops_at_median"] >= 0


def test_benchmark_can_explicitly_skip_verification() -> None:
    config = AttentionBenchmarkConfig(
        batch=1,
        heads=1,
        query_length=2,
        key_length=2,
        head_dim=2,
        value_dim=2,
        warmup=0,
        iterations=1,
        verify=False,
    )

    report = benchmark_attention(config)

    assert report["verification"] == {"performed": False}


def test_report_writer_emits_valid_json(tmp_path) -> None:
    destination = tmp_path / "nested" / "report.json"
    report = {"schema_version": 1, "value": "测试"}

    write_benchmark_report(report, destination)

    assert json.loads(destination.read_text()) == report
    assert destination.read_text().endswith("\n")


@pytest.mark.parametrize(
    "config",
    [
        AttentionBenchmarkConfig(iterations=0),
        AttentionBenchmarkConfig(warmup=-1),
        AttentionBenchmarkConfig(query_length=5, key_length=4, causal=True),
        AttentionBenchmarkConfig(dtype="int8"),  # type: ignore[arg-type]
    ],
)
def test_invalid_benchmark_configuration_is_rejected(config: AttentionBenchmarkConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_invalid_latency_samples_are_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_latencies([])
    with pytest.raises(ValueError):
        summarize_latencies([1.0, float("nan")])
    with pytest.raises(ValueError):
        summarize_latencies([-1.0])
