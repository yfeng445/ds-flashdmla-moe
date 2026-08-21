import json
import sys
from typing import get_args

import pytest
import torch

from benchmarks import attention as attention_cli
from ds_flash_mla_moe import benchmarking
from ds_flash_mla_moe.benchmarking import (
    AttentionBenchmarkBackend,
    AttentionBenchmarkConfig,
    attention_work_estimate,
    benchmark_attention,
    summarize_latencies,
    write_benchmark_report,
)


def test_attention_benchmark_backend_names_cover_facade_and_baselines() -> None:
    assert get_args(AttentionBenchmarkBackend) == (
        "auto",
        "cuda",
        "cuda_rowwise",
        "reference",
        "blockwise",
        "fa1",
        "fa2",
        "sdpa",
        "flash-attn-4",
    )


def test_formal_fa_benchmark_requires_cuda_float16() -> None:
    with pytest.raises(ValueError, match="CUDA float16"):
        AttentionBenchmarkConfig(backend="fa1", device="cpu", dtype="float16").validate()
    with pytest.raises(ValueError, match="CUDA float16"):
        AttentionBenchmarkConfig(backend="fa2", device="cuda", dtype="float32").validate()


def test_paired_benchmark_uses_the_same_configuration(monkeypatch) -> None:
    seen = []

    def fake_benchmark(config):
        seen.append(config)
        return {
            "configuration": {"backend": config.backend},
            "raw_samples_ms": [1.0],
        }

    monkeypatch.setattr(benchmarking, "benchmark_attention", fake_benchmark)
    base = AttentionBenchmarkConfig(
        backend="fa1",
        device="cuda",
        dtype="float16",
        seed=17,
        query_length=31,
        key_length=47,
    )
    report = benchmarking.benchmark_attention_backends(base, ("fa1", "fa2"))

    assert [config.backend for config in seen] == ["fa1", "fa2"]
    assert all(config.seed == 17 for config in seen)
    assert all(config.query_length == 31 for config in seen)
    assert all(config.key_length == 47 for config in seen)
    assert report == {
        "schema_version": 1,
        "comparison_backends": ["fa1", "fa2"],
        "shared_seed": 17,
        "reports": {
            "fa1": {
                "configuration": {"backend": "fa1"},
                "raw_samples_ms": [1.0],
            },
            "fa2": {
                "configuration": {"backend": "fa2"},
                "raw_samples_ms": [1.0],
            },
        },
    }


@pytest.mark.parametrize("backends", [(), ("fa1", "fa1")])
def test_paired_benchmark_requires_nonempty_unique_backends(backends) -> None:
    with pytest.raises(ValueError, match="non-empty and unique"):
        benchmarking.benchmark_attention_backends(AttentionBenchmarkConfig(), backends)


def test_cli_parses_fa_backends_and_comparison_flag(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["attention.py", "--backend", "fa2"])
    assert attention_cli.parse_args().backend == "fa2"

    monkeypatch.setattr(sys, "argv", ["attention.py", "--compare-fa1-fa2"])
    arguments = attention_cli.parse_args()
    assert arguments.backend == "auto"
    assert arguments.compare_fa1_fa2 is True


def test_cli_rejects_comparison_with_conflicting_backend(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["attention.py", "--backend", "fa1", "--compare-fa1-fa2"],
    )

    with pytest.raises(SystemExit, match="2"):
        attention_cli.parse_args()

    assert "cannot be combined" in capsys.readouterr().err


def test_cli_comparison_writes_paired_report(monkeypatch) -> None:
    captured = {}

    def fake_paired_benchmark(config, backends):
        captured["config"] = config
        captured["backends"] = backends
        return {"comparison_backends": list(backends)}

    def fake_writer(report, path):
        captured["report"] = report
        captured["path"] = path

    monkeypatch.setattr(sys, "argv", ["attention.py", "--compare-fa1-fa2"])
    monkeypatch.setattr(attention_cli, "benchmark_attention_backends", fake_paired_benchmark)
    monkeypatch.setattr(
        attention_cli,
        "benchmark_attention",
        lambda _config: pytest.fail("comparison must not run a single backend report"),
    )
    monkeypatch.setattr(attention_cli, "write_benchmark_report", fake_writer)

    attention_cli.main()

    assert captured["config"].backend == "auto"
    assert captured["backends"] == ("fa1", "fa2")
    assert captured["report"] == {"comparison_backends": ["fa1", "fa2"]}
    assert captured["path"] is None


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


def test_sdpa_baseline_matches_right_aligned_causal_reference() -> None:
    report = benchmark_attention(
        AttentionBenchmarkConfig(
            batch=1,
            heads=2,
            query_length=2,
            key_length=5,
            head_dim=4,
            value_dim=3,
            device="cpu",
            dtype="float64",
            causal=True,
            backend="sdpa",
            warmup=0,
            iterations=1,
        )
    )

    assert report["configuration"]["backend"] == "sdpa"
    assert report["verification"]["performed"] is True
    assert report["verification"]["max_absolute_error"] < 1e-9


def test_flash_attn_4_configuration_is_validated_without_importing_package() -> None:
    AttentionBenchmarkConfig(
        backend="flash-attn-4",
        device="cuda",
        dtype="bfloat16",
    ).validate()


def test_flash_attn_4_adapter_translates_layout_and_tuple_output(monkeypatch) -> None:
    batch, heads, query_length, key_length, head_dim, value_dim = 2, 3, 5, 7, 4, 2
    q = torch.randn(batch, heads, query_length, head_dim)
    k = torch.randn(batch, heads, key_length, head_dim)
    v = torch.randn(batch, heads, key_length, value_dim)

    def fake_flash_attn_4(q_bshd, k_bshd, v_bshd, *, causal):
        assert q_bshd.shape == (batch, query_length, heads, head_dim)
        assert k_bshd.shape == (batch, key_length, heads, head_dim)
        assert v_bshd.shape == (batch, key_length, heads, value_dim)
        assert causal is True
        return v_bshd[:, :query_length], None

    monkeypatch.setattr(
        benchmarking,
        "_load_flash_attn_4",
        lambda: pytest.fail("the injected implementation should avoid a timed import"),
    )

    actual = benchmarking._flash_attn_4_attention_baseline(
        q,
        k,
        v,
        causal=True,
        implementation=fake_flash_attn_4,
    )

    torch.testing.assert_close(actual, v[:, :, :query_length])
    assert actual.is_contiguous()


def test_flash_attn_4_loader_fails_loudly_when_optional_package_is_missing(
    monkeypatch,
) -> None:
    def missing_module(_name: str):
        raise ModuleNotFoundError("flash_attn")

    monkeypatch.setattr(benchmarking, "import_module", missing_module)

    with pytest.raises(RuntimeError, match="requires a working optional flash-attn-4"):
        benchmarking._load_flash_attn_4()


def test_report_writer_emits_valid_json(tmp_path) -> None:
    destination = tmp_path / "nested" / "report.json"
    report = {"schema_version": 1, "value": "测试"}

    write_benchmark_report(report, destination)

    rendered = destination.read_text(encoding="utf-8")
    assert json.loads(rendered) == report
    assert rendered.endswith("\n")


@pytest.mark.parametrize(
    "config",
    [
        AttentionBenchmarkConfig(iterations=0),
        AttentionBenchmarkConfig(warmup=-1),
        AttentionBenchmarkConfig(query_length=5, key_length=4, causal=True),
        AttentionBenchmarkConfig(dtype="int8"),  # type: ignore[arg-type]
        AttentionBenchmarkConfig(backend="unsupported"),  # type: ignore[arg-type]
        AttentionBenchmarkConfig(
            backend="flash-attn-4",
            device="cpu",
            dtype="float16",
        ),
        AttentionBenchmarkConfig(
            backend="flash-attn-4",
            device="cuda",
            dtype="float32",
        ),
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
