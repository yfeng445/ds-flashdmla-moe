import pytest
import torch

from ds_flash_mla_moe.mla_benchmarking import (
    MLABenchmarkConfig,
    _error_report,
    benchmark_mla,
    mla_cache_storage_estimate,
    mla_work_estimate,
)


@pytest.mark.parametrize("implementation", ["naive", "absorbed", "cuda"])
def test_mla_error_report_uses_composed_fp32_tolerance(implementation: str) -> None:
    expected = torch.tensor([1.0], dtype=torch.float32)
    actual = expected + 4e-4

    report = _error_report(actual, expected, implementation=implementation)  # type: ignore[arg-type]

    assert report["rtol"] == report["atol"] == 5e-4
    assert report["max_tolerance_ratio"] <= 1


@pytest.mark.parametrize(
    ("dtype", "expected_rtol", "expected_atol"),
    [
        (torch.float16, 1e-2, 2e-2),
        (torch.bfloat16, 5e-2, 3e-1),
    ],
)
def test_mla_error_report_uses_composed_low_precision_tolerance(
    dtype: torch.dtype,
    expected_rtol: float,
    expected_atol: float,
) -> None:
    expected = torch.tensor([1.0], dtype=dtype)
    report = _error_report(expected, expected, implementation="cuda")

    assert report["rtol"] == expected_rtol
    assert report["atol"] == expected_atol
    assert report["max_tolerance_ratio"] == 0


def _small_config(**overrides) -> MLABenchmarkConfig:
    values = {
        "batch": 2,
        "sequence_length": 5,
        "page_size": 2,
        "model_dim": 8,
        "n_heads": 3,
        "q_lora_rank": 4,
        "kv_lora_rank": 3,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "dtype": "float64",
        "device": "cpu",
        "warmup": 0,
        "iterations": 1,
        "seed": 157,
    }
    values.update(overrides)
    return MLABenchmarkConfig(**values)


def test_mla_cache_storage_estimate_matches_tensor_layout_formula() -> None:
    config = _small_config()

    estimate = mla_cache_storage_estimate(config)

    latent_elements = 2 * 5 * (3 + 2)
    expanded_elements = 2 * 5 * 3 * (2 + 2 + 2)
    assert estimate == {
        "latent_payload_elements": latent_elements,
        "expanded_kv_payload_elements": expanded_elements,
        "latent_payload_bytes": latent_elements * 8,
        "expanded_kv_payload_bytes": expanded_elements * 8,
        "position_metadata_bytes": 5 * 8,
        "latent_total_bytes_with_positions": latent_elements * 8 + 5 * 8,
        "expanded_total_bytes_with_positions": expanded_elements * 8 + 5 * 8,
        "payload_compression_ratio": expanded_elements / latent_elements,
    }


def test_mla_work_estimate_distinguishes_prefill_decode_and_append_copy() -> None:
    prefill = mla_work_estimate(_small_config(workload="prefill_attention"))
    decode = mla_work_estimate(_small_config(workload="decode_attention"))
    append = mla_work_estimate(_small_config(workload="decode_with_append"))
    static = mla_work_estimate(_small_config(workload="decode_with_static_write"))
    paged = mla_work_estimate(_small_config(workload="decode_with_paged_write"))

    assert prefill["query_length"] == 5
    assert decode["query_length"] == 1
    assert prefill["key_length"] == decode["key_length"] == 5
    assert prefill["cache_projection_matrix_flops"] == 0
    assert decode["cache_projection_matrix_flops"] == 0
    assert append["cache_projection_matrix_flops"] == 2 * 2 * 1 * 8 * (3 + 2)
    assert static["cache_projection_matrix_flops"] == append["cache_projection_matrix_flops"]
    assert paged["cache_projection_matrix_flops"] == append["cache_projection_matrix_flops"]
    assert append["functional_append_copy_bytes_lower_bound"] == (
        2 * 2 * 5 * (3 + 2) * 8 + 2 * 5 * 8
    )
    assert prefill["total_matrix_flops"] > decode["total_matrix_flops"]
    assert append["static_cache_storage_write_bytes"] == 0
    assert static["functional_append_copy_bytes_lower_bound"] == 0
    assert static["static_cache_storage_write_bytes"] == 2 * (3 + 2) * 8 + 8
    assert paged["paged_cache_storage_write_bytes"] == 2 * (3 + 2) * 8 + 2 * 2 * 8
    assert paged["page_table_metadata_bytes"] == 2 * (3 + 1) * 8


@pytest.mark.parametrize(
    "workload",
    [
        "prefill_attention",
        "prefill_with_cache",
        "decode_attention",
        "decode_with_append",
        "decode_with_static_write",
    ],
)
@pytest.mark.parametrize("implementation", ["naive", "absorbed"])
def test_small_mla_benchmark_reports_verified_workloads(
    workload: str,
    implementation: str,
) -> None:
    config = _small_config(workload=workload, implementation=implementation)

    report = benchmark_mla(config)

    query_length = 1 if workload.startswith("decode") else config.sequence_length
    assert report["schema_version"] == 1
    assert report["benchmark"] == "multi_head_latent_attention_reference"
    assert report["configuration"]["workload"] == workload
    assert report["configuration"]["implementation"] == implementation
    assert report["output"]["shape"] == [config.batch, query_length, config.model_dim]
    assert report["verification"]["performed"] is True
    assert report["verification"]["max_tolerance_ratio"] <= 1
    assert report["latency"]["count"] == 1
    assert len(report["raw_samples_ms"]) == 1
    assert report["cache_storage"]["payload_compression_ratio"] > 1


def test_small_paged_mla_benchmark_reports_verified_reference_workload() -> None:
    config = _small_config(workload="decode_with_paged_write", implementation="absorbed")

    report = benchmark_mla(config)

    assert report["configuration"]["page_size"] == 2
    assert report["output"]["shape"] == [config.batch, 1, config.model_dim]
    assert report["verification"]["performed"] is True
    assert report["verification"]["max_tolerance_ratio"] <= 1


def test_mla_benchmark_can_skip_verification() -> None:
    report = benchmark_mla(_small_config(verify=False))
    assert report["verification"] == {"performed": False}


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_cuda_mla_benchmark_accepts_supported_storage_dtypes(dtype: str) -> None:
    _small_config(device="cuda", implementation="cuda", dtype=dtype).validate()


def test_cuda_mla_benchmark_rejects_float64_storage() -> None:
    with pytest.raises(ValueError, match="float16, bfloat16, or float32"):
        _small_config(device="cuda", implementation="cuda", dtype="float64").validate()


@pytest.mark.parametrize(
    "config",
    [
        _small_config(sequence_length=0),
        _small_config(page_size=0),
        _small_config(q_lora_rank=-1),
        _small_config(qk_rope_head_dim=3),
        _small_config(iterations=0),
        _small_config(workload="unknown"),  # type: ignore[arg-type]
        _small_config(implementation="unknown"),  # type: ignore[arg-type]
        _small_config(workload="decode_with_paged_write", implementation="naive"),
    ],
)
def test_invalid_mla_benchmark_configuration_is_rejected(config: MLABenchmarkConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()
