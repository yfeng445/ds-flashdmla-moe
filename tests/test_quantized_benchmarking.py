import pytest

from ds_flash_mla_moe.quantized_benchmarking import (
    QuantizedGEMMBenchmarkConfig,
    benchmark_quantized_gemm,
    quantized_gemm_work_estimate,
)


def _small_config(**overrides) -> QuantizedGEMMBenchmarkConfig:
    values = {
        "m": 5,
        "n": 7,
        "k": 3,
        "format": "int8",
        "backend": "reference",
        "device": "cpu",
        "warmup": 0,
        "iterations": 1,
        "seed": 809,
    }
    values.update(overrides)
    return QuantizedGEMMBenchmarkConfig(**values)


@pytest.mark.parametrize(
    ("quantization_format", "value_bytes"),
    [("int8", 5 * 3 + 7 * 3), ("fp8_e4m3fn", 5 * 3 + 7 * 3)],
)
def test_quantized_work_estimate_separates_payload_scales_and_fp32_output(
    quantization_format: str,
    value_bytes: int,
) -> None:
    estimate = quantized_gemm_work_estimate(
        _small_config(format=quantization_format)  # type: ignore[arg-type]
    )

    assert estimate["matrix_flops"] == 2 * 5 * 7 * 3
    assert estimate["quantized_payload_bytes"] == value_bytes
    assert estimate["scale_bytes"] == (5 + 7) * 4
    assert estimate["fp32_output_bytes"] == 5 * 7 * 4
    assert estimate["analytical_only"] is True


@pytest.mark.parametrize("quantization_format", ["int8", "fp8_e4m3fn"])
def test_small_quantized_benchmark_records_paired_reference_and_raw_samples(
    quantization_format: str,
) -> None:
    report = benchmark_quantized_gemm(
        _small_config(format=quantization_format)  # type: ignore[arg-type]
    )

    assert report["schema_version"] == 1
    assert report["benchmark"] == "dequantized_quantized_linear"
    assert report["configuration"]["format"] == quantization_format
    assert report["executed_backend"] == "reference"
    assert report["output"] == {"shape": [5, 7], "dtype": "float32", "device": "cpu"}
    assert report["verification"]["paired_dequantized_reference"]["performed"] is True
    assert report["verification"]["paired_dequantized_reference"]["max_tolerance_ratio"] <= 1.0
    assert report["verification"]["original_fp32_linear"]["performed"] is True
    assert report["latency"]["count"] == 1
    assert len(report["raw_samples_ms"]) == 1
    assert report["performance_claim"] is False


def test_quantized_benchmark_can_skip_verification() -> None:
    report = benchmark_quantized_gemm(_small_config(verify=False))

    assert report["verification"] == {"performed": False}


@pytest.mark.parametrize(
    "config",
    [
        _small_config(m=0),
        _small_config(n=0),
        _small_config(k=0),
        _small_config(iterations=0),
        _small_config(warmup=-1),
        _small_config(format="int4"),  # type: ignore[arg-type]
        _small_config(backend="unknown"),  # type: ignore[arg-type]
        _small_config(device="meta"),
        _small_config(backend="cuda", device="cpu"),
    ],
)
def test_invalid_quantized_benchmark_configuration_is_rejected(
    config: QuantizedGEMMBenchmarkConfig,
) -> None:
    with pytest.raises(ValueError):
        config.validate()
