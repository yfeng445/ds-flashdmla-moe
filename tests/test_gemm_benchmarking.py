import pytest

from ds_flash_mla_moe.gemm_benchmarking import (
    GEMMBenchmarkConfig,
    benchmark_gemm,
    gemm_work_estimate,
)


def _small_config(**overrides) -> GEMMBenchmarkConfig:
    values = {
        "m": 5,
        "n": 7,
        "k": 3,
        "tile_m": 4,
        "tile_n": 3,
        "tile_k": 2,
        "dtype": "float64",
        "device": "cpu",
        "warmup": 0,
        "iterations": 1,
        "seed": 229,
    }
    values.update(overrides)
    return GEMMBenchmarkConfig(**values)


def test_gemm_work_estimate_exposes_tail_and_traffic_models() -> None:
    estimate = gemm_work_estimate(_small_config(beta=0.5))

    assert estimate["matrix_flops"] == 2 * 5 * 7 * 3
    assert estimate["compulsory_tensor_bytes_lower_bound"] == (5 * 3 + 3 * 7 + 2 * 5 * 7) * 8
    assert estimate["one_output_thread_global_bytes_model"] == (2 * 5 * 7 * 3 + 2 * 5 * 7) * 8
    tiled_inputs = 3 * 5 * 3 + 2 * 3 * 7
    assert estimate["one_output_tile_global_bytes_model"] == (tiled_inputs + 2 * 5 * 7) * 8
    assert estimate["modeled_input_reuse_ratio"] == 2 * 5 * 7 * 3 / tiled_inputs
    assert estimate["tile_count_m"] == 2
    assert estimate["tile_count_n"] == 3
    assert estimate["tile_count_k"] == 2
    assert estimate["last_tile_m"] == 1
    assert estimate["last_tile_n"] == 1
    assert estimate["last_tile_k"] == 1
    assert estimate["shared_memory_bytes_per_stage_model"] == (4 * 2 + 2 * 3) * 8


@pytest.mark.parametrize("implementation", ["torch", "tiled"])
def test_small_gemm_benchmark_is_verified(implementation: str) -> None:
    report = benchmark_gemm(_small_config(implementation=implementation, beta=-0.25))

    assert report["schema_version"] == 1
    assert report["benchmark"] == "general_matrix_multiplication"
    assert report["configuration"]["implementation"] == implementation
    assert report["output"] == {"shape": [5, 7], "dtype": "float64", "device": "cpu"}
    assert report["verification"]["performed"] is True
    assert report["verification"]["max_tolerance_ratio"] <= 1
    assert report["latency"]["count"] == 1
    assert len(report["raw_samples_ms"]) == 1


def test_gemm_benchmark_can_skip_verification() -> None:
    report = benchmark_gemm(_small_config(verify=False))
    assert report["verification"] == {"performed": False}


def test_cuda_gemm_benchmark_rejects_a_cpu_device_before_measurement() -> None:
    with pytest.raises(ValueError, match="CUDA benchmark device"):
        benchmark_gemm(
            _small_config(
                implementation="cuda",
                tile_m=16,
                tile_n=16,
                tile_k=16,
            )
        )


@pytest.mark.parametrize(
    "config",
    [
        _small_config(m=0),
        _small_config(tile_k=0),
        _small_config(iterations=0),
        _small_config(warmup=-1),
        _small_config(dtype="int8"),  # type: ignore[arg-type]
        _small_config(implementation="unknown"),  # type: ignore[arg-type]
        _small_config(implementation="cuda"),
        _small_config(alpha=float("nan")),
    ],
)
def test_invalid_gemm_benchmark_configuration_is_rejected(
    config: GEMMBenchmarkConfig,
) -> None:
    with pytest.raises(ValueError):
        config.validate()
