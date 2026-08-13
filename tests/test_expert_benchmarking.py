import pytest
import torch

from ds_flash_mla_moe.expert_benchmarking import (
    ExpertBenchmarkConfig,
    _normalized_error,
    benchmark_experts,
    expert_grouped_tile_model,
    expert_initialization_model,
    expert_native_numeric_model,
    expert_work_estimate,
)


def _small_config(**overrides) -> ExpertBenchmarkConfig:
    values = {
        "expert_counts": (2, 0, 5),
        "model_dim": 5,
        "hidden_dim": 4,
        "dtype": "float64",
        "device": "cpu",
        "backend": "reference",
        "warmup": 0,
        "iterations": 1,
        "seed": 311,
    }
    values.update(overrides)
    return ExpertBenchmarkConfig(**values)


def test_work_estimate_distinguishes_active_rows_from_padding() -> None:
    estimate = expert_work_estimate(_small_config())

    assert estimate["expert_route_rows"]["values"] == [2, 0, 5]
    assert estimate["expert_route_rows"]["zero_load_count"] == 1
    assert estimate["expert_count"] == 3
    assert estimate["active_expert_count"] == 2
    assert estimate["active_route_rows"] == 7
    assert estimate["padded_capacity_per_expert"] == 5
    assert estimate["padded_expert_slots"] == 15
    assert estimate["padding_rows"] == 8
    assert estimate["padding_utilization"] == 7 / 15
    assert estimate["forward_active_row_matrix_flops"] == 6 * 7 * 5 * 4
    assert estimate["forward_padded_matrix_flops"] == 6 * 15 * 5 * 4
    assert estimate["native_grouped_tile_model"] == {
        "analytical_only": True,
        "tile_shape": [16, 16, 16],
        "active_expert_row_tiles": 2,
        "hidden_output_tiles": 1,
        "model_output_tiles": 1,
        "hidden_projection_tasks": 2,
        "down_projection_tasks": 2,
        "allocated_row_lanes": 32,
        "inactive_tail_row_lanes": 25,
        "row_lane_utilization": 7 / 32,
    }
    assert estimate["native_numeric_model"] == {
        "applies_to_backend": "cuda",
        "configuration_dtype": "float64",
        "supported": False,
        "forward_engine": None,
        "multiplicand_dtype": None,
        "accumulator_dtype": None,
        "materialized_hidden_dtype": None,
        "minimum_compute_capability": None,
    }
    assert (
        estimate["active_row_compulsory_tensor_bytes_lower_bound"]
        == (2 * 7 * 5 + 3 * 2 * 5 * 4) * 8
    )


def test_grouped_tile_model_counts_each_expert_tail_independently() -> None:
    estimate = expert_work_estimate(
        _small_config(expert_counts=(17, 0, 5, 31), model_dim=33, hidden_dim=65)
    )
    model = estimate["native_grouped_tile_model"]

    assert model["active_expert_row_tiles"] == 2 + 0 + 1 + 2
    assert model["hidden_output_tiles"] == 5
    assert model["model_output_tiles"] == 3
    assert model["hidden_projection_tasks"] == 25
    assert model["down_projection_tasks"] == 15
    assert model["allocated_row_lanes"] == 80
    assert model["inactive_tail_row_lanes"] == 27
    assert model["row_lane_utilization"] == 53 / 80


def test_native_numeric_model_distinguishes_wmma_from_cuda_cores() -> None:
    half = expert_native_numeric_model("float16")
    single = expert_native_numeric_model("float32")

    assert half["forward_engine"] == "wmma_tensor_cores"
    assert half["accumulator_dtype"] == "float32"
    assert half["materialized_hidden_dtype"] == "float16"
    assert half["minimum_compute_capability"] == "7.0"
    assert single["forward_engine"] == "shared_memory_cuda_cores"
    assert single["minimum_compute_capability"] is None


def test_expert_initialization_is_fan_in_scaled_and_reported() -> None:
    model = expert_initialization_model(model_dim=16, hidden_dim=64)

    assert model == {
        "distribution": "normal",
        "activation_standard_deviation": 1.0,
        "gate_up_weight_standard_deviation": 0.25,
        "down_weight_standard_deviation": 0.125,
    }
    report = benchmark_experts(_small_config())
    assert report["initialization"] == expert_initialization_model(model_dim=5, hidden_dim=4)


@pytest.mark.parametrize("invalid", [torch.nan, torch.inf, -torch.inf])
def test_expert_error_metric_rejects_matching_nonfinite_values(invalid: float) -> None:
    error = _normalized_error(torch.tensor([invalid]), torch.tensor([invalid]), 1e-3, 1e-3)
    assert torch.isinf(error).all()


@pytest.mark.parametrize(
    "arguments",
    [
        ((), 4, 5),
        ((0, 0), 4, 5),
        ((1, -1), 4, 5),
        ((1,), 0, 5),
        ((1,), 4, 0),
    ],
)
def test_grouped_tile_model_rejects_invalid_shapes(
    arguments: tuple[tuple[int, ...], int, int],
) -> None:
    counts, model_dim, hidden_dim = arguments
    with pytest.raises(ValueError):
        expert_grouped_tile_model(counts, model_dim=model_dim, hidden_dim=hidden_dim)


@pytest.mark.parametrize("backward", [False, True])
def test_reference_expert_benchmark_is_verified(backward: bool) -> None:
    report = benchmark_experts(_small_config(backward=backward))

    assert report["schema_version"] == 1
    assert report["benchmark"] == "swiglu_experts_expert_major"
    assert report["output"] == {"shape": [7, 5], "dtype": "float64", "device": "cpu"}
    assert report["verification"]["output"]["max_tolerance_ratio"] <= 1
    assert report["verification"]["gradients"]["performed"] is backward
    assert report["verification"]["gradients"]["max_tolerance_ratio"] <= 1
    assert report["latency"]["count"] == 1
    assert len(report["raw_samples_ms"]) == 1


def test_expert_benchmark_can_skip_verification() -> None:
    report = benchmark_experts(_small_config(verify=False))
    assert report["verification"] == {"performed": False}


@pytest.mark.parametrize(
    "config",
    [
        _small_config(expert_counts=()),
        _small_config(expert_counts=(0, 0)),
        _small_config(expert_counts=(2, -1)),
        _small_config(model_dim=0),
        _small_config(hidden_dim=0),
        _small_config(iterations=0),
        _small_config(warmup=-1),
        _small_config(dtype="int8"),  # type: ignore[arg-type]
        _small_config(backend="unknown"),  # type: ignore[arg-type]
        _small_config(backend="cuda"),
        _small_config(backend="cuda", device="cuda", dtype="bfloat16"),
    ],
)
def test_invalid_expert_benchmark_config_is_rejected(config: ExpertBenchmarkConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()
