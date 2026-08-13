import pytest

from ds_flash_mla_moe.router_benchmarking import (
    RouterBenchmarkConfig,
    benchmark_router,
    router_work_estimate,
)


def _small_config(**overrides) -> RouterBenchmarkConfig:
    values = {
        "tokens": 7,
        "model_dim": 5,
        "experts": 8,
        "topk": 3,
        "n_groups": 4,
        "topk_groups": 2,
        "hot_expert_bias": 0.4,
        "route_scale": 1.3,
        "dtype": "float64",
        "device": "cpu",
        "backend": "reference",
        "warmup": 0,
        "iterations": 1,
        "seed": 419,
    }
    values.update(overrides)
    return RouterBenchmarkConfig(**values)


def test_router_work_estimate_keeps_projection_and_candidates_separate() -> None:
    estimate = router_work_estimate(_small_config())

    assert estimate == {
        "router_projection_matrix_flops": 2 * 7 * 5 * 8,
        "logit_elements": 7 * 8,
        "group_score_candidates": 7 * 4,
        "retained_expert_candidates": 7 * 4,
        "selected_routes": 7 * 3,
    }


@pytest.mark.parametrize("backward", [False, True])
def test_reference_router_benchmark_is_verified(backward: bool) -> None:
    report = benchmark_router(_small_config(backward=backward))

    assert report["schema_version"] == 1
    assert report["benchmark"] == "deepseek_grouped_topk"
    assert report["output"] == {
        "weights_shape": [7, 3],
        "indices_shape": [7, 3],
        "dtype": "float64",
        "device": "cpu",
    }
    assert report["verification"]["indices_exact"] is True
    assert report["verification"]["weights"]["max_tolerance_ratio"] <= 1
    assert report["verification"]["gradients"]["performed"] is backward
    assert report["verification"]["gradients"]["max_tolerance_ratio"] <= 1
    assert report["expert_load"]["total"] == 21
    assert len(report["expert_load"]["values"]) == 8
    assert report["latency"]["count"] == 1


def test_router_benchmark_can_skip_verification() -> None:
    report = benchmark_router(_small_config(verify=False))
    assert report["verification"] == {"performed": False}


@pytest.mark.parametrize(
    "config",
    [
        _small_config(tokens=0),
        _small_config(model_dim=0),
        _small_config(experts=7),
        _small_config(topk=5),
        _small_config(topk_groups=0),
        _small_config(hot_expert_bias=-1),
        _small_config(route_scale=float("nan")),
        _small_config(iterations=0),
        _small_config(warmup=-1),
        _small_config(dtype="int8"),  # type: ignore[arg-type]
        _small_config(backend="unknown"),  # type: ignore[arg-type]
        _small_config(backend="cuda"),
    ],
)
def test_invalid_router_benchmark_config_is_rejected(config: RouterBenchmarkConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()
