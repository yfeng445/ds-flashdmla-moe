from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ds_flash_mla_moe.moe_benchmarking import (
    MoEForwardBenchmarkConfig,
    benchmark_moe_forward,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _small_config(**overrides: object) -> MoEForwardBenchmarkConfig:
    values: dict[str, object] = {
        "tokens": 7,
        "model_dim": 5,
        "hidden_dim": 9,
        "experts": 4,
        "topk": 2,
        "n_groups": 2,
        "topk_groups": 1,
        "dtype": "float64",
        "device": "cpu",
        "backend": "reference",
        "seed": 521,
        "warmup": 0,
        "iterations": 1,
        "route_scale": 0.75,
        "score_bias": True,
    }
    values.update(overrides)
    return MoEForwardBenchmarkConfig(**values)  # type: ignore[arg-type]


def test_reference_whole_layer_report_records_reproducible_evidence() -> None:
    report = benchmark_moe_forward(_small_config())

    assert report["schema_version"] == 1
    assert report["benchmark"] == "deepseek_moe_forward"
    assert report["implementation"] == "single_device_staged"
    assert report["performance_claim"] is False
    assert report["configuration"] == {
        "tokens": 7,
        "model_dim": 5,
        "hidden_dim": 9,
        "experts": 4,
        "topk": 2,
        "n_groups": 2,
        "topk_groups": 1,
        "dtype": "float64",
        "device": "cpu",
        "backend": "reference",
        "seed": 521,
        "warmup": 0,
        "iterations": 1,
        "route_scale": 0.75,
        "score_bias": True,
        "verify": True,
    }
    assert report["initialization"] == {
        "distribution": "normal",
        "activation_standard_deviation": 1.0,
        "gate_weight_standard_deviation": 1 / 5**0.5,
        "expert_gate_up_weight_standard_deviation": 1 / 5**0.5,
        "expert_down_weight_standard_deviation": 1 / 9**0.5,
        "score_bias_standard_deviation": 0.5,
    }
    assert report["output"] == {
        "shape": [7, 5],
        "dtype": "float64",
        "device": "cpu",
        "contiguous": True,
    }
    assert report["verification"]["performed"] is True
    assert report["verification"]["reference"] == "deepseek_moe_reference"
    assert report["verification"]["output"]["max_tolerance_ratio"] <= 1
    assert report["latency"]["count"] == 1
    assert len(report["raw_samples_ms"]) == 1

    distribution = report["route_distribution"]
    assert len(distribution["values"]) == 4
    assert sum(distribution["values"]) == 14
    assert distribution["total"] == 14
    assert distribution["active_experts"] + distribution["empty_experts"] == 4
    assert distribution["peak_to_mean"] == distribution["maximum"] / 3.5


def test_intermediate_bytes_are_hand_counted_from_staged_route_rows() -> None:
    report = benchmark_moe_forward(_small_config())

    assert report["intermediate_bytes"] == {
        "analytical_only": True,
        "floating_dtype": "float64",
        "floating_element_size": 8,
        "index_dtype": "int64",
        "index_element_size": 8,
        "route_rows": 14,
        "dense_scores": 7 * 4 * 8,
        "packed_activations": 14 * 5 * 8,
        "packed_weights": 14 * 8,
        "packed_indices": 2 * 14 * 8,
        "expert_hidden_state": 14 * 9 * 8,
        "contributions": 14 * 5 * 8,
        "total_major_intermediates": (7 * 4 + 14 * 5 + 14 + 2 * 14 + 14 * 9 + 14 * 5)
        * 8,
    }


def test_bfloat16_intermediate_bytes_keep_compute_buffers_in_float32() -> None:
    report = benchmark_moe_forward(_small_config(dtype="bfloat16"))
    intermediate_bytes = report["intermediate_bytes"]

    assert intermediate_bytes["floating_dtype"] == "bfloat16"
    assert intermediate_bytes["floating_element_size"] == 2
    assert intermediate_bytes["index_dtype"] == "int64"
    assert intermediate_bytes["index_element_size"] == 8
    assert intermediate_bytes["route_rows"] == 14
    assert intermediate_bytes["dense_scores"] == 7 * 4 * 4
    assert intermediate_bytes["packed_activations"] == 14 * 5 * 2
    assert intermediate_bytes["packed_weights"] == 14 * 2
    assert intermediate_bytes["packed_indices"] == 2 * 14 * 8
    assert intermediate_bytes["expert_hidden_state"] == 14 * 9 * 4
    assert intermediate_bytes["contributions"] == 14 * 5 * 2
    assert intermediate_bytes["total_major_intermediates"] == (
        7 * 4 * 4 + 14 * 5 * 2 + 14 * 2 + 2 * 14 * 8 + 14 * 9 * 4 + 14 * 5 * 2
    )


def test_whole_layer_benchmark_can_skip_reference_verification() -> None:
    report = benchmark_moe_forward(_small_config(verify=False))

    assert report["verification"] == {"performed": False}
    assert report["route_distribution"]["total"] == 14


def test_whole_layer_benchmark_config_is_frozen() -> None:
    config = _small_config()

    with pytest.raises(FrozenInstanceError):
        config.tokens = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    "config",
    [
        _small_config(tokens=0),
        _small_config(model_dim=0),
        _small_config(hidden_dim=0),
        _small_config(experts=0),
        _small_config(topk=0),
        _small_config(experts=5),
        _small_config(topk=3),
        _small_config(topk_groups=0),
        _small_config(route_scale=float("nan")),
        _small_config(iterations=0),
        _small_config(warmup=-1),
        _small_config(dtype="int8"),
        _small_config(backend="unknown"),
        _small_config(backend="cuda"),
        _small_config(backend="cuda", device="cuda", dtype="float64"),
    ],
)
def test_invalid_whole_layer_benchmark_config_is_rejected(
    config: MoEForwardBenchmarkConfig,
) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_cli_no_verify_emits_json_without_reference_result() -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "benchmarks" / "moe.py"),
        "--device",
        "cpu",
        "--dtype",
        "float64",
        "--backend",
        "reference",
        "--tokens",
        "3",
        "--model-dim",
        "2",
        "--hidden-dim",
        "4",
        "--experts",
        "2",
        "--topk",
        "1",
        "--n-groups",
        "1",
        "--topk-groups",
        "1",
        "--warmup",
        "0",
        "--iterations",
        "1",
        "--score-bias",
        "--no-verify",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["configuration"]["score_bias"] is True
    assert report["configuration"]["verify"] is False
    assert report["verification"] == {"performed": False}
