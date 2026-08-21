from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ds_flash_mla_moe import moe_benchmarking
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
    assert report["implementation"] == "reference"
    assert report["executed_backend"] == "reference"
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

    intermediate_bytes = report["intermediate_bytes"]
    legacy_fields = {
        key: intermediate_bytes[key]
        for key in (
            "analytical_only",
            "floating_dtype",
            "floating_element_size",
            "index_dtype",
            "index_element_size",
            "route_rows",
            "dense_scores",
            "packed_activations",
            "packed_weights",
            "packed_indices",
            "expert_hidden_state",
            "contributions",
            "total_major_intermediates",
        )
    }

    assert legacy_fields == {
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
        "total_major_intermediates": (7 * 4 + 14 * 5 + 14 + 2 * 14 + 14 * 9 + 14 * 5) * 8,
    }
    assert intermediate_bytes["legacy_flat_fields"] == {
        "historical_compatibility": True,
        "modeled_backend": "cuda_staged",
        "executed_backend": "reference",
        "materialized_by_execution": False,
    }


def test_reference_report_qualifies_cuda_inventories_as_unexecuted_analytical_models() -> None:
    report = benchmark_moe_forward(_small_config())

    intermediate_bytes = report["intermediate_bytes"]
    assert intermediate_bytes["executed_backend"] == "reference"
    assert set(intermediate_bytes["inventories"]) == {
        "cuda_staged",
        "cuda_fused",
        "cuda_persistent",
    }
    assert {
        name: {
            key: inventory[key]
            for key in (
                "modeled_backend",
                "executed_backend",
                "evidence_kind",
                "materialized_by_execution",
            )
        }
        for name, inventory in intermediate_bytes["inventories"].items()
    } == {
        "cuda_staged": {
            "modeled_backend": "cuda_staged",
            "executed_backend": "reference",
            "evidence_kind": "analytical_model",
            "materialized_by_execution": False,
        },
        "cuda_fused": {
            "modeled_backend": "cuda_fused",
            "executed_backend": "reference",
            "evidence_kind": "analytical_model",
            "materialized_by_execution": False,
        },
        "cuda_persistent": {
            "modeled_backend": "cuda_persistent",
            "executed_backend": "reference",
            "evidence_kind": "analytical_model",
            "materialized_by_execution": False,
        },
    }


def test_auto_benchmark_resolution_matches_fused_then_staged_public_policy(monkeypatch) -> None:
    queried: list[str] = []

    def has_cuda_kernel(operator: str) -> bool:
        queried.append(operator)
        return operator == "deepseek_moe_forward"

    monkeypatch.setattr(moe_benchmarking.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(moe_benchmarking, "_operator_has_cuda_kernel", has_cuda_kernel)

    actual = moe_benchmarking._resolved_benchmark_backend(
        _small_config(device="cuda", dtype="float32", backend="auto")
    )

    assert actual == "cuda_staged"
    assert queried == ["deepseek_moe_forward_fused", "deepseek_moe_forward"]


def test_named_staged_inventory_counts_every_materialized_metadata_buffer() -> None:
    report = benchmark_moe_forward(_small_config())

    staged = report["intermediate_bytes"]["inventories"]["cuda_staged"]
    assert staged["implementation"] == "single_device_cuda_staged"
    assert staged["buffers"] == {
        "gate_logits": {
            "shape": [7, 4],
            "dtype": "float64",
            "kind": "scratch",
            "bytes": 7 * 4 * 8,
        },
        "dense_scores": {
            "shape": [7, 4],
            "dtype": "float64",
            "kind": "scratch",
            "bytes": 7 * 4 * 8,
        },
        "selected_route_weights": {
            "shape": [7, 2],
            "dtype": "float64",
            "kind": "metadata",
            "bytes": 14 * 8,
        },
        "selected_expert_indices": {
            "shape": [7, 2],
            "dtype": "int64",
            "kind": "metadata",
            "bytes": 14 * 8,
        },
        "expert_owner": {
            "shape": [4],
            "dtype": "int64",
            "kind": "metadata",
            "bytes": 4 * 8,
        },
        "route_key_counts": {
            "shape": [4],
            "dtype": "int64",
            "kind": "metadata",
            "bytes": 4 * 8,
        },
        "route_key_offsets": {
            "shape": [5],
            "dtype": "int64",
            "kind": "metadata",
            "bytes": 5 * 8,
        },
        "route_key_cursors": {
            "shape": [4],
            "dtype": "int64",
            "kind": "metadata",
            "bytes": 4 * 8,
        },
        "counts_per_expert": {
            "shape": [4],
            "dtype": "int64",
            "kind": "metadata",
            "bytes": 4 * 8,
        },
        "rank_counts": {
            "shape": [1],
            "dtype": "int64",
            "kind": "metadata",
            "bytes": 8,
        },
        "packed_activations": {
            "shape": [14, 5],
            "dtype": "float64",
            "kind": "activation",
            "bytes": 14 * 5 * 8,
        },
        "packed_route_weights": {
            "shape": [14],
            "dtype": "float64",
            "kind": "metadata",
            "bytes": 14 * 8,
        },
        "packed_route_indices": {
            "shape": [14],
            "dtype": "int64",
            "kind": "metadata",
            "bytes": 14 * 8,
        },
        "packed_expert_indices": {
            "shape": [14],
            "dtype": "int64",
            "kind": "metadata",
            "bytes": 14 * 8,
        },
        "expert_count_prefix_sum": {
            "shape": [4],
            "dtype": "int64",
            "kind": "metadata",
            "bytes": 4 * 8,
        },
        "expert_offset_seed": {
            "shape": [1],
            "dtype": "int64",
            "kind": "metadata",
            "bytes": 8,
        },
        "expert_offsets": {
            "shape": [5],
            "dtype": "int64",
            "kind": "metadata",
            "bytes": 5 * 8,
        },
        "packed_token_indices": {
            "shape": [14],
            "dtype": "int64",
            "kind": "metadata",
            "bytes": 14 * 8,
        },
        "hidden_task_offsets": {
            "shape": [5],
            "dtype": "int64",
            "kind": "metadata",
            "bytes": 5 * 8,
        },
        "down_task_offsets": {
            "shape": [5],
            "dtype": "int64",
            "kind": "metadata",
            "bytes": 5 * 8,
        },
        "expert_hidden_state": {
            "shape": [14, 9],
            "dtype": "float64",
            "kind": "activation",
            "bytes": 14 * 9 * 8,
        },
        "contributions": {
            "shape": [14, 5],
            "dtype": "float64",
            "kind": "activation",
            "bytes": 14 * 5 * 8,
        },
    }
    assert staged["metadata_bytes"] == 1_008
    assert staged["total_bytes"] == 3_584


def test_named_fused_inventory_reports_implemented_scratch_and_removed_intermediates() -> None:
    report = benchmark_moe_forward(_small_config())

    fused = report["intermediate_bytes"]["inventories"]["cuda_fused"]
    assert fused["implementation"] == "single_device_cuda_fused"
    assert set(fused["buffers"]) == {
        "gate_logits",
        "dense_scores",
        "selected_route_weights",
        "selected_expert_indices",
        "packed_activations",
        "packed_route_weights",
        "packed_token_indices",
        "expert_offsets",
        "route_pack_cursors",
        "expert_hidden_state",
    }
    assert fused["metadata_bytes"] == 520
    assert fused["total_bytes"] == 2_536
    assert (
        fused["total_bytes"]
        < report["intermediate_bytes"]["inventories"]["cuda_staged"]["total_bytes"]
    )


def test_named_persistent_inventory_adds_only_the_bounded_device_queue() -> None:
    report = benchmark_moe_forward(_small_config())

    inventories = report["intermediate_bytes"]["inventories"]
    fused = inventories["cuda_fused"]
    persistent = inventories["cuda_persistent"]
    assert persistent["implementation"] == "single_device_cuda_persistent"
    assert set(persistent["buffers"]) == {*fused["buffers"], "persistent_task_queue"}
    assert persistent["buffers"]["persistent_task_queue"] == {
        "shape": [1],
        "dtype": "int64",
        "kind": "metadata",
        "bytes": 8,
    }
    assert persistent["metadata_bytes"] == fused["metadata_bytes"] + 8
    assert persistent["total_bytes"] == fused["total_bytes"] + 8
    assert persistent["scheduler_policy"] == {
        "small_work_fallback_max_routes": 8,
        "persistent_queue_materialized": True,
    }


def test_persistent_inventory_reflects_small_work_fused_fallback() -> None:
    report = benchmark_moe_forward(_small_config(tokens=4))

    inventories = report["intermediate_bytes"]["inventories"]
    fused = inventories["cuda_fused"]
    persistent = inventories["cuda_persistent"]
    assert persistent["buffers"] == fused["buffers"]
    assert persistent["metadata_bytes"] == fused["metadata_bytes"]
    assert persistent["total_bytes"] == fused["total_bytes"]
    assert persistent["scheduler_policy"] == {
        "small_work_fallback_max_routes": 8,
        "persistent_queue_materialized": False,
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


def test_cli_lists_every_explicit_whole_layer_native_backend() -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "moe.py"), "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    for backend in ("cuda_staged", "cuda_fused", "cuda_persistent"):
        assert backend in completed.stdout


def test_cli_kineto_mode_rejects_a_cpu_device_as_a_profile_request() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks" / "moe.py"),
            "--mode",
            "kineto",
            "--device",
            "cpu",
            "--warmup",
            "0",
            "--iterations",
            "1",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "MoE profiling requires a CUDA device" in completed.stderr


def test_cli_nvtx_mode_rejects_a_kineto_trace_path() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks" / "moe.py"),
            "--mode",
            "nvtx",
            "--trace",
            "unused.json",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--trace is only available with --mode kineto" in completed.stderr
