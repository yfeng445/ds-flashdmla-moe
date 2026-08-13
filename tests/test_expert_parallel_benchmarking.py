from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from ds_flash_mla_moe.expert_parallel_benchmarking import (
    ExpertParallelBenchmarkConfig,
    _normalized_error,
    expert_parallel_chunked_tile_model,
    expert_parallel_load_analysis,
    expert_parallel_overlap_model,
    expert_parallel_work_estimate,
    summarize_rank_latency_samples,
)


def test_expert_parallel_work_estimate_uses_only_cross_rank_routes() -> None:
    config = ExpertParallelBenchmarkConfig(
        tokens_per_rank=3,
        token_skew=1,
        model_dim=4,
        hidden_dim=5,
        experts=4,
        topk=2,
        dtype="float64",
        iterations=1,
        backward=True,
    )

    estimate = expert_parallel_work_estimate(
        config,
        world_size=2,
        counts_matrix=[[1, 5], [4, 4]],
    )

    assert estimate == {
        "forward_matrix_flops": 7 * (2 * 4 * 4 + 6 * 2 * 4 * 5),
        "forward_router_matrix_flops": 2 * 7 * 4 * 4,
        "forward_routed_expert_matrix_flops": 6 * 14 * 4 * 5,
        "forward_shared_expert_matrix_flops": 0,
        "total_tokens": 7,
        "total_routes": 14,
        "cross_rank_route_rows": 9,
        "forward_cross_rank_activation_bytes": 2 * 9 * 4 * 8,
        "measured_step_cross_rank_activation_bytes": 4 * 9 * 4 * 8,
        "forward_cross_rank_expert_id_bytes": 9 * 8,
    }


@pytest.mark.parametrize(
    "config",
    [
        ExpertParallelBenchmarkConfig(tokens_per_rank=-1),
        ExpertParallelBenchmarkConfig(tokens_per_rank=0, token_skew=0),
        ExpertParallelBenchmarkConfig(token_skew=-1),
        ExpertParallelBenchmarkConfig(shared_experts=-1),
        ExpertParallelBenchmarkConfig(hot_expert_bias=-1),
        ExpertParallelBenchmarkConfig(hot_expert_bias=float("inf")),
        ExpertParallelBenchmarkConfig(capacity_factor=0),
        ExpertParallelBenchmarkConfig(capacity_factor=float("nan")),
        ExpertParallelBenchmarkConfig(symmetric_cell_capacity=0),
        ExpertParallelBenchmarkConfig(pipeline_chunks=0),
        ExpertParallelBenchmarkConfig(pipeline_chunks=2, backend="gloo"),
        ExpertParallelBenchmarkConfig(experts=3, n_groups=2),
        ExpertParallelBenchmarkConfig(experts=4, topk=3, n_groups=2, topk_groups=1),
        ExpertParallelBenchmarkConfig(dtype="float16", backend="gloo"),
        ExpertParallelBenchmarkConfig(route_backend="cuda", backend="gloo"),
        ExpertParallelBenchmarkConfig(router_backend="cuda", backend="gloo"),
        ExpertParallelBenchmarkConfig(expert_backend="cuda", backend="gloo"),
        ExpertParallelBenchmarkConfig(expert_backend="cuda", backend="nccl", dtype="bfloat16"),
        ExpertParallelBenchmarkConfig(expert_backend="unknown"),  # type: ignore[arg-type]
    ],
)
def test_invalid_expert_parallel_benchmark_config_is_rejected(
    config: ExpertParallelBenchmarkConfig,
) -> None:
    with pytest.raises(ValueError):
        config.validate(world_size=2)


def test_non_integer_pipeline_chunks_are_rejected() -> None:
    with pytest.raises(TypeError, match="integer"):
        ExpertParallelBenchmarkConfig(pipeline_chunks=True).validate(world_size=2)


def test_work_estimate_rejects_a_counts_matrix_with_missing_routes() -> None:
    config = ExpertParallelBenchmarkConfig(
        tokens_per_rank=3,
        token_skew=1,
        experts=4,
        topk=2,
        iterations=1,
    )
    with pytest.raises(ValueError, match="row sums"):
        expert_parallel_work_estimate(
            config,
            world_size=2,
            counts_matrix=[[1, 4], [4, 4]],
        )


def test_fp16_nccl_allows_reference_routing_with_native_experts() -> None:
    ExpertParallelBenchmarkConfig(
        backend="nccl",
        dtype="float16",
        router_backend="reference",
        route_backend="reference",
        expert_backend="cuda",
    ).validate(world_size=2)


def test_chunked_pipeline_is_an_explicit_nccl_configuration() -> None:
    ExpertParallelBenchmarkConfig(
        backend="nccl",
        pipeline_chunks=3,
    ).validate(world_size=2)


def test_chunked_tile_model_counts_real_per_chunk_expert_tails() -> None:
    model = expert_parallel_chunked_tile_model(
        [
            [[17, 0], [0, 0], [1, 5]],
            [[0, 16], [0, 1], [0, 0]],
        ],
        model_dim=33,
        hidden_dim=65,
    )

    assert model["pipeline_chunks"] == 3
    assert model["chunked_task_counts_are_reported"] is True
    assert model["chunked_aggregate"] == {
        "active_route_rows": 40,
        "active_expert_row_tiles": 6,
        "hidden_projection_tasks": 30,
        "down_projection_tasks": 18,
        "allocated_row_lanes": 96,
        "inactive_tail_row_lanes": 56,
        "row_lane_utilization": 40 / 96,
    }
    empty_chunk = model["per_rank_chunks"][0]["chunks"][1]
    assert empty_chunk["expert_counts"] == [0, 0]
    assert empty_chunk["active_expert_row_tiles"] == 0
    assert empty_chunk["row_lane_utilization"] == 1.0


@pytest.mark.parametrize(
    "counts",
    [[], [[]], [[[1]], [[1], [0]]], [[[1], [-1]]]],
)
def test_chunked_tile_model_rejects_invalid_counts(counts: list[list[list[int]]]) -> None:
    with pytest.raises(ValueError):
        expert_parallel_chunked_tile_model(counts, model_dim=4, hidden_dim=4)


def test_distributed_error_metric_rejects_matching_nonfinite_values() -> None:
    error = _normalized_error(
        torch.tensor([torch.nan]),
        torch.tensor([torch.nan]),
        rtol=1e-3,
        atol=1e-3,
    )
    assert torch.isinf(error).all()


def test_zero_base_tokens_are_valid_when_another_rank_has_tokens() -> None:
    config = ExpertParallelBenchmarkConfig(
        tokens_per_rank=0,
        token_skew=3,
        experts=2,
        topk=1,
        iterations=1,
    )
    config.validate(world_size=2)
    estimate = expert_parallel_work_estimate(
        config,
        world_size=2,
        counts_matrix=[[0, 0], [2, 1]],
    )
    assert estimate["total_tokens"] == 3
    assert estimate["total_routes"] == 3


def test_work_estimate_counts_widened_shared_expert_flops() -> None:
    config = ExpertParallelBenchmarkConfig(
        tokens_per_rank=3,
        token_skew=1,
        model_dim=4,
        hidden_dim=5,
        shared_experts=2,
        experts=4,
        topk=2,
        iterations=1,
    )
    estimate = expert_parallel_work_estimate(
        config,
        world_size=2,
        counts_matrix=[[1, 5], [4, 4]],
    )

    assert estimate["forward_shared_expert_matrix_flops"] == 6 * 7 * 4 * (2 * 5)
    assert estimate["forward_matrix_flops"] == (
        estimate["forward_router_matrix_flops"]
        + estimate["forward_routed_expert_matrix_flops"]
        + estimate["forward_shared_expert_matrix_flops"]
    )


def test_rank_latency_summary_preserves_rank_mean_and_max() -> None:
    summary = summarize_rank_latency_samples([[1.0, 3.0], [4.0, 2.0]])

    assert summary["rank_count"] == 2
    assert summary["iteration_count"] == 2
    assert summary["per_rank"][0]["raw_samples_ms"] == [1.0, 4.0]
    assert summary["per_rank"][1]["raw_samples_ms"] == [3.0, 2.0]
    assert summary["rank_mean"]["raw_samples_ms"] == [2.0, 3.0]
    assert summary["rank_max"]["raw_samples_ms"] == [3.0, 4.0]


@pytest.mark.parametrize(
    "samples",
    [[], [[]], [[1.0], [1.0, 2.0]], [[float("nan")]], [[-1.0]]],
)
def test_invalid_rank_latency_samples_are_rejected(samples: list[list[float]]) -> None:
    with pytest.raises(ValueError):
        summarize_rank_latency_samples(samples)


def test_load_analysis_quantifies_rank_skew_padding_and_capacity_drop() -> None:
    config = ExpertParallelBenchmarkConfig(
        tokens_per_rank=4,
        model_dim=4,
        hidden_dim=5,
        experts=4,
        topk=2,
        iterations=1,
        capacity_factor=1.0,
    )

    analysis = expert_parallel_load_analysis(
        config,
        world_size=2,
        counts_matrix=[[6, 2], [5, 3]],
        expert_counts=[10, 1, 1, 4],
    )

    assert analysis["rank_send_route_rows"]["values"] == [8, 8]
    assert analysis["rank_receive_route_rows"]["values"] == [11, 5]
    assert analysis["rank_receive_route_rows"]["peak_to_mean"] == pytest.approx(11 / 8)
    assert analysis["expert_route_rows"]["coefficient_of_variation"] == pytest.approx(13.5**0.5 / 4)
    assert analysis["owner_local_padding_model"] == {
        "capacity_per_rank": [10, 4],
        "allocated_expert_slots_per_rank": [20, 8],
        "allocated_expert_slots": 28,
        "padding_rows": 12,
        "utilization": 16 / 28,
    }
    assert analysis["uniform_capacity_model"] == {
        "capacity_factor": 1.0,
        "mean_route_rows_per_expert": 4.0,
        "capacity_per_expert": 4,
        "minimum_capacity_per_expert_without_drop": 10,
        "capacity_factor_sufficient_without_drop": 2.5,
        "accepted_route_rows": 10,
        "dropped_route_rows": 6,
        "dropped_route_fraction": 6 / 16,
        "allocated_expert_slots": 16,
        "padding_rows": 6,
        "utilization": 10 / 16,
        "analytical_only": True,
    }


@pytest.mark.parametrize(
    ("expert_counts", "message"),
    [
        ([8, 8], "one value per expert"),
        ([10, 1, 1, -4], "negative"),
        ([10, 1, 1, 3], "every routed row"),
        ([9, 2, 1, 4], "destination-rank"),
    ],
)
def test_load_analysis_rejects_inconsistent_expert_counts(
    expert_counts: list[int], message: str
) -> None:
    config = ExpertParallelBenchmarkConfig(
        tokens_per_rank=4,
        experts=4,
        topk=2,
        iterations=1,
    )
    with pytest.raises(ValueError, match=message):
        expert_parallel_load_analysis(
            config,
            world_size=2,
            counts_matrix=[[6, 2], [5, 3]],
            expert_counts=expert_counts,
        )


def test_overlap_model_keeps_communication_on_one_resource() -> None:
    stages = {
        "dispatch": {"rank_raw_samples_ms": [[2.0, 1.0], [3.0, 2.0]]},
        "expert_compute": {"rank_raw_samples_ms": [[5.0, 4.0], [4.0, 3.0]]},
        "restore": {"rank_raw_samples_ms": [[1.0, 1.0], [2.0, 1.0]]},
    }

    model = expert_parallel_overlap_model(stages)

    assert model["communication_stage_max_latency"]["raw_samples_ms"] == [3.0, 5.0]
    assert model["serialized_stage_max_core_latency"]["raw_samples_ms"] == [8.0, 9.0]
    assert model["optimistic_steady_state_lower_bound"]["raw_samples_ms"] == [5.0, 5.0]
    assert model["maximum_overlap_opportunity"]["raw_samples_ms"] == [3.0, 4.0]
    assert model["derived_at_median"]["optimistic_pipeline_speedup"] == pytest.approx(1.7)
    assert model["derived_at_median"]["maximum_overlap_fraction"] == pytest.approx(3.5 / 8.5)


def test_overlap_model_rejects_mismatched_sample_counts() -> None:
    stages = {
        "dispatch": {"rank_raw_samples_ms": [[1.0]]},
        "expert_compute": {"rank_raw_samples_ms": [[1.0], [2.0]]},
        "restore": {"rank_raw_samples_ms": [[1.0]]},
    }
    with pytest.raises(ValueError, match="sample counts"):
        expert_parallel_overlap_model(stages)


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _loopback_interface() -> str | None:
    available = {name for _, name in socket.if_nameindex()}
    for candidate in ("lo", "lo0"):
        if candidate in available:
            return candidate
    return None


def _run_gloo_benchmark(tmp_path: Path, *arguments: str) -> dict[str, object]:
    loopback = _loopback_interface()
    if loopback is None:
        pytest.skip("no loopback network interface is available")
    output = tmp_path / f"ep-report-{len(arguments)}.json"
    benchmark = Path(__file__).parents[1] / "benchmarks" / "expert_parallel.py"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--master-addr=127.0.0.1",
        f"--master-port={_free_loopback_port()}",
        "--nproc-per-node=2",
        str(benchmark),
        "--backend=gloo",
        "--route-backend=reference",
        "--dtype=float64",
        "--warmup=0",
        "--iterations=1",
        *arguments,
        f"--output={output}",
    ]
    environment = os.environ.copy()
    environment["GLOO_SOCKET_IFNAME"] = loopback
    result = subprocess.run(
        command,
        cwd=benchmark.parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(output.read_text())


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo distributed backend is unavailable",
)
def test_torchrun_cli_emits_verified_backward_report(tmp_path: Path) -> None:
    report = _run_gloo_benchmark(
        tmp_path,
        "--expert-backend=padded",
        "--tokens-per-rank=3",
        "--token-skew=1",
        "--model-dim=4",
        "--hidden-dim=5",
        "--shared-experts=2",
        "--experts=4",
        "--topk=1",
        "--hot-expert-bias=100",
        "--capacity-factor=1",
        "--symmetric-cell-capacity=2",
        "--backward",
    )
    assert report["benchmark"] == "deepseek_moe_expert_parallel_reference"
    assert report["distributed"]["backend"] == "gloo"
    assert report["distributed"]["world_size"] == 2
    assert report["configuration"]["expert_backend"] == "padded"
    assert report["configuration"]["shared_experts"] == 2
    assert report["configuration"]["hot_expert_bias"] == 100
    assert report["configuration"]["symmetric_cell_capacity"] == 2
    assert report["initialization"]["gate_up_weight_standard_deviation"] == 0.5
    assert report["initialization"]["down_weight_standard_deviation"] == 1 / (5**0.5)
    assert report["initialization"]["shared_down_weight_standard_deviation"] == 1 / (10**0.5)
    assert len(report["distributed"]["counts_matrix"]) == 2
    assert len(report["distributed"]["ranks"]) == 2
    assert report["work_estimate"]["total_tokens"] == 7
    assert report["work_estimate"]["total_routes"] == 7
    assert report["expert_compute"]["active_route_rows"] == 7
    assert report["expert_compute"]["theoretical_padded_expert_slots"] >= 7
    assert (
        report["expert_compute"]["backend_executed_expert_slots"]
        == report["expert_compute"]["theoretical_padded_expert_slots"]
    )
    assert 0 < report["expert_compute"]["theoretical_padding_utilization"] <= 1
    assert report["expert_compute"]["native_grouped_tile_model"] == {
        "applies_to_backend": "cuda",
        "analytical_only": True,
        "tile_shape": [16, 16, 16],
        "active_expert_row_tiles": 1,
        "hidden_output_tiles": 1,
        "model_output_tiles": 1,
        "hidden_projection_tasks": 1,
        "down_projection_tasks": 1,
        "allocated_row_lanes": 16,
        "inactive_tail_row_lanes": 9,
        "row_lane_utilization": 7 / 16,
    }
    assert report["expert_compute"]["native_numeric_model"]["supported"] is False
    assert report["shared_expert_compute"] == {
        "enabled": True,
        "shared_expert_count": 2,
        "effective_hidden_dim": 10,
        "active_token_rows": 7,
        "forward_matrix_flops": 6 * 7 * 4 * 10,
        "replicated_across_ranks": True,
        "gradient_reduction_included_in_measured_step": False,
    }
    assert report["load_balance"]["expert_route_rows"]["maximum"] == 7
    assert report["load_balance"]["expert_route_rows"]["zero_load_count"] == 3
    assert report["load_balance"]["rank_receive_route_rows"]["values"] == [7, 0]
    capacity = report["load_balance"]["uniform_capacity_model"]
    assert capacity["analytical_only"] is True
    assert capacity["capacity_per_expert"] == 2
    assert capacity["dropped_route_rows"] == 5
    symmetric = report["symmetric_buffer_model"]
    assert symmetric["source_expert_counts_matrix"] == [[3, 0, 0, 0], [4, 0, 0, 0]]
    assert symmetric["tensor_shape_per_rank"] == [2, 2, 2, 2, 2, 4]
    assert symmetric["dropped_route_rows"] == 3
    assert symmetric["analytical_only"] is True
    assert report["overlap_model"]["derived_at_median"]["optimistic_pipeline_speedup"] >= 1
    assert report["verification"]["output"]["max_tolerance_ratio"] <= 1.0
    assert report["verification"]["gradients"]["performed"] is True
    assert report["verification"]["gradients"]["max_tolerance_ratio"] <= 1.0
    assert report["latency"]["count"] == 1
    assert len(report["raw_samples_ms"]) == 1
    assert report["rank_latency"]["rank_count"] == 2
    assert report["rank_latency"]["iteration_count"] == 1
    assert len(report["rank_latency"]["per_iteration_rank_samples_ms"][0]) == 2
    assert report["rank_latency"]["rank_max"]["raw_samples_ms"] == report["raw_samples_ms"]
    assert set(report["stage_latency"]) == {
        "route_and_pack",
        "exchange_counts",
        "dispatch",
        "expert_compute",
        "restore",
        "combine",
        "shared_expert",
        "backward",
    }
    for stage in report["stage_latency"].values():
        assert stage["latency"]["count"] == 1
        assert len(stage["raw_samples_ms"]) == 1
        assert len(stage["rank_raw_samples_ms"]) == 1
        assert len(stage["rank_raw_samples_ms"][0]) == 2


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo distributed backend is unavailable",
)
def test_torchrun_cli_supports_an_empty_local_token_shard(tmp_path: Path) -> None:
    report = _run_gloo_benchmark(
        tmp_path,
        "--expert-backend=loop",
        "--tokens-per-rank=0",
        "--token-skew=3",
        "--model-dim=4",
        "--hidden-dim=5",
        "--experts=2",
        "--topk=1",
        "--backward",
    )

    assert report["work_estimate"]["total_tokens"] == 3
    assert report["work_estimate"]["total_routes"] == 3
    assert [rank["tokens"] for rank in report["distributed"]["ranks"]] == [0, 3]
    assert report["load_balance"]["rank_send_route_rows"]["values"] == [0, 3]
    assert report["verification"]["output"]["max_tolerance_ratio"] <= 1.0
    assert report["verification"]["gradients"]["max_tolerance_ratio"] <= 1.0
