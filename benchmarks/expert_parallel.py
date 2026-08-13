"""torchrun CLI for Expert Parallel MoE correctness and latency reports."""

from __future__ import annotations

import argparse
import datetime
import os

import torch
import torch.distributed as dist

from ds_flash_mla_moe.benchmarking import write_benchmark_report
from ds_flash_mla_moe.expert_parallel_benchmarking import (
    ExpertParallelBenchmarkConfig,
    benchmark_expert_parallel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument(
        "--router-backend",
        choices=("auto", "cuda", "reference"),
        default="auto",
    )
    parser.add_argument(
        "--route-backend",
        choices=("auto", "cuda", "reference"),
        default="auto",
    )
    parser.add_argument("--expert-backend", choices=("loop", "padded", "cuda"), default="loop")
    parser.add_argument("--tokens-per-rank", type=int, default=32)
    parser.add_argument("--token-skew", type=int, default=0)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--shared-experts",
        type=int,
        default=0,
        help="number of replicated shared experts represented by one widened SwiGLU",
    )
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--n-groups", type=int, default=1)
    parser.add_argument("--topk-groups", type=int)
    parser.add_argument(
        "--hot-expert-bias",
        type=float,
        default=0.0,
        help="non-negative selection-only bias added to expert 0",
    )
    parser.add_argument(
        "--capacity-factor",
        type=float,
        default=1.0,
        help="analytical uniform expert-capacity factor; routes are not dropped",
    )
    parser.add_argument(
        "--symmetric-cell-capacity",
        type=int,
        help="analytical per-source/expert symmetric cell capacity; routes are not dropped",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32", "float64"),
        default="float32",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--pipeline-chunks",
        type=int,
        default=1,
        help="NCCL only: split every rank pair into this many overlapped chunks",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--output", help="rank zero writes JSON to this path")
    return parser.parse_args()


def _distributed_environment() -> tuple[int, int, int]:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"missing {names}; launch this program with torchrun")
    return tuple(int(os.environ[name]) for name in required)  # type: ignore[return-value]


def main() -> None:
    arguments = parse_args()
    rank, _world_size, local_rank = _distributed_environment()
    if arguments.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")

    if arguments.backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL benchmark requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        backend=arguments.backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=arguments.timeout_seconds),
    )
    try:
        config = ExpertParallelBenchmarkConfig(
            tokens_per_rank=arguments.tokens_per_rank,
            token_skew=arguments.token_skew,
            model_dim=arguments.model_dim,
            hidden_dim=arguments.hidden_dim,
            shared_experts=arguments.shared_experts,
            experts=arguments.experts,
            topk=arguments.topk,
            n_groups=arguments.n_groups,
            topk_groups=arguments.topk_groups,
            hot_expert_bias=arguments.hot_expert_bias,
            capacity_factor=arguments.capacity_factor,
            symmetric_cell_capacity=arguments.symmetric_cell_capacity,
            dtype=arguments.dtype,
            backend=arguments.backend,
            router_backend=arguments.router_backend,
            route_backend=arguments.route_backend,
            expert_backend=arguments.expert_backend,
            pipeline_chunks=arguments.pipeline_chunks,
            warmup=arguments.warmup,
            iterations=arguments.iterations,
            seed=arguments.seed,
            backward=arguments.backward,
            verify=not arguments.no_verify,
        )
        report = benchmark_expert_parallel(
            config,
            device=device,
            local_rank=local_rank,
        )
        if rank == 0:
            if report is None:
                raise RuntimeError("rank zero did not receive an EP benchmark report")
            write_benchmark_report(report, arguments.output)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
