"""CLI for reproducible DeepSeek grouped-router experiments."""

from __future__ import annotations

import argparse

from ds_flash_mla_moe.benchmarking import write_benchmark_report
from ds_flash_mla_moe.router_benchmarking import (
    RouterBenchmarkConfig,
    benchmark_router,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--n-groups", type=int, default=1)
    parser.add_argument("--topk-groups", type=int)
    parser.add_argument("--hot-expert-bias", type=float, default=0.0)
    parser.add_argument("--route-scale", type=float, default=1.0)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32", "float64"),
        default="float32",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("auto", "cuda", "reference"), default="reference")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--output", help="write JSON to this path instead of stdout")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = RouterBenchmarkConfig(
        tokens=arguments.tokens,
        model_dim=arguments.model_dim,
        experts=arguments.experts,
        topk=arguments.topk,
        n_groups=arguments.n_groups,
        topk_groups=arguments.topk_groups,
        hot_expert_bias=arguments.hot_expert_bias,
        route_scale=arguments.route_scale,
        dtype=arguments.dtype,
        device=arguments.device,
        backend=arguments.backend,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        seed=arguments.seed,
        backward=arguments.backward,
        verify=not arguments.no_verify,
    )
    write_benchmark_report(benchmark_router(config), arguments.output)


if __name__ == "__main__":
    main()
