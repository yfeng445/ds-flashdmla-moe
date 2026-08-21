"""CLI for reproducible whole-layer DeepSeek MoE forward experiments."""

from __future__ import annotations

import argparse

from ds_flash_mla_moe.benchmarking import write_benchmark_report
from ds_flash_mla_moe.moe_benchmarking import (
    MoEForwardBenchmarkConfig,
    benchmark_moe_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--n-groups", type=int, default=1)
    parser.add_argument("--topk-groups", type=int)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32", "float64"),
        default="float32",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--backend",
        choices=("auto", "cuda", "reference"),
        default="reference",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--route-scale", type=float, default=1.0)
    parser.add_argument(
        "--score-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="initialize and pass a deterministic correction-bias vector",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compare the output with deepseek_moe_reference",
    )
    parser.add_argument("--output", help="write JSON to this path instead of stdout")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = MoEForwardBenchmarkConfig(
        tokens=arguments.tokens,
        model_dim=arguments.model_dim,
        hidden_dim=arguments.hidden_dim,
        experts=arguments.experts,
        topk=arguments.topk,
        n_groups=arguments.n_groups,
        topk_groups=arguments.topk_groups,
        dtype=arguments.dtype,
        device=arguments.device,
        backend=arguments.backend,
        seed=arguments.seed,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        route_scale=arguments.route_scale,
        score_bias=arguments.score_bias,
        verify=arguments.verify,
    )
    write_benchmark_report(benchmark_moe_forward(config), arguments.output)


if __name__ == "__main__":
    main()
