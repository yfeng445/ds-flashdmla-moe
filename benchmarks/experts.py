"""CLI for reproducible expert-major SwiGLU experiments."""

from __future__ import annotations

import argparse

from ds_flash_mla_moe.benchmarking import write_benchmark_report
from ds_flash_mla_moe.expert_benchmarking import (
    ExpertBenchmarkConfig,
    benchmark_experts,
)


def _expert_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expert counts must be comma-separated integers"
        ) from error
    if not counts:
        raise argparse.ArgumentTypeError("expert counts must not be empty")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-counts", type=_expert_counts, default=(32, 32, 32, 32))
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
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
    config = ExpertBenchmarkConfig(
        expert_counts=arguments.expert_counts,
        model_dim=arguments.model_dim,
        hidden_dim=arguments.hidden_dim,
        dtype=arguments.dtype,
        device=arguments.device,
        backend=arguments.backend,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        seed=arguments.seed,
        backward=arguments.backward,
        verify=not arguments.no_verify,
    )
    write_benchmark_report(benchmark_experts(config), arguments.output)


if __name__ == "__main__":
    main()
