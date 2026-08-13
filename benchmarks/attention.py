"""CLI for reproducible attention latency reports."""

from __future__ import annotations

import argparse

from ds_flash_mla_moe.benchmarking import (
    AttentionBenchmarkConfig,
    benchmark_attention,
    write_benchmark_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--query-length", type=int, default=128)
    parser.add_argument("--key-length", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32", "float64"),
        default="float32",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--backend", choices=("auto", "cuda", "reference"), default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-block-size", type=int, default=64)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the independent materialized reference check",
    )
    parser.add_argument("--output", help="write JSON to this path instead of stdout")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = AttentionBenchmarkConfig(
        batch=arguments.batch,
        heads=arguments.heads,
        query_length=arguments.query_length,
        key_length=arguments.key_length,
        head_dim=arguments.head_dim,
        value_dim=arguments.value_dim,
        dtype=arguments.dtype,
        device=arguments.device,
        causal=arguments.causal,
        backend=arguments.backend,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        seed=arguments.seed,
        reference_block_size=arguments.reference_block_size,
        verify=not arguments.no_verify,
    )
    write_benchmark_report(benchmark_attention(config), arguments.output)


if __name__ == "__main__":
    main()
