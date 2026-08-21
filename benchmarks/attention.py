"""CLI for reproducible attention latency reports."""

from __future__ import annotations

import argparse

from ds_flash_mla_moe.benchmarking import (
    AttentionBenchmarkConfig,
    benchmark_attention,
    benchmark_attention_backends,
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
    parser.add_argument(
        "--backend",
        choices=(
            "auto",
            "cuda",
            "cuda_rowwise",
            "reference",
            "blockwise",
            "fa1",
            "fa2",
            "fa3",
            "sdpa",
            "flash-attn-4",
        ),
        default="auto",
    )
    parser.add_argument(
        "--compare-fa1-fa2",
        action="store_true",
        help="benchmark formal FA1 and FA2 with the same seed and dimensions",
    )
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
    arguments = parser.parse_args()
    if arguments.compare_fa1_fa2 and arguments.backend != parser.get_default("backend"):
        parser.error("--compare-fa1-fa2 cannot be combined with a non-default --backend")
    return arguments


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
    report = (
        benchmark_attention_backends(config, ("fa1", "fa2"))
        if arguments.compare_fa1_fa2
        else benchmark_attention(config)
    )
    write_benchmark_report(report, arguments.output)


if __name__ == "__main__":
    main()
