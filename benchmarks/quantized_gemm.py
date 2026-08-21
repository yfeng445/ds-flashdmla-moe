"""CLI for reproducible FP8 E4M3FN and symmetric INT8 forward experiments."""

from __future__ import annotations

import argparse

from ds_flash_mla_moe.benchmarking import write_benchmark_report
from ds_flash_mla_moe.quantized_benchmarking import (
    QuantizedGEMMBenchmarkConfig,
    benchmark_quantized_gemm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--format", choices=("int8", "fp8_e4m3fn"), default="int8")
    parser.add_argument("--backend", choices=("reference", "cuda", "auto"), default="reference")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--output", help="write JSON to this path instead of stdout")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = QuantizedGEMMBenchmarkConfig(
        m=arguments.m,
        n=arguments.n,
        k=arguments.k,
        format=arguments.format,
        backend=arguments.backend,
        device=arguments.device,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        seed=arguments.seed,
        verify=not arguments.no_verify,
    )
    write_benchmark_report(benchmark_quantized_gemm(config), arguments.output)


if __name__ == "__main__":
    main()
