"""CLI for reproducible GEMM tiling experiments."""

from __future__ import annotations

import argparse

from ds_flash_mla_moe.benchmarking import write_benchmark_report
from ds_flash_mla_moe.gemm_benchmarking import GEMMBenchmarkConfig, benchmark_gemm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--tile-m", type=int, default=32)
    parser.add_argument("--tile-n", type=int, default=32)
    parser.add_argument("--tile-k", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32", "float64"),
        default="float32",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--implementation",
        choices=("torch", "tiled", "cuda"),
        default="tiled",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--output", help="write JSON to this path instead of stdout")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = GEMMBenchmarkConfig(
        m=arguments.m,
        n=arguments.n,
        k=arguments.k,
        tile_m=arguments.tile_m,
        tile_n=arguments.tile_n,
        tile_k=arguments.tile_k,
        alpha=arguments.alpha,
        beta=arguments.beta,
        dtype=arguments.dtype,
        device=arguments.device,
        implementation=arguments.implementation,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        seed=arguments.seed,
        verify=not arguments.no_verify,
    )
    write_benchmark_report(benchmark_gemm(config), arguments.output)


if __name__ == "__main__":
    main()
