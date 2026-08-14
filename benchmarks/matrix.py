"""CLI for paired single-GPU native/baseline shape matrices."""

from __future__ import annotations

import argparse

from ds_flash_mla_moe.benchmarking import write_benchmark_report
from ds_flash_mla_moe.matrix_benchmarking import (
    BenchmarkMatrixConfig,
    benchmark_matrix_manifest,
    benchmark_operator_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--profile",
        choices=("smoke", "representative", "flash-attn-4", "mla-low-precision"),
        default="representative",
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=("gemm", "attention", "mla", "experts", "router"),
        help="select one or more families; defaults to all",
    )
    parser.add_argument("--case", action="append", help="select an exact case name")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="write the selected manifest without requiring a GPU",
    )
    parser.add_argument("--output", help="write JSON to this path instead of stdout")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = BenchmarkMatrixConfig(
        device=arguments.device,
        profile=arguments.profile,
        families=tuple(arguments.family) if arguments.family else BenchmarkMatrixConfig.families,
        cases=tuple(arguments.case) if arguments.case else (),
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        seed=arguments.seed,
        verify=not arguments.no_verify,
        fail_fast=arguments.fail_fast,
    )
    report = (
        benchmark_matrix_manifest(config)
        if arguments.list_cases
        else benchmark_operator_matrix(config)
    )
    write_benchmark_report(report, arguments.output)
    if not arguments.list_cases and report["summary"]["failed_case_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
