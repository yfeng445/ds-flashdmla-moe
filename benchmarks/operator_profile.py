"""Profile one deterministic matrix case with Kineto or NVTX ranges."""

from __future__ import annotations

import argparse

from ds_flash_mla_moe.benchmarking import write_benchmark_report
from ds_flash_mla_moe.profiling import (
    OperatorProfileConfig,
    profile_operator_case,
    run_nvtx_operator_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="exact case from benchmarks/matrix.py")
    parser.add_argument("--side", choices=("native", "baseline"), default="native")
    parser.add_argument("--mode", choices=("torch", "nvtx"), default="torch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--no-record-shapes", action="store_true")
    parser.add_argument("--row-limit", type=int, default=50)
    parser.add_argument("--trace", help="optional Chrome trace path for mode=torch")
    parser.add_argument("--output", help="write the JSON report to this path")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.mode == "nvtx" and arguments.trace:
        raise SystemExit("--trace is only available with --mode torch")
    config = OperatorProfileConfig(
        case=arguments.case,
        side=arguments.side,
        device=arguments.device,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        seed=arguments.seed,
        verify=arguments.verify,
        record_shapes=not arguments.no_record_shapes,
        row_limit=arguments.row_limit,
    )
    report = (
        profile_operator_case(config, trace_path=arguments.trace)
        if arguments.mode == "torch"
        else run_nvtx_operator_case(config)
    )
    write_benchmark_report(report, arguments.output)


if __name__ == "__main__":
    main()
