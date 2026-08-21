"""Benchmark eager launch versus one static-shape CUDA Graph replay bucket."""

from __future__ import annotations

import argparse

import torch

from ds_flash_mla_moe.benchmarking import summarize_latencies, write_benchmark_report
from ds_flash_mla_moe.cuda_graph import SingleOutputCUDAGraphRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="write JSON to this path instead of stdout")
    return parser.parse_args()


def _measure(operation, *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        operation()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def main() -> None:
    arguments = parse_args()
    if arguments.batch <= 0 or arguments.width <= 0 or arguments.iterations <= 0:
        raise SystemExit("--batch, --width, and --iterations must be positive")
    if arguments.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark requires CUDA")

    torch.manual_seed(arguments.seed)
    x = torch.randn(arguments.batch, arguments.width, device="cuda")
    weight = torch.randn(arguments.width, arguments.width, device="cuda")

    def operation(value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(value @ weight)

    runner = SingleOutputCUDAGraphRunner.capture(operation, (x,), warmup=arguments.warmup)
    expected = operation(x)
    actual = runner.replay(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
    output_pointer = actual.data_ptr()
    assert runner.replay(x).data_ptr() == output_pointer

    eager_samples = _measure(
        lambda: operation(x),
        warmup=arguments.warmup,
        iterations=arguments.iterations,
    )
    replay_samples = _measure(
        lambda: runner.replay(x),
        warmup=arguments.warmup,
        iterations=arguments.iterations,
    )
    report = {
        "schema_version": 1,
        "benchmark": "single_output_cuda_graph",
        "config": {
            "batch": arguments.batch,
            "width": arguments.width,
            "dtype": str(x.dtype),
            "device": str(x.device),
            "warmup": arguments.warmup,
            "iterations": arguments.iterations,
            "seed": arguments.seed,
        },
        "contract": {
            "caller_inputs_are_copied": True,
            "input_addresses_may_change": True,
            "input_shape_dtype_device_are_static": True,
            "output_address_is_stable": True,
            "forward_only": True,
        },
        "eager": {
            "raw_samples_ms": eager_samples,
            "latency": summarize_latencies(eager_samples),
        },
        "graph_replay_with_input_copy": {
            "raw_samples_ms": replay_samples,
            "latency": summarize_latencies(replay_samples),
        },
        "claims": {"speedup_claimed": False},
    }
    write_benchmark_report(report, arguments.output)


if __name__ == "__main__":
    main()
