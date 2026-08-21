"""Emit a deterministic trace from the minimal FIFO continuous-batching scheduler."""

from __future__ import annotations

import argparse
import time

from ds_flash_mla_moe.benchmarking import summarize_latencies, write_benchmark_report
from ds_flash_mla_moe.scheduler import ContinuousBatchingScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--num-pages", type=int, default=64)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--output", help="write JSON to this path instead of stdout")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    positive = (
        arguments.requests,
        arguments.prompt_length,
        arguments.max_new_tokens,
        arguments.page_size,
        arguments.num_pages,
        arguments.max_batch_size,
    )
    if any(value <= 0 for value in positive):
        raise SystemExit("all scheduler benchmark dimensions must be positive")

    scheduler = ContinuousBatchingScheduler(
        page_size=arguments.page_size,
        num_pages=arguments.num_pages,
        max_batch_size=arguments.max_batch_size,
    )
    for index in range(arguments.requests):
        scheduler.submit(
            f"request-{index}",
            prompt_length=arguments.prompt_length,
            max_new_tokens=arguments.max_new_tokens,
        )

    iterations: list[dict[str, object]] = []
    samples_ms: list[float] = []
    while True:
        start = time.perf_counter_ns()
        batch = scheduler.schedule()
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        if batch is None:
            break
        samples_ms.append(elapsed_ms)
        iterations.append(
            {
                "batch_id": batch.batch_id,
                "phase": batch.phase,
                "request_ids": list(batch.request_ids),
                "token_counts": list(batch.token_counts),
                "sequence_lengths": list(batch.sequence_lengths),
                "block_tables": [list(row) for row in batch.block_tables],
                "slot_mappings": [list(row) for row in batch.slot_mappings],
            }
        )
        scheduler.complete(batch)

    report = {
        "schema_version": 1,
        "benchmark": "continuous_batching_control_plane",
        "config": {
            "requests": arguments.requests,
            "prompt_length": arguments.prompt_length,
            "max_new_tokens": arguments.max_new_tokens,
            "page_size": arguments.page_size,
            "num_pages": arguments.num_pages,
            "max_batch_size": arguments.max_batch_size,
        },
        "facts": {
            "iterations": len(iterations),
            "all_pages_released": scheduler.allocator.free_page_count == arguments.num_pages,
            "homogeneous_batches": all(
                item["phase"] in {"prefill", "decode"} for item in iterations
            ),
            "model_execution_included": False,
            "gpu_execution_included": False,
        },
        "schedule_call_latency": {
            "raw_samples_ms": samples_ms,
            "summary": summarize_latencies(samples_ms),
        },
        "trace": iterations,
        "claims": {"production_serving_engine": False, "speedup_claimed": False},
    }
    write_benchmark_report(report, arguments.output)


if __name__ == "__main__":
    main()
