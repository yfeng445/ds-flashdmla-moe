"""CLI for reproducible MLA prefill, decode, and cache reports."""

from __future__ import annotations

import argparse

from ds_flash_mla_moe.benchmarking import write_benchmark_report
from ds_flash_mla_moe.mla_benchmarking import MLABenchmarkConfig, benchmark_mla


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--q-lora-rank", type=int, default=32)
    parser.add_argument("--kv-lora-rank", type=int, default=32)
    parser.add_argument("--qk-nope-head-dim", type=int, default=32)
    parser.add_argument("--qk-rope-head-dim", type=int, default=16)
    parser.add_argument("--v-head-dim", type=int, default=32)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32", "float64"),
        default="float32",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--implementation", choices=("naive", "absorbed", "cuda"), default="absorbed"
    )
    parser.add_argument(
        "--workload",
        choices=(
            "prefill_attention",
            "prefill_with_cache",
            "decode_attention",
            "decode_with_append",
            "decode_with_static_write",
        ),
        default="prefill_attention",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--output", help="write JSON to this path instead of stdout")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = MLABenchmarkConfig(
        batch=arguments.batch,
        sequence_length=arguments.sequence_length,
        model_dim=arguments.model_dim,
        n_heads=arguments.n_heads,
        q_lora_rank=arguments.q_lora_rank,
        kv_lora_rank=arguments.kv_lora_rank,
        qk_nope_head_dim=arguments.qk_nope_head_dim,
        qk_rope_head_dim=arguments.qk_rope_head_dim,
        v_head_dim=arguments.v_head_dim,
        dtype=arguments.dtype,
        device=arguments.device,
        implementation=arguments.implementation,
        workload=arguments.workload,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        seed=arguments.seed,
        verify=not arguments.no_verify,
    )
    write_benchmark_report(benchmark_mla(config), arguments.output)


if __name__ == "__main__":
    main()
