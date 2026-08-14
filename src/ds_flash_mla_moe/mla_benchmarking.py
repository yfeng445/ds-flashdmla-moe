"""Reproducible prefill and decode benchmarks for Multi-head Latent Attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from .benchmarking import (
    _dtype_from_name,
    _environment_metadata,
    _measure_cpu,
    _measure_cuda,
    _verification_tolerances,
    summarize_latencies,
)
from .mla import (
    MLABackend,
    MLAConfig,
    MLALatentCache,
    MLAStaticCache,
    MLAWeights,
    allocate_mla_static_cache,
    append_mla_cache,
    build_mla_cache,
    mla_absorbed_attention,
    mla_absorbed_attention_reference,
    mla_naive_attention_reference,
    write_mla_static_cache,
)

MLAImplementation = Literal["naive", "absorbed", "cuda"]
MLAWorkload = Literal[
    "prefill_attention",
    "prefill_with_cache",
    "decode_attention",
    "decode_with_append",
    "decode_with_static_write",
]


@dataclass(frozen=True)
class MLABenchmarkConfig:
    batch: int = 1
    sequence_length: int = 128
    model_dim: int = 128
    n_heads: int = 4
    q_lora_rank: int = 32
    kv_lora_rank: int = 32
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 16
    v_head_dim: int = 32
    dtype: str = "float32"
    device: str = "cpu"
    implementation: MLAImplementation = "absorbed"
    workload: MLAWorkload = "prefill_attention"
    warmup: int = 2
    iterations: int = 5
    seed: int = 0
    verify: bool = True

    def validate(self) -> None:
        positive = (
            self.batch,
            self.sequence_length,
            self.model_dim,
            self.n_heads,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.iterations,
        )
        if any(value <= 0 for value in positive) or self.q_lora_rank < 0:
            raise ValueError(
                "MLA dimensions and iterations must be positive; q_lora_rank may be zero"
            )
        if self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be even")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.dtype not in {"float16", "bfloat16", "float32", "float64"}:
            raise ValueError("unsupported MLA benchmark dtype")
        if self.implementation not in {"naive", "absorbed", "cuda"}:
            raise ValueError("implementation must be naive, absorbed, or cuda")
        if self.implementation == "cuda" and torch.device(self.device).type != "cuda":
            raise ValueError("the CUDA MLA implementation requires device=cuda")
        if self.implementation == "cuda" and self.dtype != "float32":
            raise ValueError("the CUDA MLA implementation currently requires float32")
        if self.workload not in {
            "prefill_attention",
            "prefill_with_cache",
            "decode_attention",
            "decode_with_append",
            "decode_with_static_write",
        }:
            raise ValueError("unsupported MLA workload")

    def mla_config(self) -> MLAConfig:
        self.validate()
        return MLAConfig(
            n_heads=self.n_heads,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
        )


def mla_cache_storage_estimate(config: MLABenchmarkConfig) -> dict[str, int | float]:
    """Compare compressed cache payload with an expanded per-head K/V cache."""

    config.validate()
    element_size = torch.empty((), dtype=_dtype_from_name(config.dtype)).element_size()
    tokens = config.batch * config.sequence_length
    latent_elements = tokens * (config.kv_lora_rank + config.qk_rope_head_dim)
    expanded_elements = (
        tokens
        * config.n_heads
        * (config.qk_nope_head_dim + config.qk_rope_head_dim + config.v_head_dim)
    )
    position_metadata_bytes = (
        config.sequence_length * torch.empty((), dtype=torch.long).element_size()
    )
    latent_payload_bytes = latent_elements * element_size
    expanded_payload_bytes = expanded_elements * element_size
    return {
        "latent_payload_elements": latent_elements,
        "expanded_kv_payload_elements": expanded_elements,
        "latent_payload_bytes": latent_payload_bytes,
        "expanded_kv_payload_bytes": expanded_payload_bytes,
        "position_metadata_bytes": position_metadata_bytes,
        "latent_total_bytes_with_positions": latent_payload_bytes + position_metadata_bytes,
        "expanded_total_bytes_with_positions": expanded_payload_bytes + position_metadata_bytes,
        "payload_compression_ratio": expanded_payload_bytes / latent_payload_bytes,
    }


def mla_work_estimate(config: MLABenchmarkConfig) -> dict[str, int]:
    """Count matrix FLOPs for the selected path and cache-update boundary."""

    config.validate()
    batch = config.batch
    key_length = config.sequence_length
    query_length = 1 if config.workload.startswith("decode") else key_length
    qk_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    if config.q_lora_rank == 0:
        query_projection = 2 * batch * query_length * config.model_dim * config.n_heads * qk_dim
    else:
        query_projection = (
            2
            * batch
            * query_length
            * (config.model_dim * config.q_lora_rank + config.q_lora_rank * config.n_heads * qk_dim)
        )

    if config.implementation == "naive":
        kv_up_projection = (
            2
            * batch
            * key_length
            * config.kv_lora_rank
            * config.n_heads
            * (config.qk_nope_head_dim + config.v_head_dim)
        )
        attention = (
            2 * batch * config.n_heads * query_length * key_length * (qk_dim + config.v_head_dim)
        )
        path_flops = kv_up_projection + attention
    else:
        query_absorption = (
            2
            * batch
            * query_length
            * config.n_heads
            * config.qk_nope_head_dim
            * config.kv_lora_rank
        )
        content_and_position_scores = (
            2
            * batch
            * config.n_heads
            * query_length
            * key_length
            * (config.kv_lora_rank + config.qk_rope_head_dim)
        )
        latent_value_reduction = (
            2 * batch * config.n_heads * query_length * key_length * config.kv_lora_rank
        )
        value_up_projection = (
            2 * batch * query_length * config.n_heads * config.kv_lora_rank * config.v_head_dim
        )
        path_flops = (
            query_absorption
            + content_and_position_scores
            + latent_value_reduction
            + value_up_projection
        )

    output_projection = (
        2 * batch * query_length * config.n_heads * config.v_head_dim * config.model_dim
    )
    cache_tokens = 0
    if config.workload == "prefill_with_cache":
        cache_tokens = key_length
    elif config.workload in {"decode_with_append", "decode_with_static_write"}:
        cache_tokens = 1
    cache_projection = (
        2
        * batch
        * cache_tokens
        * config.model_dim
        * (config.kv_lora_rank + config.qk_rope_head_dim)
    )

    dtype_size = torch.empty((), dtype=_dtype_from_name(config.dtype)).element_size()
    append_copy_bytes = 0
    if config.workload == "decode_with_append" and key_length > 1:
        latent_width = config.kv_lora_rank + config.qk_rope_head_dim
        append_copy_bytes = (
            2 * batch * key_length * latent_width * dtype_size
            + 2 * key_length * torch.empty((), dtype=torch.long).element_size()
        )
    static_write_bytes = 0
    if config.workload == "decode_with_static_write":
        static_write_bytes = (
            batch * (config.kv_lora_rank + config.qk_rope_head_dim) * dtype_size
            + torch.empty((), dtype=torch.long).element_size()
        )
    total = query_projection + path_flops + output_projection + cache_projection
    return {
        "query_length": query_length,
        "key_length": key_length,
        "query_projection_matrix_flops": query_projection,
        "attention_path_matrix_flops": path_flops,
        "output_projection_matrix_flops": output_projection,
        "cache_projection_matrix_flops": cache_projection,
        "total_matrix_flops": total,
        "functional_append_copy_bytes_lower_bound": append_copy_bytes,
        "static_cache_storage_write_bytes": static_write_bytes,
    }


def _make_weights(
    config: MLABenchmarkConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> MLAWeights:
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    def normal(*shape: int) -> Tensor:
        return torch.randn(*shape, dtype=dtype, generator=generator).to(device)

    common: dict[str, Tensor] = {
        "wkv_a": normal(
            config.kv_lora_rank + config.qk_rope_head_dim,
            config.model_dim,
        ),
        "kv_norm_weight": normal(config.kv_lora_rank),
        "wkv_b": normal(
            config.n_heads * (config.qk_nope_head_dim + config.v_head_dim),
            config.kv_lora_rank,
        ),
        "wo": normal(config.model_dim, config.n_heads * config.v_head_dim),
    }
    qk_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    if config.q_lora_rank == 0:
        return MLAWeights(
            **common,
            wq=normal(config.n_heads * qk_dim, config.model_dim),
        )
    return MLAWeights(
        **common,
        wq_a=normal(config.q_lora_rank, config.model_dim),
        q_norm_weight=normal(config.q_lora_rank),
        wq_b=normal(config.n_heads * qk_dim, config.q_lora_rank),
    )


def _attention(
    implementation: MLAImplementation,
    query: Tensor,
    cache: MLALatentCache,
    mla_config: MLAConfig,
    weights: MLAWeights,
    query_positions: Tensor,
) -> Tensor:
    if implementation == "cuda":
        return mla_absorbed_attention(
            query,
            cache,
            mla_config,
            weights,
            query_positions=query_positions,
            causal=True,
            backend="cuda",
        )
    operation = (
        mla_naive_attention_reference
        if implementation == "naive"
        else (mla_absorbed_attention_reference)
    )
    return operation(
        query,
        cache,
        mla_config,
        weights,
        query_positions=query_positions,
        causal=True,
    )


def _projection_backend(implementation: MLAImplementation) -> MLABackend:
    return "cuda" if implementation == "cuda" else "reference"


def _error_report(
    actual: Tensor,
    expected: Tensor,
    *,
    implementation: MLAImplementation,
) -> dict[str, float | bool | str]:
    rtol, atol = _verification_tolerances(actual.dtype)
    if actual.dtype == torch.float32:
        # MLA comparison spans several projections, softmax, and reductions.
        # The CUDA path additionally uses online softmax and serial FMA while
        # the references delegate reductions to BLAS. These paths remain
        # within ordinary FP32 error of a float64 oracle at smoke-test sizes.
        rtol, atol = 5e-4, 5e-4
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    difference = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
    tolerance = atol + rtol * expected.to(torch.float64).abs()
    return {
        "performed": True,
        "reference": "the alternate naive/absorbed MLA path",
        "rtol": rtol,
        "atol": atol,
        "max_absolute_error": difference.max().item() if difference.numel() else 0.0,
        "max_tolerance_ratio": (difference / tolerance).max().item() if difference.numel() else 0.0,
    }


def benchmark_mla(config: MLABenchmarkConfig) -> dict[str, Any]:
    """Benchmark one fixed MLA workload and return a JSON-serializable report."""

    config.validate()
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA benchmark device is not available")
    dtype = _dtype_from_name(config.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU MLA is not a supported benchmark configuration")
    mla_config = config.mla_config()
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    context = torch.randn(
        config.batch,
        config.sequence_length,
        config.model_dim,
        dtype=dtype,
        generator=generator,
    ).to(device)
    weights = _make_weights(config, dtype=dtype, device=device)
    positions = torch.arange(config.sequence_length, device=device)
    configured_backend = _projection_backend(config.implementation)
    full_cache = build_mla_cache(
        context,
        mla_config,
        weights,
        positions=positions,
        backend=configured_backend,
    )
    prefix_cache = (
        build_mla_cache(
            context[:, :-1],
            mla_config,
            weights,
            positions=positions[:-1],
            backend=configured_backend,
        )
        if config.sequence_length > 1
        else None
    )
    static_cache: MLAStaticCache | None = None
    static_prefix_length = max(config.sequence_length - 1, 0)
    if config.workload == "decode_with_static_write":
        static_cache = allocate_mla_static_cache(
            batch_size=config.batch,
            capacity=config.sequence_length,
            config=mla_config,
            device=device,
            dtype=dtype,
        )
        if static_prefix_length:
            with torch.inference_mode():
                write_mla_static_cache(
                    static_cache,
                    context[:, :-1],
                    mla_config,
                    weights,
                    positions=positions[:-1],
                    backend=configured_backend,
                )

    is_decode = config.workload.startswith("decode")
    query = context[:, -1:] if is_decode else context
    query_positions = positions[-1:] if is_decode else positions

    def operation_for(implementation: MLAImplementation) -> Tensor:
        projection_backend = _projection_backend(implementation)
        if config.workload == "prefill_with_cache":
            cache = build_mla_cache(
                context,
                mla_config,
                weights,
                positions=positions,
                backend=projection_backend,
            )
        elif config.workload == "decode_with_append":
            cache = append_mla_cache(
                prefix_cache,
                query,
                mla_config,
                weights,
                positions=query_positions,
                backend=projection_backend,
            )
        elif config.workload == "decode_with_static_write":
            assert static_cache is not None
            if static_cache.valid_length > static_prefix_length:
                static_cache.truncate(static_prefix_length)
            cache = write_mla_static_cache(
                static_cache,
                query,
                mla_config,
                weights,
                positions=query_positions,
                backend=projection_backend,
            )
        else:
            cache = full_cache
        return _attention(
            implementation,
            query,
            cache,
            mla_config,
            weights,
            query_positions,
        )

    with torch.inference_mode():
        output = operation_for(config.implementation)
        if config.verify:
            alternate: MLAImplementation = (
                "absorbed" if config.implementation in {"naive", "cuda"} else "naive"
            )
            verification = _error_report(
                output,
                operation_for(alternate),
                implementation=config.implementation,
            )
        else:
            verification = {"performed": False}
        samples = (
            _measure_cuda(
                lambda: operation_for(config.implementation),
                config.warmup,
                config.iterations,
                device,
            )
            if device.type == "cuda"
            else _measure_cpu(
                lambda: operation_for(config.implementation),
                config.warmup,
                config.iterations,
            )
        )

    latency = summarize_latencies(samples)
    work = mla_work_estimate(config)
    median_seconds = float(latency["median_ms"]) / 1000.0
    if median_seconds <= 0:
        raise RuntimeError("measured median latency must be positive")
    actual_cache_bytes = (
        full_cache.kv.numel() * full_cache.kv.element_size()
        + full_cache.pe.numel() * full_cache.pe.element_size()
        + full_cache.positions.numel() * full_cache.positions.element_size()
    )
    cache_estimate = mla_cache_storage_estimate(config)
    if actual_cache_bytes != cache_estimate["latent_total_bytes_with_positions"]:
        raise RuntimeError("actual latent cache storage does not match the capacity model")
    return {
        "schema_version": 1,
        "benchmark": (
            "multi_head_latent_attention_cuda"
            if config.implementation == "cuda"
            else "multi_head_latent_attention_reference"
        ),
        "configuration": asdict(config),
        "environment": _environment_metadata(device),
        "output": {
            "shape": list(output.shape),
            "dtype": str(output.dtype).removeprefix("torch."),
            "device": str(output.device),
        },
        "verification": verification,
        "cache_storage": cache_estimate,
        "work_estimate": work,
        "latency": latency,
        "derived": {
            "matrix_tflops_equivalent_at_median": (
                work["total_matrix_flops"] / median_seconds / 1e12
            ),
        },
        "raw_samples_ms": samples,
        "notes": [
            "matrix FLOPs omit RMSNorm, RoPE, softmax, masking, and cache concatenation",
            "cache payload bytes are a storage model, not measured memory traffic",
            "decode_with_append uses functional torch.cat and therefore copies the prefix cache",
            "decode_with_static_write reuses fixed storage and writes only the new cache entry",
            (
                "cuda covers query projection, cache projection/static write, absorbed online "
                "attention, and output projection with native FP32 operators"
                if config.implementation == "cuda"
                else "naive and absorbed paths are correctness references, not fused MLA kernels"
            ),
        ],
    }
