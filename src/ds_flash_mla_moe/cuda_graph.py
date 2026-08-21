"""Forward-only CUDA Graph helpers with explicit static-shape contracts.

CUDA graphs replay work against captured addresses.  These helpers therefore
own static input copies and one stable output tensor.  Caller inputs may have a
new address on every replay, but their shape, dtype, and device must match the
capture bucket exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .mla import (
    MLAConfig,
    MLAPagedCache,
    MLAWeights,
    _validate_batched_positions,
    _validate_paged_logical_cache,
    mla_paged_attention,
)


@dataclass(frozen=True)
class StaticTensorSpec:
    """Shape/dtype/device contract for one copied graph input."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> StaticTensorSpec:
        if not isinstance(tensor, Tensor):
            raise TypeError("graph inputs must be tensors")
        return cls(tuple(tensor.shape), tensor.dtype, tensor.device)

    def is_compatible(self, tensor: Tensor) -> bool:
        return (
            isinstance(tensor, Tensor)
            and tuple(tensor.shape) == self.shape
            and tensor.dtype == self.dtype
            and tensor.device == self.device
        )

    def validate(self, tensor: Tensor, *, name: str) -> None:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if tuple(tensor.shape) != self.shape:
            raise ValueError(f"{name} shape must be {self.shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != self.dtype:
            raise TypeError(f"{name} dtype must be {self.dtype}, got {tensor.dtype}")
        if tensor.device != self.device:
            raise ValueError(f"{name} device must be {self.device}, got {tensor.device}")
        if tensor.requires_grad:
            raise RuntimeError(
                f"{name} requires gradients, but CUDA graph runners are forward-only"
            )


class SingleOutputCUDAGraphRunner:
    """Capture and replay one tensor-valued CUDA operation.

    The runner copies caller values into captured buffers before replay.  The
    returned tensor is owned by the graph and keeps the same address; its value
    is overwritten by the next replay.
    """

    def __init__(
        self,
        *,
        graph: torch.cuda.CUDAGraph,
        static_inputs: tuple[Tensor, ...],
        output: Tensor,
        input_specs: tuple[StaticTensorSpec, ...],
        input_names: tuple[str, ...],
    ) -> None:
        self._graph = graph
        self._static_inputs = static_inputs
        self._output = output
        self._input_specs = input_specs
        self._input_names = input_names

    @classmethod
    def capture(
        cls,
        operation: Callable[..., Tensor],
        example_inputs: Sequence[Tensor],
        *,
        warmup: int = 3,
        input_names: Sequence[str] | None = None,
        pool: Any | None = None,
    ) -> SingleOutputCUDAGraphRunner:
        """Warm up and capture ``operation`` using owned static input buffers."""

        inputs = tuple(example_inputs)
        if not inputs:
            raise ValueError("CUDA graph capture requires at least one tensor input")
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        names = (
            tuple(input_names)
            if input_names is not None
            else tuple(f"input[{index}]" for index in range(len(inputs)))
        )
        if len(names) != len(inputs):
            raise ValueError("input_names must contain one name per graph input")
        specs = tuple(StaticTensorSpec.from_tensor(tensor) for tensor in inputs)
        for spec, tensor, name in zip(specs, inputs, names, strict=True):
            spec.validate(tensor, name=name)
        if any(tensor.device.type != "cuda" for tensor in inputs):
            raise ValueError("CUDA graph inputs must be CUDA tensors")
        device = inputs[0].device
        if any(tensor.device != device for tensor in inputs):
            raise ValueError("all CUDA graph inputs must share one CUDA device")

        with torch.cuda.device(device), torch.no_grad():
            # Keep replay-copy targets as ordinary tensors.  Tensors allocated
            # inside inference mode reject later in-place updates outside that
            # context, while CUDA graph inputs must be overwritten per replay.
            static_inputs = tuple(
                tensor.clone(memory_format=torch.preserve_format) for tensor in inputs
            )
        with torch.cuda.device(device), torch.inference_mode():
            capture_stream = torch.cuda.Stream(device=device)
            current_stream = torch.cuda.current_stream(device)
            capture_stream.wait_stream(current_stream)
            eager_output: object = None
            with torch.cuda.stream(capture_stream):
                eager_output = operation(*static_inputs)
                for _ in range(warmup):
                    eager_output = operation(*static_inputs)
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize(device)
            cls._validate_output(eager_output, device)

            graph = torch.cuda.CUDAGraph()
            graph_kwargs: dict[str, Any] = {"stream": capture_stream}
            if pool is not None:
                graph_kwargs["pool"] = pool
            with torch.cuda.graph(graph, **graph_kwargs):
                output = operation(*static_inputs)
            cls._validate_output(output, device)

        assert isinstance(output, Tensor)
        return cls(
            graph=graph,
            static_inputs=static_inputs,
            output=output,
            input_specs=specs,
            input_names=names,
        )

    @staticmethod
    def _validate_output(output: object, device: torch.device) -> None:
        if not isinstance(output, Tensor):
            raise TypeError("captured operation must return a single Tensor")
        if output.device != device:
            raise ValueError("captured output must share the graph input CUDA device")
        if output.requires_grad:
            raise RuntimeError(
                "captured output requires gradients, but graph runners are forward-only"
            )

    @property
    def output(self) -> Tensor:
        return self._output

    @property
    def static_inputs(self) -> tuple[Tensor, ...]:
        return self._static_inputs

    @property
    def input_specs(self) -> tuple[StaticTensorSpec, ...]:
        return self._input_specs

    def is_compatible(self, *inputs: Tensor) -> bool:
        return len(inputs) == len(self._input_specs) and all(
            spec.is_compatible(tensor)
            for spec, tensor in zip(self._input_specs, inputs, strict=True)
        )

    def validate_inputs(self, *inputs: Tensor) -> None:
        if len(inputs) != len(self._input_specs):
            raise ValueError(
                f"graph replay expected {len(self._input_specs)} inputs, got {len(inputs)}"
            )
        for spec, tensor, name in zip(
            self._input_specs,
            inputs,
            self._input_names,
            strict=True,
        ):
            spec.validate(tensor, name=name)

    def replay(self, *inputs: Tensor) -> Tensor:
        """Validate every input, copy values, replay, and return stable output."""

        self.validate_inputs(*inputs)
        with torch.no_grad():
            for static, source in zip(self._static_inputs, inputs, strict=True):
                static.copy_(source, non_blocking=True)
            self._graph.replay()
        return self._output

    __call__ = replay


@dataclass(frozen=True)
class MLAPagedDecodeGraphBucket:
    """Exact static-shape bucket for one-token MLA paged decode."""

    batch_size: int
    max_logical_pages: int
    model_dim: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.max_logical_pages <= 0 or self.model_dim <= 0:
            raise ValueError("MLA decode graph bucket dimensions must be positive")

    def is_compatible(
        self,
        query_x: Tensor,
        block_table: Tensor,
        sequence_lengths: Tensor,
        query_positions: Tensor,
    ) -> bool:
        return (
            tuple(query_x.shape) == (self.batch_size, 1, self.model_dim)
            and tuple(block_table.shape) == (self.batch_size, self.max_logical_pages)
            and tuple(sequence_lengths.shape) == (self.batch_size,)
            and tuple(query_positions.shape) == (self.batch_size, 1)
        )

    @classmethod
    def from_inputs(
        cls,
        query_x: Tensor,
        block_table: Tensor,
        sequence_lengths: Tensor,
        query_positions: Tensor,
    ) -> MLAPagedDecodeGraphBucket:
        if query_x.ndim != 3 or query_x.shape[1] != 1:
            raise ValueError("MLA paged decode query must have shape [batch, 1, model_dim]")
        if block_table.ndim != 2:
            raise ValueError("MLA paged decode block_table must have shape [batch, pages]")
        bucket = cls(query_x.shape[0], block_table.shape[1], query_x.shape[2])
        if not bucket.is_compatible(query_x, block_table, sequence_lengths, query_positions):
            raise ValueError("MLA paged decode metadata does not match the static graph bucket")
        return bucket


@dataclass(frozen=True)
class _CapturedAddress:
    name: str
    tensor: Tensor
    spec: StaticTensorSpec
    pointer: int

    @classmethod
    def record(cls, name: str, tensor: Tensor) -> _CapturedAddress:
        return cls(name, tensor, StaticTensorSpec.from_tensor(tensor), tensor.data_ptr())

    def validate(self) -> None:
        self.spec.validate(self.tensor, name=self.name)
        if self.tensor.data_ptr() != self.pointer:
            raise RuntimeError(f"{self.name} address changed after CUDA graph capture")


class MLAPagedDecodeGraphRunner:
    """Captured native MLA paged-decode pipeline for one exact graph bucket.

    Cache and weight tensors remain at their captured addresses.  Replay
    metadata is fully validated against the paged cache before any static input
    is overwritten, then the already-prevalidated raw query, paged-attention,
    and output-projection operators are replayed.
    """

    def __init__(
        self,
        *,
        runner: SingleOutputCUDAGraphRunner,
        bucket: MLAPagedDecodeGraphBucket,
        cache: MLAPagedCache,
        captured_addresses: tuple[_CapturedAddress, ...],
        captured_cache_tensors: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        self._runner = runner
        self.bucket = bucket
        self._cache = cache
        self._captured_addresses = captured_addresses
        self._captured_cache_tensors = captured_cache_tensors

    @classmethod
    def capture(
        cls,
        query_x: Tensor,
        cache: MLAPagedCache,
        block_table: Tensor,
        sequence_lengths: Tensor,
        config: MLAConfig,
        weights: MLAWeights,
        *,
        query_positions: Tensor,
        causal: bool = True,
        warmup: int = 3,
        pool: Any | None = None,
    ) -> MLAPagedDecodeGraphRunner:
        bucket = MLAPagedDecodeGraphBucket.from_inputs(
            query_x,
            block_table,
            sequence_lengths,
            query_positions,
        )
        floating_weights = tuple(
            tensor
            for tensor in (
                weights.wkv_a,
                weights.kv_norm_weight,
                weights.wkv_b,
                weights.wo,
                weights.wq,
                weights.wq_a,
                weights.q_norm_weight,
                weights.wq_b,
            )
            if tensor is not None
        )
        if query_x.requires_grad or any(tensor.requires_grad for tensor in floating_weights):
            raise RuntimeError("MLA paged decode graph capture is forward-only")
        dynamic_inputs = (query_x, block_table, sequence_lengths, query_positions)
        if not all(tensor.is_contiguous() for tensor in dynamic_inputs):
            raise ValueError("MLA paged decode graph inputs must be contiguous")

        # This public call performs capability, layout, logical-page, and
        # absolute-position validation before raw operators enter the graph.
        with torch.inference_mode():
            mla_paged_attention(
                query_x,
                cache,
                block_table,
                sequence_lengths,
                config,
                weights,
                query_positions=query_positions,
                causal=causal,
                backend="cuda",
            )

        up = weights.wkv_b.reshape(
            config.n_heads,
            config.qk_nope_head_dim + config.v_head_dim,
            config.kv_lora_rank,
        )
        key_up = up[:, : config.qk_nope_head_dim]
        value_up = up[:, config.qk_nope_head_dim :]

        def operation(
            static_query: Tensor,
            static_table: Tensor,
            static_lengths: Tensor,
            static_positions: Tensor,
        ) -> Tensor:
            if config.q_lora_rank == 0:
                assert weights.wq is not None
                q_nope, q_pe = torch.ops.ds_flash_mla_moe.mla_query_projection.default(
                    static_query,
                    weights.wq,
                    static_positions,
                    config.n_heads,
                    config.qk_nope_head_dim,
                    config.qk_rope_head_dim,
                    config.rope_theta,
                )
            else:
                assert weights.wq_a is not None
                assert weights.q_norm_weight is not None
                assert weights.wq_b is not None
                q_nope, q_pe = torch.ops.ds_flash_mla_moe.mla_query_lora_projection.default(
                    static_query,
                    weights.wq_a,
                    weights.q_norm_weight,
                    weights.wq_b,
                    static_positions,
                    config.n_heads,
                    config.qk_nope_head_dim,
                    config.qk_rope_head_dim,
                    config.rope_theta,
                    config.rms_norm_eps,
                )
            heads = torch.ops.ds_flash_mla_moe.mla_paged_absorbed_attention.default(
                q_nope,
                q_pe,
                cache.kv_storage,
                cache.pe_storage,
                cache.position_storage,
                static_table,
                static_lengths,
                key_up,
                value_up,
                static_positions,
                True,
                causal,
                config.qk_head_dim**-0.5,
            )
            return torch.ops.ds_flash_mla_moe.mla_output_projection.default(heads, weights.wo)

        runner = SingleOutputCUDAGraphRunner.capture(
            operation,
            (query_x, block_table, sequence_lengths, query_positions),
            warmup=warmup,
            input_names=("query_x", "block_table", "sequence_lengths", "query_positions"),
            pool=pool,
        )
        fixed_tensors = (
            ("cache.kv_storage", cache.kv_storage),
            ("cache.pe_storage", cache.pe_storage),
            ("cache.position_storage", cache.position_storage),
            *((f"weight[{index}]", tensor) for index, tensor in enumerate(floating_weights)),
        )
        captured_addresses = tuple(
            _CapturedAddress.record(name, tensor) for name, tensor in fixed_tensors
        )
        return cls(
            runner=runner,
            bucket=bucket,
            cache=cache,
            captured_addresses=captured_addresses,
            captured_cache_tensors=(
                cache.kv_storage,
                cache.pe_storage,
                cache.position_storage,
            ),
        )

    @property
    def output(self) -> Tensor:
        return self._runner.output

    @property
    def static_inputs(self) -> tuple[Tensor, ...]:
        return self._runner.static_inputs

    def replay(
        self,
        query_x: Tensor,
        block_table: Tensor,
        sequence_lengths: Tensor,
        *,
        query_positions: Tensor,
    ) -> Tensor:
        inputs = (query_x, block_table, sequence_lengths, query_positions)
        self._runner.validate_inputs(*inputs)
        current_cache_tensors = (
            self._cache.kv_storage,
            self._cache.pe_storage,
            self._cache.position_storage,
        )
        if any(
            current is not captured
            for current, captured in zip(
                current_cache_tensors,
                self._captured_cache_tensors,
                strict=True,
            )
        ):
            raise RuntimeError("captured MLA cache storage was replaced after graph capture")
        for captured in self._captured_addresses:
            captured.validate()
        _validate_paged_logical_cache(self._cache, block_table, sequence_lengths)
        _validate_batched_positions(
            query_positions,
            self.bucket.batch_size,
            1,
            query_x.device,
        )
        return self._runner.replay(*inputs)

    __call__ = replay
