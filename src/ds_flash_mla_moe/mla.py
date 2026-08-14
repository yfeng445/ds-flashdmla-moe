"""Executable specifications for DeepSeek-style Multi-head Latent Attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from .attention import _broadcast_mask, _stable_probabilities

MLABackend = Literal["auto", "cuda", "reference"]


@dataclass(frozen=True)
class MLAConfig:
    """Shape and numerical parameters for a compact MLA layer."""

    n_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def validate(self) -> None:
        integer_fields = (
            self.n_heads,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        if any(value <= 0 for value in integer_fields) or self.q_lora_rank < 0:
            raise ValueError("MLA dimensions must be positive; q_lora_rank may also be zero")
        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be even for rotary embeddings")
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("rope_theta and rms_norm_eps must be positive")


@dataclass(frozen=True)
class MLAWeights:
    """MLA weights in PyTorch ``F.linear`` layout: ``[out_features, in_features]``."""

    wkv_a: Tensor
    kv_norm_weight: Tensor
    wkv_b: Tensor
    wo: Tensor
    wq: Tensor | None = None
    wq_a: Tensor | None = None
    q_norm_weight: Tensor | None = None
    wq_b: Tensor | None = None


@dataclass(frozen=True)
class MLALatentCache:
    """Compressed content and positional key cache shared by naive/absorbed paths."""

    kv: Tensor
    pe: Tensor
    positions: Tensor

    @property
    def sequence_length(self) -> int:
        return self.kv.shape[-2]


@dataclass
class MLAStaticCache:
    """Preallocated inference cache with one shared valid prefix for the batch."""

    kv_storage: Tensor
    pe_storage: Tensor
    position_storage: Tensor
    valid_length: int = 0

    @property
    def capacity(self) -> int:
        return self.kv_storage.shape[1]

    def as_latent_cache(
        self,
        *,
        _validated_query_positions: Tensor | None = None,
    ) -> MLALatentCache:
        return _validated_latent_cache(
            kv=self.kv_storage[:, : self.valid_length],
            pe=self.pe_storage[:, : self.valid_length],
            positions=self.position_storage[: self.valid_length],
            recent_positions=_validated_query_positions,
            positions_validated=_static_cache_positions_are_current(self),
        )

    def truncate(self, valid_length: int = 0) -> None:
        """Rewind the inference cursor without reallocating or clearing storage."""

        if not 0 <= valid_length <= self.valid_length:
            raise ValueError("static cache can only truncate within its valid prefix")
        self.valid_length = valid_length


@dataclass
class MLAPagedCache:
    """Inference cache stored as globally addressable fixed-size pages.

    ``position_storage == -1`` marks an unwritten physical slot. Logical
    sequences are described separately by a block table and per-row lengths.
    """

    kv_storage: Tensor
    pe_storage: Tensor
    position_storage: Tensor

    @property
    def num_pages(self) -> int:
        return self.kv_storage.shape[0]

    @property
    def page_size(self) -> int:
        return self.kv_storage.shape[1]

    @property
    def capacity(self) -> int:
        return self.num_pages * self.page_size

    def clear(self) -> None:
        """Mark every physical slot unused without reallocating storage."""

        with torch.no_grad():
            self.position_storage.fill_(-1)


@dataclass(frozen=True)
class MLAPagedCacheView:
    """Padded logical view materialized from a paged cache."""

    kv: Tensor
    pe: Tensor
    positions: Tensor
    sequence_lengths: Tensor

    @property
    def max_sequence_length(self) -> int:
        return self.kv.shape[1]

    @property
    def valid_mask(self) -> Tensor:
        logical_indices = torch.arange(self.max_sequence_length, device=self.kv.device)
        return logical_indices.unsqueeze(0) < self.sequence_lengths.unsqueeze(1)


def _compute_dtype(tensor: Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    compute_dtype = _compute_dtype(x)
    x_compute = x.to(compute_dtype)
    normalized = x_compute * torch.rsqrt(x_compute.square().mean(dim=-1, keepdim=True) + eps)
    return (normalized * weight.to(compute_dtype)).to(x.dtype)


def _validate_weights(config: MLAConfig, weights: MLAWeights, model_dim: int) -> None:
    config.validate()
    expected_kv_a = (config.kv_lora_rank + config.qk_rope_head_dim, model_dim)
    expected_kv_b = (
        config.n_heads * (config.qk_nope_head_dim + config.v_head_dim),
        config.kv_lora_rank,
    )
    if weights.wkv_a.shape != expected_kv_a:
        raise ValueError(f"wkv_a must have shape {expected_kv_a}")
    if weights.kv_norm_weight.shape != (config.kv_lora_rank,):
        raise ValueError("kv_norm_weight must have shape [kv_lora_rank]")
    if weights.wkv_b.shape != expected_kv_b:
        raise ValueError(f"wkv_b must have shape {expected_kv_b}")
    if weights.wo.shape != (model_dim, config.n_heads * config.v_head_dim):
        raise ValueError("wo must have shape [model_dim, n_heads * v_head_dim]")

    projected_q_dim = config.n_heads * config.qk_head_dim
    if config.q_lora_rank == 0:
        if weights.wq is None or weights.wq.shape != (projected_q_dim, model_dim):
            raise ValueError("q_lora_rank=0 requires wq=[n_heads*qk_head_dim, model_dim]")
        if any(value is not None for value in (weights.wq_a, weights.q_norm_weight, weights.wq_b)):
            raise ValueError("direct query projection must not also provide LoRA query weights")
    else:
        if weights.wq is not None:
            raise ValueError("q_lora_rank>0 uses wq_a/q_norm_weight/wq_b instead of wq")
        if weights.wq_a is None or weights.wq_a.shape != (config.q_lora_rank, model_dim):
            raise ValueError("wq_a has an invalid shape")
        if weights.q_norm_weight is None or weights.q_norm_weight.shape != (config.q_lora_rank,):
            raise ValueError("q_norm_weight has an invalid shape")
        if weights.wq_b is None or weights.wq_b.shape != (
            projected_q_dim,
            config.q_lora_rank,
        ):
            raise ValueError("wq_b has an invalid shape")


def _validate_positions(
    positions: Tensor,
    length: int,
    device: torch.device,
    *,
    values_validated: bool = False,
    minimum_exclusive: Tensor | int | None = None,
    order_error: str = "positions must follow the existing prefix",
) -> Tensor:
    if positions.ndim != 1 or positions.numel() != length:
        raise ValueError("positions must be a one-dimensional tensor matching sequence length")
    positions = positions.to(device=device, dtype=torch.long)
    if not positions.numel():
        return positions

    checks: list[Tensor] = []
    first_nonnegative: Tensor | None = None
    strictly_increasing: Tensor | None = None
    follows_prefix: Tensor | None = None
    if not values_validated:
        first_nonnegative = positions[0] >= 0
        checks.append(first_nonnegative)
        if positions.numel() > 1:
            strictly_increasing = torch.all(positions[1:] > positions[:-1])
            checks.append(strictly_increasing)
    if minimum_exclusive is not None:
        follows_prefix = positions[0] > minimum_exclusive
        checks.append(follows_prefix)

    if checks:
        valid = checks[0]
        for check in checks[1:]:
            valid = valid & check
        if not bool(valid):
            if first_nonnegative is not None and not bool(first_nonnegative):
                raise ValueError("positions must be non-negative")
            if strictly_increasing is not None and not bool(strictly_increasing):
                raise ValueError("positions must be strictly increasing")
            if follows_prefix is not None and not bool(follows_prefix):
                raise ValueError(order_error)
    return positions


def _validate_batched_positions(
    positions: Tensor,
    batch_size: int,
    length: int,
    device: torch.device,
    *,
    values_validated: bool = False,
) -> Tensor:
    """Validate shared ``[S]`` or per-row ``[B,S]`` token positions."""

    if positions.ndim == 1:
        return _validate_positions(
            positions,
            length,
            device,
            values_validated=values_validated,
        )
    if positions.ndim != 2 or tuple(positions.shape) != (batch_size, length):
        raise ValueError("positions must have shape [sequence] or [batch, sequence]")
    positions = positions.to(device=device, dtype=torch.long)
    if not positions.numel() or values_validated:
        return positions

    first_nonnegative = torch.all(positions[:, 0] >= 0)
    strictly_increasing = (
        torch.all(positions[:, 1:] > positions[:, :-1])
        if length > 1
        else torch.ones((), device=device, dtype=torch.bool)
    )
    if not bool(first_nonnegative & strictly_increasing):
        if not bool(first_nonnegative):
            raise ValueError("positions must be non-negative")
        raise ValueError("positions must be strictly increasing within each batch row")
    return positions


def _tensor_version(tensor: Tensor) -> int | None:
    try:
        return tensor._version
    except RuntimeError:
        # Tensors created in inference mode do not expose a version counter, so
        # their value validation cannot safely be reused across API calls.
        return None


def _validation_is_current(tensor: Tensor, version: int | None) -> bool:
    return version is not None and _tensor_version(tensor) == version


def _remember_cache_positions(cache: MLALatentCache) -> None:
    object.__setattr__(
        cache,
        "_positions_validation_version",
        _tensor_version(cache.positions),
    )


def _static_cache_positions_are_current(cache: MLAStaticCache) -> bool:
    validated_length = getattr(cache, "_positions_validation_length", -1)
    return validated_length >= cache.valid_length and _validation_is_current(
        cache.position_storage,
        getattr(cache, "_positions_validation_version", None),
    )


def _remember_static_cache_positions(cache: MLAStaticCache) -> None:
    object.__setattr__(
        cache,
        "_positions_validation_version",
        _tensor_version(cache.position_storage),
    )
    object.__setattr__(cache, "_positions_validation_length", cache.valid_length)


def _remember_recent_positions(cache: MLALatentCache, positions: Tensor) -> None:
    object.__setattr__(cache, "_recent_positions", positions)
    object.__setattr__(
        cache,
        "_recent_positions_validation_version",
        _tensor_version(positions),
    )


def _validated_latent_cache(
    *,
    kv: Tensor,
    pe: Tensor,
    positions: Tensor,
    recent_positions: Tensor | None = None,
    positions_validated: bool = True,
) -> MLALatentCache:
    cache = MLALatentCache(kv=kv, pe=pe, positions=positions)
    if positions_validated:
        _remember_cache_positions(cache)
    if recent_positions is not None:
        _remember_recent_positions(cache, recent_positions)
    elif positions_validated:
        _remember_recent_positions(cache, positions)
    return cache


def _apply_rope(
    x: Tensor,
    positions: Tensor,
    theta: float,
    *,
    _positions_validated: bool = False,
) -> Tensor:
    """Apply interleaved-pair RoPE to ``[batch, sequence, heads, rope_dim]``."""

    if x.ndim != 4 or x.shape[-1] % 2 != 0:
        raise ValueError("RoPE input must be [batch, sequence, heads, even_dim]")
    positions = _validate_batched_positions(
        positions,
        x.shape[0],
        x.shape[1],
        x.device,
        values_validated=_positions_validated,
    )
    compute_dtype = _compute_dtype(x)
    pair_index = torch.arange(0, x.shape[-1], 2, device=x.device, dtype=compute_dtype)
    inverse_frequency = theta ** (-pair_index / x.shape[-1])
    angles = positions.to(compute_dtype).unsqueeze(-1) * inverse_frequency
    if positions.ndim == 1:
        angles = angles.unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(2)
    sin = torch.sin(angles).unsqueeze(2)

    x_compute = x.to(compute_dtype)
    even = x_compute[..., 0::2]
    odd = x_compute[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def _project_query(
    x: Tensor,
    config: MLAConfig,
    weights: MLAWeights,
    positions: Tensor,
    *,
    backend: MLABackend = "reference",
    _positions_validated: bool = False,
) -> tuple[Tensor, Tensor]:
    model_dim = x.shape[-1]
    _validate_weights(config, weights, model_dim)
    if x.ndim != 3:
        raise ValueError("MLA query input must have shape [batch, sequence, model_dim]")
    positions = _validate_batched_positions(
        positions,
        x.shape[0],
        x.shape[1],
        x.device,
        values_validated=_positions_validated,
    )
    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be 'auto', 'cuda', or 'reference'")

    if backend != "reference":
        from .ops import mla_query_lora_projection, mla_query_projection

        if config.q_lora_rank == 0:
            assert weights.wq is not None
            return mla_query_projection(
                x,
                weights.wq,
                positions,
                n_heads=config.n_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                rope_theta=config.rope_theta,
                backend=backend,
            )
        assert weights.wq_a is not None
        assert weights.q_norm_weight is not None
        assert weights.wq_b is not None
        return mla_query_lora_projection(
            x,
            weights.wq_a,
            weights.q_norm_weight,
            weights.wq_b,
            positions,
            n_heads=config.n_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            backend=backend,
        )

    compute_dtype = _compute_dtype(x)
    x_compute = x.to(compute_dtype)

    if config.q_lora_rank == 0:
        assert weights.wq is not None
        q = F.linear(x_compute, weights.wq.to(compute_dtype)).to(x.dtype)
    else:
        assert weights.wq_a is not None
        assert weights.q_norm_weight is not None
        assert weights.wq_b is not None
        q_latent = F.linear(x_compute, weights.wq_a.to(compute_dtype)).to(x.dtype)
        q_latent = _rms_norm(q_latent, weights.q_norm_weight, config.rms_norm_eps)
        q = F.linear(q_latent.to(compute_dtype), weights.wq_b.to(compute_dtype)).to(x.dtype)

    q = q.reshape(x.shape[0], x.shape[1], config.n_heads, config.qk_head_dim)
    q_nope, q_pe = torch.split(
        q,
        [config.qk_nope_head_dim, config.qk_rope_head_dim],
        dim=-1,
    )
    q_pe = _apply_rope(
        q_pe,
        positions,
        config.rope_theta,
        _positions_validated=True,
    )
    return q_nope.to(x.dtype).contiguous(), q_pe.to(x.dtype).contiguous()


def build_mla_cache(
    x: Tensor,
    config: MLAConfig,
    weights: MLAWeights,
    *,
    positions: Tensor | None = None,
    backend: MLABackend = "auto",
) -> MLALatentCache:
    """Project model states into normalized latent content and positional cache entries."""

    if x.ndim != 3:
        raise ValueError("MLA cache input must have shape [batch, sequence, model_dim]")
    _validate_weights(config, weights, x.shape[-1])
    positions_generated = positions is None
    if positions_generated:
        positions = torch.arange(x.shape[1], device=x.device)
    assert positions is not None
    positions = _validate_positions(
        positions,
        x.shape[1],
        x.device,
        values_validated=positions_generated,
    )
    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be 'auto', 'cuda', or 'reference'")

    if backend != "reference":
        from .ops import mla_cache_projection

        kv, k_pe = mla_cache_projection(
            x,
            weights.wkv_a,
            weights.kv_norm_weight,
            positions,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            backend=backend,
        )
        return _validated_latent_cache(kv=kv, pe=k_pe, positions=positions)

    compute_dtype = _compute_dtype(x)
    projected = F.linear(x.to(compute_dtype), weights.wkv_a.to(compute_dtype)).to(x.dtype)
    kv, k_pe = torch.split(
        projected,
        [config.kv_lora_rank, config.qk_rope_head_dim],
        dim=-1,
    )
    kv = _rms_norm(kv, weights.kv_norm_weight, config.rms_norm_eps)
    k_pe = _apply_rope(
        k_pe.unsqueeze(2),
        positions,
        config.rope_theta,
        _positions_validated=True,
    ).squeeze(2)
    return _validated_latent_cache(
        kv=kv.to(x.dtype).contiguous(),
        pe=k_pe.to(x.dtype).contiguous(),
        positions=positions,
    )


def append_mla_cache(
    cache: MLALatentCache | None,
    x: Tensor,
    config: MLAConfig,
    weights: MLAWeights,
    *,
    positions: Tensor | None = None,
    backend: MLABackend = "auto",
) -> MLALatentCache:
    """Append projected entries while enforcing batch, device, dtype, and position order."""

    if positions is None:
        start = 0 if cache is None else int(cache.positions[-1].item()) + 1
        positions = torch.arange(start, start + x.shape[1], device=x.device)
    new = build_mla_cache(x, config, weights, positions=positions, backend=backend)
    if cache is None:
        return new
    if cache.kv.shape[0] != new.kv.shape[0]:
        raise ValueError("cache append must preserve batch size")
    if cache.kv.device != new.kv.device or cache.kv.dtype != new.kv.dtype:
        raise ValueError("cache append must preserve device and dtype")
    cache_positions_validated = _validation_is_current(
        cache.positions,
        getattr(cache, "_positions_validation_version", None),
    )
    _validate_positions(
        cache.positions,
        cache.sequence_length,
        cache.kv.device,
        values_validated=cache_positions_validated,
    )
    if not cache_positions_validated:
        _remember_cache_positions(cache)
    if new.positions.numel() and cache.positions.numel():
        _validate_positions(
            new.positions,
            new.positions.numel(),
            new.positions.device,
            values_validated=True,
            minimum_exclusive=cache.positions[-1],
            order_error="appended positions must follow existing cache positions",
        )
    return _validated_latent_cache(
        kv=torch.cat((cache.kv, new.kv), dim=1),
        pe=torch.cat((cache.pe, new.pe), dim=1),
        positions=torch.cat((cache.positions, new.positions), dim=0),
        recent_positions=new.positions,
    )


def allocate_mla_static_cache(
    *,
    batch_size: int,
    capacity: int,
    config: MLAConfig,
    device: torch.device | str,
    dtype: torch.dtype,
) -> MLAStaticCache:
    """Allocate fixed storage for inference-time latent cache writes."""

    config.validate()
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if capacity <= 0:
        raise ValueError("static cache capacity must be positive")
    device = torch.device(device)
    if not dtype.is_floating_point:
        raise TypeError("static cache dtype must be floating point")
    return MLAStaticCache(
        kv_storage=torch.empty(
            batch_size,
            capacity,
            config.kv_lora_rank,
            device=device,
            dtype=dtype,
        ),
        pe_storage=torch.empty(
            batch_size,
            capacity,
            config.qk_rope_head_dim,
            device=device,
            dtype=dtype,
        ),
        position_storage=torch.empty(capacity, device=device, dtype=torch.long),
    )


def write_mla_static_cache(
    cache: MLAStaticCache,
    x: Tensor,
    config: MLAConfig,
    weights: MLAWeights,
    *,
    positions: Tensor | None = None,
    backend: MLABackend = "auto",
) -> MLALatentCache:
    """Project and write a contiguous chunk at the current inference cursor."""

    if torch.is_grad_enabled() and (
        x.requires_grad
        or any(
            tensor is not None and tensor.requires_grad
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
        )
    ):
        raise RuntimeError("MLAStaticCache is inference-only and does not support autograd")
    if x.ndim != 3:
        raise ValueError("static cache input must have shape [batch, sequence, model_dim]")
    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be 'auto', 'cuda', or 'reference'")
    _validate_weights(config, weights, x.shape[-1])
    if x.shape[0] != cache.kv_storage.shape[0]:
        raise ValueError("static cache write must preserve batch size")
    if x.device != cache.kv_storage.device or x.dtype != cache.kv_storage.dtype:
        raise ValueError("static cache input must match storage device and dtype")
    if cache.pe_storage.device != x.device or cache.pe_storage.dtype != x.dtype:
        raise ValueError("static cache K/V and position storage are inconsistent")
    if cache.position_storage.device != x.device:
        raise ValueError("static cache position storage must share the input device")
    end = cache.valid_length + x.shape[1]
    if end > cache.capacity:
        raise ValueError("static cache write exceeds capacity")
    if cache.valid_length and not _static_cache_positions_are_current(cache):
        _validate_positions(
            cache.position_storage[: cache.valid_length],
            cache.valid_length,
            x.device,
        )
    positions_generated = positions is None
    if positions_generated:
        positions = torch.arange(x.shape[1], device=x.device)
        if cache.valid_length:
            positions = positions + cache.position_storage[cache.valid_length - 1] + 1
    assert positions is not None
    minimum_exclusive = (
        cache.position_storage[cache.valid_length - 1]
        if cache.valid_length and not positions_generated
        else None
    )
    positions = _validate_positions(
        positions,
        x.shape[1],
        x.device,
        values_validated=positions_generated,
        minimum_exclusive=minimum_exclusive,
        order_error="static cache positions must follow the valid prefix",
    )

    start = cache.valid_length
    from .ops import mla_cache_projection_write

    with torch.no_grad():
        mla_cache_projection_write(
            x,
            weights.wkv_a,
            weights.kv_norm_weight,
            positions,
            cache.kv_storage,
            cache.pe_storage,
            cache.position_storage,
            start=start,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            backend=backend,
        )
    cache.valid_length = end
    _remember_static_cache_positions(cache)
    return cache.as_latent_cache(_validated_query_positions=positions)


def allocate_mla_paged_cache(
    *,
    num_pages: int,
    page_size: int,
    config: MLAConfig,
    device: torch.device | str,
    dtype: torch.dtype,
) -> MLAPagedCache:
    """Allocate inference storage addressed by global physical token slots."""

    config.validate()
    if num_pages <= 0:
        raise ValueError("paged cache num_pages must be positive")
    if page_size <= 0:
        raise ValueError("paged cache page_size must be positive")
    if not dtype.is_floating_point:
        raise TypeError("paged cache dtype must be floating point")
    device = torch.device(device)
    return MLAPagedCache(
        kv_storage=torch.empty(
            num_pages,
            page_size,
            config.kv_lora_rank,
            device=device,
            dtype=dtype,
        ),
        pe_storage=torch.empty(
            num_pages,
            page_size,
            config.qk_rope_head_dim,
            device=device,
            dtype=dtype,
        ),
        position_storage=torch.full(
            (num_pages, page_size),
            -1,
            device=device,
            dtype=torch.long,
        ),
    )


def _validate_paged_cache_layout(cache: MLAPagedCache, config: MLAConfig | None = None) -> None:
    if cache.kv_storage.ndim != 3 or cache.pe_storage.ndim != 3:
        raise ValueError("paged K/V and positional storage must be rank 3")
    if cache.position_storage.ndim != 2:
        raise ValueError("paged position storage must be rank 2")
    if cache.kv_storage.shape[:2] != cache.pe_storage.shape[:2] or tuple(
        cache.kv_storage.shape[:2]
    ) != tuple(cache.position_storage.shape):
        raise ValueError("paged cache page dimensions are inconsistent")
    if cache.num_pages <= 0 or cache.page_size <= 0:
        raise ValueError("paged cache must contain positive page and page-size dimensions")
    if not cache.kv_storage.is_floating_point() or not cache.pe_storage.is_floating_point():
        raise TypeError("paged cache K/V and positional payloads must be floating point")
    if cache.position_storage.dtype != torch.long:
        raise TypeError("paged cache position storage must use int64")
    if (
        cache.kv_storage.device != cache.pe_storage.device
        or cache.kv_storage.device != cache.position_storage.device
    ):
        raise ValueError("paged cache storage tensors must share a device")
    if cache.kv_storage.dtype != cache.pe_storage.dtype:
        raise ValueError("paged cache payload tensors must share a dtype")
    if not all(
        tensor.is_contiguous()
        for tensor in (cache.kv_storage, cache.pe_storage, cache.position_storage)
    ):
        raise ValueError("paged cache storage tensors must be contiguous")
    if config is not None:
        if cache.kv_storage.shape[-1] != config.kv_lora_rank:
            raise ValueError("paged cache latent rank does not match config")
        if cache.pe_storage.shape[-1] != config.qk_rope_head_dim:
            raise ValueError("paged cache positional dimension does not match config")


def _paged_value_tensor_is_current(cache: MLAPagedCache, name: str, tensor: Tensor) -> bool:
    return tensor is getattr(cache, f"_{name}", None) and _validation_is_current(
        tensor,
        getattr(cache, f"_{name}_validation_version", None),
    )


def _remember_paged_value_tensor(cache: MLAPagedCache, name: str, tensor: Tensor) -> None:
    object.__setattr__(cache, f"_{name}", tensor)
    object.__setattr__(
        cache,
        f"_{name}_validation_version",
        _tensor_version(tensor),
    )


def _paged_metadata_is_current(
    cache: MLAPagedCache,
    block_table: Tensor,
    sequence_lengths: Tensor,
) -> bool:
    return (
        block_table is getattr(cache, "_paged_block_table", None)
        and sequence_lengths is getattr(cache, "_paged_sequence_lengths", None)
        and _validation_is_current(
            block_table,
            getattr(cache, "_paged_block_table_validation_version", None),
        )
        and _validation_is_current(
            sequence_lengths,
            getattr(cache, "_paged_sequence_lengths_validation_version", None),
        )
        and getattr(cache, "_paged_metadata_num_pages", None) == cache.num_pages
        and getattr(cache, "_paged_metadata_page_size", None) == cache.page_size
    )


def _remember_paged_metadata(
    cache: MLAPagedCache,
    block_table: Tensor,
    sequence_lengths: Tensor,
    length_values: list[int],
) -> None:
    object.__setattr__(cache, "_paged_block_table", block_table)
    object.__setattr__(
        cache,
        "_paged_block_table_validation_version",
        _tensor_version(block_table),
    )
    object.__setattr__(cache, "_paged_sequence_lengths", sequence_lengths)
    object.__setattr__(
        cache,
        "_paged_sequence_lengths_validation_version",
        _tensor_version(sequence_lengths),
    )
    object.__setattr__(cache, "_paged_metadata_num_pages", cache.num_pages)
    object.__setattr__(cache, "_paged_metadata_page_size", cache.page_size)
    object.__setattr__(cache, "_paged_length_values", tuple(length_values))


def _paged_logical_validation_is_current(
    cache: MLAPagedCache,
    block_table: Tensor,
    sequence_lengths: Tensor,
) -> bool:
    return _paged_metadata_is_current(
        cache,
        block_table,
        sequence_lengths,
    ) and _validation_is_current(
        cache.position_storage,
        getattr(cache, "_paged_logical_positions_validation_version", None),
    )


def _remember_paged_logical_validation(cache: MLAPagedCache) -> None:
    object.__setattr__(
        cache,
        "_paged_logical_positions_validation_version",
        _tensor_version(cache.position_storage),
    )


def _validate_paged_metadata(
    cache: MLAPagedCache,
    block_table: Tensor,
    sequence_lengths: Tensor,
) -> list[int]:
    _validate_paged_cache_layout(cache)
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [batch, logical_pages]")
    if sequence_lengths.ndim != 1 or sequence_lengths.numel() != block_table.shape[0]:
        raise ValueError("sequence_lengths must contain one entry per block-table row")
    if block_table.dtype != torch.long or sequence_lengths.dtype != torch.long:
        raise TypeError("block_table and sequence_lengths must use int64")
    if (
        block_table.device != cache.kv_storage.device
        or sequence_lengths.device != block_table.device
    ):
        raise ValueError("paged metadata and cache storage must share a device")
    if _paged_metadata_is_current(cache, block_table, sequence_lengths):
        return list(cache._paged_length_values)

    length_values = [int(value) for value in sequence_lengths.detach().cpu().tolist()]
    table_values = block_table.detach().cpu().tolist()
    logical_capacity = block_table.shape[1] * cache.page_size
    for row_index, (length, row) in enumerate(zip(length_values, table_values, strict=True)):
        if not 0 <= length <= logical_capacity:
            raise ValueError(f"sequence_lengths[{row_index}] exceeds the block-table capacity")
        required_pages = (length + cache.page_size - 1) // cache.page_size
        used_pages = row[:required_pages]
        if any(page < 0 or page >= cache.num_pages for page in used_pages):
            raise ValueError(f"block_table row {row_index} contains an out-of-range page")
        if len(set(used_pages)) != len(used_pages):
            raise ValueError(f"block_table row {row_index} repeats a physical page")
        if any(page != -1 for page in row[required_pages:]):
            raise ValueError(f"unused block_table entries in row {row_index} must be -1")
    _remember_paged_metadata(cache, block_table, sequence_lengths, length_values)
    return length_values


def _paged_logical_slots(
    cache: MLAPagedCache,
    block_table: Tensor,
    sequence_lengths: Tensor,
    *,
    _length_values: list[int] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    length_values = (
        _validate_paged_metadata(cache, block_table, sequence_lengths)
        if _length_values is None
        else _length_values
    )
    values_validated = _paged_logical_validation_is_current(
        cache,
        block_table,
        sequence_lengths,
    )
    batch_size = block_table.shape[0]
    max_length = max(length_values, default=0)
    if max_length == 0:
        empty_slots = torch.empty((batch_size, 0), device=block_table.device, dtype=torch.long)
        empty_mask = torch.empty((batch_size, 0), device=block_table.device, dtype=torch.bool)
        empty_positions = torch.empty_like(empty_slots)
        if not values_validated:
            _remember_paged_logical_validation(cache)
        return empty_slots, empty_mask, empty_positions

    logical_indices = torch.arange(max_length, device=block_table.device)
    logical_pages = torch.div(logical_indices, cache.page_size, rounding_mode="floor")
    offsets = logical_indices.remainder(cache.page_size)
    pages = block_table[:, logical_pages]
    valid_mask = logical_indices.unsqueeze(0) < sequence_lengths.unsqueeze(1)
    safe_pages = torch.where(valid_mask, pages, torch.zeros_like(pages))
    physical_slots = safe_pages * cache.page_size + offsets.unsqueeze(0)
    flat_positions = cache.position_storage.view(-1)
    positions = flat_positions[physical_slots]
    positions = torch.where(valid_mask, positions, torch.full_like(positions, -1))

    if not values_validated:
        has_unwritten = torch.any(valid_mask & (positions < 0))
        has_nonmonotonic = torch.zeros((), device=positions.device, dtype=torch.bool)
        if max_length > 1:
            adjacent_valid = logical_indices[:-1].unsqueeze(0) < (sequence_lengths.unsqueeze(1) - 1)
            increasing = positions[:, 1:] > positions[:, :-1]
            has_nonmonotonic = torch.any(adjacent_valid & ~increasing)
        if not bool(~has_unwritten & ~has_nonmonotonic):
            if bool(has_unwritten):
                raise ValueError("block_table references an unwritten paged-cache slot")
            raise ValueError("paged cache positions must increase within each logical sequence")
        _remember_paged_logical_validation(cache)
    return physical_slots, valid_mask, positions


def _validate_paged_logical_cache(
    cache: MLAPagedCache,
    block_table: Tensor,
    sequence_lengths: Tensor,
) -> None:
    length_values = _validate_paged_metadata(cache, block_table, sequence_lengths)
    if not _paged_logical_validation_is_current(cache, block_table, sequence_lengths):
        _paged_logical_slots(
            cache,
            block_table,
            sequence_lengths,
            _length_values=length_values,
        )


def materialize_mla_paged_cache(
    cache: MLAPagedCache,
    block_table: Tensor,
    sequence_lengths: Tensor,
) -> MLAPagedCacheView:
    """Gather a padded logical view for validation and reference execution."""

    physical_slots, valid_mask, positions = _paged_logical_slots(
        cache,
        block_table,
        sequence_lengths,
    )
    batch_size, max_length = physical_slots.shape
    latent_dim = cache.kv_storage.shape[-1]
    rope_dim = cache.pe_storage.shape[-1]
    if max_length == 0:
        kv = cache.kv_storage.new_empty((batch_size, 0, latent_dim))
        pe = cache.pe_storage.new_empty((batch_size, 0, rope_dim))
    else:
        kv = cache.kv_storage.view(-1, latent_dim)[physical_slots]
        pe = cache.pe_storage.view(-1, rope_dim)[physical_slots]
        kv = torch.where(valid_mask.unsqueeze(-1), kv, torch.zeros_like(kv))
        pe = torch.where(valid_mask.unsqueeze(-1), pe, torch.zeros_like(pe))
    return MLAPagedCacheView(
        kv=kv.contiguous(),
        pe=pe.contiguous(),
        positions=positions.contiguous(),
        sequence_lengths=sequence_lengths,
    )


def write_mla_paged_cache(
    cache: MLAPagedCache,
    x: Tensor,
    config: MLAConfig,
    weights: MLAWeights,
    *,
    positions: Tensor,
    slot_mapping: Tensor,
    backend: MLABackend = "auto",
) -> MLAPagedCache:
    """Project tokens and overwrite distinct global physical slots.

    Duplicate slots in one call are rejected because parallel writes would race.
    A later call may intentionally overwrite a previously populated slot; all
    latent, positional, and position fields are replaced together.
    """

    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be 'auto', 'cuda', or 'reference'")
    if torch.is_grad_enabled() and (
        x.requires_grad
        or any(
            tensor is not None and tensor.requires_grad
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
        )
    ):
        raise RuntimeError("MLAPagedCache is inference-only and does not support autograd")
    if x.ndim != 3:
        raise ValueError("paged cache input must have shape [batch, sequence, model_dim]")
    _validate_weights(config, weights, x.shape[-1])
    _validate_paged_cache_layout(cache, config)
    if x.device != cache.kv_storage.device or x.dtype != cache.kv_storage.dtype:
        raise ValueError("paged cache input must match storage device and dtype")

    positions_are_current = _paged_value_tensor_is_current(
        cache,
        "paged_write_positions",
        positions,
    )
    positions = _validate_batched_positions(
        positions,
        x.shape[0],
        x.shape[1],
        x.device,
        values_validated=positions_are_current,
    )
    validated_positions = positions
    if slot_mapping.ndim != 2 or tuple(slot_mapping.shape) != tuple(x.shape[:2]):
        raise ValueError("slot_mapping must have shape [batch, sequence]")
    if slot_mapping.dtype != torch.long:
        raise TypeError("slot_mapping must use int64")
    if slot_mapping.device != x.device:
        raise ValueError("slot_mapping must share the input device")
    slots_are_current = _paged_value_tensor_is_current(
        cache,
        "paged_write_slot_mapping",
        slot_mapping,
    )
    if not slots_are_current:
        slot_values = [int(value) for value in slot_mapping.detach().cpu().reshape(-1).tolist()]
        if any(slot < 0 or slot >= cache.capacity for slot in slot_values):
            raise ValueError("slot_mapping contains an out-of-range physical slot")
        if len(set(slot_values)) != len(slot_values):
            raise ValueError("slot_mapping must not contain duplicate physical slots in one write")

    repeated_position_write = (
        positions_are_current
        and slots_are_current
        and _validation_is_current(
            cache.position_storage,
            getattr(cache, "_paged_logical_positions_validation_version", None),
        )
    )

    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(x.shape[0], -1)
    from .ops import mla_cache_projection_write_slots

    with torch.no_grad():
        mla_cache_projection_write_slots(
            x,
            weights.wkv_a,
            weights.kv_norm_weight,
            positions.contiguous(),
            slot_mapping.contiguous(),
            cache.kv_storage,
            cache.pe_storage,
            cache.position_storage,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            backend=backend,
            _metadata_validated=True,
        )
    _remember_paged_value_tensor(cache, "paged_write_positions", validated_positions)
    _remember_paged_value_tensor(cache, "paged_write_slot_mapping", slot_mapping)
    if repeated_position_write:
        _remember_paged_logical_validation(cache)
    return cache


def _attention_probabilities(
    scores: Tensor,
    query_positions: Tensor,
    key_positions: Tensor,
    *,
    causal: bool,
    attn_mask: Tensor | None,
) -> Tensor:
    if causal:
        keep = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~keep, -torch.inf)
    mask = _broadcast_mask(attn_mask, tuple(scores.shape), scores.device)
    if mask is not None:
        scores = (
            scores.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else scores + mask
        )
    probabilities, _ = _stable_probabilities(scores)
    return probabilities


def _validate_attention_request(
    query_x: Tensor,
    cache: MLALatentCache,
    query_positions: Tensor,
    config: MLAConfig,
    weights: MLAWeights,
) -> Tensor:
    if query_x.ndim != 3:
        raise ValueError("query_x must have shape [batch, sequence, model_dim]")
    _validate_weights(config, weights, query_x.shape[-1])
    if cache.kv.ndim != 3 or cache.pe.ndim != 3:
        raise ValueError("cache tensors must be rank 3")
    if cache.kv.shape[:2] != cache.pe.shape[:2] or cache.kv.shape[0] != query_x.shape[0]:
        raise ValueError("cache batch/sequence dimensions are inconsistent")
    if cache.kv.shape[-1] != config.kv_lora_rank:
        raise ValueError("cache latent rank does not match config")
    if cache.pe.shape[-1] != config.qk_rope_head_dim:
        raise ValueError("cache positional dimension does not match config")
    if cache.positions.numel() != cache.sequence_length:
        raise ValueError("cache positions do not match cache sequence length")
    if (
        cache.kv.device != query_x.device
        or cache.pe.device != query_x.device
        or cache.positions.device != query_x.device
    ):
        raise ValueError("query and cache must be on the same device")
    cache_positions_validated = _validation_is_current(
        cache.positions,
        getattr(cache, "_positions_validation_version", None),
    )
    _validate_positions(
        cache.positions,
        cache.sequence_length,
        query_x.device,
        values_validated=cache_positions_validated,
    )
    if not cache_positions_validated:
        _remember_cache_positions(cache)
        cache_positions_validated = True

    query_positions_validated = (
        query_positions is cache.positions and cache_positions_validated
    ) or (
        query_positions is getattr(cache, "_recent_positions", None)
        and _validation_is_current(
            query_positions,
            getattr(cache, "_recent_positions_validation_version", None),
        )
    )
    query_positions = _validate_positions(
        query_positions,
        query_x.shape[1],
        query_x.device,
        values_validated=query_positions_validated,
    )
    _remember_recent_positions(cache, query_positions)
    return query_positions


def mla_naive_attention_reference(
    query_x: Tensor,
    cache: MLALatentCache,
    config: MLAConfig,
    weights: MLAWeights,
    *,
    query_positions: Tensor,
    causal: bool = True,
    attn_mask: Tensor | None = None,
) -> Tensor:
    """Expand latent K/V per head, then evaluate ordinary attention."""

    query_positions = _validate_attention_request(query_x, cache, query_positions, config, weights)
    compute_dtype = _compute_dtype(query_x)
    q_nope, q_pe = _project_query(
        query_x,
        config,
        weights,
        query_positions,
        _positions_validated=True,
    )
    q = torch.cat((q_nope, q_pe), dim=-1).to(compute_dtype)

    expanded = F.linear(cache.kv.to(compute_dtype), weights.wkv_b.to(compute_dtype))
    expanded = expanded.reshape(
        cache.kv.shape[0],
        cache.sequence_length,
        config.n_heads,
        config.qk_nope_head_dim + config.v_head_dim,
    )
    k_nope, value = torch.split(
        expanded,
        [config.qk_nope_head_dim, config.v_head_dim],
        dim=-1,
    )
    k_pe = cache.pe.unsqueeze(2).expand(-1, -1, config.n_heads, -1).to(compute_dtype)
    key = torch.cat((k_nope, k_pe), dim=-1)

    scores = torch.einsum("bshd,bthd->bhst", q, key) * (config.qk_head_dim**-0.5)
    probabilities = _attention_probabilities(
        scores,
        query_positions,
        cache.positions,
        causal=causal,
        attn_mask=attn_mask,
    )
    heads = torch.einsum("bhst,bthd->bshd", probabilities, value)
    output = F.linear(
        heads.flatten(2).to(compute_dtype),
        weights.wo.to(compute_dtype),
    )
    return output.to(query_x.dtype)


def mla_absorbed_attention_reference(
    query_x: Tensor,
    cache: MLALatentCache,
    config: MLAConfig,
    weights: MLAWeights,
    *,
    query_positions: Tensor,
    causal: bool = True,
    attn_mask: Tensor | None = None,
) -> Tensor:
    """Evaluate attention in latent space using absorbed KV up-projection weights."""

    query_positions = _validate_attention_request(query_x, cache, query_positions, config, weights)
    compute_dtype = _compute_dtype(query_x)
    q_nope, q_pe = _project_query(
        query_x,
        config,
        weights,
        query_positions,
        _positions_validated=True,
    )
    q_nope = q_nope.to(compute_dtype)
    q_pe = q_pe.to(compute_dtype)
    kv = cache.kv.to(compute_dtype)
    pe = cache.pe.to(compute_dtype)

    up = weights.wkv_b.to(compute_dtype).reshape(
        config.n_heads,
        config.qk_nope_head_dim + config.v_head_dim,
        config.kv_lora_rank,
    )
    key_up = up[:, : config.qk_nope_head_dim]
    value_up = up[:, config.qk_nope_head_dim :]

    q_latent = torch.einsum("bshd,hdr->bshr", q_nope, key_up)
    content_scores = torch.einsum("bshr,btr->bhst", q_latent, kv)
    position_scores = torch.einsum("bshd,btd->bhst", q_pe, pe)
    scores = (content_scores + position_scores) * (config.qk_head_dim**-0.5)
    probabilities = _attention_probabilities(
        scores,
        query_positions,
        cache.positions,
        causal=causal,
        attn_mask=attn_mask,
    )

    latent_output = torch.einsum("bhst,btr->bshr", probabilities, kv)
    heads = torch.einsum("bshr,hdr->bshd", latent_output, value_up).to(query_x.dtype)
    output = F.linear(heads.flatten(2).to(compute_dtype), weights.wo.to(compute_dtype))
    return output.to(query_x.dtype)


def mla_absorbed_attention(
    query_x: Tensor,
    cache: MLALatentCache,
    config: MLAConfig,
    weights: MLAWeights,
    *,
    query_positions: Tensor,
    causal: bool = True,
    attn_mask: Tensor | None = None,
    backend: MLABackend = "auto",
) -> Tensor:
    """Run absorbed MLA with an optional end-to-end native CUDA pipeline.

    The native path covers direct/LoRA query projection, RMSNorm, RoPE, absorbed
    attention over the compressed cache, and output projection. Cache builders
    select the matching native projection and static-write operators separately.
    """

    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be 'auto', 'cuda', or 'reference'")
    if attn_mask is not None:
        if backend == "cuda":
            raise RuntimeError("CUDA MLA is unavailable: explicit attention masks are unsupported")
        return mla_absorbed_attention_reference(
            query_x,
            cache,
            config,
            weights,
            query_positions=query_positions,
            causal=causal,
            attn_mask=attn_mask,
        )

    query_positions = _validate_attention_request(query_x, cache, query_positions, config, weights)
    q_nope, q_pe = _project_query(
        query_x,
        config,
        weights,
        query_positions,
        backend=backend,
        _positions_validated=True,
    )
    up = weights.wkv_b.reshape(
        config.n_heads,
        config.qk_nope_head_dim + config.v_head_dim,
        config.kv_lora_rank,
    )
    key_up = up[:, : config.qk_nope_head_dim]
    value_up = up[:, config.qk_nope_head_dim :]

    from .ops import mla_absorbed_attention as _mla_attention_core

    heads = _mla_attention_core(
        q_nope,
        q_pe,
        cache.kv,
        cache.pe,
        key_up,
        value_up,
        query_positions=query_positions,
        key_positions=cache.positions,
        causal=causal,
        scale=config.qk_head_dim**-0.5,
        backend=backend,
    )
    from .ops import mla_output_projection

    return mla_output_projection(heads, weights.wo, backend=backend).to(query_x.dtype)


def mla_paged_attention(
    query_x: Tensor,
    cache: MLAPagedCache,
    block_table: Tensor,
    sequence_lengths: Tensor,
    config: MLAConfig,
    weights: MLAWeights,
    *,
    query_positions: Tensor,
    causal: bool = True,
    backend: MLABackend = "auto",
) -> Tensor:
    """Run inference-time absorbed MLA directly over a paged latent cache."""

    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be 'auto', 'cuda', or 'reference'")
    if torch.is_grad_enabled() and (
        query_x.requires_grad
        or any(
            tensor is not None and tensor.requires_grad
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
        )
    ):
        raise RuntimeError("MLAPagedCache is inference-only and does not support autograd")
    if query_x.ndim != 3:
        raise ValueError("query_x must have shape [batch, sequence, model_dim]")
    _validate_weights(config, weights, query_x.shape[-1])
    _validate_paged_cache_layout(cache, config)
    if block_table.shape[0] != query_x.shape[0]:
        raise ValueError("block_table batch dimension must match query_x")
    if query_x.device != cache.kv_storage.device or query_x.dtype != cache.kv_storage.dtype:
        raise ValueError("query and paged cache must share device and dtype")
    _validate_paged_logical_cache(cache, block_table, sequence_lengths)
    query_positions_are_current = _paged_value_tensor_is_current(
        cache,
        "paged_query_positions",
        query_positions,
    ) or _paged_value_tensor_is_current(
        cache,
        "paged_write_positions",
        query_positions,
    )
    query_positions = _validate_batched_positions(
        query_positions,
        query_x.shape[0],
        query_x.shape[1],
        query_x.device,
        values_validated=query_positions_are_current,
    )
    _remember_paged_value_tensor(cache, "paged_query_positions", query_positions)
    if query_positions.ndim == 1:
        query_positions = query_positions.unsqueeze(0).expand(query_x.shape[0], -1)
    query_positions = query_positions.contiguous()

    q_nope, q_pe = _project_query(
        query_x,
        config,
        weights,
        query_positions,
        backend=backend,
        _positions_validated=True,
    )
    up = weights.wkv_b.reshape(
        config.n_heads,
        config.qk_nope_head_dim + config.v_head_dim,
        config.kv_lora_rank,
    )
    key_up = up[:, : config.qk_nope_head_dim]
    value_up = up[:, config.qk_nope_head_dim :]

    from .ops import mla_output_projection, mla_paged_absorbed_attention

    heads = mla_paged_absorbed_attention(
        q_nope,
        q_pe,
        cache.kv_storage,
        cache.pe_storage,
        cache.position_storage,
        block_table.contiguous(),
        sequence_lengths.contiguous(),
        key_up,
        value_up,
        query_positions=query_positions,
        causal=causal,
        scale=config.qk_head_dim**-0.5,
        backend=backend,
        _metadata_validated=True,
    )
    return mla_output_projection(heads, weights.wo, backend=backend).to(query_x.dtype)
