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
    positions = _validate_positions(
        positions,
        x.shape[1],
        x.device,
        values_validated=_positions_validated,
    )
    compute_dtype = _compute_dtype(x)
    pair_index = torch.arange(0, x.shape[-1], 2, device=x.device, dtype=compute_dtype)
    inverse_frequency = theta ** (-pair_index / x.shape[-1])
    angles = positions.to(compute_dtype).unsqueeze(-1) * inverse_frequency.unsqueeze(0)
    cos = torch.cos(angles).view(1, x.shape[1], 1, -1)
    sin = torch.sin(angles).view(1, x.shape[1], 1, -1)

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
    positions = _validate_positions(
        positions,
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
