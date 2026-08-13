"""High-precision attention specifications used to validate future CUDA kernels."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _validate_attention_inputs(q: Tensor, k: Tensor, v: Tensor) -> None:
    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        raise ValueError("q, k, and v must have at least two dimensions")
    if q.shape[:-2] != k.shape[:-2] or k.shape[:-2] != v.shape[:-2]:
        raise ValueError("q, k, and v must have identical batch/head dimensions")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same sequence length")
    if q.shape[-1] == 0:
        raise ValueError("attention head dimension must be positive")
    if k.shape[-2] == 0:
        raise ValueError("key sequence length must be positive")
    if q.device != k.device or k.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if not (q.is_floating_point() and k.is_floating_point() and v.is_floating_point()):
        raise TypeError("q, k, and v must be floating-point tensors")


def _compute_dtype(tensor: Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _effective_scale(q: Tensor, scale: float | None) -> float:
    value = float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    if not math.isfinite(value):
        raise ValueError("scale must be finite")
    return value


def _broadcast_mask(attn_mask: Tensor | None, shape: tuple[int, ...], device: torch.device):
    if attn_mask is None:
        return None
    if attn_mask.device != device:
        attn_mask = attn_mask.to(device)
    if attn_mask.dtype != torch.bool and not attn_mask.is_floating_point():
        raise TypeError("attn_mask must be boolean or floating point")
    try:
        return torch.broadcast_to(attn_mask, shape)
    except RuntimeError as exc:
        raise ValueError(
            f"attn_mask with shape {attn_mask.shape} cannot broadcast to {shape}"
        ) from exc


def _causal_keep_mask(query_length: int, key_length: int, device: torch.device) -> Tensor:
    if query_length > key_length:
        raise ValueError("right-aligned causal attention requires query_length <= key_length")
    query_positions = torch.arange(query_length, device=device) + key_length - query_length
    key_positions = torch.arange(key_length, device=device)
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


def _apply_masks(
    scores: Tensor,
    *,
    causal: bool,
    attn_mask: Tensor | None,
) -> Tensor:
    query_length, key_length = scores.shape[-2:]
    if causal:
        scores = scores.masked_fill(
            ~_causal_keep_mask(query_length, key_length, scores.device),
            -torch.inf,
        )
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, -torch.inf)
        else:
            scores = scores + attn_mask.to(scores.dtype)
    return scores


def _stable_probabilities(scores: Tensor) -> tuple[Tensor, Tensor]:
    row_max = scores.amax(dim=-1, keepdim=True)
    finite_row = torch.isfinite(row_max)
    shifted = torch.where(finite_row, scores - row_max, torch.full_like(scores, -torch.inf))
    numerators = torch.exp(shifted)
    denominator = numerators.sum(dim=-1, keepdim=True)
    probabilities = torch.where(
        denominator > 0,
        numerators / denominator.clamp_min(torch.finfo(scores.dtype).tiny),
        torch.zeros_like(numerators),
    )
    lse = torch.where(
        denominator > 0,
        row_max + torch.log(denominator.clamp_min(torch.finfo(scores.dtype).tiny)),
        torch.full_like(row_max, -torch.inf),
    ).squeeze(-1)
    return probabilities, lse


def scaled_dot_product_attention_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    attn_mask: Tensor | None = None,
    return_lse: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Materialized attention reference with stable FP32/FP64 softmax.

    Boolean masks use ``True`` for retained positions. When ``causal=True`` and
    the query is shorter than the key sequence, the causal mask is aligned to
    the lower-right corner; a one-token decode query can therefore see the full
    cache.
    """

    _validate_attention_inputs(q, k, v)
    compute_dtype = _compute_dtype(q)
    q_compute = q.to(compute_dtype)
    k_compute = k.to(compute_dtype)
    v_compute = v.to(compute_dtype)
    effective_scale = _effective_scale(q, scale)

    scores = torch.matmul(q_compute, k_compute.transpose(-1, -2)) * effective_scale
    mask = _broadcast_mask(attn_mask, tuple(scores.shape), scores.device)
    scores = _apply_masks(scores, causal=causal, attn_mask=mask)
    probabilities, lse = _stable_probabilities(scores)
    output = torch.matmul(probabilities, v_compute).to(v.dtype)
    return (output, lse) if return_lse else output


def scaled_dot_product_attention_backward_reference(
    grad_output: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    attn_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Analytic ``dQ``, ``dK``, and ``dV`` specification for exact attention.

    The implementation materializes probabilities and is intended as a
    high-precision oracle for tiled/native backward kernels. Boolean masks use
    ``True`` for retained positions; fully masked rows contribute zero to all
    three gradients.
    """

    _validate_attention_inputs(q, k, v)
    expected_grad_shape = (*q.shape[:-1], v.shape[-1])
    if tuple(grad_output.shape) != expected_grad_shape:
        raise ValueError(f"grad_output must have shape {expected_grad_shape}")
    if grad_output.device != q.device:
        raise ValueError("grad_output and attention inputs must be on the same device")
    if not grad_output.is_floating_point():
        raise TypeError("grad_output must be a floating-point tensor")

    compute_dtype = _compute_dtype(q)
    q_compute = q.to(compute_dtype)
    k_compute = k.to(compute_dtype)
    v_compute = v.to(compute_dtype)
    grad_compute = grad_output.to(compute_dtype)
    effective_scale = _effective_scale(q, scale)

    scores = torch.matmul(q_compute, k_compute.transpose(-1, -2)) * effective_scale
    mask = _broadcast_mask(attn_mask, tuple(scores.shape), scores.device)
    scores = _apply_masks(scores, causal=causal, attn_mask=mask)
    probabilities, _ = _stable_probabilities(scores)

    grad_probabilities = torch.matmul(grad_compute, v_compute.transpose(-1, -2))
    row_correction = (grad_probabilities * probabilities).sum(dim=-1, keepdim=True)
    grad_scores = probabilities * (grad_probabilities - row_correction)

    grad_q = torch.matmul(grad_scores, k_compute) * effective_scale
    grad_k = torch.matmul(grad_scores.transpose(-1, -2), q_compute) * effective_scale
    grad_v = torch.matmul(probabilities.transpose(-1, -2), grad_compute)
    return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype)


def blockwise_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    attn_mask: Tensor | None = None,
    block_size: int = 64,
    return_lse: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Online-softmax attention that never materializes the full score matrix.

    This is deliberately written with regular PyTorch operations. It specifies
    the recurrence that a tiled CUDA kernel must reproduce; it is not intended
    to outperform PyTorch's fused attention implementation.
    """

    _validate_attention_inputs(q, k, v)
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    compute_dtype = _compute_dtype(q)
    q_compute = q.to(compute_dtype)
    k_compute = k.to(compute_dtype)
    v_compute = v.to(compute_dtype)
    effective_scale = _effective_scale(q, scale)

    query_length = q.shape[-2]
    key_length = k.shape[-2]
    output_dim = v.shape[-1]
    state_shape = (*q.shape[:-2], query_length)
    accumulator_shape = (*state_shape, output_dim)

    running_max = torch.full(state_shape, -torch.inf, device=q.device, dtype=compute_dtype)
    running_sum = torch.zeros(state_shape, device=q.device, dtype=compute_dtype)
    accumulator = torch.zeros(accumulator_shape, device=q.device, dtype=compute_dtype)
    full_mask = _broadcast_mask(attn_mask, (*state_shape, key_length), q.device)
    causal_mask = _causal_keep_mask(query_length, key_length, q.device) if causal else None

    for start in range(0, key_length, block_size):
        end = min(start + block_size, key_length)
        k_tile = k_compute[..., start:end, :]
        v_tile = v_compute[..., start:end, :]
        scores = torch.matmul(q_compute, k_tile.transpose(-1, -2)) * effective_scale

        if causal_mask is not None:
            scores = scores.masked_fill(~causal_mask[:, start:end], -torch.inf)
        if full_mask is not None:
            tile_mask = full_mask[..., start:end]
            if tile_mask.dtype == torch.bool:
                scores = scores.masked_fill(~tile_mask, -torch.inf)
            else:
                scores = scores + tile_mask.to(compute_dtype)

        tile_max = scores.amax(dim=-1)
        new_max = torch.maximum(running_max, tile_max)
        finite_new_max = torch.isfinite(new_max)
        old_scale = torch.where(
            torch.isfinite(running_max) & finite_new_max,
            torch.exp(running_max - new_max),
            torch.zeros_like(new_max),
        )
        shifted = torch.where(
            finite_new_max.unsqueeze(-1),
            scores - new_max.unsqueeze(-1),
            torch.full_like(scores, -torch.inf),
        )
        tile_weights = torch.exp(shifted)
        tile_sum = tile_weights.sum(dim=-1)

        accumulator = accumulator * old_scale.unsqueeze(-1) + torch.matmul(tile_weights, v_tile)
        running_sum = running_sum * old_scale + tile_sum
        running_max = new_max

    output = torch.where(
        (running_sum > 0).unsqueeze(-1),
        accumulator / running_sum.clamp_min(torch.finfo(compute_dtype).tiny).unsqueeze(-1),
        torch.zeros_like(accumulator),
    ).to(v.dtype)
    lse = torch.where(
        running_sum > 0,
        running_max + torch.log(running_sum.clamp_min(torch.finfo(compute_dtype).tiny)),
        torch.full_like(running_sum, -torch.inf),
    )
    return (output, lse) if return_lse else output
