"""Dispatch boundary between executable specifications and optional native kernels."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from .attention import (
    _broadcast_mask,
    _causal_keep_mask,
    _stable_probabilities,
    _validate_attention_inputs,
    blockwise_attention,
    scaled_dot_product_attention_backward_reference,
    scaled_dot_product_attention_reference,
)
from .gemm import _gemm_compute_dtype, _validate_gemm_inputs, gemm_reference
from .moe import RoutingResult, deepseek_grouped_topk, pack_routes_reference

AttentionBackend = Literal[
    "auto",
    "cuda",
    "cuda_rowwise",
    "reference",
    "blockwise",
    "fa1",
    "fa2",
]
NativeAttentionBackend = Literal["cuda_rowwise", "fa1", "fa2"]
GEMMBackend = Literal["auto", "cuda", "reference"]
MLABackend = Literal["auto", "cuda", "reference"]

_LIBRARY_HANDLES: list[torch.library.Library] = []
_NATIVE_EXTENSION_LOADED = False


def _native_library_candidates() -> list[Path]:
    package_dir = Path(__file__).resolve().parent
    native_suffixes = {".so", ".pyd", ".dll", ".dylib"}
    return sorted(
        path
        for path in package_dir.glob("_C*")
        if path.is_file() and path.suffix.lower() in native_suffixes
    )


def _load_native_library() -> bool:
    candidates = _native_library_candidates()
    if not candidates:
        return False
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates)
        raise ImportError(
            f"expected exactly one ds_flash_mla_moe native library, found: {rendered}"
        )
    try:
        torch.ops.load_library(str(candidates[0]))
    except OSError as exc:
        raise ImportError(f"failed to load native operator library {candidates[0]}") from exc
    return True


def _operator_is_defined(operator: str) -> bool:
    try:
        torch._C._dispatch_find_schema_or_throw(  # type: ignore[attr-defined]
            f"ds_flash_mla_moe::{operator}", ""
        )
    except RuntimeError:
        return False
    return True


def _operator_has_cuda_kernel(operator: str) -> bool:
    if not _operator_is_defined(operator):
        return False
    return torch._C._dispatch_has_kernel_for_dispatch_key(  # type: ignore[attr-defined]
        f"ds_flash_mla_moe::{operator}", "CUDA"
    )


_NATIVE_EXTENSION_LOADED = _load_native_library()

_FORWARD_SCHEMA = (
    "attention_forward(Tensor q, Tensor k, Tensor v, bool causal, float scale) -> Tensor"
)
_FA1_FORWARD_SCHEMA = (
    "attention_fa1_forward(Tensor q, Tensor k, Tensor v, bool causal, float scale) -> Tensor"
)
_FA2_FORWARD_SCHEMA = (
    "attention_fa2_forward(Tensor q, Tensor k, Tensor v, bool causal, float scale) -> Tensor"
)
_BACKWARD_SCHEMA = (
    "attention_backward(Tensor grad_output, Tensor q, Tensor k, Tensor v, "
    "bool causal, float scale) -> (Tensor, Tensor, Tensor)"
)
_ROUTE_PACK_SCHEMA = (
    "route_pack(Tensor x, Tensor route_weights, Tensor expert_indices, "
    "Tensor expert_owner, int world_size) -> "
    "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
)
_ROUTE_COMBINE_SCHEMA = (
    "route_combine(Tensor contributions, Tensor route_weights, "
    "Tensor token_indices, int token_count) -> Tensor"
)
_TILED_GEMM_SCHEMA = "tiled_gemm(Tensor a, Tensor b, Tensor? c, float alpha, float beta) -> Tensor"
_SWIGLU_EXPERTS_SCHEMA = (
    "swiglu_experts(Tensor activations, Tensor expert_offsets, Tensor expert_w1, "
    "Tensor expert_w2, Tensor expert_w3) -> Tensor"
)
_EXPERT_MAJOR_PACK_SCHEMA = (
    "expert_major_pack(Tensor activations, Tensor expert_indices, "
    "Tensor local_expert_ids) -> (Tensor, Tensor, Tensor)"
)
_GROUPED_TOPK_SCHEMA = (
    "grouped_topk(Tensor x, Tensor gate_weight, int topk, int n_groups, "
    "int topk_groups, Tensor? score_bias, float route_scale) -> (Tensor, Tensor)"
)
_DEEPSEEK_MOE_FORWARD_SCHEMA = (
    "deepseek_moe_forward(Tensor x, Tensor gate_weight, Tensor expert_w1, "
    "Tensor expert_w2, Tensor expert_w3, int topk, int n_groups, "
    "int topk_groups, Tensor? score_bias, float route_scale) -> Tensor"
)
_MLA_ABSORBED_ATTENTION_SCHEMA = (
    "mla_absorbed_attention(Tensor q_nope, Tensor q_pe, Tensor kv, Tensor pe, "
    "Tensor key_up, Tensor value_up, Tensor query_positions, Tensor key_positions, "
    "bool causal, float scale) -> Tensor"
)
_MLA_QUERY_PROJECTION_SCHEMA = (
    "mla_query_projection(Tensor x, Tensor wq, Tensor positions, int n_heads, "
    "int qk_nope_head_dim, int qk_rope_head_dim, float rope_theta) -> (Tensor, Tensor)"
)
_MLA_QUERY_LORA_PROJECTION_SCHEMA = (
    "mla_query_lora_projection(Tensor x, Tensor wq_a, Tensor q_norm_weight, Tensor wq_b, "
    "Tensor positions, int n_heads, int qk_nope_head_dim, int qk_rope_head_dim, "
    "float rope_theta, float rms_norm_eps) -> (Tensor, Tensor)"
)
_MLA_CACHE_PROJECTION_SCHEMA = (
    "mla_cache_projection(Tensor x, Tensor wkv_a, Tensor kv_norm_weight, "
    "Tensor positions, int kv_lora_rank, float rope_theta, float rms_norm_eps) "
    "-> (Tensor, Tensor)"
)
_MLA_CACHE_PROJECTION_WRITE_SCHEMA = (
    "mla_cache_projection_write(Tensor x, Tensor wkv_a, Tensor kv_norm_weight, "
    "Tensor positions, Tensor(a!) kv_storage, Tensor(b!) pe_storage, "
    "Tensor(c!) position_storage, int start, float rope_theta, float rms_norm_eps) -> ()"
)
_MLA_CACHE_PROJECTION_WRITE_SLOTS_SCHEMA = (
    "mla_cache_projection_write_slots(Tensor x, Tensor wkv_a, Tensor kv_norm_weight, "
    "Tensor positions, Tensor slot_mapping, Tensor(a!) kv_storage, Tensor(b!) pe_storage, "
    "Tensor(c!) position_storage, bool metadata_validated, float rope_theta, "
    "float rms_norm_eps) -> ()"
)
_MLA_PAGED_ABSORBED_ATTENTION_SCHEMA = (
    "mla_paged_absorbed_attention(Tensor q_nope, Tensor q_pe, Tensor kv_storage, "
    "Tensor pe_storage, Tensor position_storage, Tensor block_table, Tensor sequence_lengths, "
    "Tensor key_up, Tensor value_up, Tensor query_positions, bool metadata_validated, "
    "bool causal, float scale) -> Tensor"
)
_MLA_OUTPUT_PROJECTION_SCHEMA = "mla_output_projection(Tensor heads, Tensor wo) -> Tensor"
_SCHEMAS = {
    "attention_forward": _FORWARD_SCHEMA,
    "attention_fa1_forward": _FA1_FORWARD_SCHEMA,
    "attention_fa2_forward": _FA2_FORWARD_SCHEMA,
    "attention_backward": _BACKWARD_SCHEMA,
    "route_pack": _ROUTE_PACK_SCHEMA,
    "route_combine": _ROUTE_COMBINE_SCHEMA,
    "tiled_gemm": _TILED_GEMM_SCHEMA,
    "swiglu_experts": _SWIGLU_EXPERTS_SCHEMA,
    "expert_major_pack": _EXPERT_MAJOR_PACK_SCHEMA,
    "grouped_topk": _GROUPED_TOPK_SCHEMA,
    "deepseek_moe_forward": _DEEPSEEK_MOE_FORWARD_SCHEMA,
    "mla_absorbed_attention": _MLA_ABSORBED_ATTENTION_SCHEMA,
    "mla_query_projection": _MLA_QUERY_PROJECTION_SCHEMA,
    "mla_query_lora_projection": _MLA_QUERY_LORA_PROJECTION_SCHEMA,
    "mla_cache_projection": _MLA_CACHE_PROJECTION_SCHEMA,
    "mla_cache_projection_write": _MLA_CACHE_PROJECTION_WRITE_SCHEMA,
    "mla_cache_projection_write_slots": _MLA_CACHE_PROJECTION_WRITE_SLOTS_SCHEMA,
    "mla_paged_absorbed_attention": _MLA_PAGED_ABSORBED_ATTENTION_SCHEMA,
    "mla_output_projection": _MLA_OUTPUT_PROJECTION_SCHEMA,
}
_missing_schemas = {
    operator: schema for operator, schema in _SCHEMAS.items() if not _operator_is_defined(operator)
}
if len(_missing_schemas) == len(_SCHEMAS):
    definition = torch.library.Library("ds_flash_mla_moe", "DEF")
    for schema in _missing_schemas.values():
        definition.define(schema)
    _LIBRARY_HANDLES.append(definition)
elif _missing_schemas:
    fragment = torch.library.Library("ds_flash_mla_moe", "FRAGMENT")
    for schema in _missing_schemas.values():
        fragment.define(schema)
    _LIBRARY_HANDLES.append(fragment)


def _composite_attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool,
    scale: float,
) -> Tensor:
    return blockwise_attention(q, k, v, causal=causal, scale=scale)


def _fake_attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool,
    scale: float,
) -> Tensor:
    torch._check(q.ndim >= 2 and k.ndim >= 2 and v.ndim >= 2)
    torch._check(q.shape[:-2] == k.shape[:-2])
    torch._check(k.shape[:-2] == v.shape[:-2])
    torch._check(q.shape[-1] == k.shape[-1])
    torch._check(k.shape[-2] == v.shape[-2])
    torch._check(q.device == k.device and k.device == v.device)
    torch._check(q.is_floating_point() and k.is_floating_point() and v.is_floating_point())
    torch._check(q.shape[-1] > 0)
    torch._check((not causal) or q.shape[-2] <= k.shape[-2])
    return v.new_empty((*q.shape[:-1], v.shape[-1]))


def _fake_formal_attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool,
    scale: float,
) -> Tensor:
    output = _fake_attention_forward(q, k, v, causal, scale)
    torch._check(
        not any(tensor.requires_grad for tensor in (q, k, v)),
        lambda: (
            "formal FA1/FA2 forward kernels are forward-only and do not accept "
            "requires_grad tensors"
        ),
    )
    torch._check(q.ndim == 4 and k.ndim == 4 and v.ndim == 4)
    torch._check(q.dtype == torch.float16)
    torch._check(k.dtype == q.dtype and v.dtype == q.dtype)
    torch._check(q.shape[-1] <= 128 and v.shape[-1] <= 128)
    torch._check(k.shape[-2] > 0)
    return output


def _composite_attention_backward(
    grad_output: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    return scaled_dot_product_attention_backward_reference(
        grad_output,
        q,
        k,
        v,
        causal=causal,
        scale=scale,
    )


def _composite_route_pack(
    x: Tensor,
    route_weights: Tensor,
    expert_indices: Tensor,
    expert_owner: Tensor,
    world_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    packed = pack_routes_reference(
        x,
        RoutingResult(route_weights, expert_indices),
        n_experts=expert_owner.numel(),
        expert_owner=expert_owner,
        world_size=world_size,
    )
    route_indices = packed.token_indices * route_weights.shape[1] + packed.slot_indices
    return (
        packed.activations,
        packed.route_weights,
        route_indices,
        packed.expert_indices,
        packed.counts_per_expert,
        packed.rank_counts,
    )


def _composite_route_combine(
    contributions: Tensor,
    route_weights: Tensor,
    token_indices: Tensor,
    token_count: int,
) -> Tensor:
    compute_dtype = torch.float64 if contributions.dtype == torch.float64 else torch.float32
    weighted = contributions.to(compute_dtype) * route_weights.to(compute_dtype).unsqueeze(-1)
    output = torch.zeros(
        (token_count, contributions.shape[-1]),
        dtype=compute_dtype,
        device=contributions.device,
    ).index_add(0, token_indices, weighted)
    return output.to(contributions.dtype)


def _composite_tiled_gemm(
    a: Tensor,
    b: Tensor,
    c: Tensor | None,
    alpha: float,
    beta: float,
) -> Tensor:
    return gemm_reference(a, b, c, alpha=alpha, beta=beta)


def _composite_swiglu_experts(
    activations: Tensor,
    expert_offsets: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
) -> Tensor:
    return _tensorized_swiglu_experts_reference(
        activations,
        expert_offsets,
        expert_w1,
        expert_w2,
        expert_w3,
    )


def _composite_expert_major_pack(
    activations: Tensor,
    expert_indices: Tensor,
    local_expert_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if expert_indices.numel() == 0:
        return (
            activations,
            expert_indices.new_zeros(local_expert_ids.numel() + 1),
            expert_indices.new_empty(0),
        )
    if expert_indices.numel() and local_expert_ids.numel() == 0:
        raise ValueError("non-empty activations require at least one local expert")
    matches = expert_indices.unsqueeze(1) == local_expert_ids.unsqueeze(0)
    if torch.any(matches.sum(dim=1) != 1):
        raise ValueError("every row expert must appear exactly once in local_expert_ids")
    local_indices = matches.to(torch.long).argmax(dim=1)
    permutation = torch.argsort(local_indices, stable=True)
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation.scatter_(
        0,
        permutation,
        torch.arange(permutation.numel(), device=permutation.device),
    )
    counts = torch.bincount(local_indices, minlength=local_expert_ids.numel())
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return activations.index_select(0, permutation), offsets, inverse_permutation


def _composite_grouped_topk(
    x: Tensor,
    gate_weight: Tensor,
    topk: int,
    n_groups: int,
    topk_groups: int,
    score_bias: Tensor | None,
    route_scale: float,
) -> tuple[Tensor, Tensor]:
    routing = deepseek_grouped_topk(
        x,
        gate_weight,
        topk=topk,
        n_groups=n_groups,
        topk_groups=topk_groups,
        score_func="sigmoid",
        score_bias=score_bias,
        route_scale=route_scale,
    )
    return routing.weights, routing.indices


def _composite_mla_absorbed_attention(
    q_nope: Tensor,
    q_pe: Tensor,
    kv: Tensor,
    pe: Tensor,
    key_up: Tensor,
    value_up: Tensor,
    query_positions: Tensor,
    key_positions: Tensor,
    causal: bool,
    scale: float,
) -> Tensor:
    compute_dtype = torch.float64 if q_nope.dtype == torch.float64 else torch.float32
    q_nope_compute = q_nope.to(compute_dtype)
    q_pe_compute = q_pe.to(compute_dtype)
    kv_compute = kv.to(compute_dtype)
    pe_compute = pe.to(compute_dtype)
    q_latent = torch.einsum("bshd,hdr->bshr", q_nope_compute, key_up.to(compute_dtype))
    scores = (
        torch.einsum("bshr,btr->bhst", q_latent, kv_compute)
        + torch.einsum("bshd,btd->bhst", q_pe_compute, pe_compute)
    ) * scale
    if causal:
        keep = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~keep, -torch.inf)
    row_max = scores.amax(dim=-1, keepdim=True)
    finite_row = torch.isfinite(row_max)
    shifted = torch.where(finite_row, scores - row_max, torch.full_like(scores, -torch.inf))
    numerators = torch.exp(shifted)
    denominator = numerators.sum(dim=-1, keepdim=True)
    probabilities = torch.where(
        denominator > 0,
        numerators / denominator.clamp_min(torch.finfo(compute_dtype).tiny),
        torch.zeros_like(numerators),
    )
    latent_output = torch.einsum("bhst,btr->bshr", probabilities, kv_compute)
    return (
        torch.einsum("bshr,hdr->bshd", latent_output, value_up.to(compute_dtype))
        .to(q_nope.dtype)
        .contiguous()
    )


def _mla_projection_compute_dtype(tensor: Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _composite_mla_rms_norm(x: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    compute_dtype = _mla_projection_compute_dtype(x)
    x_compute = x.to(compute_dtype)
    inverse_rms = torch.rsqrt(x_compute.square().mean(dim=-1, keepdim=True) + epsilon)
    return x_compute * inverse_rms * weight.to(compute_dtype)


def _composite_mla_rope(x: Tensor, positions: Tensor, theta: float) -> Tensor:
    compute_dtype = _mla_projection_compute_dtype(x)
    pair_index = torch.arange(0, x.shape[-1], 2, device=x.device, dtype=compute_dtype)
    inverse_frequency = theta ** (-pair_index / x.shape[-1])
    angles = positions.to(compute_dtype).unsqueeze(-1) * inverse_frequency
    if positions.ndim == 1:
        angles = angles.unsqueeze(0)
    cosine = torch.cos(angles).unsqueeze(2)
    sine = torch.sin(angles).unsqueeze(2)
    x_compute = x.to(compute_dtype)
    even = x_compute[..., 0::2]
    odd = x_compute[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        dim=-1,
    ).flatten(-2)


def _composite_mla_query_projection(
    x: Tensor,
    wq: Tensor,
    positions: Tensor,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    rope_theta: float,
) -> tuple[Tensor, Tensor]:
    compute_dtype = _mla_projection_compute_dtype(x)
    head_dim = qk_nope_head_dim + qk_rope_head_dim
    projected = torch.nn.functional.linear(
        x.to(compute_dtype),
        wq.to(compute_dtype),
    ).to(x.dtype)
    projected = projected.reshape(x.shape[0], x.shape[1], n_heads, head_dim)
    q_nope, q_pe = torch.split(
        projected,
        [qk_nope_head_dim, qk_rope_head_dim],
        dim=-1,
    )
    return (
        q_nope.to(x.dtype).contiguous(),
        _composite_mla_rope(q_pe, positions, rope_theta).to(x.dtype).contiguous(),
    )


def _composite_mla_query_lora_projection(
    x: Tensor,
    wq_a: Tensor,
    q_norm_weight: Tensor,
    wq_b: Tensor,
    positions: Tensor,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    rope_theta: float,
    rms_norm_eps: float,
) -> tuple[Tensor, Tensor]:
    compute_dtype = _mla_projection_compute_dtype(x)
    latent = torch.nn.functional.linear(
        x.to(compute_dtype),
        wq_a.to(compute_dtype),
    ).to(x.dtype)
    latent = _composite_mla_rms_norm(latent, q_norm_weight, rms_norm_eps).to(x.dtype)
    projected = torch.nn.functional.linear(
        latent.to(compute_dtype),
        wq_b.to(compute_dtype),
    ).to(x.dtype)
    head_dim = qk_nope_head_dim + qk_rope_head_dim
    projected = projected.reshape(x.shape[0], x.shape[1], n_heads, head_dim)
    q_nope, q_pe = torch.split(
        projected,
        [qk_nope_head_dim, qk_rope_head_dim],
        dim=-1,
    )
    return (
        q_nope.to(x.dtype).contiguous(),
        _composite_mla_rope(q_pe, positions, rope_theta).to(x.dtype).contiguous(),
    )


def _composite_mla_cache_projection(
    x: Tensor,
    wkv_a: Tensor,
    kv_norm_weight: Tensor,
    positions: Tensor,
    kv_lora_rank: int,
    rope_theta: float,
    rms_norm_eps: float,
) -> tuple[Tensor, Tensor]:
    compute_dtype = _mla_projection_compute_dtype(x)
    projected = torch.nn.functional.linear(
        x.to(compute_dtype),
        wkv_a.to(compute_dtype),
    ).to(x.dtype)
    kv, pe = torch.split(
        projected,
        [kv_lora_rank, projected.shape[-1] - kv_lora_rank],
        dim=-1,
    )
    kv = _composite_mla_rms_norm(kv, kv_norm_weight, rms_norm_eps).to(x.dtype).contiguous()
    pe = (
        _composite_mla_rope(pe.unsqueeze(2), positions, rope_theta)
        .squeeze(2)
        .to(x.dtype)
        .contiguous()
    )
    return kv, pe


def _composite_mla_cache_projection_write(
    x: Tensor,
    wkv_a: Tensor,
    kv_norm_weight: Tensor,
    positions: Tensor,
    kv_storage: Tensor,
    pe_storage: Tensor,
    position_storage: Tensor,
    start: int,
    rope_theta: float,
    rms_norm_eps: float,
) -> None:
    kv, pe = _composite_mla_cache_projection(
        x,
        wkv_a,
        kv_norm_weight,
        positions,
        kv_norm_weight.numel(),
        rope_theta,
        rms_norm_eps,
    )
    end = start + x.shape[1]
    kv_storage[:, start:end].copy_(kv)
    pe_storage[:, start:end].copy_(pe)
    position_storage[start:end].copy_(positions)


def _composite_mla_cache_projection_write_slots(
    x: Tensor,
    wkv_a: Tensor,
    kv_norm_weight: Tensor,
    positions: Tensor,
    slot_mapping: Tensor,
    kv_storage: Tensor,
    pe_storage: Tensor,
    position_storage: Tensor,
    metadata_validated: bool,
    rope_theta: float,
    rms_norm_eps: float,
) -> None:
    del metadata_validated
    token_count = x.shape[0] * x.shape[1]
    flat_x = x.reshape(1, token_count, x.shape[2])
    flat_positions = positions.reshape(token_count)
    kv, pe = _composite_mla_cache_projection(
        flat_x,
        wkv_a,
        kv_norm_weight,
        flat_positions,
        kv_norm_weight.numel(),
        rope_theta,
        rms_norm_eps,
    )
    flat_slots = slot_mapping.reshape(token_count)
    kv_storage.view(-1, kv_storage.shape[-1]).index_copy_(
        0,
        flat_slots,
        kv.reshape(token_count, kv.shape[-1]),
    )
    pe_storage.view(-1, pe_storage.shape[-1]).index_copy_(
        0,
        flat_slots,
        pe.reshape(token_count, pe.shape[-1]),
    )
    position_storage.view(-1).index_copy_(0, flat_slots, flat_positions)


def _composite_mla_paged_absorbed_attention(
    q_nope: Tensor,
    q_pe: Tensor,
    kv_storage: Tensor,
    pe_storage: Tensor,
    position_storage: Tensor,
    block_table: Tensor,
    sequence_lengths: Tensor,
    key_up: Tensor,
    value_up: Tensor,
    query_positions: Tensor,
    metadata_validated: bool,
    causal: bool,
    scale: float,
) -> Tensor:
    del metadata_validated
    batch_size = q_nope.shape[0]
    max_key_length = int(sequence_lengths.max().item()) if batch_size else 0
    value_dim = value_up.shape[1]
    if max_key_length == 0:
        return q_nope.new_zeros((batch_size, q_nope.shape[1], q_nope.shape[2], value_dim))

    page_size = kv_storage.shape[1]
    logical_indices = torch.arange(max_key_length, device=q_nope.device)
    logical_pages = torch.div(logical_indices, page_size, rounding_mode="floor")
    offsets = logical_indices.remainder(page_size)
    valid_keys = logical_indices.unsqueeze(0) < sequence_lengths.unsqueeze(1)
    pages = block_table[:, logical_pages]
    safe_pages = torch.where(valid_keys, pages, torch.zeros_like(pages))
    physical_slots = safe_pages * page_size + offsets.unsqueeze(0)
    kv = kv_storage.view(-1, kv_storage.shape[-1])[physical_slots]
    pe = pe_storage.view(-1, pe_storage.shape[-1])[physical_slots]
    key_positions = position_storage.view(-1)[physical_slots]

    compute_dtype = _mla_projection_compute_dtype(q_nope)
    q_latent = torch.einsum(
        "bshd,hdr->bshr",
        q_nope.to(compute_dtype),
        key_up.to(compute_dtype),
    )
    scores = torch.einsum("bshr,btr->bhst", q_latent, kv.to(compute_dtype))
    scores = scores + torch.einsum(
        "bshd,btd->bhst",
        q_pe.to(compute_dtype),
        pe.to(compute_dtype),
    )
    scores = scores * scale
    keep = valid_keys[:, None, None, :]
    if causal:
        keep = keep & (key_positions[:, None, None, :] <= query_positions[:, None, :, None])
    scores = scores.masked_fill(~keep, -torch.inf)
    probabilities, _ = _stable_probabilities(scores)
    latent_output = torch.einsum("bhst,btr->bshr", probabilities, kv.to(compute_dtype))
    return (
        torch.einsum(
            "bshr,hdr->bshd",
            latent_output,
            value_up.to(compute_dtype),
        )
        .to(q_nope.dtype)
        .contiguous()
    )


def _composite_mla_output_projection(heads: Tensor, wo: Tensor) -> Tensor:
    compute_dtype = _mla_projection_compute_dtype(heads)
    return (
        torch.nn.functional.linear(heads.flatten(2).to(compute_dtype), wo.to(compute_dtype))
        .to(heads.dtype)
        .contiguous()
    )


def _tensorized_swiglu_experts_reference(
    activations: Tensor,
    expert_offsets: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
) -> Tensor:
    """Traceable active-row expert reference used at the dispatcher boundary."""

    rows = activations.shape[0]
    experts = expert_w1.shape[0]
    counts = expert_offsets[1:] - expert_offsets[:-1]
    row_experts = torch.repeat_interleave(
        torch.arange(experts, device=expert_offsets.device),
        counts,
        output_size=rows,
    )
    compute_dtype = torch.float64 if activations.dtype == torch.float64 else torch.float32
    activation_compute = activations.to(compute_dtype)
    row_w1 = expert_w1.to(compute_dtype).index_select(0, row_experts)
    row_w2 = expert_w2.to(compute_dtype).index_select(0, row_experts)
    row_w3 = expert_w3.to(compute_dtype).index_select(0, row_experts)
    gate = torch.bmm(row_w1, activation_compute.unsqueeze(-1)).squeeze(-1)
    up = torch.bmm(row_w3, activation_compute.unsqueeze(-1)).squeeze(-1)
    hidden = torch.nn.functional.silu(gate) * up
    if activations.dtype == torch.float16:
        hidden = hidden.to(activations.dtype).to(compute_dtype)
    return torch.bmm(row_w2, hidden.unsqueeze(-1)).squeeze(-1).to(activations.dtype)


def _fake_route_pack(
    x: Tensor,
    route_weights: Tensor,
    expert_indices: Tensor,
    expert_owner: Tensor,
    world_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    torch._check(x.ndim == 2)
    torch._check(route_weights.ndim == 2 and expert_indices.ndim == 2)
    torch._check(route_weights.shape == expert_indices.shape)
    torch._check(x.shape[0] == route_weights.shape[0])
    torch._check(expert_owner.ndim == 1)
    torch._check(world_size > 0)
    route_count = route_weights.numel()
    long_options = {"dtype": torch.long, "device": x.device}
    return (
        x.new_empty((route_count, x.shape[1])),
        route_weights.new_empty(route_count),
        torch.empty(route_count, **long_options),
        torch.empty(route_count, **long_options),
        torch.empty(expert_owner.numel(), **long_options),
        torch.empty(world_size, **long_options),
    )


def _fake_route_combine(
    contributions: Tensor,
    route_weights: Tensor,
    token_indices: Tensor,
    token_count: int,
) -> Tensor:
    torch._check(contributions.ndim == 2)
    torch._check(route_weights.ndim == 1 and token_indices.ndim == 1)
    torch._check(contributions.shape[0] == route_weights.numel())
    torch._check(contributions.shape[0] == token_indices.numel())
    torch._check(token_count >= 0)
    return contributions.new_empty((token_count, contributions.shape[1]))


def _fake_tiled_gemm(
    a: Tensor,
    b: Tensor,
    c: Tensor | None,
    alpha: float,
    beta: float,
) -> Tensor:
    torch._check(a.ndim == 2 and b.ndim == 2)
    torch._check(a.shape[1] == b.shape[0])
    torch._check(a.device == b.device and a.dtype == b.dtype)
    torch._check(a.is_floating_point() and b.is_floating_point())
    torch._check(math.isfinite(alpha) and math.isfinite(beta))
    if c is None:
        torch._check(beta == 0.0)
    else:
        torch._check(c.shape == (a.shape[0], b.shape[1]))
        torch._check(c.device == a.device and c.dtype == a.dtype)
    return a.new_empty((a.shape[0], b.shape[1]))


def _fake_swiglu_experts(
    activations: Tensor,
    expert_offsets: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
) -> Tensor:
    torch._check(activations.ndim == 2)
    torch._check(expert_offsets.ndim == 1 and expert_offsets.dtype == torch.long)
    torch._check(expert_w1.ndim == 3 and expert_w2.ndim == 3 and expert_w3.ndim == 3)
    torch._check(expert_w1.shape == expert_w3.shape)
    torch._check(expert_w2.shape[0] == expert_w1.shape[0])
    torch._check(expert_w2.shape[1] == expert_w1.shape[2])
    torch._check(expert_w2.shape[2] == expert_w1.shape[1])
    torch._check(expert_offsets.shape[0] == expert_w1.shape[0] + 1)
    torch._check(activations.shape[1] == expert_w1.shape[2])
    torch._check(
        activations.device
        == expert_offsets.device
        == expert_w1.device
        == expert_w2.device
        == expert_w3.device
    )
    torch._check(activations.dtype == expert_w1.dtype == expert_w2.dtype == expert_w3.dtype)
    return activations.new_empty(activations.shape)


def _fake_expert_major_pack(
    activations: Tensor,
    expert_indices: Tensor,
    local_expert_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    torch._check(activations.ndim == 2)
    torch._check(expert_indices.ndim == 1 and local_expert_ids.ndim == 1)
    torch._check(expert_indices.dtype == torch.long and local_expert_ids.dtype == torch.long)
    torch._check(activations.shape[0] == expert_indices.shape[0])
    torch._check(activations.device == expert_indices.device == local_expert_ids.device)
    long_options = {"dtype": torch.long, "device": activations.device}
    return (
        activations.new_empty(activations.shape),
        torch.empty(local_expert_ids.shape[0] + 1, **long_options),
        torch.empty(expert_indices.shape, **long_options),
    )


def _fake_grouped_topk(
    x: Tensor,
    gate_weight: Tensor,
    topk: int,
    n_groups: int,
    topk_groups: int,
    score_bias: Tensor | None,
    route_scale: float,
) -> tuple[Tensor, Tensor]:
    torch._check(x.ndim == 2 and gate_weight.ndim == 2)
    torch._check(x.shape[1] == gate_weight.shape[1])
    torch._check(x.device == gate_weight.device and x.dtype == gate_weight.dtype)
    torch._check(x.is_floating_point() and gate_weight.is_floating_point())
    torch._check(n_groups > 0 and gate_weight.shape[0] % n_groups == 0)
    torch._check(1 <= topk_groups <= n_groups)
    torch._check(1 <= topk <= gate_weight.shape[0])
    torch._check(topk <= topk_groups * (gate_weight.shape[0] // n_groups))
    torch._check(math.isfinite(route_scale))
    if score_bias is not None:
        torch._check(score_bias.shape == (gate_weight.shape[0],))
        torch._check(score_bias.device == x.device and score_bias.dtype == x.dtype)
    return (
        x.new_empty((x.shape[0], topk)),
        torch.empty((x.shape[0], topk), dtype=torch.long, device=x.device),
    )


def _fake_deepseek_moe_forward(
    x: Tensor,
    gate_weight: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
    topk: int,
    n_groups: int,
    topk_groups: int,
    score_bias: Tensor | None,
    route_scale: float,
) -> Tensor:
    tensors = (x, gate_weight, expert_w1, expert_w2, expert_w3)
    floating_tensors = tensors if score_bias is None else (*tensors, score_bias)
    torch._check(
        not any(tensor.requires_grad for tensor in floating_tensors),
        lambda: (
            "the DeepSeek MoE forward operator is forward-only and does not accept "
            "requires_grad tensors"
        ),
    )
    torch._check(x.ndim == 2 and gate_weight.ndim == 2)
    torch._check(expert_w1.ndim == 3 and expert_w2.ndim == 3 and expert_w3.ndim == 3)

    experts, model_dim = gate_weight.shape
    hidden = expert_w1.shape[1]
    torch._check(experts > 0 and hidden > 0 and model_dim > 0)
    torch._check(x.shape[1] == model_dim)
    torch._check(expert_w1.shape == (experts, hidden, model_dim))
    torch._check(expert_w2.shape == (experts, model_dim, hidden))
    torch._check(expert_w3.shape == (experts, hidden, model_dim))

    torch._check(all(tensor.is_floating_point() for tensor in floating_tensors))
    torch._check(all(tensor.dtype == x.dtype for tensor in floating_tensors))
    torch._check(all(tensor.device == x.device for tensor in floating_tensors))
    torch._check(all(tensor.is_contiguous() for tensor in floating_tensors))
    if score_bias is not None:
        torch._check(score_bias.shape == (experts,))

    torch._check(n_groups > 0 and experts % n_groups == 0)
    torch._check(1 <= topk_groups <= n_groups)
    torch._check(1 <= topk <= experts)
    torch._check(topk <= topk_groups * (experts // n_groups))
    torch._check(math.isfinite(route_scale))
    return x.new_empty(x.shape)


def _fake_mla_absorbed_attention(
    q_nope: Tensor,
    q_pe: Tensor,
    kv: Tensor,
    pe: Tensor,
    key_up: Tensor,
    value_up: Tensor,
    query_positions: Tensor,
    key_positions: Tensor,
    causal: bool,
    scale: float,
) -> Tensor:
    del causal
    torch._check(q_nope.ndim == 4 and q_pe.ndim == 4)
    torch._check(kv.ndim == 3 and pe.ndim == 3)
    torch._check(key_up.ndim == 3 and value_up.ndim == 3)
    torch._check(query_positions.ndim == 1 and key_positions.ndim == 1)
    torch._check(q_nope.shape[:3] == q_pe.shape[:3])
    torch._check(kv.shape[:2] == pe.shape[:2])
    torch._check(q_nope.shape[0] == kv.shape[0])
    torch._check(q_nope.shape[2] == key_up.shape[0])
    torch._check(key_up.shape[0] == value_up.shape[0])
    torch._check(q_nope.shape[3] == key_up.shape[1])
    torch._check(q_pe.shape[3] == pe.shape[2])
    torch._check(kv.shape[2] == key_up.shape[2])
    torch._check(key_up.shape[2] == value_up.shape[2])
    torch._check(query_positions.shape[0] == q_nope.shape[1])
    torch._check(key_positions.shape[0] == kv.shape[1])
    torch._check(
        q_nope.device
        == q_pe.device
        == kv.device
        == pe.device
        == key_up.device
        == value_up.device
        == query_positions.device
        == key_positions.device
    )
    for tensor in (q_nope, q_pe, kv, pe, key_up, value_up):
        torch._check(tensor.is_floating_point())
    torch._check(query_positions.dtype == torch.long and key_positions.dtype == torch.long)
    torch._check(math.isfinite(scale))
    return q_nope.new_empty((q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], value_up.shape[1]))


def _fake_mla_query_outputs(
    x: Tensor,
    positions: Tensor,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    rope_theta: float,
) -> tuple[Tensor, Tensor]:
    torch._check(x.ndim == 3 and positions.ndim in {1, 2})
    if positions.ndim == 1:
        torch._check(positions.shape[0] == x.shape[1])
    else:
        torch._check(positions.shape[0] == x.shape[0] and positions.shape[1] == x.shape[1])
    torch._check(x.device == positions.device)
    torch._check(x.is_floating_point() and positions.dtype == torch.long)
    torch._check(n_heads > 0 and qk_nope_head_dim > 0)
    torch._check(qk_rope_head_dim > 0 and qk_rope_head_dim % 2 == 0)
    torch._check(math.isfinite(rope_theta) and rope_theta > 0)
    return (
        x.new_empty((x.shape[0], x.shape[1], n_heads, qk_nope_head_dim)),
        x.new_empty((x.shape[0], x.shape[1], n_heads, qk_rope_head_dim)),
    )


def _fake_mla_query_projection(
    x: Tensor,
    wq: Tensor,
    positions: Tensor,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    rope_theta: float,
) -> tuple[Tensor, Tensor]:
    torch._check(wq.ndim == 2)
    torch._check(wq.shape[1] == x.shape[2])
    torch._check(wq.shape[0] == n_heads * (qk_nope_head_dim + qk_rope_head_dim))
    torch._check(wq.device == x.device and wq.dtype == x.dtype)
    return _fake_mla_query_outputs(
        x,
        positions,
        n_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        rope_theta,
    )


def _fake_mla_query_lora_projection(
    x: Tensor,
    wq_a: Tensor,
    q_norm_weight: Tensor,
    wq_b: Tensor,
    positions: Tensor,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    rope_theta: float,
    rms_norm_eps: float,
) -> tuple[Tensor, Tensor]:
    torch._check(wq_a.ndim == 2 and q_norm_weight.ndim == 1 and wq_b.ndim == 2)
    torch._check(wq_a.shape[1] == x.shape[2])
    torch._check(wq_a.shape[0] == q_norm_weight.shape[0] == wq_b.shape[1])
    torch._check(wq_b.shape[0] == n_heads * (qk_nope_head_dim + qk_rope_head_dim))
    torch._check(
        wq_a.device == q_norm_weight.device == wq_b.device == x.device
        and wq_a.dtype == q_norm_weight.dtype == wq_b.dtype == x.dtype
    )
    torch._check(math.isfinite(rms_norm_eps) and rms_norm_eps > 0)
    return _fake_mla_query_outputs(
        x,
        positions,
        n_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        rope_theta,
    )


def _fake_mla_cache_projection(
    x: Tensor,
    wkv_a: Tensor,
    kv_norm_weight: Tensor,
    positions: Tensor,
    kv_lora_rank: int,
    rope_theta: float,
    rms_norm_eps: float,
) -> tuple[Tensor, Tensor]:
    torch._check(x.ndim == 3 and wkv_a.ndim == 2 and kv_norm_weight.ndim == 1)
    torch._check(positions.ndim == 1 and positions.shape[0] == x.shape[1])
    torch._check(wkv_a.shape[1] == x.shape[2])
    torch._check(kv_norm_weight.shape[0] == kv_lora_rank)
    rope_dim = wkv_a.shape[0] - kv_lora_rank
    torch._check(kv_lora_rank > 0 and rope_dim > 0 and rope_dim % 2 == 0)
    torch._check(
        x.device == wkv_a.device == kv_norm_weight.device == positions.device
        and x.dtype == wkv_a.dtype == kv_norm_weight.dtype
    )
    torch._check(positions.dtype == torch.long)
    torch._check(math.isfinite(rope_theta) and rope_theta > 0)
    torch._check(math.isfinite(rms_norm_eps) and rms_norm_eps > 0)
    return (
        x.new_empty((x.shape[0], x.shape[1], kv_lora_rank)),
        x.new_empty((x.shape[0], x.shape[1], rope_dim)),
    )


def _fake_mla_cache_projection_write(
    x: Tensor,
    wkv_a: Tensor,
    kv_norm_weight: Tensor,
    positions: Tensor,
    kv_storage: Tensor,
    pe_storage: Tensor,
    position_storage: Tensor,
    start: int,
    rope_theta: float,
    rms_norm_eps: float,
) -> None:
    kv, pe = _fake_mla_cache_projection(
        x,
        wkv_a,
        kv_norm_weight,
        positions,
        kv_norm_weight.shape[0],
        rope_theta,
        rms_norm_eps,
    )
    torch._check(kv_storage.ndim == 3 and pe_storage.ndim == 3)
    torch._check(position_storage.ndim == 1 and position_storage.dtype == torch.long)
    torch._check(kv_storage.shape[0] == pe_storage.shape[0] == x.shape[0])
    torch._check(kv_storage.shape[1] == pe_storage.shape[1] == position_storage.shape[0])
    torch._check(kv_storage.shape[2] == kv.shape[2] and pe_storage.shape[2] == pe.shape[2])
    torch._check(
        kv_storage.device == pe_storage.device == position_storage.device == x.device
        and kv_storage.dtype == pe_storage.dtype == x.dtype
    )
    torch._check(start >= 0 and start + x.shape[1] <= kv_storage.shape[1])


def _fake_mla_cache_projection_write_slots(
    x: Tensor,
    wkv_a: Tensor,
    kv_norm_weight: Tensor,
    positions: Tensor,
    slot_mapping: Tensor,
    kv_storage: Tensor,
    pe_storage: Tensor,
    position_storage: Tensor,
    metadata_validated: bool,
    rope_theta: float,
    rms_norm_eps: float,
) -> None:
    del metadata_validated
    torch._check(x.ndim == 3 and positions.ndim == 2 and slot_mapping.ndim == 2)
    torch._check(positions.shape == x.shape[:2] and slot_mapping.shape == x.shape[:2])
    token_count = x.shape[0] * x.shape[1]
    flat_x = x.reshape(1, token_count, x.shape[2])
    kv, pe = _fake_mla_cache_projection(
        flat_x,
        wkv_a,
        kv_norm_weight,
        positions.reshape(token_count),
        kv_norm_weight.shape[0],
        rope_theta,
        rms_norm_eps,
    )
    torch._check(kv_storage.ndim == 3 and pe_storage.ndim == 3)
    torch._check(position_storage.ndim == 2 and position_storage.dtype == torch.long)
    torch._check(kv_storage.shape[:2] == pe_storage.shape[:2])
    torch._check(kv_storage.shape[:2] == position_storage.shape)
    torch._check(kv_storage.shape[2] == kv.shape[2] and pe_storage.shape[2] == pe.shape[2])
    torch._check(slot_mapping.dtype == torch.long)
    torch._check(
        kv_storage.device
        == pe_storage.device
        == position_storage.device
        == positions.device
        == slot_mapping.device
        == x.device
        and kv_storage.dtype == pe_storage.dtype == x.dtype
    )


def _fake_mla_paged_absorbed_attention(
    q_nope: Tensor,
    q_pe: Tensor,
    kv_storage: Tensor,
    pe_storage: Tensor,
    position_storage: Tensor,
    block_table: Tensor,
    sequence_lengths: Tensor,
    key_up: Tensor,
    value_up: Tensor,
    query_positions: Tensor,
    metadata_validated: bool,
    causal: bool,
    scale: float,
) -> Tensor:
    del metadata_validated, causal
    torch._check(q_nope.ndim == 4 and q_pe.ndim == 4)
    torch._check(kv_storage.ndim == 3 and pe_storage.ndim == 3)
    torch._check(position_storage.ndim == 2 and block_table.ndim == 2)
    torch._check(sequence_lengths.ndim == 1 and query_positions.ndim == 2)
    torch._check(q_nope.shape[:3] == q_pe.shape[:3])
    torch._check(kv_storage.shape[:2] == pe_storage.shape[:2])
    torch._check(kv_storage.shape[:2] == position_storage.shape)
    torch._check(block_table.shape[0] == sequence_lengths.shape[0] == q_nope.shape[0])
    torch._check(query_positions.shape[:2] == q_nope.shape[:2])
    torch._check(q_nope.shape[2] == key_up.shape[0] == value_up.shape[0])
    torch._check(q_nope.shape[3] == key_up.shape[1])
    torch._check(q_pe.shape[3] == pe_storage.shape[2])
    torch._check(kv_storage.shape[2] == key_up.shape[2] == value_up.shape[2])
    torch._check(
        position_storage.dtype
        == block_table.dtype
        == sequence_lengths.dtype
        == query_positions.dtype
        == torch.long
    )
    torch._check(
        q_nope.device
        == q_pe.device
        == kv_storage.device
        == pe_storage.device
        == position_storage.device
        == block_table.device
        == sequence_lengths.device
        == key_up.device
        == value_up.device
        == query_positions.device
    )
    torch._check(
        q_nope.dtype
        == q_pe.dtype
        == kv_storage.dtype
        == pe_storage.dtype
        == key_up.dtype
        == value_up.dtype
    )
    torch._check(math.isfinite(scale))
    return q_nope.new_empty((q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], value_up.shape[1]))


def _fake_mla_output_projection(heads: Tensor, wo: Tensor) -> Tensor:
    torch._check(heads.ndim == 4 and wo.ndim == 2)
    torch._check(wo.shape[1] == heads.shape[2] * heads.shape[3])
    torch._check(heads.device == wo.device and heads.dtype == wo.dtype)
    torch._check(heads.is_floating_point() and wo.is_floating_point())
    return heads.new_empty((heads.shape[0], heads.shape[1], wo.shape[0]))


composite_explicit = torch.library.Library("ds_flash_mla_moe", "IMPL", "CompositeExplicitAutograd")
composite_explicit.impl("attention_forward", _composite_attention_forward)
composite_explicit.impl("route_pack", _composite_route_pack)
composite_explicit.impl("route_combine", _composite_route_combine)
composite_explicit.impl("tiled_gemm", _composite_tiled_gemm)
composite_explicit.impl("swiglu_experts", _composite_swiglu_experts)
composite_explicit.impl("expert_major_pack", _composite_expert_major_pack)
composite_explicit.impl("grouped_topk", _composite_grouped_topk)
composite_explicit.impl("mla_absorbed_attention", _composite_mla_absorbed_attention)
composite_explicit.impl("mla_query_projection", _composite_mla_query_projection)
composite_explicit.impl("mla_query_lora_projection", _composite_mla_query_lora_projection)
composite_explicit.impl("mla_cache_projection", _composite_mla_cache_projection)
composite_explicit.impl("mla_cache_projection_write", _composite_mla_cache_projection_write)
composite_explicit.impl(
    "mla_cache_projection_write_slots", _composite_mla_cache_projection_write_slots
)
composite_explicit.impl("mla_paged_absorbed_attention", _composite_mla_paged_absorbed_attention)
composite_explicit.impl("mla_output_projection", _composite_mla_output_projection)
_LIBRARY_HANDLES.append(composite_explicit)

composite_implicit = torch.library.Library("ds_flash_mla_moe", "IMPL", "CompositeImplicitAutograd")
composite_implicit.impl("attention_backward", _composite_attention_backward)
_LIBRARY_HANDLES.append(composite_implicit)

torch.library.register_fake("ds_flash_mla_moe::attention_forward", _fake_attention_forward)
torch.library.register_fake(
    "ds_flash_mla_moe::attention_fa1_forward", _fake_formal_attention_forward
)
torch.library.register_fake(
    "ds_flash_mla_moe::attention_fa2_forward", _fake_formal_attention_forward
)
torch.library.register_fake("ds_flash_mla_moe::route_pack", _fake_route_pack)
torch.library.register_fake("ds_flash_mla_moe::route_combine", _fake_route_combine)
torch.library.register_fake("ds_flash_mla_moe::tiled_gemm", _fake_tiled_gemm)
torch.library.register_fake("ds_flash_mla_moe::swiglu_experts", _fake_swiglu_experts)
torch.library.register_fake("ds_flash_mla_moe::expert_major_pack", _fake_expert_major_pack)
torch.library.register_fake("ds_flash_mla_moe::grouped_topk", _fake_grouped_topk)
torch.library.register_fake("ds_flash_mla_moe::deepseek_moe_forward", _fake_deepseek_moe_forward)
torch.library.register_fake(
    "ds_flash_mla_moe::mla_absorbed_attention", _fake_mla_absorbed_attention
)
torch.library.register_fake("ds_flash_mla_moe::mla_query_projection", _fake_mla_query_projection)
torch.library.register_fake(
    "ds_flash_mla_moe::mla_query_lora_projection", _fake_mla_query_lora_projection
)
torch.library.register_fake("ds_flash_mla_moe::mla_cache_projection", _fake_mla_cache_projection)
torch.library.register_fake(
    "ds_flash_mla_moe::mla_cache_projection_write", _fake_mla_cache_projection_write
)
torch.library.register_fake(
    "ds_flash_mla_moe::mla_cache_projection_write_slots",
    _fake_mla_cache_projection_write_slots,
)
torch.library.register_fake(
    "ds_flash_mla_moe::mla_paged_absorbed_attention",
    _fake_mla_paged_absorbed_attention,
)
torch.library.register_fake("ds_flash_mla_moe::mla_output_projection", _fake_mla_output_projection)


def _attention_setup_context(ctx, inputs, output) -> None:
    q, k, v, causal, scale = inputs
    ctx.save_for_backward(q, k, v)
    ctx.causal = causal
    ctx.scale = scale


def _attention_backward(context, grad_output: Tensor):
    q, k, v = context.saved_tensors
    use_native = (
        _NATIVE_EXTENSION_LOADED
        and _operator_has_cuda_kernel("attention_backward")
        and q.device.type == "cuda"
        and q.ndim == 4
        and q.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and grad_output.dtype == q.dtype
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and not torch.is_grad_enabled()
        and not torch.are_deterministic_algorithms_enabled()
    )
    if use_native:
        return (
            *torch.ops.ds_flash_mla_moe.attention_backward.default(
                grad_output, q, k, v, context.causal, context.scale
            ),
            None,
            None,
        )
    with torch.enable_grad():
        gradients = _composite_attention_backward(
            grad_output, q, k, v, context.causal, context.scale
        )
    return (*gradients, None, None)


torch.library.register_autograd(
    "ds_flash_mla_moe::attention_forward",
    _attention_backward,
    setup_context=_attention_setup_context,
)


def _route_pack_setup_context(ctx, inputs, output) -> None:
    x, route_weights, _expert_indices, _expert_owner, _world_size = inputs
    route_indices = output[2]
    ctx.mark_non_differentiable(*output[2:])
    ctx.save_for_backward(x, route_weights, route_indices)


def _route_pack_backward(context, *grad_outputs: Tensor | None):
    x, route_weights, route_indices = context.saved_tensors
    grad_activations, grad_packed_weights = grad_outputs[:2]
    topk = route_weights.shape[1]
    flat_grad_x = torch.zeros_like(x)
    if grad_activations is not None:
        route_major_grad = grad_activations.new_zeros(
            (route_indices.numel(), grad_activations.shape[-1])
        ).index_copy(0, route_indices, grad_activations)
        flat_grad_x = route_major_grad.reshape(x.shape[0], topk, x.shape[1]).sum(dim=1)
    grad_route_weights = torch.zeros_like(route_weights).reshape(-1)
    if grad_packed_weights is not None:
        grad_route_weights = grad_route_weights.index_copy(
            0,
            route_indices,
            grad_packed_weights,
        )
    return flat_grad_x, grad_route_weights.reshape_as(route_weights), None, None, None


torch.library.register_autograd(
    "ds_flash_mla_moe::route_pack",
    _route_pack_backward,
    setup_context=_route_pack_setup_context,
)


def _route_combine_setup_context(ctx, inputs, output) -> None:
    contributions, route_weights, token_indices, _token_count = inputs
    ctx.save_for_backward(contributions, route_weights, token_indices)


def _route_combine_backward(context, grad_output: Tensor):
    contributions, route_weights, token_indices = context.saved_tensors
    selected_grad = grad_output.index_select(0, token_indices)
    compute_dtype = torch.float64 if contributions.dtype == torch.float64 else torch.float32
    selected_compute = selected_grad.to(compute_dtype)
    contribution_compute = contributions.to(compute_dtype)
    weight_compute = route_weights.to(compute_dtype)
    grad_contributions = (selected_compute * weight_compute.unsqueeze(-1)).to(contributions.dtype)
    grad_route_weights = (
        (selected_compute * contribution_compute).sum(dim=-1).to(route_weights.dtype)
    )
    return grad_contributions, grad_route_weights, None, None


torch.library.register_autograd(
    "ds_flash_mla_moe::route_combine",
    _route_combine_backward,
    setup_context=_route_combine_setup_context,
)


def _tiled_gemm_setup_context(ctx, inputs, output) -> None:
    a, b, c, alpha, beta = inputs
    ctx.has_c = c is not None
    ctx.alpha = alpha
    ctx.beta = beta
    ctx.save_for_backward(a, b)


def _tiled_gemm_backward(context, grad_output: Tensor):
    a, b = context.saved_tensors[:2]
    compute_dtype = _gemm_compute_dtype(a.dtype)
    grad_compute = grad_output.to(compute_dtype)
    a_compute = a.to(compute_dtype)
    b_compute = b.to(compute_dtype)
    grad_a = (context.alpha * (grad_compute @ b_compute.mT)).to(a.dtype)
    grad_b = (context.alpha * (a_compute.mT @ grad_compute)).to(b.dtype)
    grad_c = (context.beta * grad_output) if context.has_c else None
    return grad_a, grad_b, grad_c, None, None


torch.library.register_autograd(
    "ds_flash_mla_moe::tiled_gemm",
    _tiled_gemm_backward,
    setup_context=_tiled_gemm_setup_context,
)


def _swiglu_experts_setup_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _swiglu_experts_backward(context, grad_output: Tensor):
    activations, expert_offsets, expert_w1, expert_w2, expert_w3 = context.saved_tensors
    differentiable = (activations, expert_w1, expert_w2, expert_w3)
    input_positions = (0, 2, 3, 4)
    requested = [
        tensor
        for tensor, position in zip(differentiable, input_positions, strict=True)
        if context.needs_input_grad[position]
    ]
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        reference = _tensorized_swiglu_experts_reference(
            activations,
            expert_offsets,
            expert_w1,
            expert_w2,
            expert_w3,
        )
        requested_gradients = torch.autograd.grad(
            reference,
            requested,
            grad_output,
            create_graph=create_graph,
            allow_unused=True,
        )
    gradients: list[Tensor | None] = []
    requested_iterator = iter(requested_gradients)
    for position in input_positions:
        gradients.append(next(requested_iterator) if context.needs_input_grad[position] else None)
    return gradients[0], None, gradients[1], gradients[2], gradients[3]


torch.library.register_autograd(
    "ds_flash_mla_moe::swiglu_experts",
    _swiglu_experts_backward,
    setup_context=_swiglu_experts_setup_context,
)


def _expert_major_pack_setup_context(ctx, inputs, output) -> None:
    del inputs
    ctx.mark_non_differentiable(output[1], output[2])
    ctx.save_for_backward(output[2])


def _expert_major_pack_backward(context, *grad_outputs: Tensor | None):
    (inverse_permutation,) = context.saved_tensors
    grad_activations = grad_outputs[0]
    if grad_activations is not None:
        grad_activations = grad_activations.index_select(0, inverse_permutation)
    return grad_activations, None, None


torch.library.register_autograd(
    "ds_flash_mla_moe::expert_major_pack",
    _expert_major_pack_backward,
    setup_context=_expert_major_pack_setup_context,
)


def _grouped_topk_setup_context(ctx, inputs, output) -> None:
    x, gate_weight, _topk, _n_groups, _topk_groups, score_bias, route_scale = inputs
    ctx.mark_non_differentiable(output[1])
    ctx.save_for_backward(x, gate_weight, output[1])
    del score_bias
    ctx.route_scale = route_scale


def _grouped_topk_backward(context, grad_weights: Tensor | None, _grad_indices: None):
    x, gate_weight, indices = context.saved_tensors
    if grad_weights is None:
        return None, None, None, None, None, None, None
    requested = []
    requested_positions = []
    for tensor, position in ((x, 0), (gate_weight, 1)):
        if context.needs_input_grad[position]:
            requested.append(tensor)
            requested_positions.append(position)
    if not requested:
        return None, None, None, None, None, None, None
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        scores = torch.sigmoid(x.to(compute_dtype) @ gate_weight.to(compute_dtype).transpose(0, 1))
        selected = scores.gather(1, indices)
        weights = selected / selected.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(compute_dtype).tiny
        )
        weights = (weights * context.route_scale).to(x.dtype)
        requested_gradients = torch.autograd.grad(
            weights,
            requested,
            grad_weights,
            create_graph=create_graph,
            allow_unused=True,
        )
    gradients: list[Tensor | None] = [None, None]
    for position, gradient in zip(requested_positions, requested_gradients, strict=True):
        gradients[position] = gradient
    return gradients[0], gradients[1], None, None, None, None, None


torch.library.register_autograd(
    "ds_flash_mla_moe::grouped_topk",
    _grouped_topk_backward,
    setup_context=_grouped_topk_setup_context,
)


def _mla_absorbed_attention_setup_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs[:8])
    ctx.causal = inputs[8]
    ctx.scale = inputs[9]


def _mla_absorbed_attention_backward(context, grad_output: Tensor):
    tensors = context.saved_tensors
    requested = [
        tensor for index, tensor in enumerate(tensors[:6]) if context.needs_input_grad[index]
    ]
    requested_positions = [index for index in range(6) if context.needs_input_grad[index]]
    if not requested:
        return (None,) * 10
    with torch.enable_grad():
        reference = _composite_mla_absorbed_attention(
            *tensors,
            context.causal,
            context.scale,
        )
        requested_gradients = torch.autograd.grad(
            reference,
            requested,
            grad_output,
            create_graph=torch.is_grad_enabled(),
            allow_unused=True,
        )
    gradients: list[Tensor | None] = [None] * 10
    for position, gradient in zip(requested_positions, requested_gradients, strict=True):
        gradients[position] = gradient
    return tuple(gradients)


torch.library.register_autograd(
    "ds_flash_mla_moe::mla_absorbed_attention",
    _mla_absorbed_attention_backward,
    setup_context=_mla_absorbed_attention_setup_context,
)


def _mla_query_projection_setup_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs[:3])
    ctx.n_heads, ctx.nope_dim, ctx.rope_dim, ctx.theta = inputs[3:]


def _mla_query_projection_backward(context, grad_q_nope: Tensor | None, grad_q_pe: Tensor | None):
    x, wq, positions = context.saved_tensors
    requested = [tensor for index, tensor in enumerate((x, wq)) if context.needs_input_grad[index]]
    requested_positions = [index for index in range(2) if context.needs_input_grad[index]]
    if not requested:
        return (None,) * 7
    with torch.enable_grad():
        q_nope, q_pe = _composite_mla_query_projection(
            x,
            wq,
            positions,
            context.n_heads,
            context.nope_dim,
            context.rope_dim,
            context.theta,
        )
        requested_gradients = torch.autograd.grad(
            (q_nope, q_pe),
            requested,
            (
                torch.zeros_like(q_nope) if grad_q_nope is None else grad_q_nope,
                torch.zeros_like(q_pe) if grad_q_pe is None else grad_q_pe,
            ),
            create_graph=torch.is_grad_enabled(),
            allow_unused=True,
        )
    gradients: list[Tensor | None] = [None] * 7
    for position, gradient in zip(requested_positions, requested_gradients, strict=True):
        gradients[position] = gradient
    return tuple(gradients)


torch.library.register_autograd(
    "ds_flash_mla_moe::mla_query_projection",
    _mla_query_projection_backward,
    setup_context=_mla_query_projection_setup_context,
)


def _mla_query_lora_projection_setup_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs[:5])
    (
        ctx.n_heads,
        ctx.nope_dim,
        ctx.rope_dim,
        ctx.theta,
        ctx.epsilon,
    ) = inputs[5:]


def _mla_query_lora_projection_backward(
    context,
    grad_q_nope: Tensor | None,
    grad_q_pe: Tensor | None,
):
    x, wq_a, q_norm_weight, wq_b, positions = context.saved_tensors
    differentiable = (x, wq_a, q_norm_weight, wq_b)
    requested = [
        tensor for index, tensor in enumerate(differentiable) if context.needs_input_grad[index]
    ]
    requested_positions = [index for index in range(4) if context.needs_input_grad[index]]
    if not requested:
        return (None,) * 10
    with torch.enable_grad():
        q_nope, q_pe = _composite_mla_query_lora_projection(
            x,
            wq_a,
            q_norm_weight,
            wq_b,
            positions,
            context.n_heads,
            context.nope_dim,
            context.rope_dim,
            context.theta,
            context.epsilon,
        )
        requested_gradients = torch.autograd.grad(
            (q_nope, q_pe),
            requested,
            (
                torch.zeros_like(q_nope) if grad_q_nope is None else grad_q_nope,
                torch.zeros_like(q_pe) if grad_q_pe is None else grad_q_pe,
            ),
            create_graph=torch.is_grad_enabled(),
            allow_unused=True,
        )
    gradients: list[Tensor | None] = [None] * 10
    for position, gradient in zip(requested_positions, requested_gradients, strict=True):
        gradients[position] = gradient
    return tuple(gradients)


torch.library.register_autograd(
    "ds_flash_mla_moe::mla_query_lora_projection",
    _mla_query_lora_projection_backward,
    setup_context=_mla_query_lora_projection_setup_context,
)


def _mla_cache_projection_setup_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs[:4])
    ctx.latent_dim, ctx.theta, ctx.epsilon = inputs[4:]


def _mla_cache_projection_backward(context, grad_kv: Tensor | None, grad_pe: Tensor | None):
    x, wkv_a, kv_norm_weight, positions = context.saved_tensors
    differentiable = (x, wkv_a, kv_norm_weight)
    requested = [
        tensor for index, tensor in enumerate(differentiable) if context.needs_input_grad[index]
    ]
    requested_positions = [index for index in range(3) if context.needs_input_grad[index]]
    if not requested:
        return (None,) * 7
    with torch.enable_grad():
        kv, pe = _composite_mla_cache_projection(
            x,
            wkv_a,
            kv_norm_weight,
            positions,
            context.latent_dim,
            context.theta,
            context.epsilon,
        )
        requested_gradients = torch.autograd.grad(
            (kv, pe),
            requested,
            (
                torch.zeros_like(kv) if grad_kv is None else grad_kv,
                torch.zeros_like(pe) if grad_pe is None else grad_pe,
            ),
            create_graph=torch.is_grad_enabled(),
            allow_unused=True,
        )
    gradients: list[Tensor | None] = [None] * 7
    for position, gradient in zip(requested_positions, requested_gradients, strict=True):
        gradients[position] = gradient
    return tuple(gradients)


torch.library.register_autograd(
    "ds_flash_mla_moe::mla_cache_projection",
    _mla_cache_projection_backward,
    setup_context=_mla_cache_projection_setup_context,
)


def _mla_output_projection_setup_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _mla_output_projection_backward(context, grad_output: Tensor):
    heads, wo = context.saved_tensors
    requested = [
        tensor for index, tensor in enumerate((heads, wo)) if context.needs_input_grad[index]
    ]
    requested_positions = [index for index in range(2) if context.needs_input_grad[index]]
    if not requested:
        return None, None
    with torch.enable_grad():
        reference = _composite_mla_output_projection(heads, wo)
        requested_gradients = torch.autograd.grad(
            reference,
            requested,
            grad_output,
            create_graph=torch.is_grad_enabled(),
            allow_unused=True,
        )
    gradients: list[Tensor | None] = [None, None]
    for position, gradient in zip(requested_positions, requested_gradients, strict=True):
        gradients[position] = gradient
    return tuple(gradients)


torch.library.register_autograd(
    "ds_flash_mla_moe::mla_output_projection",
    _mla_output_projection_backward,
    setup_context=_mla_output_projection_setup_context,
)


def native_extension_loaded() -> bool:
    """Return whether this installation contains and loaded the native library."""

    return _NATIVE_EXTENSION_LOADED


_ATTENTION_OPERATOR = {
    "cuda_rowwise": "attention_forward",
    "fa1": "attention_fa1_forward",
    "fa2": "attention_fa2_forward",
}


def cuda_attention_backend_available(
    backend: NativeAttentionBackend = "cuda_rowwise",
) -> bool:
    """Return whether the selected native attention backend can execute."""

    if backend not in _ATTENTION_OPERATOR:
        raise ValueError("native attention backend must be cuda_rowwise, fa1, or fa2")
    return (
        _NATIVE_EXTENSION_LOADED
        and torch.cuda.is_available()
        and _operator_has_cuda_kernel(_ATTENTION_OPERATOR[backend])
    )


def cuda_kernel_available() -> bool:
    """Return whether the row-wise native attention backend can execute."""

    return cuda_attention_backend_available("cuda_rowwise")


def cuda_gemm_available() -> bool:
    """Return whether the native tiled GEMM kernel can be executed."""

    return (
        _NATIVE_EXTENSION_LOADED
        and torch.cuda.is_available()
        and _operator_has_cuda_kernel("tiled_gemm")
    )


def cuda_mla_available() -> bool:
    """Return whether the complete native MLA prefill/decode pipeline can execute."""

    return (
        _NATIVE_EXTENSION_LOADED
        and torch.cuda.is_available()
        and all(
            _operator_has_cuda_kernel(operator)
            for operator in (
                "mla_query_projection",
                "mla_query_lora_projection",
                "mla_cache_projection",
                "mla_cache_projection_write",
                "mla_absorbed_attention",
                "mla_output_projection",
            )
        )
    )


def cuda_paged_mla_available() -> bool:
    """Return whether native paged-cache write and attention kernels can execute."""

    return cuda_mla_available() and all(
        _operator_has_cuda_kernel(operator)
        for operator in (
            "mla_cache_projection_write_slots",
            "mla_paged_absorbed_attention",
        )
    )


def _validate_mla_backend(backend: MLABackend) -> None:
    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be 'auto', 'cuda', or 'reference'")


def _mla_projection_cuda_ineligibility_reason(
    operator: str,
    floating_tensors: tuple[Tensor, ...],
    positions: Tensor | None = None,
    *,
    contiguous_tensors: tuple[Tensor, ...] = (),
) -> str | None:
    if not _NATIVE_EXTENSION_LOADED:
        return "the native extension is not installed"
    tensors = floating_tensors + (() if positions is None else (positions,))
    if any(tensor.device.type != "cuda" for tensor in tensors):
        return "all MLA projection tensors must be CUDA tensors"
    storage_dtype = floating_tensors[0].dtype
    if storage_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return "the CUDA MLA projection kernels support float16, bfloat16, and float32"
    if any(tensor.dtype != storage_dtype for tensor in floating_tensors[1:]):
        return "all floating-point MLA projection tensors must have the same dtype"
    if positions is not None and positions.dtype != torch.long:
        return "MLA positions must use int64"
    if len({tensor.device for tensor in tensors}) != 1:
        return "all MLA projection tensors must share a CUDA device"
    if any(not tensor.is_contiguous() for tensor in contiguous_tensors):
        return "the CUDA MLA projection weights and storage must be contiguous"
    if not _operator_has_cuda_kernel(operator):
        return f"the loaded native extension does not register {operator}"
    return None


def mla_query_projection(
    x: Tensor,
    wq: Tensor,
    positions: Tensor,
    *,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    rope_theta: float,
    backend: MLABackend = "auto",
) -> tuple[Tensor, Tensor]:
    """Project direct MLA queries with native CUDA when its storage contract holds."""

    _validate_mla_backend(backend)
    reason = _mla_projection_cuda_ineligibility_reason(
        "mla_query_projection",
        (x, wq),
        positions,
        contiguous_tensors=(wq,),
    )
    arguments = (
        x,
        wq,
        positions,
        n_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        rope_theta,
    )
    if backend == "cuda":
        if reason is not None:
            raise RuntimeError(f"CUDA MLA is unavailable: query projection: {reason}")
        return torch.ops.ds_flash_mla_moe.mla_query_projection.default(*arguments)
    if backend == "auto" and reason is None:
        return torch.ops.ds_flash_mla_moe.mla_query_projection.default(*arguments)
    return _composite_mla_query_projection(*arguments)


def mla_query_lora_projection(
    x: Tensor,
    wq_a: Tensor,
    q_norm_weight: Tensor,
    wq_b: Tensor,
    positions: Tensor,
    *,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    rope_theta: float,
    rms_norm_eps: float,
    backend: MLABackend = "auto",
) -> tuple[Tensor, Tensor]:
    """Project LoRA MLA queries, including RMSNorm and RoPE, through one operator."""

    _validate_mla_backend(backend)
    reason = _mla_projection_cuda_ineligibility_reason(
        "mla_query_lora_projection",
        (x, wq_a, q_norm_weight, wq_b),
        positions,
        contiguous_tensors=(wq_a, q_norm_weight, wq_b),
    )
    arguments = (
        x,
        wq_a,
        q_norm_weight,
        wq_b,
        positions,
        n_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        rope_theta,
        rms_norm_eps,
    )
    if backend == "cuda":
        if reason is not None:
            raise RuntimeError(f"CUDA MLA is unavailable: query projection: {reason}")
        return torch.ops.ds_flash_mla_moe.mla_query_lora_projection.default(*arguments)
    if backend == "auto" and reason is None:
        return torch.ops.ds_flash_mla_moe.mla_query_lora_projection.default(*arguments)
    return _composite_mla_query_lora_projection(*arguments)


def mla_cache_projection(
    x: Tensor,
    wkv_a: Tensor,
    kv_norm_weight: Tensor,
    positions: Tensor,
    *,
    kv_lora_rank: int,
    rope_theta: float,
    rms_norm_eps: float,
    backend: MLABackend = "auto",
) -> tuple[Tensor, Tensor]:
    """Build compressed MLA cache entries with native projection, RMSNorm, and RoPE."""

    _validate_mla_backend(backend)
    reason = _mla_projection_cuda_ineligibility_reason(
        "mla_cache_projection",
        (x, wkv_a, kv_norm_weight),
        positions,
        contiguous_tensors=(wkv_a, kv_norm_weight),
    )
    arguments = (
        x,
        wkv_a,
        kv_norm_weight,
        positions,
        kv_lora_rank,
        rope_theta,
        rms_norm_eps,
    )
    if backend == "cuda":
        if reason is not None:
            raise RuntimeError(f"CUDA MLA is unavailable: cache projection: {reason}")
        return torch.ops.ds_flash_mla_moe.mla_cache_projection.default(*arguments)
    if backend == "auto" and reason is None:
        return torch.ops.ds_flash_mla_moe.mla_cache_projection.default(*arguments)
    return _composite_mla_cache_projection(*arguments)


def mla_cache_projection_write(
    x: Tensor,
    wkv_a: Tensor,
    kv_norm_weight: Tensor,
    positions: Tensor,
    kv_storage: Tensor,
    pe_storage: Tensor,
    position_storage: Tensor,
    *,
    start: int,
    rope_theta: float,
    rms_norm_eps: float,
    backend: MLABackend = "auto",
) -> None:
    """Project a chunk directly into preallocated inference cache storage."""

    _validate_mla_backend(backend)
    floating = (x, wkv_a, kv_norm_weight, kv_storage, pe_storage)
    reason = _mla_projection_cuda_ineligibility_reason(
        "mla_cache_projection_write",
        floating,
        positions,
        contiguous_tensors=(
            wkv_a,
            kv_norm_weight,
            kv_storage,
            pe_storage,
            position_storage,
        ),
    )
    if reason is None and (
        position_storage.device.type != "cuda"
        or position_storage.device != x.device
        or position_storage.dtype != torch.long
    ):
        reason = "MLA position storage must be an int64 CUDA tensor"
    arguments = (
        x,
        wkv_a,
        kv_norm_weight,
        positions,
        kv_storage,
        pe_storage,
        position_storage,
        start,
        rope_theta,
        rms_norm_eps,
    )
    if backend == "cuda":
        if reason is not None:
            raise RuntimeError(f"CUDA MLA is unavailable: static cache write: {reason}")
        torch.ops.ds_flash_mla_moe.mla_cache_projection_write.default(*arguments)
        return
    if backend == "auto" and reason is None:
        torch.ops.ds_flash_mla_moe.mla_cache_projection_write.default(*arguments)
        return
    _composite_mla_cache_projection_write(*arguments)


def mla_cache_projection_write_slots(
    x: Tensor,
    wkv_a: Tensor,
    kv_norm_weight: Tensor,
    positions: Tensor,
    slot_mapping: Tensor,
    kv_storage: Tensor,
    pe_storage: Tensor,
    position_storage: Tensor,
    *,
    rope_theta: float,
    rms_norm_eps: float,
    backend: MLABackend = "auto",
    _metadata_validated: bool = False,
) -> None:
    """Project tokens into distinct physical slots of a paged cache."""

    _validate_mla_backend(backend)
    floating = (x, wkv_a, kv_norm_weight, kv_storage, pe_storage)
    reason = _mla_projection_cuda_ineligibility_reason(
        "mla_cache_projection_write_slots",
        floating,
        positions,
        contiguous_tensors=(
            wkv_a,
            kv_norm_weight,
            positions,
            slot_mapping,
            kv_storage,
            pe_storage,
            position_storage,
        ),
    )
    integer_tensors = (positions, slot_mapping, position_storage)
    if reason is None and (
        any(tensor.device.type != "cuda" or tensor.device != x.device for tensor in integer_tensors)
        or any(tensor.dtype != torch.long for tensor in integer_tensors)
    ):
        reason = "paged MLA positions and slot mappings must be int64 CUDA tensors"
    arguments = (
        x,
        wkv_a,
        kv_norm_weight,
        positions,
        slot_mapping,
        kv_storage,
        pe_storage,
        position_storage,
        _metadata_validated,
        rope_theta,
        rms_norm_eps,
    )
    if backend == "cuda":
        if reason is not None:
            raise RuntimeError(f"CUDA MLA is unavailable: paged cache write: {reason}")
        torch.ops.ds_flash_mla_moe.mla_cache_projection_write_slots.default(*arguments)
        return
    if backend == "auto" and reason is None:
        torch.ops.ds_flash_mla_moe.mla_cache_projection_write_slots.default(*arguments)
        return
    _composite_mla_cache_projection_write_slots(*arguments)


def mla_output_projection(
    heads: Tensor,
    wo: Tensor,
    *,
    backend: MLABackend = "auto",
) -> Tensor:
    """Project MLA heads back to model width using the selected operator backend."""

    _validate_mla_backend(backend)
    reason = _mla_projection_cuda_ineligibility_reason(
        "mla_output_projection",
        (heads, wo),
        contiguous_tensors=(heads, wo),
    )
    if backend == "cuda":
        if reason is not None:
            raise RuntimeError(f"CUDA MLA is unavailable: output projection: {reason}")
        return torch.ops.ds_flash_mla_moe.mla_output_projection.default(heads, wo)
    if backend == "auto" and reason is None:
        return torch.ops.ds_flash_mla_moe.mla_output_projection.default(heads, wo)
    return _composite_mla_output_projection(heads, wo)


def _mla_cuda_ineligibility_reason(
    q_nope: Tensor,
    q_pe: Tensor,
    kv: Tensor,
    pe: Tensor,
    key_up: Tensor,
    value_up: Tensor,
    query_positions: Tensor,
    key_positions: Tensor,
) -> str | None:
    tensors = (q_nope, q_pe, kv, pe, key_up, value_up, query_positions, key_positions)
    if not _NATIVE_EXTENSION_LOADED:
        return "the native extension is not installed"
    if any(tensor.device.type != "cuda" for tensor in tensors):
        return "all MLA tensors must be CUDA tensors"
    storage_dtype = q_nope.dtype
    if storage_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return "the CUDA MLA kernel supports float16, bfloat16, and float32"
    if any(tensor.dtype != storage_dtype for tensor in tensors[1:6]):
        return "all floating-point MLA tensors must have the same dtype"
    if query_positions.dtype != torch.long or key_positions.dtype != torch.long:
        return "MLA positions must use int64"
    if len({tensor.device for tensor in tensors}) != 1:
        return "all MLA tensors must share a CUDA device"
    if not _operator_has_cuda_kernel("mla_absorbed_attention"):
        return "the loaded native extension does not register a CUDA MLA kernel"
    return None


def mla_absorbed_attention(
    q_nope: Tensor,
    q_pe: Tensor,
    kv: Tensor,
    pe: Tensor,
    key_up: Tensor,
    value_up: Tensor,
    *,
    query_positions: Tensor,
    key_positions: Tensor,
    causal: bool = True,
    scale: float,
    backend: MLABackend = "auto",
) -> Tensor:
    """Evaluate absorbed MLA attention, selecting the native fused CUDA path when eligible."""

    _validate_mla_backend(backend)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    reason = _mla_cuda_ineligibility_reason(
        q_nope,
        q_pe,
        kv,
        pe,
        key_up,
        value_up,
        query_positions,
        key_positions,
    )
    if backend == "cuda":
        if reason is not None:
            raise RuntimeError(f"CUDA MLA is unavailable: {reason}")
        return torch.ops.ds_flash_mla_moe.mla_absorbed_attention.default(
            q_nope,
            q_pe,
            kv,
            pe,
            key_up,
            value_up,
            query_positions,
            key_positions,
            causal,
            scale,
        )
    if backend == "auto" and reason is None:
        return torch.ops.ds_flash_mla_moe.mla_absorbed_attention.default(
            q_nope,
            q_pe,
            kv,
            pe,
            key_up,
            value_up,
            query_positions,
            key_positions,
            causal,
            scale,
        )
    return _composite_mla_absorbed_attention(
        q_nope,
        q_pe,
        kv,
        pe,
        key_up,
        value_up,
        query_positions,
        key_positions,
        causal,
        scale,
    )


def _mla_paged_cuda_ineligibility_reason(
    q_nope: Tensor,
    q_pe: Tensor,
    kv_storage: Tensor,
    pe_storage: Tensor,
    position_storage: Tensor,
    block_table: Tensor,
    sequence_lengths: Tensor,
    key_up: Tensor,
    value_up: Tensor,
    query_positions: Tensor,
) -> str | None:
    floating = (q_nope, q_pe, kv_storage, pe_storage, key_up, value_up)
    integer = (position_storage, block_table, sequence_lengths, query_positions)
    if not _NATIVE_EXTENSION_LOADED:
        return "the native extension is not installed"
    if any(tensor.device.type != "cuda" for tensor in (*floating, *integer)):
        return "all paged MLA tensors must be CUDA tensors"
    if q_nope.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return "the CUDA paged MLA kernel supports float16, bfloat16, and float32"
    if any(tensor.dtype != q_nope.dtype for tensor in floating[1:]):
        return "all floating-point paged MLA tensors must have the same dtype"
    if any(tensor.dtype != torch.long for tensor in integer):
        return "paged MLA metadata must use int64"
    if len({tensor.device for tensor in (*floating, *integer)}) != 1:
        return "all paged MLA tensors must share a CUDA device"
    if not _operator_has_cuda_kernel("mla_paged_absorbed_attention"):
        return "the loaded native extension does not register paged MLA attention"
    return None


def mla_paged_absorbed_attention(
    q_nope: Tensor,
    q_pe: Tensor,
    kv_storage: Tensor,
    pe_storage: Tensor,
    position_storage: Tensor,
    block_table: Tensor,
    sequence_lengths: Tensor,
    key_up: Tensor,
    value_up: Tensor,
    *,
    query_positions: Tensor,
    causal: bool = True,
    scale: float,
    backend: MLABackend = "auto",
    _metadata_validated: bool = False,
) -> Tensor:
    """Evaluate absorbed MLA directly through a logical-to-physical page table."""

    _validate_mla_backend(backend)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    reason = _mla_paged_cuda_ineligibility_reason(
        q_nope,
        q_pe,
        kv_storage,
        pe_storage,
        position_storage,
        block_table,
        sequence_lengths,
        key_up,
        value_up,
        query_positions,
    )
    arguments = (
        q_nope,
        q_pe,
        kv_storage,
        pe_storage,
        position_storage,
        block_table,
        sequence_lengths,
        key_up,
        value_up,
        query_positions,
        _metadata_validated,
        causal,
        scale,
    )
    if backend == "cuda":
        if reason is not None:
            raise RuntimeError(f"CUDA MLA is unavailable: paged attention: {reason}")
        return torch.ops.ds_flash_mla_moe.mla_paged_absorbed_attention.default(*arguments)
    if backend == "auto" and reason is None:
        return torch.ops.ds_flash_mla_moe.mla_paged_absorbed_attention.default(*arguments)
    return _composite_mla_paged_absorbed_attention(*arguments)


def _gemm_cuda_ineligibility_reason(a: Tensor, b: Tensor, c: Tensor | None) -> str | None:
    if not _NATIVE_EXTENSION_LOADED:
        return "the native extension is not installed"
    if a.device.type != "cuda" or b.device.type != "cuda":
        return "a and b must be CUDA tensors"
    if c is not None and c.device.type != "cuda":
        return "c must be a CUDA tensor"
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        return "the CUDA tiled GEMM kernel currently supports float32 only"
    if c is not None and c.dtype != torch.float32:
        return "the CUDA tiled GEMM kernel currently supports float32 only"
    if not (a.is_contiguous() and b.is_contiguous()):
        return "the CUDA tiled GEMM kernel requires contiguous a and b tensors"
    if c is not None and not c.is_contiguous():
        return "the CUDA tiled GEMM kernel requires a contiguous c tensor"
    if not _operator_has_cuda_kernel("tiled_gemm"):
        return "the loaded native extension does not register a CUDA tiled GEMM kernel"
    return None


def tiled_gemm(
    a: Tensor,
    b: Tensor,
    c: Tensor | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    backend: GEMMBackend = "auto",
) -> Tensor:
    """Run fixed-tile native CUDA GEMM when eligible, else the specification."""

    _validate_gemm_inputs(a, b, c, alpha=alpha, beta=beta)
    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be 'auto', 'cuda', or 'reference'")
    reason = _gemm_cuda_ineligibility_reason(a, b, c)
    if backend == "cuda":
        if reason is not None:
            raise RuntimeError(f"CUDA tiled GEMM is unavailable: {reason}")
        return torch.ops.ds_flash_mla_moe.tiled_gemm.default(a, b, c, alpha, beta)
    if backend == "auto" and reason is None:
        return torch.ops.ds_flash_mla_moe.tiled_gemm.default(a, b, c, alpha, beta)
    return gemm_reference(a, b, c, alpha=alpha, beta=beta)


def _attention_backend_ineligibility_reason(
    backend: NativeAttentionBackend,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    attn_mask: Tensor | None,
) -> str | None:
    if backend in {"fa1", "fa2"} and any(t.requires_grad for t in (q, k, v)):
        return f"{backend} is forward-only and does not accept requires_grad tensors"
    if q.ndim != 4:
        return "the CUDA kernel requires [batch, heads, sequence, dimension] tensors"
    supported = (
        {torch.float16}
        if backend in {"fa1", "fa2"}
        else {torch.float16, torch.bfloat16, torch.float32}
    )
    if q.dtype not in supported:
        rendered = "float16" if backend in {"fa1", "fa2"} else "float16, bfloat16, or float32"
        return f"{backend} supports {rendered}"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return "q, k, and v must have the same dtype"
    if not all(t.is_contiguous() for t in (q, k, v)):
        return "the CUDA kernel requires contiguous tensors"
    if attn_mask is not None:
        return "the CUDA kernel does not support an explicit attention mask"
    if backend in {"fa1", "fa2"} and (q.shape[-1] > 128 or v.shape[-1] > 128):
        return "formal FA1/FA2 currently require head_dim <= 128 and value_dim <= 128"
    if not _NATIVE_EXTENSION_LOADED:
        return "the native extension is not installed"
    if any(t.device.type != "cuda" for t in (q, k, v)):
        return "q, k, and v must be CUDA tensors"
    operator = _ATTENTION_OPERATOR[backend]
    if not _operator_has_cuda_kernel(operator):
        return f"the loaded native extension does not register {operator}"
    return None


def _validate_attention_mask_request(
    q: Tensor,
    k: Tensor,
    *,
    causal: bool,
    attn_mask: Tensor | None,
) -> None:
    if causal:
        _causal_keep_mask(q.shape[-2], k.shape[-2], q.device)
    _broadcast_mask(attn_mask, (*q.shape[:-2], q.shape[-2], k.shape[-2]), q.device)


def flash_attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    attn_mask: Tensor | None = None,
    backend: AttentionBackend = "auto",
    reference_block_size: int = 64,
) -> Tensor:
    """Dispatch attention to a strict backend or the automatic row-wise fallback."""

    _validate_attention_inputs(q, k, v)
    valid = {"auto", "cuda", "cuda_rowwise", "reference", "blockwise", "fa1", "fa2"}
    if backend not in valid:
        raise ValueError("backend must be auto, cuda_rowwise, reference, blockwise, fa1, or fa2")
    if backend == "cuda":
        warnings.warn(
            "backend='cuda' is deprecated; use backend='cuda_rowwise'",
            FutureWarning,
            stacklevel=2,
        )
        backend = "cuda_rowwise"
    if causal and q.shape[-2] > k.shape[-2]:
        raise ValueError("right-aligned causal attention requires query_length <= key_length")
    _validate_attention_mask_request(q, k, causal=causal, attn_mask=attn_mask)

    effective_scale = float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    if not math.isfinite(effective_scale):
        raise ValueError("scale must be finite")

    if backend == "reference":
        return scaled_dot_product_attention_reference(
            q,
            k,
            v,
            causal=causal,
            scale=effective_scale,
            attn_mask=attn_mask,
        )
    if backend == "blockwise":
        return blockwise_attention(
            q,
            k,
            v,
            causal=causal,
            scale=effective_scale,
            attn_mask=attn_mask,
            block_size=reference_block_size,
        )

    selected = "cuda_rowwise" if backend == "auto" else backend
    reason = _attention_backend_ineligibility_reason(selected, q, k, v, attn_mask=attn_mask)
    if reason is None:
        operator = getattr(torch.ops.ds_flash_mla_moe, _ATTENTION_OPERATOR[selected]).default
        return operator(q, k, v, causal, effective_scale)
    if backend != "auto":
        raise RuntimeError(f"{selected} attention is unavailable: {reason}")
    return blockwise_attention(
        q,
        k,
        v,
        causal=causal,
        scale=effective_scale,
        attn_mask=attn_mask,
        block_size=reference_block_size,
    )
