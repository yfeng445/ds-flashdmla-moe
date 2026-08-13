"""Dispatch boundary between executable specifications and optional native kernels."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from .attention import (
    _broadcast_mask,
    _causal_keep_mask,
    _validate_attention_inputs,
    blockwise_attention,
    scaled_dot_product_attention_backward_reference,
)
from .gemm import _gemm_compute_dtype, _validate_gemm_inputs, gemm_reference
from .moe import RoutingResult, deepseek_grouped_topk, pack_routes_reference

AttentionBackend = Literal["auto", "cuda", "reference"]
GEMMBackend = Literal["auto", "cuda", "reference"]

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
_SCHEMAS = {
    "attention_forward": _FORWARD_SCHEMA,
    "attention_backward": _BACKWARD_SCHEMA,
    "route_pack": _ROUTE_PACK_SCHEMA,
    "route_combine": _ROUTE_COMBINE_SCHEMA,
    "tiled_gemm": _TILED_GEMM_SCHEMA,
    "swiglu_experts": _SWIGLU_EXPERTS_SCHEMA,
    "expert_major_pack": _EXPERT_MAJOR_PACK_SCHEMA,
    "grouped_topk": _GROUPED_TOPK_SCHEMA,
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


composite_explicit = torch.library.Library("ds_flash_mla_moe", "IMPL", "CompositeExplicitAutograd")
composite_explicit.impl("attention_forward", _composite_attention_forward)
composite_explicit.impl("route_pack", _composite_route_pack)
composite_explicit.impl("route_combine", _composite_route_combine)
composite_explicit.impl("tiled_gemm", _composite_tiled_gemm)
composite_explicit.impl("swiglu_experts", _composite_swiglu_experts)
composite_explicit.impl("expert_major_pack", _composite_expert_major_pack)
composite_explicit.impl("grouped_topk", _composite_grouped_topk)
_LIBRARY_HANDLES.append(composite_explicit)

composite_implicit = torch.library.Library("ds_flash_mla_moe", "IMPL", "CompositeImplicitAutograd")
composite_implicit.impl("attention_backward", _composite_attention_backward)
_LIBRARY_HANDLES.append(composite_implicit)

torch.library.register_fake("ds_flash_mla_moe::attention_forward", _fake_attention_forward)
torch.library.register_fake("ds_flash_mla_moe::route_pack", _fake_route_pack)
torch.library.register_fake("ds_flash_mla_moe::route_combine", _fake_route_combine)
torch.library.register_fake("ds_flash_mla_moe::tiled_gemm", _fake_tiled_gemm)
torch.library.register_fake("ds_flash_mla_moe::swiglu_experts", _fake_swiglu_experts)
torch.library.register_fake("ds_flash_mla_moe::expert_major_pack", _fake_expert_major_pack)
torch.library.register_fake("ds_flash_mla_moe::grouped_topk", _fake_grouped_topk)


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
        and q.dtype == torch.float32
        and k.dtype == torch.float32
        and v.dtype == torch.float32
        and grad_output.dtype == torch.float32
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
        gradients = torch.ops.ds_flash_mla_moe.attention_backward.default(
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


def native_extension_loaded() -> bool:
    """Return whether this installation contains and loaded the native library."""

    return _NATIVE_EXTENSION_LOADED


def cuda_kernel_available() -> bool:
    """Return whether the native library is loaded and a CUDA device is usable."""

    return _NATIVE_EXTENSION_LOADED and torch.cuda.is_available()


def cuda_gemm_available() -> bool:
    """Return whether the native tiled GEMM kernel can be executed."""

    return (
        _NATIVE_EXTENSION_LOADED
        and torch.cuda.is_available()
        and _operator_has_cuda_kernel("tiled_gemm")
    )


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


def _cuda_ineligibility_reason(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    attn_mask: Tensor | None,
) -> str | None:
    if not _NATIVE_EXTENSION_LOADED:
        return "the native extension is not installed"
    if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
        return "q, k, and v must be CUDA tensors"
    if q.ndim != 4:
        return "the CUDA kernel currently requires [batch, heads, sequence, dimension] tensors"
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        return "the CUDA kernel currently supports float32 only"
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return "the CUDA kernel currently requires contiguous tensors"
    if attn_mask is not None:
        return "the CUDA kernel does not support an explicit attention mask yet"
    if not _operator_has_cuda_kernel("attention_forward"):
        return "the loaded native extension does not register a CUDA forward kernel"
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
    """Run the native CUDA forward kernel when eligible, otherwise use the specification.

    The first native kernel intentionally has a narrow contract: contiguous
    float32 rank-four CUDA tensors and no explicit mask.
    ``backend="cuda"`` enforces that contract. ``backend="auto"`` falls back
    to the differentiable blockwise PyTorch specification for unsupported
    shapes, dtypes, layouts, devices, or masks. Until a native backward exists,
    autograd recomputes the differentiable specification during backward.
    """

    _validate_attention_inputs(q, k, v)
    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be 'auto', 'cuda', or 'reference'")
    if causal and q.shape[-2] > k.shape[-2]:
        raise ValueError("right-aligned causal attention requires query_length <= key_length")
    _validate_attention_mask_request(q, k, causal=causal, attn_mask=attn_mask)

    effective_scale = float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    if not math.isfinite(effective_scale):
        raise ValueError("scale must be finite")

    reason = _cuda_ineligibility_reason(q, k, v, attn_mask=attn_mask)
    if backend == "cuda":
        if reason is not None:
            raise RuntimeError(f"CUDA attention is unavailable: {reason}")
        return torch.ops.ds_flash_mla_moe.attention_forward.default(
            q, k, v, causal, effective_scale
        )

    if backend == "auto" and reason is None:
        return torch.ops.ds_flash_mla_moe.attention_forward.default(
            q, k, v, causal, effective_scale
        )

    return blockwise_attention(
        q,
        k,
        v,
        causal=causal,
        scale=effective_scale,
        attn_mask=attn_mask,
        block_size=reference_block_size,
    )
