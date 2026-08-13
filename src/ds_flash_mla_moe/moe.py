"""DeepSeek-style grouped routing and MoE executable specifications."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


class RoutingResult(NamedTuple):
    """Selected expert indices and their unbiased, normalized routing weights."""

    weights: Tensor
    indices: Tensor


class PackedRoutes(NamedTuple):
    """Rank-major, expert-major rows and the metadata needed to restore them.

    ``expert_offsets`` index segments in ``expert_order`` rather than raw
    expert-id order. This keeps every destination-rank region contiguous even
    when the expert-to-rank assignment is not monotonic.
    """

    activations: Tensor
    route_weights: Tensor
    token_indices: Tensor
    slot_indices: Tensor
    expert_indices: Tensor
    destination_ranks: Tensor
    counts_per_expert: Tensor
    expert_order: Tensor
    expert_offsets: Tensor
    rank_counts: Tensor
    rank_offsets: Tensor
    token_shape: tuple[int, ...]


class ExpertMajorLayout(NamedTuple):
    """Stable expert-major rows plus the inverse permutation to source order."""

    activations: Tensor
    expert_indices: Tensor
    counts_per_expert: Tensor
    expert_offsets: Tensor
    permutation: Tensor
    inverse_permutation: Tensor


def _compute_dtype(tensor: Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _stable_topk_indices(values: Tensor, k: int) -> Tensor:
    """Select descending Top-K with smaller indices winning exact ties."""

    return torch.argsort(values, dim=-1, descending=True, stable=True)[..., :k].contiguous()


def deepseek_grouped_topk(
    x: Tensor,
    gate_weight: Tensor,
    *,
    topk: int,
    n_groups: int = 1,
    topk_groups: int | None = None,
    score_func: str = "sigmoid",
    score_bias: Tensor | None = None,
    route_scale: float = 1.0,
) -> RoutingResult:
    """Reference implementation of the DeepSeek-V3 group-limited gate.

    ``score_bias`` changes expert selection but never the gathered routing
    weights. Sigmoid scores are renormalized over the selected experts before
    applying ``route_scale``; softmax scores follow the official reference and
    are not renormalized after Top-K.
    """

    if x.ndim < 1 or gate_weight.ndim != 2:
        raise ValueError("x must end in model_dim and gate_weight must be [experts, model_dim]")
    experts, model_dim = gate_weight.shape
    if x.shape[-1] != model_dim:
        raise ValueError("x model dimension does not match gate_weight")
    if experts <= 0 or n_groups <= 0 or experts % n_groups != 0:
        raise ValueError("number of experts must be positive and divisible by n_groups")
    if not 1 <= topk <= experts:
        raise ValueError("topk must be in [1, number_of_experts]")
    if topk_groups is None:
        topk_groups = n_groups
    if not 1 <= topk_groups <= n_groups:
        raise ValueError("topk_groups must be in [1, n_groups]")
    experts_per_group = experts // n_groups
    if topk > topk_groups * experts_per_group:
        raise ValueError("topk exceeds the number of experts retained by group selection")
    if score_func not in {"sigmoid", "softmax"}:
        raise ValueError("score_func must be 'sigmoid' or 'softmax'")
    if score_bias is not None and score_bias.shape != (experts,):
        raise ValueError("score_bias must have shape [number_of_experts]")

    original_shape = x.shape[:-1]
    compute_dtype = _compute_dtype(x)
    flat_x = x.reshape(-1, model_dim).to(compute_dtype)
    logits = F.linear(flat_x, gate_weight.to(compute_dtype))
    original_scores = (
        torch.sigmoid(logits) if score_func == "sigmoid" else torch.softmax(logits, dim=-1)
    )
    selection_scores = original_scores
    if score_bias is not None:
        selection_scores = selection_scores + score_bias.to(compute_dtype)

    if n_groups > 1:
        grouped_scores = selection_scores.reshape(-1, n_groups, experts_per_group)
        if score_bias is None:
            group_scores = grouped_scores.amax(dim=-1)
        else:
            group_scores = grouped_scores.topk(min(2, experts_per_group), dim=-1).values.sum(dim=-1)
        selected_groups = _stable_topk_indices(group_scores, topk_groups)
        group_keep = torch.zeros_like(group_scores, dtype=torch.bool)
        group_keep.scatter_(1, selected_groups, True)
        expert_keep = group_keep.unsqueeze(-1).expand_as(grouped_scores).reshape(-1, experts)
        selection_scores = selection_scores.masked_fill(~expert_keep, -torch.inf)

    indices = _stable_topk_indices(selection_scores, topk)
    weights = original_scores.gather(1, indices)
    if score_func == "sigmoid":
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
    weights = weights * float(route_scale)

    result_shape = (*original_shape, topk)
    return RoutingResult(weights.to(x.dtype).reshape(result_shape), indices.reshape(result_shape))


def pack_routes_reference(
    x: Tensor,
    routing: RoutingResult,
    *,
    n_experts: int,
    expert_owner: Tensor | None = None,
    world_size: int | None = None,
) -> PackedRoutes:
    """Pack raw token activations by destination rank and expert.

    Routing weights remain metadata and are not applied to ``activations``.
    DeepSeek experts are nonlinear, so applying a weight before the expert
    would change ``alpha * E(x)`` into ``E(alpha * x)``.
    """

    integer_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    if x.ndim < 2:
        raise ValueError("x must have shape [..., tokens, model_dim]")
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")
    if routing.weights.shape != routing.indices.shape:
        raise ValueError("routing weights and indices must have identical shapes")
    if routing.indices.ndim < 1 or routing.indices.shape[:-1] != x.shape[:-1]:
        raise ValueError("routing leading dimensions must match x token dimensions")
    if routing.indices.dtype not in integer_dtypes:
        raise TypeError("routing indices must use an integer dtype")
    if routing.indices.device != x.device or routing.weights.device != x.device:
        raise ValueError("x and routing tensors must be on the same device")

    if expert_owner is None:
        if world_size not in (None, 1):
            raise ValueError("world_size > 1 requires expert_owner")
        world_size = 1
        expert_owner = torch.zeros(n_experts, device=x.device, dtype=torch.long)
    else:
        if expert_owner.shape != (n_experts,) or expert_owner.device != x.device:
            raise ValueError("expert_owner must be [n_experts] on the same device as x")
        if expert_owner.dtype not in integer_dtypes:
            raise TypeError("expert_owner must use an integer dtype")
        expert_owner = expert_owner.to(torch.long)
        inferred_world_size = int(expert_owner.max().item()) + 1
        world_size = inferred_world_size if world_size is None else world_size
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if torch.any(expert_owner < 0) or torch.any(expert_owner >= world_size):
            raise ValueError("expert_owner contains a rank outside world_size")

    flat_x = x.reshape(-1, x.shape[-1])
    topk = routing.indices.shape[-1]
    flat_experts = routing.indices.reshape(-1).to(torch.long)
    if torch.any(flat_experts < 0) or torch.any(flat_experts >= n_experts):
        raise ValueError("routing contains an expert outside [0, n_experts)")

    route_count = flat_experts.numel()
    token_indices = torch.arange(flat_x.shape[0], device=x.device).repeat_interleave(topk)
    slot_indices = torch.arange(topk, device=x.device).repeat(flat_x.shape[0])
    destination_ranks = expert_owner[flat_experts]
    permutation = torch.argsort(destination_ranks * n_experts + flat_experts, stable=True)

    counts_per_expert = torch.bincount(flat_experts, minlength=n_experts)
    expert_ids = torch.arange(n_experts, device=x.device)
    expert_order = torch.argsort(expert_owner * n_experts + expert_ids, stable=True)
    expert_offsets = torch.cat(
        (
            torch.zeros(1, device=x.device, dtype=torch.long),
            counts_per_expert[expert_order].cumsum(0),
        )
    )
    rank_counts = torch.bincount(destination_ranks, minlength=world_size)
    rank_offsets = torch.cat(
        (torch.zeros(1, device=x.device, dtype=torch.long), rank_counts.cumsum(0))
    )
    if int(expert_offsets[-1]) != route_count or int(rank_offsets[-1]) != route_count:
        raise RuntimeError("route counts and offsets are inconsistent")

    return PackedRoutes(
        activations=flat_x[token_indices[permutation]],
        route_weights=routing.weights.reshape(-1)[permutation],
        token_indices=token_indices[permutation],
        slot_indices=slot_indices[permutation],
        expert_indices=flat_experts[permutation],
        destination_ranks=destination_ranks[permutation],
        counts_per_expert=counts_per_expert,
        expert_order=expert_order,
        expert_offsets=expert_offsets,
        rank_counts=rank_counts,
        rank_offsets=rank_offsets,
        token_shape=tuple(x.shape[:-1]),
    )


def combine_packed_routes(contributions: Tensor, packed: PackedRoutes) -> Tensor:
    """Apply routing weights and restore expert outputs to token order."""

    if contributions.ndim != 2:
        raise ValueError("contributions must have shape [packed_rows, model_dim]")
    if contributions.shape[0] != packed.token_indices.numel():
        raise ValueError("contribution row count does not match packed metadata")
    if contributions.device != packed.token_indices.device:
        raise ValueError("contributions and packed metadata must be on the same device")

    token_count = 1
    for dimension in packed.token_shape:
        token_count *= dimension
    compute_dtype = _compute_dtype(contributions)
    weighted = contributions.to(compute_dtype) * packed.route_weights.to(compute_dtype).unsqueeze(
        -1
    )
    restored = torch.zeros(
        (token_count, contributions.shape[-1]),
        device=contributions.device,
        dtype=compute_dtype,
    ).index_add(0, packed.token_indices, weighted)
    return restored.to(contributions.dtype).reshape(*packed.token_shape, contributions.shape[-1])


def swiglu_expert(x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
    """Compute ``W2(SiLU(W1(x)) * W3(x))`` with high-precision accumulation."""

    if x.ndim < 1 or w1.ndim != 2 or w2.ndim != 2 or w3.ndim != 2:
        raise ValueError("x must be [..., D] and expert weights must be matrices")
    hidden, model_dim = w1.shape
    if w3.shape != (hidden, model_dim) or w2.shape != (model_dim, hidden):
        raise ValueError("expected w1/w3=[hidden, D] and w2=[D, hidden]")
    if x.shape[-1] != model_dim:
        raise ValueError("x model dimension does not match expert weights")

    compute_dtype = _compute_dtype(x)
    x_compute = x.to(compute_dtype)
    hidden_state = F.silu(F.linear(x_compute, w1.to(compute_dtype))) * F.linear(
        x_compute, w3.to(compute_dtype)
    )
    if x.dtype == torch.float16:
        hidden_state = hidden_state.to(x.dtype).to(compute_dtype)
    return F.linear(hidden_state, w2.to(compute_dtype)).to(x.dtype)


def to_expert_major_reference(
    activations: Tensor,
    expert_indices: Tensor,
    *,
    n_experts: int,
) -> ExpertMajorLayout:
    """Stably group arbitrary input rows by global expert id."""

    if activations.ndim != 2:
        raise ValueError("activations must have shape [rows, model_dim]")
    if expert_indices.ndim != 1 or expert_indices.shape[0] != activations.shape[0]:
        raise ValueError("expert_indices must have one entry per activation row")
    if expert_indices.dtype != torch.long:
        raise TypeError("expert_indices must use int64")
    if expert_indices.device != activations.device:
        raise ValueError("activations and expert_indices must be on the same device")
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")
    if expert_indices.numel() > 0 and (
        torch.any(expert_indices < 0) or torch.any(expert_indices >= n_experts)
    ):
        raise ValueError("expert_indices contains an expert outside [0, n_experts)")

    permutation = torch.argsort(expert_indices, stable=True)
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation.scatter_(
        0,
        permutation,
        torch.arange(permutation.numel(), device=permutation.device),
    )
    counts_per_expert = torch.bincount(expert_indices, minlength=n_experts)
    expert_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=expert_indices.device),
            counts_per_expert.cumsum(0),
        )
    )
    return ExpertMajorLayout(
        activations=activations.index_select(0, permutation),
        expert_indices=expert_indices.index_select(0, permutation),
        counts_per_expert=counts_per_expert,
        expert_offsets=expert_offsets,
        permutation=permutation,
        inverse_permutation=inverse_permutation,
    )


def swiglu_experts_expert_major_reference(
    activations: Tensor,
    expert_offsets: Tensor,
    expert_ids: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
) -> Tensor:
    """Evaluate contiguous expert segments and preserve expert-major row order."""

    integer_inputs = (expert_offsets, expert_ids)
    if any(tensor.dtype != torch.long for tensor in integer_inputs):
        raise TypeError("expert_offsets and expert_ids must use int64")
    if activations.ndim != 2:
        raise ValueError("activations must have shape [rows, model_dim]")
    if expert_ids.ndim != 1 or expert_offsets.shape != (expert_ids.numel() + 1,):
        raise ValueError("expert_offsets must have shape [number_of_expert_ids + 1]")
    if expert_w1.ndim != 3 or expert_w2.ndim != 3 or expert_w3.ndim != 3:
        raise ValueError("expert weights must be rank-3 tensors")
    local_experts, hidden, model_dim = expert_w1.shape
    if expert_w3.shape != (local_experts, hidden, model_dim):
        raise ValueError("expert_w3 shape must match expert_w1")
    if expert_w2.shape != (local_experts, model_dim, hidden):
        raise ValueError("expert_w2 must have shape [local_experts, model_dim, hidden]")
    if expert_ids.numel() != local_experts:
        raise ValueError("expert_ids and local expert weights must have the same length")
    if activations.shape[-1] != model_dim:
        raise ValueError("activation model dimension does not match expert weights")
    if any(
        tensor.device != activations.device
        for tensor in (*integer_inputs, expert_w1, expert_w2, expert_w3)
    ):
        raise ValueError("activations, metadata, and expert weights must share a device")
    if expert_offsets.numel() > 0 and (
        int(expert_offsets[0]) != 0
        or int(expert_offsets[-1]) != activations.shape[0]
        or torch.any(expert_offsets[1:] < expert_offsets[:-1])
    ):
        raise ValueError("expert_offsets must be monotonic and cover every activation row")

    output = activations * 0
    for local_index in range(local_experts):
        start = int(expert_offsets[local_index])
        end = int(expert_offsets[local_index + 1])
        if start == end:
            continue
        expert_output = swiglu_expert(
            activations[start:end],
            expert_w1[local_index],
            expert_w2[local_index],
            expert_w3[local_index],
        )
        output = torch.cat((output[:start], expert_output, output[end:]), dim=0)
    if torch.is_grad_enabled() and any(
        weight.requires_grad for weight in (expert_w1, expert_w2, expert_w3)
    ):
        parameter_zero = (
            expert_w1.reshape(-1)[:1].sum()
            + expert_w2.reshape(-1)[:1].sum()
            + expert_w3.reshape(-1)[:1].sum()
        ) * 0
        output = output + parameter_zero.to(output.dtype)
    return output


def swiglu_experts_padded_reference(
    activations: Tensor,
    expert_offsets: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
) -> Tensor:
    """Evaluate variable expert segments as three padded batched matrix multiplies."""

    if activations.ndim != 2:
        raise ValueError("activations must have shape [rows, model_dim]")
    if expert_offsets.dtype != torch.long or expert_offsets.ndim != 1:
        raise TypeError("expert_offsets must be an int64 vector")
    if expert_w1.ndim != 3 or expert_w2.ndim != 3 or expert_w3.ndim != 3:
        raise ValueError("expert weights must be rank-3 tensors")
    local_experts, hidden, model_dim = expert_w1.shape
    if expert_w3.shape != (local_experts, hidden, model_dim):
        raise ValueError("expert_w3 shape must match expert_w1")
    if expert_w2.shape != (local_experts, model_dim, hidden):
        raise ValueError("expert_w2 must have shape [local_experts, model_dim, hidden]")
    if expert_offsets.shape != (local_experts + 1,):
        raise ValueError("expert_offsets must have shape [local_experts + 1]")
    if activations.shape[-1] != model_dim:
        raise ValueError("activation model dimension does not match expert weights")
    if any(
        tensor.device != activations.device
        for tensor in (expert_offsets, expert_w1, expert_w2, expert_w3)
    ):
        raise ValueError("activations, offsets, and expert weights must share a device")
    if (
        int(expert_offsets[0]) != 0
        or int(expert_offsets[-1]) != activations.shape[0]
        or torch.any(expert_offsets[1:] < expert_offsets[:-1])
    ):
        raise ValueError("expert_offsets must be monotonic and cover every activation row")
    if local_experts == 0:
        parameter_zero = (expert_w1.sum() + expert_w2.sum() + expert_w3.sum()) * 0
        return activations * 0 + parameter_zero.to(activations.dtype)

    counts = expert_offsets[1:] - expert_offsets[:-1]
    capacity = int(counts.max())
    padded_rows = []
    for expert in range(local_experts):
        start = int(expert_offsets[expert])
        end = int(expert_offsets[expert + 1])
        rows = activations[start:end]
        padded_rows.append(F.pad(rows, (0, 0, 0, capacity - rows.shape[0])))
    padded = torch.stack(padded_rows)

    compute_dtype = _compute_dtype(activations)
    padded_compute = padded.to(compute_dtype)
    hidden_gate = torch.bmm(padded_compute, expert_w1.to(compute_dtype).transpose(1, 2))
    hidden_up = torch.bmm(padded_compute, expert_w3.to(compute_dtype).transpose(1, 2))
    hidden_state = F.silu(hidden_gate) * hidden_up
    if activations.dtype == torch.float16:
        hidden_state = hidden_state.to(activations.dtype).to(compute_dtype)
    expert_output = torch.bmm(
        hidden_state,
        expert_w2.to(compute_dtype).transpose(1, 2),
    ).to(activations.dtype)

    segments = [expert_output[expert, : int(counts[expert])] for expert in range(local_experts)]
    output = torch.cat(segments, dim=0)
    if torch.is_grad_enabled() and any(
        weight.requires_grad for weight in (expert_w1, expert_w2, expert_w3)
    ):
        parameter_zero = (
            expert_w1.reshape(-1)[:1].sum()
            + expert_w2.reshape(-1)[:1].sum()
            + expert_w3.reshape(-1)[:1].sum()
        ) * 0
        output = output + parameter_zero.to(output.dtype)
    return output


def deepseek_moe_reference(
    x: Tensor,
    gate_weight: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
    *,
    topk: int,
    n_groups: int = 1,
    topk_groups: int | None = None,
    score_func: str = "sigmoid",
    score_bias: Tensor | None = None,
    route_scale: float = 1.0,
    shared_w1: Tensor | None = None,
    shared_w2: Tensor | None = None,
    shared_w3: Tensor | None = None,
    return_routing: bool = False,
) -> Tensor | tuple[Tensor, RoutingResult]:
    """Unfused single-device DeepSeek-style MoE reference.

    Routed expert tensors use ``[experts, hidden, model_dim]`` for ``w1/w3``
    and ``[experts, model_dim, hidden]`` for ``w2``. Shared expert weights are
    optional matrices with the same per-expert convention.
    """

    if x.ndim < 2:
        raise ValueError("x must have shape [..., tokens, model_dim]")
    if expert_w1.ndim != 3 or expert_w2.ndim != 3 or expert_w3.ndim != 3:
        raise ValueError("routed expert weights must be rank-3 tensors")
    experts, hidden, model_dim = expert_w1.shape
    if x.shape[-1] != model_dim:
        raise ValueError("x model dimension does not match expert weights")
    if expert_w3.shape != (experts, hidden, model_dim):
        raise ValueError("expert_w3 shape must match expert_w1")
    if expert_w2.shape != (experts, model_dim, hidden):
        raise ValueError("expert_w2 must have shape [experts, model_dim, hidden]")
    if gate_weight.shape != (experts, model_dim):
        raise ValueError("gate_weight must have shape [experts, model_dim]")
    shared = (shared_w1, shared_w2, shared_w3)
    if any(weight is None for weight in shared) and not all(weight is None for weight in shared):
        raise ValueError("provide all three shared expert weights or none of them")

    original_shape = x.shape
    flat_x = x.reshape(-1, model_dim)
    routing = deepseek_grouped_topk(
        flat_x,
        gate_weight,
        topk=topk,
        n_groups=n_groups,
        topk_groups=topk_groups,
        score_func=score_func,
        score_bias=score_bias,
        route_scale=route_scale,
    )

    compute_dtype = _compute_dtype(x)
    routed_output = torch.zeros(
        flat_x.shape,
        dtype=compute_dtype,
        device=x.device,
    )
    for expert_id in range(experts):
        token_indices, slots = torch.where(routing.indices == expert_id)
        if token_indices.numel() == 0:
            continue
        expert_output = swiglu_expert(
            flat_x[token_indices],
            expert_w1[expert_id],
            expert_w2[expert_id],
            expert_w3[expert_id],
        ).to(compute_dtype)
        weighted = expert_output * routing.weights[token_indices, slots].to(
            compute_dtype
        ).unsqueeze(-1)
        routed_output = routed_output.index_add(0, token_indices, weighted)

    if shared_w1 is not None and shared_w2 is not None and shared_w3 is not None:
        routed_output = routed_output + swiglu_expert(flat_x, shared_w1, shared_w2, shared_w3).to(
            compute_dtype
        )

    output = routed_output.to(x.dtype).reshape(original_shape)
    if return_routing:
        routing_shape = (*original_shape[:-1], topk)
        shaped_routing = RoutingResult(
            routing.weights.reshape(routing_shape), routing.indices.reshape(routing_shape)
        )
        return output, shaped_routing
    return output


def deepseek_moe_packed_reference(
    x: Tensor,
    gate_weight: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
    *,
    topk: int,
    n_groups: int = 1,
    topk_groups: int | None = None,
    score_func: str = "sigmoid",
    score_bias: Tensor | None = None,
    route_scale: float = 1.0,
    shared_w1: Tensor | None = None,
    shared_w2: Tensor | None = None,
    shared_w3: Tensor | None = None,
    expert_owner: Tensor | None = None,
    world_size: int | None = None,
    return_packed: bool = False,
) -> Tensor | tuple[Tensor, PackedRoutes]:
    """Evaluate MoE through the pack/expert/combine stages used by EP."""

    if x.ndim < 2 or expert_w1.ndim != 3:
        raise ValueError("x must be [..., D] and expert_w1 must be [experts, hidden, D]")
    experts, hidden, model_dim = expert_w1.shape
    if expert_w3.shape != (experts, hidden, model_dim):
        raise ValueError("expert_w3 shape must match expert_w1")
    if expert_w2.shape != (experts, model_dim, hidden):
        raise ValueError("expert_w2 must have shape [experts, model_dim, hidden]")
    if gate_weight.shape != (experts, model_dim):
        raise ValueError("gate_weight must have shape [experts, model_dim]")
    shared = (shared_w1, shared_w2, shared_w3)
    if any(weight is None for weight in shared) and not all(weight is None for weight in shared):
        raise ValueError("provide all three shared expert weights or none of them")

    routing = deepseek_grouped_topk(
        x,
        gate_weight,
        topk=topk,
        n_groups=n_groups,
        topk_groups=topk_groups,
        score_func=score_func,
        score_bias=score_bias,
        route_scale=route_scale,
    )
    packed = pack_routes_reference(
        x,
        routing,
        n_experts=experts,
        expert_owner=expert_owner,
        world_size=world_size,
    )

    contributions = torch.empty_like(packed.activations)
    for segment, expert_id_tensor in enumerate(packed.expert_order):
        start = int(packed.expert_offsets[segment])
        end = int(packed.expert_offsets[segment + 1])
        if start == end:
            continue
        expert_id = int(expert_id_tensor)
        contributions[start:end] = swiglu_expert(
            packed.activations[start:end],
            expert_w1[expert_id],
            expert_w2[expert_id],
            expert_w3[expert_id],
        )

    output = combine_packed_routes(contributions, packed)
    if shared_w1 is not None and shared_w2 is not None and shared_w3 is not None:
        output = output + swiglu_expert(x, shared_w1, shared_w2, shared_w3)
    return (output, packed) if return_packed else output
