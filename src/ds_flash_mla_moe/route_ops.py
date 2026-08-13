"""Dispatch wrappers for MoE route packing and combination primitives."""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from torch import Tensor

from .moe import RoutingResult, pack_routes_reference
from .ops import _operator_has_cuda_kernel

RouteBackend = Literal["auto", "cuda", "reference"]


class RoutePackResult(NamedTuple):
    """Packed rows and metadata shared by reference and CUDA backends."""

    activations: Tensor
    route_weights: Tensor
    token_indices: Tensor
    slot_indices: Tensor
    expert_indices: Tensor
    counts_per_expert: Tensor
    rank_counts: Tensor


def _validate_route_pack_inputs(
    x: Tensor,
    route_weights: Tensor,
    expert_indices: Tensor,
    expert_owner: Tensor,
    world_size: int,
) -> None:
    if x.ndim != 2:
        raise ValueError("x must have shape [tokens, model_dim]")
    if x.shape[-1] <= 0:
        raise ValueError("model_dim must be positive")
    if route_weights.ndim != 2 or expert_indices.ndim != 2:
        raise ValueError("route weights and expert indices must have shape [tokens, topk]")
    if route_weights.shape != expert_indices.shape:
        raise ValueError("route weights and expert indices must have identical shapes")
    if x.shape[0] != route_weights.shape[0]:
        raise ValueError("x and routing tensors must have the same token count")
    if route_weights.shape[1] <= 0:
        raise ValueError("topk must be positive")
    if expert_owner.ndim != 1 or expert_owner.numel() <= 0:
        raise ValueError("expert_owner must be a non-empty vector")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not route_weights.is_floating_point():
        raise TypeError("route_weights must use a floating-point dtype")
    if expert_indices.dtype != torch.long or expert_owner.dtype != torch.long:
        raise TypeError("expert_indices and expert_owner must use int64")
    if not (x.device == route_weights.device == expert_indices.device == expert_owner.device):
        raise ValueError("all route-pack tensors must be on the same device")


def _validate_route_pack_values(
    expert_indices: Tensor,
    expert_owner: Tensor,
    world_size: int,
) -> None:
    if expert_indices.numel() > 0 and (
        torch.any(expert_indices < 0) or torch.any(expert_indices >= expert_owner.numel())
    ):
        raise ValueError("expert_indices contains an expert outside [0, experts)")
    if torch.any(expert_owner < 0) or torch.any(expert_owner >= world_size):
        raise ValueError("expert_owner contains a rank outside world_size")


def _cuda_pack_ineligibility_reason(
    x: Tensor,
    route_weights: Tensor,
    expert_indices: Tensor,
    expert_owner: Tensor,
) -> str | None:
    if x.device.type != "cuda":
        return "route-pack tensors must be CUDA tensors"
    if x.dtype != torch.float32 or route_weights.dtype != torch.float32:
        return "the CUDA route pack currently supports float32 activations and weights only"
    if expert_indices.dtype != torch.long or expert_owner.dtype != torch.long:
        return "the CUDA route pack requires int64 indices and ownership"
    if not all(
        tensor.is_contiguous() for tensor in (x, route_weights, expert_indices, expert_owner)
    ):
        return "the CUDA route pack requires contiguous tensors"
    if not _operator_has_cuda_kernel("route_pack"):
        return "the loaded native extension does not register a CUDA route-pack kernel"
    return None


def route_pack(
    x: Tensor,
    route_weights: Tensor,
    expert_indices: Tensor,
    expert_owner: Tensor,
    *,
    world_size: int,
    backend: RouteBackend = "auto",
) -> RoutePackResult:
    """Pack unweighted activations in destination-rank/expert order."""

    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be auto, cuda, or reference")
    _validate_route_pack_inputs(
        x,
        route_weights,
        expert_indices,
        expert_owner,
        world_size,
    )
    reason = _cuda_pack_ineligibility_reason(
        x,
        route_weights,
        expert_indices,
        expert_owner,
    )
    if backend == "cuda" and reason is not None:
        raise RuntimeError(f"CUDA route pack is unavailable: {reason}")

    use_native = backend == "cuda" or (backend == "auto" and reason is None)
    if torch.are_deterministic_algorithms_enabled():
        use_native = False
        if backend == "cuda":
            raise RuntimeError("CUDA route pack uses atomic row assignment and is nondeterministic")
    if use_native:
        (
            activations,
            packed_weights,
            route_indices,
            packed_experts,
            counts_per_expert,
            rank_counts,
        ) = torch.ops.ds_flash_mla_moe.route_pack.default(
            x,
            route_weights,
            expert_indices,
            expert_owner,
            world_size,
        )
        topk = route_weights.shape[1]
        return RoutePackResult(
            activations=activations,
            route_weights=packed_weights,
            token_indices=torch.div(route_indices, topk, rounding_mode="floor"),
            slot_indices=torch.remainder(route_indices, topk),
            expert_indices=packed_experts,
            counts_per_expert=counts_per_expert,
            rank_counts=rank_counts,
        )

    _validate_route_pack_values(expert_indices, expert_owner, world_size)
    packed = pack_routes_reference(
        x,
        RoutingResult(route_weights, expert_indices),
        n_experts=expert_owner.numel(),
        expert_owner=expert_owner,
        world_size=world_size,
    )
    return RoutePackResult(
        activations=packed.activations,
        route_weights=packed.route_weights,
        token_indices=packed.token_indices,
        slot_indices=packed.slot_indices,
        expert_indices=packed.expert_indices,
        counts_per_expert=packed.counts_per_expert,
        rank_counts=packed.rank_counts,
    )


def _validate_route_combine_inputs(
    contributions: Tensor,
    route_weights: Tensor,
    token_indices: Tensor,
    token_count: int,
) -> None:
    if contributions.ndim != 2:
        raise ValueError("contributions must have shape [rows, model_dim]")
    if contributions.shape[-1] <= 0:
        raise ValueError("model_dim must be positive")
    if route_weights.ndim != 1 or token_indices.ndim != 1:
        raise ValueError("route_weights and token_indices must be vectors")
    if (
        contributions.shape[0] != route_weights.numel()
        or contributions.shape[0] != token_indices.numel()
    ):
        raise ValueError("route-combine row counts must match")
    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    if not route_weights.is_floating_point() or not contributions.is_floating_point():
        raise TypeError("contributions and route_weights must use floating-point dtypes")
    if token_indices.dtype != torch.long:
        raise TypeError("token_indices must use int64")
    if not (contributions.device == route_weights.device == token_indices.device):
        raise ValueError("all route-combine tensors must be on the same device")


def _validate_route_combine_values(token_indices: Tensor, token_count: int) -> None:
    if token_indices.numel() > 0 and (
        torch.any(token_indices < 0) or torch.any(token_indices >= token_count)
    ):
        raise ValueError("token_indices contains a token outside [0, token_count)")


def _cuda_combine_ineligibility_reason(
    contributions: Tensor,
    route_weights: Tensor,
    token_indices: Tensor,
) -> str | None:
    if contributions.device.type != "cuda":
        return "route-combine tensors must be CUDA tensors"
    if contributions.dtype != torch.float32 or route_weights.dtype != torch.float32:
        return "the CUDA route combine currently supports float32 only"
    if token_indices.dtype != torch.long:
        return "the CUDA route combine requires int64 token indices"
    if not all(tensor.is_contiguous() for tensor in (contributions, route_weights, token_indices)):
        return "the CUDA route combine requires contiguous tensors"
    if not _operator_has_cuda_kernel("route_combine"):
        return "the loaded native extension does not register a CUDA route-combine kernel"
    return None


def _route_combine_reference(
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


def route_combine(
    contributions: Tensor,
    route_weights: Tensor,
    token_indices: Tensor,
    *,
    token_count: int,
    backend: RouteBackend = "auto",
) -> Tensor:
    """Apply route weights after expert compute and accumulate by source token."""

    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be auto, cuda, or reference")
    _validate_route_combine_inputs(
        contributions,
        route_weights,
        token_indices,
        token_count,
    )
    reason = _cuda_combine_ineligibility_reason(
        contributions,
        route_weights,
        token_indices,
    )
    if backend == "cuda" and reason is not None:
        raise RuntimeError(f"CUDA route combine is unavailable: {reason}")
    use_native = backend == "cuda" or (backend == "auto" and reason is None)
    if torch.are_deterministic_algorithms_enabled():
        use_native = False
        if backend == "cuda":
            raise RuntimeError(
                "CUDA route combine uses atomic accumulation and is nondeterministic"
            )
    if use_native:
        return torch.ops.ds_flash_mla_moe.route_combine.default(
            contributions,
            route_weights,
            token_indices,
            token_count,
        )

    _validate_route_combine_values(token_indices, token_count)
    if torch.are_deterministic_algorithms_enabled() and contributions.device.type == "cuda":
        compute_dtype = torch.float64 if contributions.dtype == torch.float64 else torch.float32
        assignment = torch.nn.functional.one_hot(
            token_indices,
            num_classes=token_count,
        ).to(compute_dtype)
        weighted = contributions.to(compute_dtype) * route_weights.to(compute_dtype).unsqueeze(-1)
        return (assignment.transpose(0, 1) @ weighted).to(contributions.dtype)
    return _route_combine_reference(contributions, route_weights, token_indices, token_count)


def cuda_route_ops_available() -> bool:
    """Return whether both native CUDA route primitives are usable."""

    return (
        torch.cuda.is_available()
        and _operator_has_cuda_kernel("route_pack")
        and _operator_has_cuda_kernel("route_combine")
    )
