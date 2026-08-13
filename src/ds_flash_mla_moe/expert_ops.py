"""Dispatch wrapper for expert-major SwiGLU compute."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from .moe import swiglu_experts_padded_reference
from .ops import _operator_has_cuda_kernel

ExpertBackend = Literal["auto", "cuda", "reference"]


def expert_major_pack(
    activations: Tensor,
    expert_indices: Tensor,
    local_expert_ids: Tensor,
    *,
    backend: ExpertBackend = "auto",
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack arbitrary rows by local expert and return an inverse permutation."""

    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be auto, cuda, or reference")
    if activations.ndim != 2:
        raise ValueError("activations must have shape [rows, model_dim]")
    if expert_indices.ndim != 1 or local_expert_ids.ndim != 1:
        raise ValueError("expert_indices and local_expert_ids must be vectors")
    if expert_indices.dtype != torch.long or local_expert_ids.dtype != torch.long:
        raise TypeError("expert_indices and local_expert_ids must use int64")
    if expert_indices.shape[0] != activations.shape[0]:
        raise ValueError("expert_indices must contain one id per activation row")
    if not (activations.device == expert_indices.device == local_expert_ids.device):
        raise ValueError("activations and expert indices must share a device")
    reason = None
    tensors = (activations, expert_indices, local_expert_ids)
    if activations.device.type != "cuda":
        reason = "expert-major pack tensors must be CUDA tensors"
    elif activations.dtype not in {torch.float16, torch.float32}:
        reason = "the CUDA expert-major pack currently supports float16 and float32 only"
    elif not all(tensor.is_contiguous() for tensor in tensors):
        reason = "the CUDA expert-major pack requires contiguous tensors"
    elif not _operator_has_cuda_kernel("expert_major_pack"):
        reason = "the loaded native extension does not register a CUDA expert-major pack"
    if torch.are_deterministic_algorithms_enabled() and reason is None:
        reason = "the CUDA expert-major pack uses nondeterministic atomic row assignment"
    if backend == "cuda" and reason is not None:
        raise RuntimeError(f"CUDA expert-major pack is unavailable: {reason}")
    if backend == "cuda" or (backend == "auto" and reason is None):
        return torch.ops.ds_flash_mla_moe.expert_major_pack.default(
            activations,
            expert_indices,
            local_expert_ids,
        )
    return _composite_expert_major_pack_reference(
        activations,
        expert_indices,
        local_expert_ids,
    )


def _composite_expert_major_pack_reference(
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
    matches = expert_indices.unsqueeze(1) == local_expert_ids.unsqueeze(0)
    if expert_indices.numel() and torch.any(matches.sum(dim=1) != 1):
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


def _validate_expert_major_inputs(
    activations: Tensor,
    expert_offsets: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
) -> None:
    if activations.ndim != 2:
        raise ValueError("activations must have shape [rows, model_dim]")
    if expert_offsets.ndim != 1 or expert_offsets.dtype != torch.long:
        raise TypeError("expert_offsets must be an int64 vector")
    if expert_w1.ndim != 3 or expert_w2.ndim != 3 or expert_w3.ndim != 3:
        raise ValueError("expert weights must be rank-3 tensors")
    experts, hidden_dim, model_dim = expert_w1.shape
    if hidden_dim <= 0 or model_dim <= 0:
        raise ValueError("hidden_dim and model_dim must be positive")
    if activations.shape[1] != model_dim:
        raise ValueError("activation model dimension does not match expert weights")
    if expert_w3.shape != expert_w1.shape:
        raise ValueError("expert_w3 shape must match expert_w1")
    if expert_w2.shape != (experts, model_dim, hidden_dim):
        raise ValueError("expert_w2 must have shape [experts, model_dim, hidden_dim]")
    if expert_offsets.shape != (experts + 1,):
        raise ValueError("expert_offsets must have shape [experts + 1]")
    tensors = (activations, expert_offsets, expert_w1, expert_w2, expert_w3)
    if any(tensor.device != activations.device for tensor in tensors):
        raise ValueError("activations, offsets, and weights must share a device")
    if not activations.is_floating_point() or any(
        not weight.is_floating_point() for weight in (expert_w1, expert_w2, expert_w3)
    ):
        raise TypeError("activations and expert weights must use floating-point dtypes")
    if any(weight.dtype != activations.dtype for weight in (expert_w1, expert_w2, expert_w3)):
        raise TypeError("activations and expert weights must share a dtype")


def _cuda_ineligibility_reason(
    activations: Tensor,
    expert_offsets: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
) -> str | None:
    tensors = (activations, expert_offsets, expert_w1, expert_w2, expert_w3)
    if activations.device.type != "cuda":
        return "expert tensors must be CUDA tensors"
    if activations.dtype not in {torch.float16, torch.float32}:
        return "the CUDA expert kernel currently supports float16 and float32 only"
    if not all(tensor.is_contiguous() for tensor in tensors):
        return "the CUDA expert kernel requires contiguous tensors"
    if not _operator_has_cuda_kernel("swiglu_experts"):
        return "the loaded native extension does not register a CUDA expert kernel"
    return None


def swiglu_experts_expert_major(
    activations: Tensor,
    expert_offsets: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
    *,
    backend: ExpertBackend = "auto",
) -> Tensor:
    """Evaluate contiguous expert segments with native CUDA when eligible.

    ``expert_offsets`` maps each local expert weight index to a contiguous row
    interval. The native path computes only active rows and does not pad or
    drop routes.
    """

    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be auto, cuda, or reference")
    _validate_expert_major_inputs(
        activations,
        expert_offsets,
        expert_w1,
        expert_w2,
        expert_w3,
    )
    reason = _cuda_ineligibility_reason(
        activations,
        expert_offsets,
        expert_w1,
        expert_w2,
        expert_w3,
    )
    if backend == "cuda" and reason is not None:
        raise RuntimeError(f"CUDA expert compute is unavailable: {reason}")
    if backend == "cuda" or (backend == "auto" and reason is None):
        return torch.ops.ds_flash_mla_moe.swiglu_experts.default(
            activations,
            expert_offsets,
            expert_w1,
            expert_w2,
            expert_w3,
        )
    return swiglu_experts_padded_reference(
        activations,
        expert_offsets,
        expert_w1,
        expert_w2,
        expert_w3,
    )


def cuda_expert_ops_available() -> bool:
    """Return whether native expert-major pack and compute kernels can execute."""

    return (
        torch.cuda.is_available()
        and _operator_has_cuda_kernel("expert_major_pack")
        and _operator_has_cuda_kernel("swiglu_experts")
    )
