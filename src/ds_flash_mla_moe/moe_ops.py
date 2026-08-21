"""Whole-layer dispatch for the single-device DeepSeek-style MoE."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from .moe import deepseek_moe_packed_reference
from .ops import _operator_has_cuda_kernel

MoEBackend = Literal["auto", "cuda", "reference"]
MoEScoreFunction = Literal["sigmoid", "softmax"]


def _validate_moe_inputs(
    x: Tensor,
    gate_weight: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
    *,
    topk: int,
    n_groups: int,
    topk_groups: int | None,
    score_func: str,
    score_bias: Tensor | None,
    route_scale: float,
) -> int:
    if x.ndim < 2:
        raise ValueError("x must have rank at least 2 and end in model_dim")
    if gate_weight.ndim != 2:
        raise ValueError("gate_weight must have shape [experts, model_dim]")
    if expert_w1.ndim != 3 or expert_w2.ndim != 3 or expert_w3.ndim != 3:
        raise ValueError("routed expert weights must be rank-3 tensors")

    experts, model_dim = gate_weight.shape
    routed_experts, hidden, expert_model_dim = expert_w1.shape
    if model_dim <= 0:
        raise ValueError("model_dim must be positive")
    if hidden <= 0:
        raise ValueError("hidden must be positive")
    if x.shape[-1] != model_dim:
        raise ValueError("x model dimension does not match gate_weight")
    if routed_experts != experts or expert_model_dim != model_dim:
        raise ValueError("expert_w1 must have shape [experts, hidden, model_dim]")
    if expert_w3.shape != (experts, hidden, model_dim):
        raise ValueError("expert_w3 shape must match expert_w1")
    if expert_w2.shape != (experts, model_dim, hidden):
        raise ValueError("expert_w2 must have shape [experts, model_dim, hidden]")
    if score_bias is not None and score_bias.shape != (experts,):
        raise ValueError("score_bias must have shape [number_of_experts]")

    tensors = (x, gate_weight, expert_w1, expert_w2, expert_w3)
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise TypeError("all MoE tensors must use a floating-point dtype")
    if any(tensor.dtype != x.dtype for tensor in tensors):
        raise ValueError("all MoE tensors must share x's dtype")
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all MoE tensors must share x's device")
    if score_bias is not None:
        if not score_bias.is_floating_point():
            raise TypeError("score_bias must use a floating-point dtype")
        if score_bias.dtype != x.dtype or score_bias.device != x.device:
            raise ValueError("score_bias must share x's dtype and device")

    if experts <= 0 or n_groups <= 0 or experts % n_groups:
        raise ValueError("number of experts must be positive and divisible by n_groups")
    effective_topk_groups = n_groups if topk_groups is None else topk_groups
    if not 1 <= effective_topk_groups <= n_groups:
        raise ValueError("topk_groups must be in [1, n_groups]")
    if not 1 <= topk <= experts:
        raise ValueError("topk must be in [1, number_of_experts]")
    if topk > effective_topk_groups * (experts // n_groups):
        raise ValueError("topk exceeds the number of experts retained by group selection")
    if score_func not in {"sigmoid", "softmax"}:
        raise ValueError("score_func must be sigmoid or softmax")
    if not math.isfinite(route_scale):
        raise ValueError("route_scale must be finite")
    return effective_topk_groups


def _cuda_moe_ineligibility_reason(
    x: Tensor,
    gate_weight: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
    score_func: str,
    score_bias: Tensor | None,
) -> str | None:
    tensors = (x, gate_weight, expert_w1, expert_w2, expert_w3)
    floating_tensors = tensors if score_bias is None else (*tensors, score_bias)
    if any(tensor.requires_grad for tensor in floating_tensors):
        return "the CUDA whole-layer operator is forward-only for requires_grad tensors"
    if any(tensor.device.type != "cuda" for tensor in floating_tensors):
        return "all MoE tensors must be CUDA tensors"
    if x.dtype != torch.float32:
        return "the CUDA whole-layer operator currently supports float32 only"
    if score_func != "sigmoid":
        return "the CUDA whole-layer operator currently supports sigmoid scores only"
    if not all(tensor.is_contiguous() for tensor in floating_tensors):
        return "the CUDA whole-layer operator requires contiguous tensors"
    if torch.are_deterministic_algorithms_enabled():
        return "deterministic algorithms are enabled"
    if not _operator_has_cuda_kernel("deepseek_moe_forward"):
        return "the loaded native extension does not register a CUDA DeepSeek MoE forward"
    return None


def _call_cuda_moe(
    x: Tensor,
    gate_weight: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
    *,
    topk: int,
    n_groups: int,
    topk_groups: int,
    score_bias: Tensor | None,
    route_scale: float,
) -> Tensor:
    original_shape = x.shape
    output = torch.ops.ds_flash_mla_moe.deepseek_moe_forward.default(
        x.reshape(-1, x.shape[-1]),
        gate_weight,
        expert_w1,
        expert_w2,
        expert_w3,
        topk,
        n_groups,
        topk_groups,
        score_bias,
        route_scale,
    )
    return output.reshape(original_shape).contiguous()


def deepseek_moe_forward(
    x: Tensor,
    gate_weight: Tensor,
    expert_w1: Tensor,
    expert_w2: Tensor,
    expert_w3: Tensor,
    *,
    topk: int,
    n_groups: int = 1,
    topk_groups: int | None = None,
    score_func: MoEScoreFunction = "sigmoid",
    score_bias: Tensor | None = None,
    route_scale: float = 1.0,
    backend: MoEBackend = "auto",
) -> Tensor:
    """Evaluate a DeepSeek-style MoE through one strict backend selection."""

    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be auto, cuda, or reference")
    effective_topk_groups = _validate_moe_inputs(
        x,
        gate_weight,
        expert_w1,
        expert_w2,
        expert_w3,
        topk=topk,
        n_groups=n_groups,
        topk_groups=topk_groups,
        score_func=score_func,
        score_bias=score_bias,
        route_scale=route_scale,
    )
    if backend == "reference":
        return deepseek_moe_packed_reference(
            x,
            gate_weight,
            expert_w1,
            expert_w2,
            expert_w3,
            topk=topk,
            n_groups=n_groups,
            topk_groups=effective_topk_groups,
            score_func=score_func,
            score_bias=score_bias,
            route_scale=route_scale,
        ).contiguous()

    reason = _cuda_moe_ineligibility_reason(
        x,
        gate_weight,
        expert_w1,
        expert_w2,
        expert_w3,
        score_func,
        score_bias,
    )
    if backend == "cuda" and reason is not None:
        raise RuntimeError(f"CUDA DeepSeek MoE is unavailable: {reason}")
    if reason is None:
        return _call_cuda_moe(
            x,
            gate_weight,
            expert_w1,
            expert_w2,
            expert_w3,
            topk=topk,
            n_groups=n_groups,
            topk_groups=effective_topk_groups,
            score_bias=score_bias,
            route_scale=route_scale,
        )
    return deepseek_moe_packed_reference(
        x,
        gate_weight,
        expert_w1,
        expert_w2,
        expert_w3,
        topk=topk,
        n_groups=n_groups,
        topk_groups=effective_topk_groups,
        score_func=score_func,
        score_bias=score_bias,
        route_scale=route_scale,
    ).contiguous()


def cuda_moe_available() -> bool:
    """Return whether the native whole-layer CUDA operator is available."""

    return bool(torch.cuda.is_available() and _operator_has_cuda_kernel("deepseek_moe_forward"))
