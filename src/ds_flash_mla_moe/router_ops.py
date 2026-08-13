"""Dispatch wrapper for DeepSeek-style group-limited Top-K routing."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from .moe import RoutingResult, deepseek_grouped_topk
from .ops import _operator_has_cuda_kernel

RouterBackend = Literal["auto", "cuda", "reference"]


def _validate_router_inputs(
    x: Tensor,
    gate_weight: Tensor,
    *,
    topk: int,
    n_groups: int,
    topk_groups: int | None,
    score_func: str,
    score_bias: Tensor | None,
    route_scale: float,
) -> int:
    if x.ndim < 1 or gate_weight.ndim != 2:
        raise ValueError("x must end in model_dim and gate_weight must be [experts, model_dim]")
    experts, model_dim = gate_weight.shape
    if x.shape[-1] != model_dim:
        raise ValueError("x model dimension does not match gate_weight")
    if x.device != gate_weight.device or x.dtype != gate_weight.dtype:
        raise ValueError("x and gate_weight must share a device and dtype")
    if not x.is_floating_point() or not gate_weight.is_floating_point():
        raise TypeError("x and gate_weight must use a floating-point dtype")
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
    if score_bias is not None:
        if score_bias.shape != (experts,):
            raise ValueError("score_bias must have shape [number_of_experts]")
        if score_bias.device != x.device or score_bias.dtype != x.dtype:
            raise ValueError("score_bias must share x's device and dtype")
    if not math.isfinite(route_scale):
        raise ValueError("route_scale must be finite")
    return effective_topk_groups


def _cuda_ineligibility_reason(
    x: Tensor,
    gate_weight: Tensor,
    score_func: str,
    score_bias: Tensor | None,
) -> str | None:
    tensors = (x, gate_weight) if score_bias is None else (x, gate_weight, score_bias)
    if x.device.type != "cuda":
        return "router tensors must be CUDA tensors"
    if x.dtype != torch.float32:
        return "the CUDA grouped router currently supports float32 only"
    if score_func != "sigmoid":
        return "the CUDA grouped router currently supports sigmoid scores only"
    if not all(tensor.is_contiguous() for tensor in tensors):
        return "the CUDA grouped router requires contiguous tensors"
    if not _operator_has_cuda_kernel("grouped_topk"):
        return "the loaded native extension does not register a CUDA grouped router"
    return None


def grouped_topk(
    x: Tensor,
    gate_weight: Tensor,
    *,
    topk: int,
    n_groups: int = 1,
    topk_groups: int | None = None,
    score_func: str = "sigmoid",
    score_bias: Tensor | None = None,
    route_scale: float = 1.0,
    backend: RouterBackend = "auto",
) -> RoutingResult:
    """Route tokens with native CUDA selection when its narrow contract holds."""

    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be auto, cuda, or reference")
    effective_topk_groups = _validate_router_inputs(
        x,
        gate_weight,
        topk=topk,
        n_groups=n_groups,
        topk_groups=topk_groups,
        score_func=score_func,
        score_bias=score_bias,
        route_scale=route_scale,
    )
    reason = _cuda_ineligibility_reason(x, gate_weight, score_func, score_bias)
    if backend == "cuda" and reason is not None:
        raise RuntimeError(f"CUDA grouped router is unavailable: {reason}")
    if backend == "cuda" or (backend == "auto" and reason is None):
        original_shape = x.shape[:-1]
        weights, indices = torch.ops.ds_flash_mla_moe.grouped_topk.default(
            x.reshape(-1, x.shape[-1]),
            gate_weight,
            topk,
            n_groups,
            effective_topk_groups,
            score_bias,
            route_scale,
        )
        route_shape = (*original_shape, topk)
        return RoutingResult(weights.reshape(route_shape), indices.reshape(route_shape))
    return deepseek_grouped_topk(
        x,
        gate_weight,
        topk=topk,
        n_groups=n_groups,
        topk_groups=effective_topk_groups,
        score_func=score_func,
        score_bias=score_bias,
        route_scale=route_scale,
    )


def cuda_router_available() -> bool:
    """Return whether the native CUDA grouped router can execute."""

    return torch.cuda.is_available() and _operator_has_cuda_kernel("grouped_topk")
