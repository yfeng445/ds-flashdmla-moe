"""Correctness-first GEMM specifications used by the CUDA teaching chapters."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _validate_gemm_inputs(
    a: Tensor,
    b: Tensor,
    c: Tensor | None,
    *,
    alpha: float,
    beta: float,
) -> tuple[int, int, int]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("GEMM inputs must be rank-2 matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError("GEMM inner dimensions must match")
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("GEMM inputs must share device and dtype")
    if not a.dtype.is_floating_point:
        raise TypeError("GEMM inputs must use a floating-point dtype")
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError("alpha and beta must be finite")

    m, k = a.shape
    n = b.shape[1]
    if c is None:
        if beta != 0.0:
            raise ValueError("a nonzero beta requires an epilogue matrix c")
    elif c.shape != (m, n):
        raise ValueError("GEMM epilogue matrix c must have shape [m, n]")
    elif c.device != a.device or c.dtype != a.dtype:
        raise ValueError("GEMM epilogue matrix c must share device and dtype")
    return m, n, k


def _gemm_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def gemm_reference(
    a: Tensor,
    b: Tensor,
    c: Tensor | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Tensor:
    """Evaluate ``alpha * (a @ b) + beta * c`` with widened accumulation."""

    _validate_gemm_inputs(a, b, c, alpha=alpha, beta=beta)
    compute_dtype = _gemm_compute_dtype(a.dtype)
    result = alpha * (a.to(compute_dtype) @ b.to(compute_dtype))
    if c is not None:
        result = result + beta * c.to(compute_dtype)
    return result.to(a.dtype)


def tiled_gemm_reference(
    a: Tensor,
    b: Tensor,
    c: Tensor | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    tile_m: int = 32,
    tile_n: int = 32,
    tile_k: int = 32,
) -> Tensor:
    """Evaluate GEMM as explicit M/N/K tiles without assuming divisible shapes.

    This function exposes the loop structure of a shared-memory CUDA GEMM while
    remaining an ordinary differentiable PyTorch specification. It does not model
    thread-level execution and is not intended to outperform ``torch.matmul``.
    """

    m, n, k = _validate_gemm_inputs(a, b, c, alpha=alpha, beta=beta)
    if min(tile_m, tile_n, tile_k) <= 0:
        raise ValueError("GEMM tile dimensions must be positive")

    compute_dtype = _gemm_compute_dtype(a.dtype)
    a_compute = a.to(compute_dtype)
    b_compute = b.to(compute_dtype)
    c_compute = c.to(compute_dtype) if c is not None else None
    anchor = (a_compute.sum() + b_compute.sum()) * 0.0
    if c_compute is not None:
        anchor = anchor + c_compute.sum() * 0.0

    if m == 0 or n == 0:
        return (torch.zeros((m, n), dtype=compute_dtype, device=a.device) + anchor).to(a.dtype)

    output_rows = []
    for row_start in range(0, m, tile_m):
        row_end = min(row_start + tile_m, m)
        output_columns = []
        for column_start in range(0, n, tile_n):
            column_end = min(column_start + tile_n, n)
            accumulator = (
                torch.zeros(
                    (row_end - row_start, column_end - column_start),
                    dtype=compute_dtype,
                    device=a.device,
                )
                + anchor
            )
            for reduction_start in range(0, k, tile_k):
                reduction_end = min(reduction_start + tile_k, k)
                accumulator = accumulator + (
                    a_compute[row_start:row_end, reduction_start:reduction_end]
                    @ b_compute[reduction_start:reduction_end, column_start:column_end]
                )
            tile = alpha * accumulator
            if c_compute is not None:
                tile = (
                    tile
                    + beta
                    * c_compute[
                        row_start:row_end,
                        column_start:column_end,
                    ]
                )
            output_columns.append(tile)
        output_rows.append(torch.cat(output_columns, dim=1))
    return torch.cat(output_rows, dim=0).to(a.dtype)
