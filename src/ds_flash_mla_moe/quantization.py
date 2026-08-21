"""Forward-only FP8 E4M3FN and symmetric INT8 matrix experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from . import ops as _ops

QuantizationFormat = Literal["fp8_e4m3fn", "int8"]
QuantizationBackend = Literal["auto", "cuda", "reference"]
ScaleGranularity = Literal["per_row", "per_output_channel"]

_FORMATS: dict[QuantizationFormat, tuple[torch.dtype, float]] = {
    "fp8_e4m3fn": (torch.uint8, 448.0),
    "int8": (torch.int8, 127.0),
}
_FORMAT_GRANULARITIES: dict[QuantizationFormat, frozenset[str]] = {
    "fp8_e4m3fn": frozenset({"per_row", "per_output_channel"}),
    "int8": frozenset({"per_row", "per_output_channel"}),
}
_QUANTIZE_OPERATOR: dict[QuantizationFormat, str] = {
    "fp8_e4m3fn": "quantize_fp8_e4m3fn_per_row",
    "int8": "quantize_int8_per_row",
}
_LINEAR_OPERATOR: dict[QuantizationFormat, str] = {
    "fp8_e4m3fn": "dequantized_linear_fp8_e4m3fn",
    "int8": "dequantized_linear_int8",
}


@dataclass(frozen=True)
class QuantizedMatrixMetadata:
    """Immutable semantics for one row-major quantized matrix.

    Scales are indexed along axis 0 and reduce axis 1. For activations this is
    called per-row scaling; for a conventional ``[out_features, in_features]``
    linear weight it is called per-output-channel scaling.
    """

    format: QuantizationFormat
    scale_granularity: ScaleGranularity
    shape: tuple[int, int]
    source_dtype: torch.dtype
    value_dtype: torch.dtype
    scale_dtype: torch.dtype
    scale_index_axis: int
    scale_reduction_axis: int
    accumulator_dtype: torch.dtype
    layout: Literal["row_major_contiguous"]
    quantized_min: float
    quantized_max: float


@dataclass(frozen=True, eq=False)
class QuantizedMatrix:
    """Quantized payload and FP32 scales carrying immutable explicit metadata."""

    values: Tensor
    scales: Tensor
    metadata: QuantizedMatrixMetadata

    def __post_init__(self) -> None:
        _validate_quantized_matrix(self)


def _format_spec(format: QuantizationFormat) -> tuple[torch.dtype, float]:
    if format not in _FORMATS:
        raise ValueError("format must be 'fp8_e4m3fn' or 'int8'")
    return _FORMATS[format]


def _validate_quantized_matrix(matrix: QuantizedMatrix) -> None:
    metadata = matrix.metadata
    expected_dtype, bound = _format_spec(metadata.format)
    if metadata.scale_granularity not in _FORMAT_GRANULARITIES[metadata.format]:
        raise ValueError("metadata scale granularity is unsupported for the quantization format")
    if metadata.source_dtype != torch.float32:
        raise ValueError("metadata source dtype must be float32")
    if len(metadata.shape) != 2 or metadata.shape[0] <= 0 or metadata.shape[1] <= 0:
        raise ValueError("metadata shape must contain two positive matrix dimensions")
    if metadata.value_dtype != expected_dtype:
        raise ValueError("metadata value dtype does not match the quantization format")
    if metadata.scale_dtype != torch.float32:
        raise ValueError("metadata scale dtype must be float32")
    if metadata.scale_index_axis != 0 or metadata.scale_reduction_axis != 1:
        raise ValueError("quantized matrices require scales indexed by axis 0 over axis 1")
    if metadata.accumulator_dtype != torch.float32:
        raise ValueError("dequantized linear accumulation must use float32")
    if metadata.layout != "row_major_contiguous":
        raise ValueError("quantized matrices require row-major contiguous layout")
    if metadata.quantized_min != -bound or metadata.quantized_max != bound:
        raise ValueError("metadata saturation bounds do not match the quantization format")

    if matrix.values.dtype != metadata.value_dtype:
        raise ValueError("values dtype does not match quantized metadata")
    if matrix.scales.dtype != metadata.scale_dtype:
        raise ValueError("scales must use the float32 dtype declared by metadata")
    if tuple(matrix.values.shape) != metadata.shape:
        raise ValueError("values shape does not match quantized metadata")
    if tuple(matrix.scales.shape) != (metadata.shape[0],):
        raise ValueError("scales must contain one value per matrix row")
    if matrix.values.device != matrix.scales.device:
        raise ValueError("values and scales must share a device")
    if not matrix.values.is_contiguous() or not matrix.scales.is_contiguous():
        raise ValueError("quantized values and scales must be contiguous")
    if matrix.values.requires_grad or matrix.scales.requires_grad:
        raise RuntimeError("quantized matrices are forward-only")
    if not bool(torch.isfinite(matrix.scales).all().item()) or not bool(
        (matrix.scales > 0).all().item()
    ):
        raise ValueError("quantized scales must be finite and strictly positive")
    if metadata.format == "fp8_e4m3fn":
        finite_codes = torch.bitwise_and(matrix.values, 0x7F) != 0x7F
        if not bool(finite_codes.all().item()):
            raise ValueError("FP8 E4M3FN payload must not contain NaN encodings")
    elif not bool((matrix.values >= -127).all().item()):
        raise ValueError("symmetric INT8 payload must stay within [-127, 127]")


def _validate_backend(backend: QuantizationBackend) -> None:
    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError("backend must be 'auto', 'cuda', or 'reference'")


def _validate_source_matrix(matrix: Tensor) -> None:
    if matrix.ndim != 2:
        raise ValueError("quantization input must be a rank-2 matrix")
    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError("quantization matrix dimensions must be positive")
    if matrix.dtype != torch.float32:
        raise TypeError("quantization input must use float32")
    if not matrix.is_contiguous():
        raise ValueError("quantization input must be row-major contiguous")
    if matrix.requires_grad:
        raise RuntimeError("quantization experiments are forward-only")


def _validate_finite_source_matrix(matrix: Tensor) -> None:
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError("quantization input must contain only finite values")


def _metadata(
    matrix: Tensor,
    format: QuantizationFormat,
    scale_granularity: ScaleGranularity,
) -> QuantizedMatrixMetadata:
    value_dtype, bound = _format_spec(format)
    return QuantizedMatrixMetadata(
        format=format,
        scale_granularity=scale_granularity,
        shape=(matrix.shape[0], matrix.shape[1]),
        source_dtype=matrix.dtype,
        value_dtype=value_dtype,
        scale_dtype=torch.float32,
        scale_index_axis=0,
        scale_reduction_axis=1,
        accumulator_dtype=torch.float32,
        layout="row_major_contiguous",
        quantized_min=-bound,
        quantized_max=bound,
    )


def _reference_quantize(matrix: Tensor, format: QuantizationFormat) -> tuple[Tensor, Tensor]:
    _, bound = _format_spec(format)
    row_maximum = matrix.abs().amax(dim=1)
    nonzero_scales = (row_maximum / bound).clamp_min(torch.finfo(torch.float32).tiny)
    scales = torch.where(row_maximum == 0, torch.ones_like(row_maximum), nonzero_scales)
    normalized = torch.clamp(matrix / scales[:, None], min=-bound, max=bound)
    if format == "int8":
        values = torch.round(normalized).clamp(min=-127, max=127).to(torch.int8)
    else:
        values = normalized.to(torch.float8_e4m3fn).view(torch.uint8)
    return values.contiguous(), scales.to(torch.float32).contiguous()


def _cuda_quantize_ineligibility_reason(matrix: Tensor, format: QuantizationFormat) -> str | None:
    if not _ops.native_extension_loaded():
        return "the native extension is not installed"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    if matrix.device.type != "cuda":
        return "the input must be a CUDA tensor"
    operator = _QUANTIZE_OPERATOR[format]
    if not _ops._operator_has_cuda_kernel(operator):
        return f"the loaded native extension does not register {operator}"
    return None


def _quantize_matrix(
    matrix: Tensor,
    *,
    format: QuantizationFormat,
    scale_granularity: ScaleGranularity,
    backend: QuantizationBackend,
) -> QuantizedMatrix:
    _format_spec(format)
    _validate_backend(backend)
    _validate_source_matrix(matrix)
    reason = _cuda_quantize_ineligibility_reason(matrix, format)
    if backend == "cuda":
        if reason is not None:
            _validate_finite_source_matrix(matrix)
            raise RuntimeError(f"CUDA {format} quantization is unavailable: {reason}")
        values, scales = getattr(torch.ops.ds_flash_mla_moe, _QUANTIZE_OPERATOR[format]).default(
            matrix
        )
    elif backend == "auto" and reason is None:
        values, scales = getattr(torch.ops.ds_flash_mla_moe, _QUANTIZE_OPERATOR[format]).default(
            matrix
        )
    else:
        _validate_finite_source_matrix(matrix)
        values, scales = _reference_quantize(matrix, format)
    return QuantizedMatrix(
        values=values,
        scales=scales,
        metadata=_metadata(matrix, format, scale_granularity),
    )


def quantize_activations(
    matrix: Tensor,
    *,
    format: QuantizationFormat,
    backend: QuantizationBackend = "auto",
) -> QuantizedMatrix:
    """Quantize contiguous FP32 ``[rows, K]`` activations with one scale per row."""

    return _quantize_matrix(
        matrix,
        format=format,
        scale_granularity="per_row",
        backend=backend,
    )


def quantize_weights(
    weight: Tensor,
    *,
    format: QuantizationFormat,
    backend: QuantizationBackend = "auto",
) -> QuantizedMatrix:
    """Quantize contiguous FP32 ``[out_features, K]`` weights per output channel."""

    return _quantize_matrix(
        weight,
        format=format,
        scale_granularity="per_output_channel",
        backend=backend,
    )


def dequantize_matrix(matrix: QuantizedMatrix) -> Tensor:
    """Materialize a quantized matrix in row-major FP32 using its explicit scales."""

    _validate_quantized_matrix(matrix)
    return _dequantize_validated(matrix)


def _dequantize_validated(matrix: QuantizedMatrix) -> Tensor:
    if matrix.metadata.format == "int8":
        normalized = matrix.values.to(torch.float32)
    else:
        normalized = matrix.values.view(torch.float8_e4m3fn).to(torch.float32)
    return normalized * matrix.scales[:, None]


def _validate_linear_inputs(activations: QuantizedMatrix, weight: QuantizedMatrix) -> None:
    _validate_quantized_matrix(activations)
    _validate_quantized_matrix(weight)
    if activations.metadata.scale_granularity != "per_row":
        raise ValueError("linear activations must use per_row scales")
    if weight.metadata.scale_granularity != "per_output_channel":
        raise ValueError("linear weights must use per_output_channel scales")
    if activations.metadata.format != weight.metadata.format:
        raise ValueError("linear activations and weights must use the same format")
    if activations.values.shape[1] != weight.values.shape[1]:
        raise ValueError("linear activation and weight inner dimensions must match")
    if activations.values.device != weight.values.device:
        raise ValueError("linear activations and weights must share a device")


def _cuda_linear_ineligibility_reason(
    activations: QuantizedMatrix,
    weight: QuantizedMatrix,
) -> str | None:
    format = activations.metadata.format
    if not _ops.native_extension_loaded():
        return "the native extension is not installed"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    if activations.values.device.type != "cuda" or weight.values.device.type != "cuda":
        return "quantized values and scales must be CUDA tensors"
    operator = _LINEAR_OPERATOR[format]
    if not _ops._operator_has_cuda_kernel(operator):
        return f"the loaded native extension does not register {operator}"
    return None


def dequantized_linear(
    activations: QuantizedMatrix,
    weight: QuantizedMatrix,
    *,
    backend: QuantizationBackend = "auto",
) -> Tensor:
    """Compute ``[M,K] @ [N,K].T`` with dequantization and FP32 accumulation."""

    _validate_backend(backend)
    _validate_linear_inputs(activations, weight)
    reason = _cuda_linear_ineligibility_reason(activations, weight)
    if backend == "cuda":
        if reason is not None:
            raise RuntimeError(
                f"CUDA {activations.metadata.format} dequantized linear is unavailable: {reason}"
            )
        operator = getattr(
            torch.ops.ds_flash_mla_moe,
            _LINEAR_OPERATOR[activations.metadata.format],
        ).default
        return operator(
            activations.values,
            activations.scales,
            weight.values,
            weight.scales,
        )
    if backend == "auto" and reason is None:
        operator = getattr(
            torch.ops.ds_flash_mla_moe,
            _LINEAR_OPERATOR[activations.metadata.format],
        ).default
        return operator(
            activations.values,
            activations.scales,
            weight.values,
            weight.scales,
        )
    return _dequantize_validated(activations) @ _dequantize_validated(weight).T


def cuda_quantization_available(format: QuantizationFormat | None = None) -> bool:
    """Return whether native quantize and linear kernels exist for selected formats."""

    formats = tuple(_FORMATS) if format is None else (format,)
    for selected in formats:
        _format_spec(selected)
    return (
        _ops.native_extension_loaded()
        and torch.cuda.is_available()
        and all(
            _ops._operator_has_cuda_kernel(operator)
            for selected in formats
            for operator in (_QUANTIZE_OPERATOR[selected], _LINEAR_OPERATOR[selected])
        )
    )
