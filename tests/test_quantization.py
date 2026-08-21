from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from ds_flash_mla_moe import (
    QuantizedMatrix,
    cuda_quantization_available,
    dequantize_matrix,
    dequantized_linear,
    quantize_activations,
    quantize_weights,
)


@pytest.mark.parametrize(
    ("quantization_format", "value_dtype", "bound"),
    [
        ("int8", torch.int8, 127.0),
        ("fp8_e4m3fn", torch.uint8, 448.0),
    ],
)
def test_reference_quantization_records_explicit_immutable_metadata(
    quantization_format: str,
    value_dtype: torch.dtype,
    bound: float,
) -> None:
    matrix = torch.tensor([[0.0, 1.0, -2.0], [3.0, -4.0, 0.0]])

    quantized = quantize_activations(
        matrix,
        format=quantization_format,  # type: ignore[arg-type]
        backend="reference",
    )

    assert quantized.values.dtype == value_dtype
    assert quantized.values.is_contiguous()
    torch.testing.assert_close(quantized.scales, torch.tensor([2.0 / bound, 4.0 / bound]))
    assert quantized.metadata.format == quantization_format
    assert quantized.metadata.scale_granularity == "per_row"
    assert quantized.metadata.shape == (2, 3)
    assert quantized.metadata.source_dtype == torch.float32
    assert quantized.metadata.value_dtype == value_dtype
    assert quantized.metadata.scale_dtype == torch.float32
    assert quantized.metadata.scale_index_axis == 0
    assert quantized.metadata.scale_reduction_axis == 1
    assert quantized.metadata.accumulator_dtype == torch.float32
    assert quantized.metadata.layout == "row_major_contiguous"
    assert quantized.metadata.quantized_min == -bound
    assert quantized.metadata.quantized_max == bound
    with pytest.raises(FrozenInstanceError):
        quantized.metadata.scale_index_axis = 1  # type: ignore[misc]


@pytest.mark.parametrize("quantization_format", ["int8", "fp8_e4m3fn"])
def test_zero_rows_use_unit_scale_and_round_trip_to_exact_zero(
    quantization_format: str,
) -> None:
    matrix = torch.zeros(2, 5)

    quantized = quantize_activations(
        matrix,
        format=quantization_format,  # type: ignore[arg-type]
        backend="reference",
    )

    torch.testing.assert_close(quantized.scales, torch.ones(2))
    torch.testing.assert_close(dequantize_matrix(quantized), matrix)


@pytest.mark.parametrize("quantization_format", ["int8", "fp8_e4m3fn"])
def test_nonzero_subnormal_rows_clamp_scale_to_smallest_normal(
    quantization_format: str,
) -> None:
    smallest = torch.finfo(torch.float32).smallest_normal
    matrix = torch.tensor([[2.0**-149, 0.0]])

    quantized = quantize_activations(
        matrix,
        format=quantization_format,  # type: ignore[arg-type]
        backend="reference",
    )

    torch.testing.assert_close(quantized.scales, torch.tensor([smallest]))
    assert torch.isfinite(dequantize_matrix(quantized)).all()


def test_int8_reference_is_symmetric_and_saturates_at_127() -> None:
    matrix = torch.tensor([[3.0, -6.0, 0.0]])

    quantized = quantize_activations(matrix, format="int8", backend="reference")

    assert quantized.values.tolist() == [[64, -127, 0]]
    assert int(quantized.values.min()) >= -127
    assert int(quantized.values.max()) <= 127


def test_fp8_reference_uses_e4m3fn_payload_bits_and_finite_saturation() -> None:
    matrix = torch.tensor([[448.0, -448.0, 1.0, 0.0]])

    quantized = quantize_activations(matrix, format="fp8_e4m3fn", backend="reference")
    decoded = quantized.values.view(torch.float8_e4m3fn).to(torch.float32)

    assert decoded.tolist() == [[448.0, -448.0, 1.0, 0.0]]
    assert torch.isfinite(decoded).all()


@pytest.mark.parametrize("quantization_format", ["int8", "fp8_e4m3fn"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_quantization_rejects_non_finite_inputs(
    quantization_format: str,
    bad: float,
) -> None:
    matrix = torch.tensor([[1.0, bad]])

    with pytest.raises(ValueError, match="finite"):
        quantize_activations(
            matrix,
            format=quantization_format,  # type: ignore[arg-type]
            backend="reference",
        )


def test_unavailable_explicit_cuda_still_rejects_non_finite_input_strictly() -> None:
    matrix = torch.tensor([[1.0, float("nan")]])

    with pytest.raises(ValueError, match="finite"):
        quantize_activations(matrix, format="int8", backend="cuda")


@pytest.mark.parametrize("quantization_format", ["int8", "fp8_e4m3fn"])
def test_dequantized_linear_matches_hand_derived_fp32_oracle(
    quantization_format: str,
) -> None:
    activations = torch.tensor([[0.25, -0.5, 1.0], [2.0, 0.5, -1.0]])
    weight = torch.tensor([[1.0, -2.0, 0.5], [-0.25, 0.75, 1.5]])
    quantized_activations = quantize_activations(
        activations,
        format=quantization_format,  # type: ignore[arg-type]
        backend="reference",
    )
    quantized_weight = quantize_weights(
        weight,
        format=quantization_format,  # type: ignore[arg-type]
        backend="reference",
    )
    if quantization_format == "int8":
        activation_codes = quantized_activations.values.to(torch.float32)
        weight_codes = quantized_weight.values.to(torch.float32)
    else:
        activation_codes = quantized_activations.values.view(torch.float8_e4m3fn).to(torch.float32)
        weight_codes = quantized_weight.values.view(torch.float8_e4m3fn).to(torch.float32)
    expected = (activation_codes * quantized_activations.scales[:, None]) @ (
        weight_codes * quantized_weight.scales[:, None]
    ).T

    actual = dequantized_linear(
        quantized_activations,
        quantized_weight,
        backend="reference",
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_weight_metadata_uses_output_channel_scales() -> None:
    weight = torch.tensor([[1.0, -2.0], [4.0, 3.0], [-0.5, 0.25]])

    quantized = quantize_weights(weight, format="int8", backend="reference")

    assert quantized.metadata.scale_granularity == "per_output_channel"
    assert quantized.metadata.scale_index_axis == 0
    assert quantized.metadata.scale_reduction_axis == 1
    torch.testing.assert_close(quantized.scales, torch.tensor([2.0, 4.0, 0.5]) / 127.0)


def test_linear_rejects_wrong_roles_formats_and_inner_dimensions() -> None:
    activation = quantize_activations(torch.randn(2, 3), format="int8", backend="reference")
    weight = quantize_weights(torch.randn(4, 3), format="int8", backend="reference")
    fp8_weight = quantize_weights(torch.randn(4, 3), format="fp8_e4m3fn", backend="reference")
    wrong_inner = quantize_weights(torch.randn(4, 5), format="int8", backend="reference")

    with pytest.raises(ValueError, match="per_row"):
        dequantized_linear(weight, weight, backend="reference")
    with pytest.raises(ValueError, match="same format"):
        dequantized_linear(activation, fp8_weight, backend="reference")
    with pytest.raises(ValueError, match="inner"):
        dequantized_linear(activation, wrong_inner, backend="reference")


def test_reference_and_auto_are_explicit_and_cuda_fails_on_cpu() -> None:
    matrix = torch.tensor([[1.0, -2.0]])
    reference = quantize_activations(matrix, format="int8", backend="reference")
    automatic = quantize_activations(matrix, format="int8", backend="auto")

    torch.testing.assert_close(automatic.values, reference.values)
    torch.testing.assert_close(automatic.scales, reference.scales)
    with pytest.raises(RuntimeError, match="CUDA int8 quantization is unavailable"):
        quantize_activations(matrix, format="int8", backend="cuda")
    with pytest.raises(ValueError, match="backend"):
        quantize_activations(matrix, format="int8", backend="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="format"):
        quantize_activations(matrix, format="int4", backend="reference")  # type: ignore[arg-type]


def test_quantization_is_forward_only() -> None:
    with pytest.raises(RuntimeError, match="forward-only"):
        quantize_activations(
            torch.randn(2, 3, requires_grad=True),
            format="int8",
            backend="reference",
        )


@pytest.mark.parametrize(
    ("operator", "value_dtype"),
    [
        ("quantize_int8_per_row", torch.int8),
        ("quantize_fp8_e4m3fn_per_row", torch.uint8),
    ],
)
def test_raw_quantizers_have_fake_tensor_shape_and_dtype_parity(
    operator: str,
    value_dtype: torch.dtype,
) -> None:
    with FakeTensorMode():
        matrix = torch.empty(3, 5, device="cuda", dtype=torch.float32)
        values, scales = getattr(torch.ops.ds_flash_mla_moe, operator).default(matrix)

    assert values.shape == (3, 5)
    assert values.dtype == value_dtype
    assert scales.shape == (3,)
    assert scales.dtype == torch.float32
    assert values.device == matrix.device == scales.device


@pytest.mark.parametrize(
    ("operator", "value_dtype"),
    [
        ("dequantized_linear_int8", torch.int8),
        ("dequantized_linear_fp8_e4m3fn", torch.uint8),
    ],
)
def test_raw_dequantized_linear_has_fake_tensor_shape_and_dtype_parity(
    operator: str,
    value_dtype: torch.dtype,
) -> None:
    with FakeTensorMode():
        activation_values = torch.empty(3, 5, device="cuda", dtype=value_dtype)
        activation_scales = torch.empty(3, device="cuda", dtype=torch.float32)
        weight_values = torch.empty(7, 5, device="cuda", dtype=value_dtype)
        weight_scales = torch.empty(7, device="cuda", dtype=torch.float32)
        output = getattr(torch.ops.ds_flash_mla_moe, operator).default(
            activation_values,
            activation_scales,
            weight_values,
            weight_scales,
        )

    assert output.shape == (3, 7)
    assert output.dtype == torch.float32
    assert output.device == activation_values.device


@pytest.mark.parametrize(
    "operator",
    [
        "quantize_int8_per_row",
        "quantize_fp8_e4m3fn_per_row",
        "dequantized_linear_int8",
        "dequantized_linear_fp8_e4m3fn",
    ],
)
def test_raw_quantization_operators_are_native_only_and_forward_only(operator: str) -> None:
    qualified = f"ds_flash_mla_moe::{operator}"
    for dispatch_key in ("CompositeExplicitAutograd", "CompositeImplicitAutograd", "Autograd"):
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(qualified, dispatch_key)


def test_quantized_matrix_rejects_mutable_or_inconsistent_payload_metadata() -> None:
    quantized = quantize_activations(torch.randn(2, 3), format="int8", backend="reference")

    with pytest.raises((TypeError, ValueError), match="values dtype"):
        QuantizedMatrix(
            values=quantized.values.to(torch.uint8),
            scales=quantized.scales,
            metadata=quantized.metadata,
        )


@pytest.mark.parametrize(
    ("metadata_override", "message"),
    [
        ({"source_dtype": torch.float64}, "source dtype"),
        ({"scale_granularity": "per_tensor"}, "scale granularity"),
    ],
)
def test_quantized_matrix_rejects_forged_public_metadata(
    metadata_override: dict[str, object],
    message: str,
) -> None:
    quantized = quantize_activations(torch.randn(2, 3), format="int8", backend="reference")
    forged = replace(quantized.metadata, **metadata_override)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        QuantizedMatrix(
            values=quantized.values,
            scales=quantized.scales,
            metadata=forged,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("requires_grad", "forward-only"),
        ("non_finite_scale", "finite"),
        ("non_positive_scale", "positive"),
        ("value_dtype", "values dtype"),
        ("scale_dtype", "scales must use"),
        ("non_contiguous_values", "contiguous"),
    ],
)
def test_dequantize_revalidates_post_construction_tensor_mutation(
    mutation: str,
    message: str,
) -> None:
    quantized = quantize_activations(torch.randn(2, 3), format="int8", backend="reference")
    if mutation == "requires_grad":
        quantized.scales.requires_grad_(True)
    elif mutation == "non_finite_scale":
        quantized.scales.fill_(float("nan"))
    elif mutation == "non_positive_scale":
        quantized.scales.zero_()
    elif mutation == "value_dtype":
        quantized.values.data = quantized.values.to(torch.float32)
    elif mutation == "scale_dtype":
        quantized.scales.data = quantized.scales.to(torch.float64)
    else:
        storage = torch.empty(2, 6, dtype=torch.int8)
        storage[:, ::2].copy_(quantized.values)
        quantized.values.set_(storage[:, ::2])

    with pytest.raises((RuntimeError, ValueError), match=message):
        dequantize_matrix(quantized)


def test_linear_revalidates_mutated_weight_before_backend_selection() -> None:
    activations = quantize_activations(torch.randn(2, 3), format="int8", backend="reference")
    weight = quantize_weights(torch.randn(4, 3), format="int8", backend="reference")
    weight.scales.requires_grad_(True)

    with pytest.raises(RuntimeError, match="forward-only"):
        dequantized_linear(activations, weight, backend="cuda")


@pytest.mark.parametrize("quantization_format", ["int8", "fp8_e4m3fn"])
def test_cuda_quantization_capability_flag_is_consistent(quantization_format: str) -> None:
    available = cuda_quantization_available(quantization_format)  # type: ignore[arg-type]
    assert isinstance(available, bool)
    if available:
        assert torch.cuda.is_available()


@pytest.mark.parametrize("quantization_format", ["int8", "fp8_e4m3fn"])
@pytest.mark.cuda
def test_cuda_quantization_and_linear_match_dequantized_reference(
    quantization_format: str,
) -> None:
    if not cuda_quantization_available(quantization_format):  # type: ignore[arg-type]
        pytest.skip(f"requires native CUDA {quantization_format} quantization")
    torch.manual_seed(801)
    activation = torch.randn(17, 23, device="cuda")
    weight = torch.randn(11, 23, device="cuda")

    activation_cuda = quantize_activations(
        activation,
        format=quantization_format,  # type: ignore[arg-type]
        backend="cuda",
    )
    weight_cuda = quantize_weights(
        weight,
        format=quantization_format,  # type: ignore[arg-type]
        backend="cuda",
    )
    activation_reference = quantize_activations(
        activation,
        format=quantization_format,  # type: ignore[arg-type]
        backend="reference",
    )
    weight_reference = quantize_weights(
        weight,
        format=quantization_format,  # type: ignore[arg-type]
        backend="reference",
    )

    torch.testing.assert_close(activation_cuda.values, activation_reference.values)
    torch.testing.assert_close(activation_cuda.scales, activation_reference.scales)
    torch.testing.assert_close(weight_cuda.values, weight_reference.values)
    torch.testing.assert_close(weight_cuda.scales, weight_reference.scales)
    actual = dequantized_linear(activation_cuda, weight_cuda, backend="cuda")
    expected = dequantized_linear(
        activation_reference,
        weight_reference,
        backend="reference",
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("quantization_format", ["int8", "fp8_e4m3fn"])
@pytest.mark.cuda
def test_cuda_quantizer_broadcasts_row_scale_to_nonzero_thread_lanes(
    quantization_format: str,
) -> None:
    if not cuda_quantization_available(quantization_format):  # type: ignore[arg-type]
        pytest.skip(f"requires native CUDA {quantization_format} quantization")
    pattern = torch.tensor([1.0, 0.5, -0.25, 0.125])
    first_row = pattern.repeat(129)[:513]
    matrix = torch.stack((first_row, first_row * 3.0)).to("cuda").contiguous()

    actual = quantize_activations(
        matrix,
        format=quantization_format,  # type: ignore[arg-type]
        backend="cuda",
    )
    expected = quantize_activations(
        matrix,
        format=quantization_format,  # type: ignore[arg-type]
        backend="reference",
    )

    # CUDA thread 0 owns columns 0/256/512; these nonzero-thread columns prove
    # that the block-wide row scale was broadcast rather than merely published
    # through a global-memory store by thread 0.
    nonzero_thread_columns = [1, 255, 257, 511]
    torch.testing.assert_close(
        actual.values[:, nonzero_thread_columns],
        expected.values[:, nonzero_thread_columns],
    )
    torch.testing.assert_close(actual.values, expected.values)
    torch.testing.assert_close(actual.scales, expected.scales)


@pytest.mark.parametrize("quantization_format", ["int8", "fp8_e4m3fn"])
@pytest.mark.cuda
def test_cuda_dequantized_linear_uses_current_stream_and_graph_replay(
    quantization_format: str,
) -> None:
    if not cuda_quantization_available(quantization_format):  # type: ignore[arg-type]
        pytest.skip(f"requires native CUDA {quantization_format} quantization")
    activation = quantize_activations(
        torch.randn(8, 16, device="cuda"),
        format=quantization_format,  # type: ignore[arg-type]
        backend="cuda",
    )
    weight = quantize_weights(
        torch.randn(12, 16, device="cuda"),
        format=quantization_format,  # type: ignore[arg-type]
        backend="cuda",
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        expected = dequantized_linear(activation, weight, backend="reference")
        eager = dequantized_linear(activation, weight, backend="cuda")
        eager.record_stream(stream)
    stream.synchronize()
    torch.testing.assert_close(eager, expected, rtol=2e-5, atol=2e-5)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = dequantized_linear(activation, weight, backend="cuda")
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, rtol=2e-5, atol=2e-5)
