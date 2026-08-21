from contextlib import contextmanager

import pytest
import torch

from ds_flash_mla_moe import (
    blockwise_attention,
    cuda_kernel_available,
    flash_attention_forward,
    native_extension_loaded,
    scaled_dot_product_attention_backward_reference,
)


def _cuda_attention_tolerances(
    dtype: torch.dtype,
    *,
    backward: bool = False,
) -> tuple[float, float]:
    if dtype == torch.float32:
        return (8e-5, 8e-5) if backward else (3e-5, 3e-5)
    if dtype == torch.float16:
        return (1e-2, 1e-2) if backward else (5e-3, 5e-3)
    return (5e-2, 5e-2) if backward else (2e-2, 2e-2)


@contextmanager
def _deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


@pytest.mark.parametrize("causal", [False, True])
def test_reference_backend_matches_blockwise_specification(causal: bool) -> None:
    torch.manual_seed(41)
    q = torch.randn(2, 3, 5, 7, dtype=torch.float64)
    k = torch.randn(2, 3, 8, 7, dtype=torch.float64)
    v = torch.randn(2, 3, 8, 4, dtype=torch.float64)

    actual = flash_attention_forward(
        q,
        k,
        v,
        causal=causal,
        scale=0.31,
        backend="reference",
        reference_block_size=3,
    )
    expected = blockwise_attention(q, k, v, causal=causal, scale=0.31, block_size=3)

    torch.testing.assert_close(actual, expected)


def test_auto_backend_preserves_reference_autograd() -> None:
    torch.manual_seed(43)
    q = torch.randn(1, 2, 4, 5, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 6, 5, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 2, 6, 3, dtype=torch.float64, requires_grad=True)
    expected_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v)]
    upstream = torch.randn(1, 2, 4, 3, dtype=torch.float64)

    actual = flash_attention_forward(q, k, v, causal=True, backend="auto")
    expected = blockwise_attention(*expected_inputs, causal=True)
    actual.backward(upstream)
    expected.backward(upstream)

    torch.testing.assert_close(actual, expected)
    for actual_input, expected_input in zip((q, k, v), expected_inputs):
        torch.testing.assert_close(actual_input.grad, expected_input.grad)


def test_masked_auto_backend_falls_back_to_reference() -> None:
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    v = torch.tensor([[[[1.0], [2.0], [4.0]]]])
    mask = torch.tensor([[False, False, False], [True, False, True]])

    actual = flash_attention_forward(q, k, v, attn_mask=mask)
    expected = blockwise_attention(q, k, v, attn_mask=mask)

    torch.testing.assert_close(actual, expected)


def test_cuda_backend_fails_loudly_without_eligible_inputs() -> None:
    q = torch.randn(1, 2, 3, 4)
    with pytest.raises(RuntimeError, match="cuda_rowwise attention is unavailable"):
        flash_attention_forward(q, q, q, backend="cuda_rowwise")


@pytest.mark.parametrize("backend", ["unknown", "CUDA", ""])
def test_invalid_backend_is_rejected(backend: str) -> None:
    q = torch.randn(1, 2, 3, 4)
    with pytest.raises(ValueError, match="backend"):
        flash_attention_forward(q, q, q, backend=backend)  # type: ignore[arg-type]


def test_invalid_scale_and_head_dimension_are_rejected() -> None:
    q = torch.randn(1, 2, 3, 4)
    with pytest.raises(ValueError, match="finite"):
        flash_attention_forward(q, q, q, scale=float("nan"))

    empty_head = torch.empty(1, 2, 3, 0)
    with pytest.raises(ValueError, match="head dimension"):
        flash_attention_forward(empty_head, empty_head, empty_head)


def test_invalid_mask_is_rejected_before_backend_selection() -> None:
    q = torch.randn(1, 2, 3, 4)
    invalid_mask = torch.ones(5, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="cannot broadcast"):
        flash_attention_forward(q, q, q, attn_mask=invalid_mask)


def test_custom_operator_registration_passes_opcheck() -> None:
    q = torch.randn(1, 2, 4, 5, dtype=torch.float64)
    result = torch.library.opcheck(
        torch.ops.ds_flash_mla_moe.attention_forward.default,
        (q, q, q, True, 5**-0.5),
    )

    assert set(result.values()) == {"SUCCESS"}


def test_backward_custom_operator_registration_passes_opcheck() -> None:
    q = torch.randn(1, 2, 4, 5, dtype=torch.float64)
    k = torch.randn(1, 2, 7, 5, dtype=torch.float64)
    v = torch.randn(1, 2, 7, 3, dtype=torch.float64)
    grad_output = torch.randn(1, 2, 4, 3, dtype=torch.float64)
    result = torch.library.opcheck(
        torch.ops.ds_flash_mla_moe.attention_backward.default,
        (grad_output, q, k, v, True, 5**-0.5),
    )

    assert set(result.values()) == {"SUCCESS"}


def test_backward_custom_operator_matches_analytic_specification() -> None:
    torch.manual_seed(44)
    q = torch.randn(1, 2, 4, 5, dtype=torch.float64)
    k = torch.randn(1, 2, 7, 5, dtype=torch.float64)
    v = torch.randn(1, 2, 7, 3, dtype=torch.float64)
    grad_output = torch.randn(1, 2, 4, 3, dtype=torch.float64)

    actual = torch.ops.ds_flash_mla_moe.attention_backward.default(grad_output, q, k, v, True, 0.37)
    expected = scaled_dot_product_attention_backward_reference(
        grad_output, q, k, v, causal=True, scale=0.37
    )

    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_raw_custom_operator_gradients_match_reference() -> None:
    torch.manual_seed(45)
    inputs = [torch.randn(1, 2, 4, 5, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    expected_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    upstream = torch.randn_like(inputs[0])

    actual = torch.ops.ds_flash_mla_moe.attention_forward.default(*inputs, True, 0.2)
    expected = blockwise_attention(*expected_inputs, causal=True, scale=0.2)
    actual.backward(upstream)
    expected.backward(upstream)

    torch.testing.assert_close(actual, expected)
    for actual_input, expected_input in zip(inputs, expected_inputs):
        torch.testing.assert_close(actual_input.grad, expected_input.grad)


def test_raw_custom_operator_handles_aliased_qkv_gradients() -> None:
    torch.manual_seed(46)
    x = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_(True)
    upstream = torch.randn_like(x)

    actual = torch.ops.ds_flash_mla_moe.attention_forward.default(x, x, x, True, 0.5)
    expected = blockwise_attention(expected_x, expected_x, expected_x, causal=True, scale=0.5)
    actual.backward(upstream)
    expected.backward(upstream)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, expected_x.grad, rtol=1e-9, atol=1e-9)


def test_raw_custom_operator_supports_second_order_gradients() -> None:
    torch.manual_seed(48)
    inputs = tuple(
        torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True) for _ in range(3)
    )

    def operation(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.ops.ds_flash_mla_moe.attention_forward.default(q, k, v, False, 0.5)

    assert torch.autograd.gradcheck(operation, inputs, rtol=1e-5, atol=1e-6)
    assert torch.autograd.gradgradcheck(operation, inputs, rtol=1e-5, atol=1e-6)


def test_custom_operator_opcheck_with_autograd_inputs() -> None:
    q = torch.randn(1, 2, 4, 5, dtype=torch.float64, requires_grad=True)
    result = torch.library.opcheck(
        torch.ops.ds_flash_mla_moe.attention_forward.default,
        (q, q, q, False, 5**-0.5),
    )

    assert set(result.values()) == {"SUCCESS"}


def test_custom_operator_opcheck_with_cross_attention_shapes() -> None:
    q = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    k = torch.randn(2, 3, 7, 5, dtype=torch.float64)
    v = torch.randn(2, 3, 7, 2, dtype=torch.float64)
    result = torch.library.opcheck(
        torch.ops.ds_flash_mla_moe.attention_forward.default,
        (q, k, v, True, 5**-0.5),
    )

    assert set(result.values()) == {"SUCCESS"}


def test_custom_operator_runs_through_torch_compile() -> None:
    q = torch.randn(1, 2, 4, 5)

    @torch.compile(fullgraph=True, backend="eager")
    def compiled_attention(x: torch.Tensor) -> torch.Tensor:
        return torch.ops.ds_flash_mla_moe.attention_forward.default(x, x, x, True, 5**-0.5)

    actual = compiled_attention(q)
    expected = blockwise_attention(q, q, q, causal=True)

    torch.testing.assert_close(actual, expected)


def test_native_capability_flags_are_consistent() -> None:
    assert isinstance(native_extension_loaded(), bool)
    assert isinstance(cuda_kernel_available(), bool)
    if cuda_kernel_available():
        assert native_extension_loaded()
        assert torch.cuda.is_available()


@pytest.mark.skipif(
    not cuda_kernel_available(),
    reason="requires a built native extension and a CUDA device",
)
@pytest.mark.cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("query_length", "key_length", "head_dim", "value_dim"),
    [(1, 7, 8, 5), (5, 5, 32, 16), (7, 11, 65, 33)],
)
def test_cuda_forward_matches_reference(
    causal: bool,
    dtype: torch.dtype,
    query_length: int,
    key_length: int,
    head_dim: int,
    value_dim: int,
) -> None:
    torch.manual_seed(47)
    q = torch.randn(2, 3, query_length, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(2, 3, key_length, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(2, 3, key_length, value_dim, device="cuda", dtype=dtype)

    with torch.no_grad():
        actual = flash_attention_forward(q, k, v, causal=causal, backend="cuda_rowwise")
        expected = blockwise_attention(q, k, v, causal=causal, block_size=3)

    rtol, atol = _cuda_attention_tolerances(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not cuda_kernel_available(),
    reason="requires a built native extension and a CUDA device",
)
@pytest.mark.cuda
def test_cuda_kernel_uses_current_stream() -> None:
    q = torch.randn(1, 2, 8, 16, device="cuda")
    stream = torch.cuda.Stream()

    with torch.no_grad(), torch.cuda.stream(stream):
        q.fill_(0.25)
        output = flash_attention_forward(q, q, q, causal=True, backend="cuda_rowwise")
        output.record_stream(stream)
    stream.synchronize()

    torch.testing.assert_close(output, torch.full_like(output, 0.25))


@pytest.mark.skipif(
    not cuda_kernel_available(),
    reason="requires a built native extension and a CUDA device",
)
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_forward_autograd_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(49)
    inputs = [
        torch.randn(1, 2, 5, 8, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)
    ]
    expected_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    upstream = torch.randn_like(inputs[0])

    actual = flash_attention_forward(*inputs, causal=True, backend="cuda_rowwise")
    expected = blockwise_attention(*expected_inputs, causal=True, block_size=3)
    actual.backward(upstream)
    expected.backward(upstream)

    forward_rtol, forward_atol = _cuda_attention_tolerances(dtype)
    backward_rtol, backward_atol = _cuda_attention_tolerances(dtype, backward=True)
    torch.testing.assert_close(actual, expected, rtol=forward_rtol, atol=forward_atol)
    for actual_input, expected_input in zip(inputs, expected_inputs):
        torch.testing.assert_close(
            actual_input.grad,
            expected_input.grad,
            rtol=backward_rtol,
            atol=backward_atol,
        )


@pytest.mark.skipif(
    not cuda_kernel_available(),
    reason="requires a built native extension and a CUDA device",
)
@pytest.mark.cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("query_length", "key_length", "head_dim", "value_dim"),
    [(1, 7, 8, 5), (5, 5, 32, 16), (7, 11, 65, 33)],
)
def test_cuda_backward_matches_analytic_reference(
    causal: bool,
    dtype: torch.dtype,
    query_length: int,
    key_length: int,
    head_dim: int,
    value_dim: int,
) -> None:
    torch.manual_seed(51)
    q = torch.randn(2, 3, query_length, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(2, 3, key_length, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(2, 3, key_length, value_dim, device="cuda", dtype=dtype)
    grad_output = torch.randn(
        2,
        3,
        query_length,
        value_dim,
        device="cuda",
        dtype=dtype,
    )

    actual = torch.ops.ds_flash_mla_moe.attention_backward.default(
        grad_output, q, k, v, causal, head_dim**-0.5
    )
    expected = scaled_dot_product_attention_backward_reference(
        grad_output,
        q,
        k,
        v,
        causal=causal,
    )

    rtol, atol = _cuda_attention_tolerances(dtype, backward=True)
    for actual_gradient, expected_gradient in zip(actual, expected):
        assert actual_gradient.dtype == dtype
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not cuda_kernel_available(),
    reason="requires a built native extension and a CUDA device",
)
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_noncontiguous_inputs_follow_the_explicit_fallback_contract(
    dtype: torch.dtype,
) -> None:
    q = torch.randn(1, 2, 5, 16, device="cuda", dtype=dtype)[..., ::2]
    k = torch.randn(1, 2, 7, 16, device="cuda", dtype=dtype)[..., ::2]
    v = torch.randn(1, 2, 7, 12, device="cuda", dtype=dtype)[..., ::2]
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()

    actual = flash_attention_forward(q, k, v, causal=True, backend="auto")
    expected = blockwise_attention(q, k, v, causal=True, block_size=3)
    rtol, atol = _cuda_attention_tolerances(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    with pytest.raises(RuntimeError, match="contiguous"):
        flash_attention_forward(q, k, v, causal=True, backend="cuda_rowwise")


@pytest.mark.skipif(
    not cuda_kernel_available(),
    reason="requires a built native extension and a CUDA device",
)
@pytest.mark.cuda
def test_cuda_attention_rejects_mixed_storage_dtypes() -> None:
    q = torch.randn(1, 2, 3, 8, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 5, 8, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 2, 5, 4, device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="same dtype"):
        flash_attention_forward(q, k, v, backend="cuda_rowwise")


@pytest.mark.skipif(
    not cuda_kernel_available(),
    reason="requires a built native extension and a CUDA device",
)
@pytest.mark.filterwarnings(
    "ignore:Attempting to run cuBLAS, but there was no current CUDA context!:UserWarning"
)
@pytest.mark.cuda
def test_deterministic_mode_uses_reference_backward() -> None:
    inputs = [torch.randn(1, 2, 5, 8, device="cuda", requires_grad=True) for _ in range(3)]
    with _deterministic_algorithms():
        output = flash_attention_forward(*inputs, causal=True, backend="cuda_rowwise")
        output.sum().backward()
    for tensor in inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.skipif(
    not cuda_kernel_available(),
    reason="requires a built native extension and a CUDA device",
)
@pytest.mark.cuda
def test_cuda_backward_uses_current_stream() -> None:
    q = torch.randn(1, 2, 5, 8, device="cuda")
    k = torch.randn(1, 2, 7, 8, device="cuda")
    v = torch.randn(1, 2, 7, 4, device="cuda")
    grad_output = torch.randn(1, 2, 5, 4, device="cuda")
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        q.fill_(0.25)
        k.fill_(0.5)
        v.fill_(0.75)
        grad_output.fill_(1.0)
        gradients = torch.ops.ds_flash_mla_moe.attention_backward.default(
            grad_output, q, k, v, True, 8**-0.5
        )
        for gradient in gradients:
            gradient.record_stream(stream)
    stream.synchronize()

    expected = scaled_dot_product_attention_backward_reference(grad_output, q, k, v, causal=True)
    for actual_gradient, expected_gradient in zip(gradients, expected):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=8e-5, atol=8e-5)
