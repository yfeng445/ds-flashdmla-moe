import pytest
import torch
import torch.nn.functional as F

from ds_flash_mla_moe import (
    blockwise_attention,
    scaled_dot_product_attention_backward_reference,
    scaled_dot_product_attention_reference,
)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("block_size", [1, 3, 8])
def test_blockwise_matches_materialized_and_pytorch(causal: bool, block_size: int) -> None:
    torch.manual_seed(11)
    q = torch.randn(2, 3, 7, 8, dtype=torch.float64)
    k = torch.randn(2, 3, 7, 8, dtype=torch.float64)
    v = torch.randn(2, 3, 7, 5, dtype=torch.float64)

    expected = scaled_dot_product_attention_reference(q, k, v, causal=causal)
    actual = blockwise_attention(q, k, v, causal=causal, block_size=block_size)
    pytorch = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(actual, pytorch, rtol=1e-10, atol=1e-10)


def test_blockwise_gradients_match_materialized_reference() -> None:
    torch.manual_seed(23)
    inputs = [torch.randn(1, 2, 5, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    q1, k1, v1 = inputs
    q2, k2, v2 = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    upstream = torch.randn(1, 2, 5, 4, dtype=torch.float64)

    expected = scaled_dot_product_attention_reference(q1, k1, v1, causal=True)
    actual = blockwise_attention(q2, k2, v2, causal=True, block_size=2)
    expected.backward(upstream)
    actual.backward(upstream)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    for actual_grad, expected_grad in zip((q2.grad, k2.grad, v2.grad), (q1.grad, k1.grad, v1.grad)):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("causal", [False, True])
def test_analytic_backward_matches_autograd_for_cross_attention(causal: bool) -> None:
    torch.manual_seed(29)
    q = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 3, 7, 5, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 3, 7, 3, dtype=torch.float64, requires_grad=True)
    grad_output = torch.randn(2, 3, 4, 3, dtype=torch.float64)

    output = scaled_dot_product_attention_reference(q, k, v, causal=causal, scale=0.37)
    expected = torch.autograd.grad(output, (q, k, v), grad_output, create_graph=True)
    actual = scaled_dot_product_attention_backward_reference(
        grad_output,
        q,
        k,
        v,
        causal=causal,
        scale=0.37,
    )

    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-10)


def test_analytic_backward_respects_mask_and_fully_masked_rows() -> None:
    torch.manual_seed(30)
    q = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 4, 2, dtype=torch.float64, requires_grad=True)
    grad_output = torch.randn(1, 2, 2, dtype=torch.float64)
    mask = torch.tensor([[False, False, False, False], [True, False, True, False]])

    output = scaled_dot_product_attention_reference(q, k, v, attn_mask=mask)
    expected = torch.autograd.grad(output, (q, k, v), grad_output)
    actual = scaled_dot_product_attention_backward_reference(
        grad_output,
        q,
        k,
        v,
        attn_mask=mask,
    )

    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-10)
        assert torch.isfinite(actual_gradient).all()
    assert torch.count_nonzero(actual[0][..., 0, :]) == 0


def test_analytic_backward_supports_second_order_gradients() -> None:
    torch.manual_seed(32)
    inputs = tuple(
        torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True) for _ in range(3)
    )
    grad_output = torch.randn_like(inputs[0], requires_grad=True)

    def operation(q, k, v, upstream):
        return scaled_dot_product_attention_backward_reference(upstream, q, k, v)

    assert torch.autograd.gradcheck(operation, (*inputs, grad_output), rtol=1e-5, atol=1e-6)
    assert torch.autograd.gradgradcheck(operation, (*inputs, grad_output), rtol=1e-5, atol=1e-6)


def test_boolean_mask_and_fully_masked_row() -> None:
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    v = torch.tensor([[[1.0], [2.0], [4.0]]])
    mask = torch.tensor([[False, False, False], [True, False, True]])

    expected, expected_lse = scaled_dot_product_attention_reference(
        q, k, v, attn_mask=mask, return_lse=True
    )
    actual, actual_lse = blockwise_attention(q, k, v, attn_mask=mask, block_size=2, return_lse=True)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_lse, expected_lse)
    assert actual[0, 0, 0].item() == 0.0
    assert torch.isneginf(actual_lse[0, 0])


def test_right_aligned_decode_causal_mask_sees_full_cache() -> None:
    q = torch.zeros(1, 1, 2)
    k = torch.zeros(1, 4, 2)
    v = torch.arange(4.0).reshape(1, 4, 1)

    output = blockwise_attention(q, k, v, causal=True, block_size=2)

    torch.testing.assert_close(output, torch.tensor([[[1.5]]]))


def test_invalid_shapes_fail_loudly() -> None:
    with pytest.raises(ValueError, match="head dimension"):
        blockwise_attention(torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 5))
    with pytest.raises(ValueError, match="block_size"):
        blockwise_attention(torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 5), block_size=0)
    with pytest.raises(ValueError, match="key sequence length"):
        blockwise_attention(
            torch.randn(1, 2, 3),
            torch.randn(1, 0, 3),
            torch.randn(1, 0, 5),
        )


def test_backward_rejects_invalid_grad_output() -> None:
    q = torch.randn(1, 2, 3)
    k = torch.randn(1, 4, 3)
    v = torch.randn(1, 4, 5)
    with pytest.raises(ValueError, match="grad_output"):
        scaled_dot_product_attention_backward_reference(torch.randn(1, 2, 4), q, k, v)


def test_empty_value_dimension_has_zero_input_gradients() -> None:
    q = torch.randn(1, 2, 3, dtype=torch.float64)
    k = torch.randn(1, 4, 3, dtype=torch.float64)
    v = torch.empty(1, 4, 0, dtype=torch.float64)
    grad_output = torch.empty(1, 2, 0, dtype=torch.float64)

    grad_q, grad_k, grad_v = scaled_dot_product_attention_backward_reference(grad_output, q, k, v)

    assert torch.count_nonzero(grad_q) == 0
    assert torch.count_nonzero(grad_k) == 0
    assert grad_v.shape == v.shape
