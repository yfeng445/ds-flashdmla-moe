import pytest
import torch

from ds_flash_mla_moe import (
    cuda_router_available,
    deepseek_grouped_topk,
    grouped_topk,
)


def _router_inputs(
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(401)
    x = torch.randn(
        7,
        5,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=requires_grad,
    )
    gate = torch.randn(
        8,
        5,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=requires_grad,
    )
    bias = torch.linspace(-0.3, 0.4, 8, dtype=dtype, device=device)
    return x, gate, bias


@pytest.mark.parametrize("score_bias", [False, True])
def test_reference_router_dispatch_matches_grouped_topk_spec(score_bias: bool) -> None:
    x, gate, bias = _router_inputs()
    selected_bias = bias if score_bias else None
    actual = grouped_topk(
        x,
        gate,
        topk=3,
        n_groups=4,
        topk_groups=2,
        score_bias=selected_bias,
        route_scale=1.7,
        backend="reference",
    )
    expected = deepseek_grouped_topk(
        x,
        gate,
        topk=3,
        n_groups=4,
        topk_groups=2,
        score_bias=selected_bias,
        route_scale=1.7,
    )

    torch.testing.assert_close(actual.indices, expected.indices)
    torch.testing.assert_close(actual.weights, expected.weights)


def test_raw_router_operator_uses_the_documented_tie_break() -> None:
    weights, indices = torch.ops.ds_flash_mla_moe.grouped_topk.default(
        torch.tensor([[100.0]]),
        torch.ones(4, 1),
        2,
        2,
        1,
        None,
        1.0,
    )

    assert indices.tolist() == [[0, 1]]
    torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5]]))


def test_router_selection_bias_does_not_receive_a_gradient() -> None:
    x, gate, bias = _router_inputs(requires_grad=True)
    bias.requires_grad_()
    weights, _indices = torch.ops.ds_flash_mla_moe.grouped_topk.default(
        x,
        gate,
        3,
        4,
        2,
        bias,
        1.2,
    )
    weights.square().sum().backward()

    assert x.grad is not None
    assert gate.grad is not None
    assert bias.grad is None


def test_raw_router_operator_matches_reference_gradients_and_opcheck() -> None:
    x, gate, bias = _router_inputs(requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    expected_gate = gate.detach().clone().requires_grad_()
    upstream = torch.randn(7, 3, dtype=torch.float64)

    actual_weights, actual_indices = torch.ops.ds_flash_mla_moe.grouped_topk.default(
        x,
        gate,
        3,
        4,
        2,
        bias,
        1.3,
    )
    expected = deepseek_grouped_topk(
        expected_x,
        expected_gate,
        topk=3,
        n_groups=4,
        topk_groups=2,
        score_bias=bias,
        route_scale=1.3,
    )
    actual_weights.backward(upstream)
    expected.weights.backward(upstream)

    torch.testing.assert_close(actual_indices, expected.indices)
    torch.testing.assert_close(actual_weights, expected.weights)
    torch.testing.assert_close(x.grad, expected_x.grad)
    torch.testing.assert_close(gate.grad, expected_gate.grad)

    result = torch.library.opcheck(
        torch.ops.ds_flash_mla_moe.grouped_topk.default,
        (x.detach(), gate.detach(), 3, 4, 2, bias, 1.3),
    )
    assert set(result.values()) == {"SUCCESS"}


def test_raw_router_operator_supports_compile_and_higher_order_gradients() -> None:
    x, gate, bias = _router_inputs(requires_grad=True)

    @torch.compile(fullgraph=True, backend="eager")
    def compiled(left: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.ds_flash_mla_moe.grouped_topk.default(
            left,
            weight,
            3,
            4,
            2,
            bias,
            1.1,
        )

    actual = compiled(x.detach(), gate.detach())
    expected = deepseek_grouped_topk(
        x.detach(),
        gate.detach(),
        topk=3,
        n_groups=4,
        topk_groups=2,
        score_bias=bias,
        route_scale=1.1,
    )
    torch.testing.assert_close(actual[0], expected.weights)
    torch.testing.assert_close(actual[1], expected.indices)

    def operation(left: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.ops.ds_flash_mla_moe.grouped_topk.default(
            left,
            weight,
            3,
            4,
            2,
            bias,
            1.1,
        )[0]

    assert torch.autograd.gradcheck(operation, (x, gate), rtol=1e-5, atol=1e-6)
    assert torch.autograd.gradgradcheck(operation, (x, gate), rtol=1e-5, atol=1e-6)


def test_router_supports_empty_token_dimension() -> None:
    x = torch.empty(0, 5)
    gate = torch.randn(8, 5)
    routing = grouped_topk(
        x,
        gate,
        topk=3,
        n_groups=4,
        topk_groups=2,
        backend="reference",
    )

    assert routing.weights.shape == (0, 3)
    assert routing.indices.shape == (0, 3)


def test_explicit_cuda_router_rejects_ineligible_inputs() -> None:
    x, gate, _bias = _router_inputs(dtype=torch.float32)
    with pytest.raises(RuntimeError, match="CUDA grouped router is unavailable"):
        grouped_topk(x, gate, topk=2, backend="cuda")
    if torch.cuda.is_available():
        cuda_x = x.cuda()
        cuda_gate = gate.cuda()
        with pytest.raises(RuntimeError, match="sigmoid"):
            grouped_topk(
                cuda_x,
                cuda_gate,
                topk=2,
                score_func="softmax",
                backend="cuda",
            )


def test_cuda_router_capability_flag_is_consistent() -> None:
    assert isinstance(cuda_router_available(), bool)
    if cuda_router_available():
        assert torch.cuda.is_available()


@pytest.mark.skipif(not cuda_router_available(), reason="requires native CUDA grouped router")
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("tokens", "experts", "topk", "n_groups", "topk_groups", "with_bias"),
    [
        (0, 8, 2, 1, 1, False),
        (7, 8, 3, 4, 2, False),
        (11, 8, 2, 4, 1, True),
        (5, 9, 4, 3, 2, True),
    ],
)
def test_cuda_router_matches_reference_for_group_and_tail_shapes(
    tokens: int,
    experts: int,
    topk: int,
    n_groups: int,
    topk_groups: int,
    with_bias: bool,
) -> None:
    torch.manual_seed(409)
    x = torch.randn(tokens, 7, device="cuda", requires_grad=True)
    gate = torch.randn(experts, 7, device="cuda", requires_grad=True)
    bias = torch.linspace(-0.25, 0.3, experts, device="cuda") if with_bias else None
    expected_x = x.detach().clone().requires_grad_()
    expected_gate = gate.detach().clone().requires_grad_()

    actual = grouped_topk(
        x,
        gate,
        topk=topk,
        n_groups=n_groups,
        topk_groups=topk_groups,
        score_bias=bias,
        route_scale=1.4,
        backend="cuda",
    )
    expected = deepseek_grouped_topk(
        expected_x,
        expected_gate,
        topk=topk,
        n_groups=n_groups,
        topk_groups=topk_groups,
        score_bias=bias,
        route_scale=1.4,
    )

    torch.testing.assert_close(actual.indices, expected.indices)
    torch.testing.assert_close(actual.weights, expected.weights, rtol=3e-5, atol=3e-5)
    if tokens:
        upstream = torch.randn_like(actual.weights)
        actual.weights.backward(upstream)
        expected.weights.backward(upstream)
        torch.testing.assert_close(x.grad, expected_x.grad, rtol=5e-5, atol=5e-5)
        torch.testing.assert_close(gate.grad, expected_gate.grad, rtol=5e-5, atol=5e-5)


@pytest.mark.skipif(not cuda_router_available(), reason="requires native CUDA grouped router")
@pytest.mark.cuda
def test_cuda_router_matches_documented_tie_break() -> None:
    x = torch.tensor([[100.0]], device="cuda")
    gate = torch.ones(4, 1, device="cuda")

    routing = grouped_topk(
        x,
        gate,
        topk=2,
        n_groups=2,
        topk_groups=1,
        backend="cuda",
    )

    assert routing.indices.tolist() == [[0, 1]]
    torch.testing.assert_close(routing.weights, torch.tensor([[0.5, 0.5]], device="cuda"))


@pytest.mark.skipif(not cuda_router_available(), reason="requires native CUDA grouped router")
@pytest.mark.cuda
def test_cuda_router_uses_current_stream() -> None:
    x = torch.empty(7, 5, device="cuda")
    gate = torch.empty(8, 5, device="cuda")
    generator = torch.Generator(device="cpu").manual_seed(421)
    gate_source = torch.randn(8, 5, generator=generator)
    stream = torch.cuda.Stream()
    with torch.no_grad(), torch.cuda.stream(stream):
        x.fill_(0.25)
        gate.copy_(gate_source)
        actual = grouped_topk(x, gate, topk=2, n_groups=4, topk_groups=2, backend="cuda")
        actual.weights.record_stream(stream)
        actual.indices.record_stream(stream)
    stream.synchronize()
    expected = deepseek_grouped_topk(
        x,
        gate,
        topk=2,
        n_groups=4,
        topk_groups=2,
    )

    torch.testing.assert_close(actual.indices, expected.indices)
    torch.testing.assert_close(actual.weights, expected.weights, rtol=3e-5, atol=3e-5)
