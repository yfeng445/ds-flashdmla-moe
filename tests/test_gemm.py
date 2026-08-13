import pytest
import torch

from ds_flash_mla_moe import (
    cuda_gemm_available,
    gemm_reference,
    tiled_gemm,
    tiled_gemm_reference,
)


@pytest.mark.parametrize(
    ("m", "n", "k", "tile_m", "tile_n", "tile_k"),
    [
        (1, 1, 1, 1, 1, 1),
        (2, 5, 1, 4, 3, 2),
        (5, 3, 7, 2, 4, 3),
        (37, 29, 23, 16, 8, 7),
    ],
)
def test_tiled_gemm_matches_materialized_reference_for_tail_shapes(
    m: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> None:
    generator = torch.Generator().manual_seed(m * 100 + n * 10 + k)
    a = torch.randn(m, k, dtype=torch.float64, generator=generator)
    b = torch.randn(k, n, dtype=torch.float64, generator=generator)

    actual = tiled_gemm_reference(
        a,
        b,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )

    torch.testing.assert_close(actual, gemm_reference(a, b), rtol=1e-12, atol=1e-12)


def test_generalized_alpha_beta_epilogue_matches_formula() -> None:
    torch.manual_seed(211)
    a = torch.randn(5, 7, dtype=torch.float64)
    b = torch.randn(7, 3, dtype=torch.float64)
    c = torch.randn(5, 3, dtype=torch.float64)

    actual = tiled_gemm_reference(
        a,
        b,
        c,
        alpha=0.75,
        beta=-0.25,
        tile_m=4,
        tile_n=2,
        tile_k=3,
    )

    torch.testing.assert_close(actual, 0.75 * (a @ b) - 0.25 * c, rtol=1e-12, atol=1e-12)


def test_tiled_gemm_supports_noncontiguous_inputs() -> None:
    torch.manual_seed(223)
    a = torch.randn(7, 5, dtype=torch.float64).T
    b = torch.randn(3, 7, dtype=torch.float64).T
    assert not a.is_contiguous() and not b.is_contiguous()

    actual = tiled_gemm_reference(a, b, tile_m=3, tile_n=2, tile_k=4)

    torch.testing.assert_close(actual, a @ b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("m", "n", "k"), [(0, 3, 2), (2, 0, 3), (2, 3, 0)])
def test_tiled_gemm_supports_empty_dimensions_and_keeps_autograd_connected(
    m: int,
    n: int,
    k: int,
) -> None:
    a = torch.randn(m, k, dtype=torch.float64, requires_grad=True)
    b = torch.randn(k, n, dtype=torch.float64, requires_grad=True)

    output = tiled_gemm_reference(a, b, tile_m=2, tile_n=2, tile_k=2)
    gradients = torch.autograd.grad(output.sum(), (a, b))

    assert output.shape == (m, n)
    torch.testing.assert_close(output, a @ b)
    assert gradients[0].shape == a.shape
    assert gradients[1].shape == b.shape


def test_tiled_gemm_passes_first_and_second_order_gradient_checks() -> None:
    torch.manual_seed(227)
    a = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    b = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    c = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)

    def operation(left, right, epilogue):
        return tiled_gemm_reference(
            left,
            right,
            epilogue,
            alpha=0.7,
            beta=0.2,
            tile_m=2,
            tile_n=3,
            tile_k=2,
        )

    assert torch.autograd.gradcheck(operation, (a, b, c))
    assert torch.autograd.gradgradcheck(operation, (a, b, c))


@pytest.mark.parametrize(
    ("a", "b", "c", "kwargs", "message"),
    [
        (torch.ones(2), torch.ones(2, 2), None, {}, "rank-2"),
        (torch.ones(2, 3), torch.ones(2, 4), None, {}, "inner"),
        (torch.ones(2, 3), torch.ones(3, 4, dtype=torch.float64), None, {}, "dtype"),
        (
            torch.ones(2, 3, dtype=torch.int64),
            torch.ones(3, 4, dtype=torch.int64),
            None,
            {},
            "floating",
        ),
        (torch.ones(2, 3), torch.ones(3, 4), None, {"beta": 1.0}, "requires"),
        (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), {}, "shape"),
        (torch.ones(2, 3), torch.ones(3, 4), None, {"tile_k": 0}, "positive"),
    ],
)
def test_tiled_gemm_rejects_invalid_contracts(a, b, c, kwargs, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        tiled_gemm_reference(a, b, c, **kwargs)


def test_dispatcher_reference_and_auto_backends_match_gemm_specification() -> None:
    torch.manual_seed(233)
    a = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    b = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    c = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    expected_inputs = [tensor.detach().clone().requires_grad_() for tensor in (a, b, c)]
    upstream = torch.randn_like(c)

    actual = tiled_gemm(a, b, c, alpha=0.7, beta=-0.2, backend="auto")
    expected = gemm_reference(
        *expected_inputs,
        alpha=0.7,
        beta=-0.2,
    )
    actual.backward(upstream)
    expected.backward(upstream)

    torch.testing.assert_close(actual, expected)
    for actual_input, expected_input in zip((a, b, c), expected_inputs):
        torch.testing.assert_close(actual_input.grad, expected_input.grad)
    torch.testing.assert_close(
        tiled_gemm(a.detach(), b.detach(), backend="reference"),
        gemm_reference(a.detach(), b.detach()),
    )


def test_explicit_cuda_gemm_rejects_cpu_tensors() -> None:
    a = torch.randn(3, 5)
    b = torch.randn(5, 4)
    with pytest.raises(RuntimeError, match="CUDA tiled GEMM is unavailable"):
        tiled_gemm(a, b, backend="cuda")


def test_invalid_dispatcher_backend_is_rejected() -> None:
    a = torch.randn(3, 5)
    b = torch.randn(5, 4)
    with pytest.raises(ValueError, match="backend"):
        tiled_gemm(a, b, backend="unknown")  # type: ignore[arg-type]


def test_raw_tiled_gemm_operator_passes_opcheck_and_torch_compile() -> None:
    a = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    b = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    c = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

    result = torch.library.opcheck(
        torch.ops.ds_flash_mla_moe.tiled_gemm.default,
        (a, b, c, 0.7, -0.2),
    )
    assert set(result.values()) == {"SUCCESS"}

    @torch.compile(fullgraph=True, backend="eager")
    def operation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.ops.ds_flash_mla_moe.tiled_gemm.default(left, right, None, 1.0, 0.0)

    torch.testing.assert_close(operation(a.detach(), b.detach()), a.detach() @ b.detach())


def test_raw_tiled_gemm_operator_supports_second_order_gradients() -> None:
    torch.manual_seed(239)
    inputs = (
        torch.randn(3, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(5, 4, dtype=torch.float64, requires_grad=True),
        torch.randn(3, 4, dtype=torch.float64, requires_grad=True),
    )

    def operation(a, b, c):
        return torch.ops.ds_flash_mla_moe.tiled_gemm.default(a, b, c, 0.6, -0.3)

    assert torch.autograd.gradcheck(operation, inputs)
    assert torch.autograd.gradgradcheck(operation, inputs)


def test_cuda_gemm_capability_flag_is_consistent() -> None:
    assert isinstance(cuda_gemm_available(), bool)
    if cuda_gemm_available():
        assert torch.cuda.is_available()


@pytest.mark.skipif(not cuda_gemm_available(), reason="requires native CUDA tiled GEMM")
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (0, 5, 3),
        (3, 0, 5),
        (3, 4, 0),
        (1, 1, 1),
        (16, 16, 16),
        (37, 29, 23),
    ],
)
def test_cuda_tiled_gemm_matches_reference_for_tail_shapes(m: int, n: int, k: int) -> None:
    torch.manual_seed(241)
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    c = torch.randn(m, n, device="cuda")

    actual = tiled_gemm(a, b, c, alpha=0.7, beta=-0.2, backend="cuda")
    expected = gemm_reference(a, b, c, alpha=0.7, beta=-0.2)

    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.skipif(not cuda_gemm_available(), reason="requires native CUDA tiled GEMM")
@pytest.mark.cuda
def test_cuda_tiled_gemm_uses_current_stream() -> None:
    a = torch.empty(37, 23, device="cuda")
    b = torch.empty(23, 29, device="cuda")
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        a.fill_(0.25)
        b.fill_(0.5)
        output = tiled_gemm(a, b, backend="cuda")
        output.record_stream(stream)
    stream.synchronize()

    torch.testing.assert_close(output, torch.full_like(output, 23 * 0.25 * 0.5))


@pytest.mark.skipif(not cuda_gemm_available(), reason="requires native CUDA tiled GEMM")
@pytest.mark.cuda
def test_cuda_tiled_gemm_uses_registered_analytic_backward() -> None:
    torch.manual_seed(251)
    inputs = [
        torch.randn(17, 23, device="cuda", requires_grad=True),
        torch.randn(23, 11, device="cuda", requires_grad=True),
        torch.randn(17, 11, device="cuda", requires_grad=True),
    ]
    expected_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    upstream = torch.randn(17, 11, device="cuda")

    actual = tiled_gemm(*inputs, alpha=0.8, beta=0.3, backend="cuda")
    expected = gemm_reference(*expected_inputs, alpha=0.8, beta=0.3)
    actual.backward(upstream)
    expected.backward(upstream)

    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    for actual_input, expected_input in zip(inputs, expected_inputs):
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=3e-5, atol=3e-5)
