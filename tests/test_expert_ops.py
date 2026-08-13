import pytest
import torch

from ds_flash_mla_moe import (
    cuda_expert_ops_available,
    expert_major_pack,
    swiglu_experts_expert_major,
    swiglu_experts_padded_reference,
)


def _pack_inputs(
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    activations = torch.arange(30, dtype=dtype, device=device).reshape(6, 5)
    activations.requires_grad_(requires_grad)
    expert_indices = torch.tensor([9, 3, 9, 7, 3, 7], device=device)
    local_expert_ids = torch.tensor([7, 9, 3], device=device)
    return activations, expert_indices, local_expert_ids


def test_reference_expert_major_pack_preserves_identity_and_gradients() -> None:
    activations, expert_indices, local_expert_ids = _pack_inputs(requires_grad=True)
    packed, offsets, inverse = expert_major_pack(
        activations,
        expert_indices,
        local_expert_ids,
        backend="reference",
    )

    assert offsets.tolist() == [0, 2, 4, 6]
    assert inverse.tolist() == [2, 4, 3, 0, 5, 1]
    torch.testing.assert_close(packed, activations[[3, 5, 0, 2, 1, 4]])
    torch.testing.assert_close(packed.index_select(0, inverse), activations)
    upstream = torch.randn_like(packed)
    packed.backward(upstream)
    torch.testing.assert_close(activations.grad, upstream.index_select(0, inverse))


def test_raw_expert_major_pack_passes_opcheck_and_torch_compile() -> None:
    inputs = _pack_inputs(requires_grad=True)
    result = torch.library.opcheck(
        torch.ops.ds_flash_mla_moe.expert_major_pack.default,
        inputs,
    )
    assert set(result.values()) == {"SUCCESS"}

    @torch.compile(fullgraph=True, backend="eager")
    def operation(*arguments: torch.Tensor) -> torch.Tensor:
        packed, _offsets, inverse = torch.ops.ds_flash_mla_moe.expert_major_pack.default(*arguments)
        return packed.index_select(0, inverse)

    torch.testing.assert_close(operation(*(tensor.detach() for tensor in inputs)), inputs[0])


def test_expert_major_pack_supports_no_rows_and_no_local_experts() -> None:
    activations = torch.empty(0, 5, requires_grad=True)
    expert_indices = torch.empty(0, dtype=torch.long)
    local_expert_ids = torch.empty(0, dtype=torch.long)

    packed, offsets, inverse = expert_major_pack(
        activations,
        expert_indices,
        local_expert_ids,
        backend="reference",
    )
    packed.sum().backward()

    assert packed.shape == (0, 5)
    assert offsets.tolist() == [0]
    assert inverse.shape == (0,)
    assert activations.grad is not None


def _expert_inputs(
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(307)
    activations = torch.randn(
        7,
        5,
        device=device,
        dtype=dtype,
        generator=generator,
        requires_grad=requires_grad,
    )
    offsets = torch.tensor([0, 2, 2, 7], device=device, dtype=torch.long)
    w1 = torch.randn(
        3,
        4,
        5,
        device=device,
        dtype=dtype,
        generator=generator,
        requires_grad=requires_grad,
    )
    w2 = torch.randn(
        3,
        5,
        4,
        device=device,
        dtype=dtype,
        generator=generator,
        requires_grad=requires_grad,
    )
    w3 = torch.randn(
        3,
        4,
        5,
        device=device,
        dtype=dtype,
        generator=generator,
        requires_grad=requires_grad,
    )
    return activations, offsets, w1, w2, w3


def test_reference_dispatch_matches_padded_specification_and_gradients() -> None:
    inputs = _expert_inputs(requires_grad=True)
    expected_inputs = tuple(
        tensor if index == 1 else tensor.detach().clone().requires_grad_()
        for index, tensor in enumerate(inputs)
    )
    actual = swiglu_experts_expert_major(*inputs, backend="reference")
    expected = swiglu_experts_padded_reference(*expected_inputs)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)

    torch.testing.assert_close(actual, expected)
    for index in (0, 2, 3, 4):
        torch.testing.assert_close(inputs[index].grad, expected_inputs[index].grad)


def test_raw_expert_operator_passes_opcheck_and_torch_compile() -> None:
    inputs = _expert_inputs(requires_grad=True)
    result = torch.library.opcheck(
        torch.ops.ds_flash_mla_moe.swiglu_experts.default,
        inputs,
    )
    assert set(result.values()) == {"SUCCESS"}

    @torch.compile(fullgraph=True, backend="eager")
    def operation(*arguments: torch.Tensor) -> torch.Tensor:
        return torch.ops.ds_flash_mla_moe.swiglu_experts.default(*arguments)

    actual = operation(*(tensor.detach() for tensor in inputs))
    expected = swiglu_experts_padded_reference(*(tensor.detach() for tensor in inputs))
    torch.testing.assert_close(actual, expected)


def test_raw_expert_operator_supports_second_order_gradients() -> None:
    inputs = _expert_inputs(requires_grad=True)

    def operation(
        activations: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.ds_flash_mla_moe.swiglu_experts.default(
            activations,
            inputs[1],
            w1,
            w2,
            w3,
        )

    assert torch.autograd.gradcheck(operation, (inputs[0], inputs[2], inputs[3], inputs[4]))
    assert torch.autograd.gradgradcheck(operation, (inputs[0], inputs[2], inputs[3], inputs[4]))


def test_empty_expert_rows_have_well_defined_shape_and_gradients() -> None:
    activations = torch.empty(0, 5, dtype=torch.float64, requires_grad=True)
    offsets = torch.tensor([0, 0, 0], dtype=torch.long)
    weights = (
        torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True),
        torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True),
    )
    output = swiglu_experts_expert_major(
        activations,
        offsets,
        *weights,
        backend="reference",
    )
    output.sum().backward()

    assert output.shape == (0, 5)
    assert activations.grad is not None
    for weight in weights:
        assert weight.grad is not None
        torch.testing.assert_close(weight.grad, torch.zeros_like(weight))


def test_explicit_cuda_expert_backend_rejects_cpu_tensors() -> None:
    with pytest.raises(RuntimeError, match="CUDA expert compute is unavailable"):
        swiglu_experts_expert_major(*_expert_inputs(dtype=torch.float32), backend="cuda")
    with pytest.raises(RuntimeError, match="CUDA expert-major pack is unavailable"):
        expert_major_pack(*_pack_inputs(), backend="cuda")


def test_explicit_cuda_expert_backend_rejects_unsupported_dtype() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA tensors to reach the dtype contract")
    with pytest.raises(RuntimeError, match="float16 and float32"):
        swiglu_experts_expert_major(
            *_expert_inputs(device="cuda", dtype=torch.bfloat16),
            backend="cuda",
        )
    with pytest.raises(RuntimeError, match="float16 and float32"):
        expert_major_pack(
            *_pack_inputs(device="cuda", dtype=torch.bfloat16),
            backend="cuda",
        )


def test_cuda_expert_capability_flag_is_consistent() -> None:
    assert isinstance(cuda_expert_ops_available(), bool)
    if cuda_expert_ops_available():
        assert torch.cuda.is_available()


@pytest.mark.skipif(
    not cuda_expert_ops_available(),
    reason="requires a native CUDA expert kernel",
)
@pytest.mark.cuda
def test_cuda_expert_forward_and_backward_match_reference_with_empty_expert() -> None:
    inputs = _expert_inputs(device="cuda", dtype=torch.float32, requires_grad=True)
    expected_inputs = tuple(
        tensor if index == 1 else tensor.detach().clone().requires_grad_()
        for index, tensor in enumerate(inputs)
    )
    actual = swiglu_experts_expert_major(*inputs, backend="cuda")
    expected = swiglu_experts_padded_reference(*expected_inputs)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    for index in (0, 2, 3, 4):
        torch.testing.assert_close(
            inputs[index].grad,
            expected_inputs[index].grad,
            rtol=3e-4,
            atol=3e-4,
        )


@pytest.mark.skipif(
    not cuda_expert_ops_available(),
    reason="requires a native CUDA expert kernel",
)
@pytest.mark.cuda
def test_cuda_expert_kernel_uses_current_stream_and_supports_tail_shapes() -> None:
    inputs = _expert_inputs(device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    with torch.no_grad(), torch.cuda.stream(stream):
        inputs[0].fill_(0.25)
        actual = swiglu_experts_expert_major(*inputs, backend="cuda")
        actual.record_stream(stream)
    stream.synchronize()
    expected = swiglu_experts_padded_reference(*inputs)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(
    not cuda_expert_ops_available(),
    reason="requires a native CUDA expert kernel",
)
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("counts", "model_dim", "hidden_dim"),
    [
        ((1,), 1, 1),
        ((0, 0), 33, 65),
        ((17, 0, 5, 31), 33, 65),
        ((0, 16, 1), 16, 17),
        ((15, 16, 17), 31, 32),
    ],
)
def test_cuda_grouped_tiled_experts_cover_skew_empty_experts_and_tile_tails(
    counts: tuple[int, ...],
    model_dim: int,
    hidden_dim: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(331)
    rows = sum(counts)
    offsets = torch.tensor(
        [0, *torch.tensor(counts, dtype=torch.long).cumsum(0).tolist()],
        device="cuda",
    )
    inputs = (
        torch.randn(rows, model_dim, device="cuda", generator=generator),
        offsets,
        torch.randn(len(counts), hidden_dim, model_dim, device="cuda", generator=generator),
        torch.randn(len(counts), model_dim, hidden_dim, device="cuda", generator=generator),
        torch.randn(len(counts), hidden_dim, model_dim, device="cuda", generator=generator),
    )

    actual = swiglu_experts_expert_major(*inputs, backend="cuda")
    expected = swiglu_experts_padded_reference(*inputs)

    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(
    not cuda_expert_ops_available(),
    reason="requires a native CUDA expert kernel",
)
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("counts", "model_dim", "hidden_dim"),
    [
        ((1,), 1, 1),
        ((0, 0), 33, 65),
        ((17, 0, 5, 31), 33, 65),
        ((0, 16, 1), 16, 17),
    ],
)
def test_cuda_fp16_wmma_experts_match_mixed_precision_reference_and_gradients(
    counts: tuple[int, ...],
    model_dim: int,
    hidden_dim: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(337)
    rows = sum(counts)
    offsets = torch.tensor(
        [0, *torch.tensor(counts, dtype=torch.long).cumsum(0).tolist()],
        device="cuda",
    )
    inputs = (
        torch.randn(
            rows,
            model_dim,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
            requires_grad=True,
        ),
        offsets,
        torch.randn(
            len(counts),
            hidden_dim,
            model_dim,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
            requires_grad=True,
        ),
        torch.randn(
            len(counts),
            model_dim,
            hidden_dim,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
            requires_grad=True,
        ),
        torch.randn(
            len(counts),
            hidden_dim,
            model_dim,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
            requires_grad=True,
        ),
    )
    expected_inputs = tuple(
        tensor if index == 1 else tensor.detach().clone().requires_grad_()
        for index, tensor in enumerate(inputs)
    )

    actual = swiglu_experts_expert_major(*inputs, backend="cuda")
    expected = swiglu_experts_padded_reference(*expected_inputs)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    for index in (0, 2, 3, 4):
        torch.testing.assert_close(
            inputs[index].grad,
            expected_inputs[index].grad,
            rtol=3e-2,
            atol=3e-2,
        )


@pytest.mark.skipif(
    not cuda_expert_ops_available(),
    reason="requires a native CUDA expert kernel",
)
@pytest.mark.cuda
def test_cuda_fp16_wmma_experts_use_current_stream() -> None:
    inputs = _expert_inputs(device="cuda", dtype=torch.float16)
    stream = torch.cuda.Stream()
    with torch.no_grad(), torch.cuda.stream(stream):
        inputs[0].fill_(0.25)
        actual = swiglu_experts_expert_major(*inputs, backend="cuda")
        actual.record_stream(stream)
    stream.synchronize()
    expected = swiglu_experts_padded_reference(*inputs)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not cuda_expert_ops_available(),
    reason="requires native CUDA expert kernels",
)
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cuda_expert_major_pack_matches_reference_and_uses_current_stream(
    dtype: torch.dtype,
) -> None:
    inputs = _pack_inputs(device="cuda", dtype=dtype, requires_grad=True)
    expected_inputs = (
        inputs[0].detach().clone().requires_grad_(),
        inputs[1],
        inputs[2],
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = expert_major_pack(*inputs, backend="cuda")
        actual[0].record_stream(stream)
    stream.synchronize()
    expected = expert_major_pack(*expected_inputs, backend="reference")

    torch.testing.assert_close(actual[1], expected[1])
    actual_restored = actual[0].index_select(0, actual[2])
    expected_restored = expected[0].index_select(0, expected[2])
    torch.testing.assert_close(actual_restored, inputs[0])
    torch.testing.assert_close(actual_restored, expected_restored)
    upstream = torch.randn_like(actual_restored)
    actual_restored.backward(upstream)
    expected_restored.backward(upstream)
    torch.testing.assert_close(inputs[0].grad, expected_inputs[0].grad)
