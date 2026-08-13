from contextlib import contextmanager

import pytest
import torch

from ds_flash_mla_moe import (
    cuda_route_ops_available,
    route_combine,
    route_pack,
)


@contextmanager
def _deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def _route_inputs(
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.arange(15, dtype=dtype, device=device).reshape(5, 3)
    weights = torch.tensor(
        [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1], [0.25, 0.75]],
        dtype=dtype,
        device=device,
    )
    indices = torch.tensor(
        [[0, 3], [2, 1], [3, 0], [1, 2], [0, 1]],
        dtype=torch.long,
        device=device,
    )
    owner = torch.tensor([1, 0, 1, 0], dtype=torch.long, device=device)
    return x, weights, indices, owner


def _route_identity(pack) -> set[tuple[int, int, int]]:
    return set(
        zip(
            pack.token_indices.tolist(),
            pack.slot_indices.tolist(),
            pack.expert_indices.tolist(),
        )
    )


def test_reference_route_pack_preserves_unweighted_activations_and_identity() -> None:
    x, weights, indices, owner = _route_inputs()

    packed = route_pack(x, weights, indices, owner, world_size=2, backend="reference")

    assert packed.rank_counts.tolist() == [5, 5]
    assert packed.counts_per_expert.tolist() == [3, 3, 2, 2]
    assert packed.expert_indices.tolist() == sorted(
        packed.expert_indices.tolist(), key=lambda expert: (int(owner[expert]), expert)
    )
    expected_identity = {
        (token, slot, int(indices[token, slot]))
        for token in range(indices.shape[0])
        for slot in range(indices.shape[1])
    }
    assert _route_identity(packed) == expected_identity
    torch.testing.assert_close(packed.activations, x[packed.token_indices])
    torch.testing.assert_close(
        packed.route_weights,
        weights[packed.token_indices, packed.slot_indices],
    )


def test_route_combine_matches_explicit_post_expert_weighting_and_gradients() -> None:
    torch.manual_seed(127)
    contributions = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(8, dtype=torch.float64, requires_grad=True)
    token_indices = torch.tensor([2, 0, 1, 2, 0, 1, 2, 0])
    expected_contributions = contributions.detach().clone().requires_grad_()
    expected_weights = weights.detach().clone().requires_grad_()

    actual = route_combine(
        contributions,
        weights,
        token_indices,
        token_count=3,
        backend="reference",
    )
    expected = torch.zeros(3, 4, dtype=torch.float64).index_add(
        0,
        token_indices,
        expected_contributions * expected_weights.unsqueeze(-1),
    )
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(contributions.grad, expected_contributions.grad)
    torch.testing.assert_close(weights.grad, expected_weights.grad)


def test_route_custom_operators_support_second_order_gradients() -> None:
    torch.manual_seed(129)
    contributions = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(5, dtype=torch.float64, requires_grad=True)
    token_indices = torch.tensor([0, 1, 0, 1, 0])

    def operation(values: torch.Tensor, route_weights: torch.Tensor) -> torch.Tensor:
        return torch.ops.ds_flash_mla_moe.route_combine.default(
            values,
            route_weights,
            token_indices,
            2,
        )

    assert torch.autograd.gradcheck(operation, (contributions, weights))
    assert torch.autograd.gradgradcheck(operation, (contributions, weights))


def test_route_pack_custom_operator_gradients_follow_route_identity() -> None:
    x, weights, indices, owner = _route_inputs()
    x.requires_grad_()
    weights.requires_grad_()
    expected_x = x.detach().clone().requires_grad_()
    expected_weights = weights.detach().clone().requires_grad_()

    packed = torch.ops.ds_flash_mla_moe.route_pack.default(x, weights, indices, owner, 2)
    upstream_x = torch.randn_like(packed[0])
    upstream_weights = torch.randn_like(packed[1])
    loss = (packed[0] * upstream_x).sum() + (packed[1] * upstream_weights).sum()
    loss.backward()

    route_indices = packed[2]
    expected_packed_x = expected_x.repeat_interleave(weights.shape[1], dim=0).index_select(
        0, route_indices
    )
    expected_packed_weights = expected_weights.reshape(-1).index_select(0, route_indices)
    expected_loss = (expected_packed_x * upstream_x).sum() + (
        expected_packed_weights * upstream_weights
    ).sum()
    expected_loss.backward()

    torch.testing.assert_close(x.grad, expected_x.grad)
    torch.testing.assert_close(weights.grad, expected_weights.grad)


def test_raw_route_custom_operators_pass_opcheck() -> None:
    x, weights, indices, owner = _route_inputs()
    pack_result = torch.library.opcheck(
        torch.ops.ds_flash_mla_moe.route_pack.default,
        (x, weights, indices, owner, 2),
    )
    contributions = torch.randn(10, 3, dtype=torch.float64)
    combine_result = torch.library.opcheck(
        torch.ops.ds_flash_mla_moe.route_combine.default,
        (contributions, weights.reshape(-1), torch.arange(10) // 2, 5),
    )

    assert set(pack_result.values()) == {"SUCCESS"}
    assert set(combine_result.values()) == {"SUCCESS"}


def test_route_ops_run_through_torch_compile() -> None:
    x, weights, indices, owner = _route_inputs(dtype=torch.float32)

    @torch.compile(fullgraph=True, backend="eager")
    def operation(
        activations: torch.Tensor,
        route_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        packed = torch.ops.ds_flash_mla_moe.route_pack.default(
            activations,
            route_weights,
            expert_indices,
            owner,
            2,
        )
        return torch.ops.ds_flash_mla_moe.route_combine.default(
            packed[0],
            packed[1],
            torch.div(packed[2], 2, rounding_mode="floor"),
            activations.shape[0],
        )

    actual = operation(x, weights, indices)
    expected = x * weights.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(actual, expected)


def test_empty_route_pack_and_combine_have_well_defined_shapes() -> None:
    x = torch.empty(0, 3)
    weights = torch.empty(0, 2)
    indices = torch.empty(0, 2, dtype=torch.long)
    owner = torch.tensor([0, 1, 0], dtype=torch.long)

    packed = route_pack(x, weights, indices, owner, world_size=2)
    combined = route_combine(
        packed.activations,
        packed.route_weights,
        packed.token_indices,
        token_count=0,
    )

    assert packed.activations.shape == (0, 3)
    assert packed.rank_counts.tolist() == [0, 0]
    assert packed.counts_per_expert.tolist() == [0, 0, 0]
    assert combined.shape == (0, 3)


def test_deterministic_empty_reference_combine_is_well_defined() -> None:
    contributions = torch.empty(0, 3)
    weights = torch.empty(0)
    token_indices = torch.empty(0, dtype=torch.long)
    with _deterministic_algorithms():
        output = route_combine(
            contributions,
            weights,
            token_indices,
            token_count=0,
            backend="auto",
        )
    assert output.shape == (0, 3)


def test_explicit_cuda_route_backend_rejects_cpu_tensors() -> None:
    x, weights, indices, owner = _route_inputs(dtype=torch.float32)
    with pytest.raises(RuntimeError, match="CUDA route pack is unavailable"):
        route_pack(x, weights, indices, owner, world_size=2, backend="cuda")
    with pytest.raises(RuntimeError, match="CUDA route combine is unavailable"):
        route_combine(x, weights[:, 0], torch.arange(5), token_count=5, backend="cuda")


def test_reference_route_ops_reject_out_of_range_metadata() -> None:
    x, weights, indices, owner = _route_inputs(dtype=torch.float32)
    invalid_experts = indices.clone()
    invalid_experts[0, 0] = owner.numel()
    with pytest.raises(ValueError, match="outside"):
        route_pack(
            x,
            weights,
            invalid_experts,
            owner,
            world_size=2,
            backend="reference",
        )
    with pytest.raises(ValueError, match="outside"):
        route_combine(
            x,
            weights[:, 0],
            torch.tensor([0, 1, 2, 3, 5]),
            token_count=5,
            backend="reference",
        )


@pytest.mark.skipif(
    not cuda_route_ops_available(),
    reason="requires native CUDA route kernels and a CUDA device",
)
@pytest.mark.cuda
def test_cuda_route_pack_matches_reference_up_to_within_expert_order() -> None:
    x, weights, indices, owner = _route_inputs(device="cuda", dtype=torch.float32)

    actual = route_pack(x, weights, indices, owner, world_size=2, backend="cuda")
    expected = route_pack(x, weights, indices, owner, world_size=2, backend="reference")

    torch.testing.assert_close(actual.rank_counts, expected.rank_counts)
    torch.testing.assert_close(actual.counts_per_expert, expected.counts_per_expert)
    assert _route_identity(actual) == _route_identity(expected)
    torch.testing.assert_close(actual.activations, x[actual.token_indices])
    torch.testing.assert_close(
        actual.route_weights,
        weights[actual.token_indices, actual.slot_indices],
    )
    expert_keys = owner[actual.expert_indices] * owner.numel() + actual.expert_indices
    assert torch.all(expert_keys[1:] >= expert_keys[:-1])


@pytest.mark.skipif(
    not cuda_route_ops_available(),
    reason="requires native CUDA route kernels and a CUDA device",
)
@pytest.mark.cuda
def test_cuda_route_pack_uses_current_stream() -> None:
    x, weights, indices, owner = _route_inputs(device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()

    with torch.no_grad(), torch.cuda.stream(stream):
        x.fill_(0.25)
        weights.fill_(0.5)
        packed = route_pack(x, weights, indices, owner, world_size=2, backend="cuda")
        packed.activations.record_stream(stream)
        packed.route_weights.record_stream(stream)
    stream.synchronize()

    torch.testing.assert_close(packed.activations, torch.full_like(packed.activations, 0.25))
    torch.testing.assert_close(packed.route_weights, torch.full_like(packed.route_weights, 0.5))


@pytest.mark.skipif(
    not cuda_route_ops_available(),
    reason="requires native CUDA route kernels and a CUDA device",
)
@pytest.mark.cuda
def test_cuda_route_combine_matches_reference_and_uses_current_stream() -> None:
    torch.manual_seed(131)
    contributions = torch.randn(10, 7, device="cuda")
    weights = torch.randn(10, device="cuda")
    token_indices = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], device="cuda")
    stream = torch.cuda.Stream()

    with torch.no_grad(), torch.cuda.stream(stream):
        contributions.fill_(0.25)
        weights.fill_(0.5)
        actual = route_combine(
            contributions,
            weights,
            token_indices,
            token_count=5,
            backend="cuda",
        )
        actual.record_stream(stream)
    stream.synchronize()
    expected = route_combine(
        contributions,
        weights,
        token_indices,
        token_count=5,
        backend="reference",
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(
    not cuda_route_ops_available(),
    reason="requires native CUDA route kernels and a CUDA device",
)
@pytest.mark.cuda
def test_cuda_route_reference_backend_does_not_redispatch_to_native_kernel() -> None:
    contributions = torch.randn(6, 3, dtype=torch.float64, device="cuda")
    weights = torch.randn(6, dtype=torch.float64, device="cuda")
    token_indices = torch.tensor([0, 1, 2, 0, 1, 2], device="cuda")

    actual = route_combine(
        contributions,
        weights,
        token_indices,
        token_count=3,
        backend="reference",
    )
    expected = torch.zeros(3, 3, dtype=torch.float64, device="cuda").index_add(
        0,
        token_indices,
        contributions * weights.unsqueeze(-1),
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not cuda_route_ops_available(),
    reason="requires native CUDA route kernels and a CUDA device",
)
@pytest.mark.cuda
def test_cuda_route_auto_backend_has_deterministic_fallback() -> None:
    x, weights, indices, owner = _route_inputs(device="cuda", dtype=torch.float32)

    with _deterministic_algorithms():
        packed = route_pack(x, weights, indices, owner, world_size=2, backend="auto")
        actual = route_combine(
            packed.activations.square(),
            packed.route_weights,
            packed.token_indices,
            token_count=x.shape[0],
            backend="auto",
        )
        with pytest.raises(RuntimeError, match="nondeterministic"):
            route_pack(x, weights, indices, owner, world_size=2, backend="cuda")
        with pytest.raises(RuntimeError, match="nondeterministic"):
            route_combine(
                packed.activations,
                packed.route_weights,
                packed.token_indices,
                token_count=x.shape[0],
                backend="cuda",
            )

    expected = x.square() * weights.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(
    not cuda_route_ops_available(),
    reason="requires native CUDA route kernels and a CUDA device",
)
@pytest.mark.cuda
def test_cuda_route_ops_support_empty_route_tensors() -> None:
    x = torch.empty(0, 3, device="cuda")
    weights = torch.empty(0, 2, device="cuda")
    indices = torch.empty(0, 2, dtype=torch.long, device="cuda")
    owner = torch.tensor([0, 1, 0], dtype=torch.long, device="cuda")

    packed = route_pack(x, weights, indices, owner, world_size=2, backend="cuda")
    output = route_combine(
        packed.activations,
        packed.route_weights,
        packed.token_indices,
        token_count=0,
        backend="cuda",
    )

    assert packed.activations.shape == (0, 3)
    assert packed.rank_counts.tolist() == [0, 0]
    assert packed.counts_per_expert.tolist() == [0, 0, 0]
    assert output.shape == (0, 3)


@pytest.mark.skipif(
    not cuda_route_ops_available(),
    reason="requires native CUDA route kernels and a CUDA device",
)
@pytest.mark.cuda
def test_cuda_route_ops_autograd_matches_reference() -> None:
    x, weights, indices, owner = _route_inputs(device="cuda", dtype=torch.float32)
    x.requires_grad_()
    weights.requires_grad_()
    expected_x = x.detach().clone().requires_grad_()
    expected_weights = weights.detach().clone().requires_grad_()

    actual_pack = route_pack(x, weights, indices, owner, world_size=2, backend="cuda")
    expected_pack = route_pack(
        expected_x,
        expected_weights,
        indices,
        owner,
        world_size=2,
        backend="reference",
    )
    actual = route_combine(
        actual_pack.activations.square(),
        actual_pack.route_weights,
        actual_pack.token_indices,
        token_count=5,
        backend="cuda",
    )
    expected = route_combine(
        expected_pack.activations.square(),
        expected_pack.route_weights,
        expected_pack.token_indices,
        token_count=5,
        backend="reference",
    )
    actual.sum().backward()
    expected.sum().backward()

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(x.grad, expected_x.grad, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(weights.grad, expected_weights.grad, rtol=2e-6, atol=2e-6)
