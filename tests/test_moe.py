import pytest
import torch
import torch.nn.functional as F

from ds_flash_mla_moe import (
    RoutingResult,
    combine_packed_routes,
    deepseek_grouped_topk,
    deepseek_moe_packed_reference,
    deepseek_moe_reference,
    pack_routes_reference,
    swiglu_expert,
    swiglu_experts_expert_major_reference,
    swiglu_experts_padded_reference,
    to_expert_major_reference,
)


def test_selection_bias_does_not_change_returned_weight_values() -> None:
    x = torch.tensor([[2.0, 1.0, -1.0]])
    gate = torch.eye(3)
    bias = torch.tensor([-10.0, 0.0, 10.0])

    routing = deepseek_grouped_topk(
        x,
        gate,
        topk=2,
        score_func="sigmoid",
        score_bias=bias,
        route_scale=1.5,
    )

    assert routing.indices.tolist() == [[2, 1]]
    raw = torch.sigmoid(x).gather(1, routing.indices)
    expected_weights = raw / raw.sum(dim=-1, keepdim=True) * 1.5
    torch.testing.assert_close(routing.weights, expected_weights)


def test_group_limited_topk_only_selects_retained_groups() -> None:
    x = torch.tensor([[8.0, 7.0, 6.0, 5.0]])
    gate = torch.eye(4)
    routing = deepseek_grouped_topk(
        x,
        gate,
        topk=2,
        n_groups=2,
        topk_groups=1,
        score_func="softmax",
    )

    assert set(routing.indices.flatten().tolist()) == {0, 1}


def test_grouped_topk_breaks_exact_ties_by_smaller_group_and_expert_id() -> None:
    routing = deepseek_grouped_topk(
        torch.tensor([[100.0]]),
        torch.ones(4, 1),
        topk=2,
        n_groups=2,
        topk_groups=1,
    )

    assert routing.indices.tolist() == [[0, 1]]
    torch.testing.assert_close(routing.weights, torch.tensor([[0.5, 0.5]]))


def test_swiglu_matches_direct_formula() -> None:
    torch.manual_seed(4)
    x = torch.randn(5, 3, dtype=torch.float64)
    w1 = torch.randn(7, 3, dtype=torch.float64)
    w2 = torch.randn(3, 7, dtype=torch.float64)
    w3 = torch.randn(7, 3, dtype=torch.float64)

    expected = F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)
    actual = swiglu_expert(x, w1, w2, w3)

    torch.testing.assert_close(actual, expected)


def test_float16_swiglu_quantizes_the_materialized_hidden_state() -> None:
    torch.manual_seed(13)
    x = torch.randn(3, 5, dtype=torch.float16)
    w1 = torch.randn(7, 5, dtype=torch.float16)
    w2 = torch.randn(5, 7, dtype=torch.float16)
    w3 = torch.randn(7, 5, dtype=torch.float16)

    gate = F.linear(x.float(), w1.float())
    up = F.linear(x.float(), w3.float())
    hidden = (F.silu(gate) * up).half().float()
    expected = F.linear(hidden, w2.float()).half()

    torch.testing.assert_close(swiglu_expert(x, w1, w2, w3), expected)


def test_widened_shared_swiglu_equals_sum_of_independent_experts() -> None:
    torch.manual_seed(17)
    x = torch.randn(4, 5, dtype=torch.float64)
    w1 = torch.randn(2, 3, 5, dtype=torch.float64)
    w2 = torch.randn(2, 5, 3, dtype=torch.float64)
    w3 = torch.randn(2, 3, 5, dtype=torch.float64)

    expected = sum(swiglu_expert(x, w1[index], w2[index], w3[index]) for index in range(2))
    actual = swiglu_expert(
        x,
        w1.reshape(6, 5),
        torch.cat((w2[0], w2[1]), dim=1),
        w3.reshape(6, 5),
    )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_moe_matches_explicit_token_loop_and_has_gradients() -> None:
    torch.manual_seed(19)
    tokens, model_dim, hidden, experts = 5, 3, 4, 4
    x = torch.randn(tokens, model_dim, dtype=torch.float64, requires_grad=True)
    gate = torch.randn(experts, model_dim, dtype=torch.float64, requires_grad=True)
    w1 = torch.randn(experts, hidden, model_dim, dtype=torch.float64, requires_grad=True)
    w2 = torch.randn(experts, model_dim, hidden, dtype=torch.float64, requires_grad=True)
    w3 = torch.randn(experts, hidden, model_dim, dtype=torch.float64, requires_grad=True)
    shared_w1 = torch.randn(hidden, model_dim, dtype=torch.float64, requires_grad=True)
    shared_w2 = torch.randn(model_dim, hidden, dtype=torch.float64, requires_grad=True)
    shared_w3 = torch.randn(hidden, model_dim, dtype=torch.float64, requires_grad=True)

    actual, routing = deepseek_moe_reference(
        x,
        gate,
        w1,
        w2,
        w3,
        topk=2,
        n_groups=2,
        topk_groups=1,
        shared_w1=shared_w1,
        shared_w2=shared_w2,
        shared_w3=shared_w3,
        return_routing=True,
    )

    expected_rows = []
    for token in range(tokens):
        row = torch.zeros(model_dim, dtype=torch.float64)
        for slot in range(2):
            expert = routing.indices[token, slot]
            row = row + routing.weights[token, slot] * swiglu_expert(
                x[token], w1[expert], w2[expert], w3[expert]
            )
        row = row + swiglu_expert(x[token], shared_w1, shared_w2, shared_w3)
        expected_rows.append(row)
    expected = torch.stack(expected_rows)

    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    for tensor in (x, gate, w1, w2, w3, shared_w1, shared_w2, shared_w3):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_invalid_group_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        deepseek_grouped_topk(torch.randn(2, 3), torch.randn(5, 3), topk=2, n_groups=2)


def test_pack_metadata_is_rank_major_expert_major_and_bijective() -> None:
    x = torch.arange(12.0).reshape(4, 3)
    routing = RoutingResult(
        weights=torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]]),
        indices=torch.tensor([[0, 3], [2, 1], [3, 0], [1, 2]]),
    )
    owner = torch.tensor([1, 0, 1, 0])

    packed = pack_routes_reference(
        x,
        routing,
        n_experts=4,
        expert_owner=owner,
        world_size=2,
    )

    assert packed.destination_ranks.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert packed.expert_indices.tolist() == [1, 1, 3, 3, 0, 0, 2, 2]
    assert packed.expert_order.tolist() == [1, 3, 0, 2]
    assert packed.expert_offsets.tolist() == [0, 2, 4, 6, 8]
    assert packed.rank_counts.tolist() == [4, 4]
    assert packed.rank_offsets.tolist() == [0, 4, 8]
    route_identity = set(zip(packed.token_indices.tolist(), packed.slot_indices.tolist()))
    assert route_identity == {(token, slot) for token in range(4) for slot in range(2)}
    torch.testing.assert_close(packed.activations, x[packed.token_indices])


def test_combine_applies_weights_after_expert() -> None:
    x = torch.tensor([[2.0], [3.0]])
    routing = RoutingResult(
        weights=torch.tensor([[0.25, 0.75], [0.6, 0.4]]),
        indices=torch.tensor([[0, 1], [0, 1]]),
    )
    packed = pack_routes_reference(x, routing, n_experts=2)
    nonlinear_contributions = packed.activations.square()

    actual = combine_packed_routes(nonlinear_contributions, packed)
    expected = x.square() * routing.weights.sum(dim=-1, keepdim=True)
    incorrectly_preweighted = combine_packed_routes(
        (packed.activations * packed.route_weights.unsqueeze(-1)).square(), packed
    )

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, incorrectly_preweighted)


def test_packed_moe_matches_token_loop_with_nonmonotonic_owner() -> None:
    torch.manual_seed(37)
    tokens, model_dim, hidden, experts = 7, 5, 6, 4
    direct_inputs = [
        torch.randn(tokens, model_dim, dtype=torch.float64, requires_grad=True),
        torch.randn(experts, model_dim, dtype=torch.float64, requires_grad=True),
        torch.randn(experts, hidden, model_dim, dtype=torch.float64, requires_grad=True),
        torch.randn(experts, model_dim, hidden, dtype=torch.float64, requires_grad=True),
        torch.randn(experts, hidden, model_dim, dtype=torch.float64, requires_grad=True),
    ]
    packed_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in direct_inputs]
    owner = torch.tensor([1, 0, 1, 0])

    expected = deepseek_moe_reference(
        *direct_inputs,
        topk=2,
        n_groups=2,
        topk_groups=1,
    )
    actual = deepseek_moe_packed_reference(
        *packed_inputs,
        topk=2,
        n_groups=2,
        topk_groups=1,
        expert_owner=owner,
        world_size=2,
    )
    upstream = torch.randn_like(actual)
    expected.backward(upstream)
    actual.backward(upstream)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    for packed_input, direct_input in zip(packed_inputs, direct_inputs):
        torch.testing.assert_close(packed_input.grad, direct_input.grad, rtol=1e-9, atol=1e-9)


def test_expert_major_layout_is_stable_and_exactly_invertible() -> None:
    activations = torch.arange(24.0).reshape(8, 3)
    expert_indices = torch.tensor([3, 1, 3, 0, 1, 3, 0, 2])

    layout = to_expert_major_reference(activations, expert_indices, n_experts=4)

    assert layout.expert_indices.tolist() == [0, 0, 1, 1, 2, 3, 3, 3]
    assert layout.counts_per_expert.tolist() == [2, 2, 1, 3]
    assert layout.expert_offsets.tolist() == [0, 2, 4, 5, 8]
    assert layout.permutation.tolist() == [3, 6, 1, 4, 7, 0, 2, 5]
    torch.testing.assert_close(
        layout.activations.index_select(0, layout.inverse_permutation),
        activations,
    )


def test_expert_major_swiglu_matches_rowwise_experts_and_gradients() -> None:
    torch.manual_seed(139)
    rows, model_dim, hidden = 9, 4, 5
    activations = torch.randn(rows, model_dim, dtype=torch.float64, requires_grad=True)
    w1 = torch.randn(2, hidden, model_dim, dtype=torch.float64, requires_grad=True)
    w2 = torch.randn(2, model_dim, hidden, dtype=torch.float64, requires_grad=True)
    w3 = torch.randn(2, hidden, model_dim, dtype=torch.float64, requires_grad=True)
    expected_inputs = [
        tensor.detach().clone().requires_grad_(True) for tensor in (activations, w1, w2, w3)
    ]
    expert_ids = torch.tensor([1, 3])
    expert_offsets = torch.tensor([0, 4, 9])

    actual = swiglu_experts_expert_major_reference(
        activations,
        expert_offsets,
        expert_ids,
        w1,
        w2,
        w3,
    )
    expected = torch.cat(
        (
            swiglu_expert(
                expected_inputs[0][:4],
                expected_inputs[1][0],
                expected_inputs[2][0],
                expected_inputs[3][0],
            ),
            swiglu_expert(
                expected_inputs[0][4:],
                expected_inputs[1][1],
                expected_inputs[2][1],
                expected_inputs[3][1],
            ),
        )
    )
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)

    torch.testing.assert_close(actual, expected)
    for actual_input, expected_input in zip((activations, w1, w2, w3), expected_inputs):
        torch.testing.assert_close(actual_input.grad, expected_input.grad)


def test_padded_batched_swiglu_matches_expert_loop_and_gradients() -> None:
    torch.manual_seed(149)
    rows, model_dim, hidden, experts = 8, 3, 4, 3
    offsets = torch.tensor([0, 2, 2, 8])
    padded_inputs = [
        torch.randn(rows, model_dim, dtype=torch.float64, requires_grad=True),
        torch.randn(experts, hidden, model_dim, dtype=torch.float64, requires_grad=True),
        torch.randn(experts, model_dim, hidden, dtype=torch.float64, requires_grad=True),
        torch.randn(experts, hidden, model_dim, dtype=torch.float64, requires_grad=True),
    ]
    loop_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in padded_inputs]

    actual = swiglu_experts_padded_reference(padded_inputs[0], offsets, *padded_inputs[1:])
    expected = swiglu_experts_expert_major_reference(
        loop_inputs[0],
        offsets,
        torch.arange(experts),
        *loop_inputs[1:],
    )
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)

    torch.testing.assert_close(actual, expected)
    for padded_input, loop_input in zip(padded_inputs, loop_inputs):
        torch.testing.assert_close(padded_input.grad, loop_input.grad)
