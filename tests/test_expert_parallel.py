from __future__ import annotations

import datetime
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ds_flash_mla_moe import (
    deepseek_moe_expert_parallel_reference,
    deepseek_moe_reference,
)
from ds_flash_mla_moe.expert_parallel import (
    _chunked_dispatch_compute_restore,
    _compute_received_experts,
    _gather_rank_major_chunk,
    _rank_major_chunk_plan,
)
from ds_flash_mla_moe.route_ops import RoutePackResult


def test_rank_major_chunk_plan_preserves_every_peer_segment() -> None:
    tensor = torch.arange(10)
    plan = _rank_major_chunk_plan((5, 0, 3, 2), chunks=3)

    assert plan.splits == ((2, 0, 1, 1), (2, 0, 1, 1), (1, 0, 1, 0))
    chunks = [_gather_rank_major_chunk(tensor, ranges) for ranges in plan.ranges]
    assert [chunk.tolist() for chunk in chunks] == [
        [0, 1, 5, 8],
        [2, 3, 6, 9],
        [4, 7],
    ]


@pytest.mark.parametrize("chunks", [0, -1])
def test_rank_major_chunk_plan_rejects_nonpositive_chunks(chunks: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        _rank_major_chunk_plan((1, 2), chunks)


class _ImmediateWork:
    def wait(self) -> bool:
        return True


def _install_immediate_all_to_all(
    monkeypatch: pytest.MonkeyPatch,
    collective_modes: list[bool],
) -> None:
    def immediate_all_to_all(
        output: torch.Tensor,
        input_tensor: torch.Tensor,
        *,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        group: object,
        async_op: bool = False,
    ) -> _ImmediateWork | None:
        del group
        assert output_split_sizes == input_split_sizes
        output.copy_(input_tensor)
        collective_modes.append(async_op)
        return _ImmediateWork() if async_op else None

    monkeypatch.setattr(dist, "all_to_all_single", immediate_all_to_all)


@pytest.mark.parametrize("chunks", [3, 8])
def test_chunked_pipeline_restores_rows_gradients_and_collective_order(
    monkeypatch: pytest.MonkeyPatch,
    chunks: int,
) -> None:
    collective_modes: list[bool] = []
    _install_immediate_all_to_all(monkeypatch, collective_modes)
    torch.manual_seed(59)
    activations = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    expert_indices = torch.tensor([1, 0, 1, 1, 0, 0, 1])
    packed = RoutePackResult(
        activations=activations,
        route_weights=torch.ones(7, dtype=torch.float64),
        token_indices=torch.arange(7),
        slot_indices=torch.zeros(7, dtype=torch.long),
        expert_indices=expert_indices,
        counts_per_expert=torch.bincount(expert_indices, minlength=2),
        rank_counts=torch.tensor([7]),
    )
    w1 = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    w2 = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    w3 = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    expected_inputs = tuple(
        tensor.detach().clone().requires_grad_() for tensor in (activations, w1, w2, w3)
    )
    local_expert_ids = torch.tensor([0, 1])
    owner = torch.tensor([0, 0])

    actual, received_indices, order_token = _chunked_dispatch_compute_restore(
        packed,
        send_splits=(7,),
        recv_splits=(7,),
        chunks=chunks,
        anchor=torch.zeros((), dtype=torch.float64, requires_grad=True),
        expert_owner=owner,
        local_expert_ids=local_expert_ids,
        local_expert_w1=w1,
        local_expert_w2=w2,
        local_expert_w3=w3,
        expert_backend="padded",
        rank=0,
        group=None,
    )
    expected = _compute_received_experts(
        expected_inputs[0],
        expert_indices,
        expert_owner=owner,
        local_expert_ids=local_expert_ids,
        local_expert_w1=expected_inputs[1],
        local_expert_w2=expected_inputs[2],
        local_expert_w3=expected_inputs[3],
        expert_backend="padded",
        rank=0,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(received_indices, expert_indices)
    assert order_token.shape == ()

    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    for tensor, reference in zip((activations, w1, w2, w3), expected_inputs, strict=True):
        torch.testing.assert_close(tensor.grad, reference.grad)

    assert collective_modes == [True] * (3 * chunks) + [False] * (2 * chunks)


def test_empty_chunked_pipeline_keeps_collective_backward_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collective_modes: list[bool] = []
    _install_immediate_all_to_all(monkeypatch, collective_modes)
    chunks = 3
    activations = torch.empty(0, 3, dtype=torch.float64, requires_grad=True)
    expert_indices = torch.empty(0, dtype=torch.long)
    packed = RoutePackResult(
        activations=activations,
        route_weights=torch.empty(0, dtype=torch.float64),
        token_indices=torch.empty(0, dtype=torch.long),
        slot_indices=torch.empty(0, dtype=torch.long),
        expert_indices=expert_indices,
        counts_per_expert=torch.zeros(2, dtype=torch.long),
        rank_counts=torch.zeros(1, dtype=torch.long),
    )
    weights = (
        torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True),
        torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True),
    )

    actual, received_indices, _order_token = _chunked_dispatch_compute_restore(
        packed,
        send_splits=(0,),
        recv_splits=(0,),
        chunks=chunks,
        anchor=torch.zeros((), dtype=torch.float64, requires_grad=True),
        expert_owner=torch.tensor([0, 0]),
        local_expert_ids=torch.tensor([0, 1]),
        local_expert_w1=weights[0],
        local_expert_w2=weights[1],
        local_expert_w3=weights[2],
        expert_backend="padded",
        rank=0,
        group=None,
    )
    actual.sum().backward()

    assert actual.shape == (0, 3)
    assert received_indices.numel() == 0
    assert activations.grad is not None
    for weight in weights:
        assert weight.grad is not None
        torch.testing.assert_close(weight.grad, torch.zeros_like(weight))
    assert collective_modes == [True] * (3 * chunks) + [False] * (2 * chunks)


def _assert_uneven_forward_and_backward(rank: int, expert_backend: str = "loop") -> None:
    dtype = torch.float64
    full_x = torch.tensor(
        [
            [8.0, 7.0, -1.0, -2.0],
            [6.0, -3.0, 5.0, -4.0],
            [-1.0, 4.0, 3.0, -2.0],
            [-3.0, 6.0, -2.0, 5.0],
            [1.0, 2.0, 8.0, 7.0],
        ],
        dtype=dtype,
    )
    token_slices = (slice(0, 3), slice(3, 5))
    owner = torch.tensor([1, 0, 1, 0])

    torch.manual_seed(71)
    gate_base = torch.eye(4, dtype=dtype)
    w1_base = torch.randn(4, 5, 4, dtype=dtype)
    w2_base = torch.randn(4, 4, 5, dtype=dtype)
    w3_base = torch.randn(4, 5, 4, dtype=dtype)
    shared_w1_base = torch.randn(5, 4, dtype=dtype)
    shared_w2_base = torch.randn(4, 5, dtype=dtype)
    shared_w3_base = torch.randn(5, 4, dtype=dtype)

    local_ids = torch.where(owner == rank)[0]
    x = full_x[token_slices[rank]].clone().requires_grad_()
    gate = gate_base.clone().requires_grad_()
    local_w1 = w1_base[local_ids].clone().requires_grad_()
    local_w2 = w2_base[local_ids].clone().requires_grad_()
    local_w3 = w3_base[local_ids].clone().requires_grad_()
    shared_w1 = shared_w1_base.clone().requires_grad_()
    shared_w2 = shared_w2_base.clone().requires_grad_()
    shared_w3 = shared_w3_base.clone().requires_grad_()

    actual, trace = deepseek_moe_expert_parallel_reference(
        x,
        gate,
        local_w1,
        local_w2,
        local_w3,
        expert_owner=owner,
        topk=2,
        shared_w1=shared_w1,
        shared_w2=shared_w2,
        shared_w3=shared_w3,
        expert_backend=expert_backend,
        return_trace=True,
    )

    expected_x = full_x.clone().requires_grad_()
    expected_gate = gate_base.clone().requires_grad_()
    expected_w1 = w1_base.clone().requires_grad_()
    expected_w2 = w2_base.clone().requires_grad_()
    expected_w3 = w3_base.clone().requires_grad_()
    expected_shared_w1 = shared_w1_base.clone().requires_grad_()
    expected_shared_w2 = shared_w2_base.clone().requires_grad_()
    expected_shared_w3 = shared_w3_base.clone().requires_grad_()
    expected = deepseek_moe_reference(
        expected_x,
        expected_gate,
        expected_w1,
        expected_w2,
        expected_w3,
        topk=2,
        shared_w1=expected_shared_w1,
        shared_w2=expected_shared_w2,
        shared_w3=expected_shared_w3,
    )

    expected_send_counts = ((2, 4), (3, 1))
    expected_recv_counts = ((2, 3), (4, 1))
    assert tuple(trace.packed.rank_counts.tolist()) == expected_send_counts[rank]
    assert tuple(trace.recv_counts.tolist()) == expected_recv_counts[rank]
    assert trace.local_expert_ids.tolist() == local_ids.tolist()
    torch.testing.assert_close(actual, expected[token_slices[rank]], rtol=1e-10, atol=1e-10)

    actual.square().sum().backward()
    expected.square().sum().backward()

    for replicated_gradient in (gate.grad, shared_w1.grad, shared_w2.grad, shared_w3.grad):
        assert replicated_gradient is not None
        dist.all_reduce(replicated_gradient)

    torch.testing.assert_close(x.grad, expected_x.grad[token_slices[rank]], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(gate.grad, expected_gate.grad, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(local_w1.grad, expected_w1.grad[local_ids], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(local_w2.grad, expected_w2.grad[local_ids], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(local_w3.grad, expected_w3.grad[local_ids], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(shared_w1.grad, expected_shared_w1.grad, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(shared_w2.grad, expected_shared_w2.grad, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(shared_w3.grad, expected_shared_w3.grad, rtol=1e-9, atol=1e-9)


def _assert_zero_receive_rank(rank: int, expert_backend: str = "loop") -> None:
    dtype = torch.float64
    owner = torch.tensor([0, 0, 0])
    full_x = torch.tensor([[4.0, 3.0, 2.0], [2.0, 5.0, 1.0], [3.0, 1.0, 6.0]], dtype=dtype)
    token_slices = (slice(0, 2), slice(2, 3))
    torch.manual_seed(83)
    gate = torch.eye(3, dtype=dtype)
    w1 = torch.randn(3, 4, 3, dtype=dtype)
    w2 = torch.randn(3, 3, 4, dtype=dtype)
    w3 = torch.randn(3, 4, 3, dtype=dtype)
    local_ids = torch.where(owner == rank)[0]

    actual, trace = deepseek_moe_expert_parallel_reference(
        full_x[token_slices[rank]],
        gate,
        w1[local_ids],
        w2[local_ids],
        w3[local_ids],
        expert_owner=owner,
        topk=2,
        expert_backend=expert_backend,
        return_trace=True,
    )
    expected = deepseek_moe_reference(full_x, gate, w1, w2, w3, topk=2)

    torch.testing.assert_close(actual, expected[token_slices[rank]], rtol=1e-10, atol=1e-10)
    assert trace.received_expert_indices.numel() == (6 if rank == 0 else 0)
    assert tuple(trace.recv_counts.tolist()) == ((4, 2) if rank == 0 else (0, 0))


def _assert_empty_source_still_participates_in_backward(rank: int) -> None:
    dtype = torch.float64
    owner = torch.tensor([1, 0, 1])
    full_x = torch.tensor([[5.0, 4.0, 1.0], [2.0, 3.0, 6.0]], dtype=dtype)
    token_slices = (slice(0, 2), slice(2, 2))
    torch.manual_seed(97)
    gate_base = torch.eye(3, dtype=dtype)
    w1_base = torch.randn(3, 4, 3, dtype=dtype)
    w2_base = torch.randn(3, 3, 4, dtype=dtype)
    w3_base = torch.randn(3, 4, 3, dtype=dtype)
    local_ids = torch.where(owner == rank)[0]

    x = full_x[token_slices[rank]].clone().requires_grad_()
    gate = gate_base.clone().requires_grad_()
    local_w1 = w1_base[local_ids].clone().requires_grad_()
    local_w2 = w2_base[local_ids].clone().requires_grad_()
    local_w3 = w3_base[local_ids].clone().requires_grad_()
    actual = deepseek_moe_expert_parallel_reference(
        x,
        gate,
        local_w1,
        local_w2,
        local_w3,
        expert_owner=owner,
        topk=2,
    )

    expected_x = full_x.clone().requires_grad_()
    expected_w1 = w1_base.clone().requires_grad_()
    expected_w2 = w2_base.clone().requires_grad_()
    expected_w3 = w3_base.clone().requires_grad_()
    expected = deepseek_moe_reference(
        expected_x,
        gate_base,
        expected_w1,
        expected_w2,
        expected_w3,
        topk=2,
    )
    torch.testing.assert_close(actual, expected[token_slices[rank]], rtol=1e-10, atol=1e-10)

    actual.square().sum().backward()
    expected.square().sum().backward()

    assert x.grad is not None
    torch.testing.assert_close(x.grad, expected_x.grad[token_slices[rank]], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(local_w1.grad, expected_w1.grad[local_ids], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(local_w2.grad, expected_w2.grad[local_ids], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(local_w3.grad, expected_w3.grad[local_ids], rtol=1e-9, atol=1e-9)


def _assert_unused_owned_expert_has_explicit_zero_gradient(rank: int) -> None:
    dtype = torch.float64
    owner = torch.tensor([0, 1])
    x = torch.tensor([[5.0, -5.0], [4.0, -4.0]], dtype=dtype)[rank : rank + 1]
    gate = torch.eye(2, dtype=dtype)
    torch.manual_seed(109)
    w1 = torch.randn(2, 3, 2, dtype=dtype)
    w2 = torch.randn(2, 2, 3, dtype=dtype)
    w3 = torch.randn(2, 3, 2, dtype=dtype)
    local_ids = torch.where(owner == rank)[0]
    local_w1 = w1[local_ids].clone().requires_grad_()
    local_w2 = w2[local_ids].clone().requires_grad_()
    local_w3 = w3[local_ids].clone().requires_grad_()

    output, trace = deepseek_moe_expert_parallel_reference(
        x,
        gate,
        local_w1,
        local_w2,
        local_w3,
        expert_owner=owner,
        topk=1,
        return_trace=True,
    )
    output.square().sum().backward()

    if rank == 1:
        assert trace.received_expert_indices.numel() == 0
        for gradient in (local_w1.grad, local_w2.grad, local_w3.grad):
            assert gradient is not None
            assert torch.count_nonzero(gradient).item() == 0


def _assert_chunked_pipeline_rejects_gloo(rank: int) -> None:
    owner = torch.tensor([0, 1])
    local_ids = torch.where(owner == rank)[0]
    with pytest.raises(ValueError, match="NCCL"):
        deepseek_moe_expert_parallel_reference(
            torch.randn(1, 2),
            torch.randn(2, 2),
            torch.randn(local_ids.numel(), 3, 2),
            torch.randn(local_ids.numel(), 2, 3),
            torch.randn(local_ids.numel(), 3, 2),
            expert_owner=owner,
            topk=1,
            pipeline_chunks=2,
        )


def _expert_parallel_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        _assert_uneven_forward_and_backward(rank)
        _assert_uneven_forward_and_backward(rank, expert_backend="padded")
        _assert_zero_receive_rank(rank)
        _assert_zero_receive_rank(rank, expert_backend="padded")
        _assert_empty_source_still_participates_in_backward(rank)
        _assert_unused_owned_expert_has_explicit_zero_gradient(rank)
        _assert_chunked_pipeline_rejects_gloo(rank)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo distributed backend is unavailable",
)
def test_two_rank_gloo_expert_parallel_protocol() -> None:
    with tempfile.TemporaryDirectory() as directory:
        init_file = os.path.join(directory, "process-group-init")
        mp.spawn(_expert_parallel_worker, args=(2, init_file), nprocs=2, join=True)


def test_expert_parallel_requires_an_initialized_process_group() -> None:
    if dist.is_initialized():
        pytest.skip("the test process already owns a process group")
    with pytest.raises(RuntimeError, match="must be initialized"):
        deepseek_moe_expert_parallel_reference(
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.randn(2, 4, 3),
            torch.randn(2, 3, 4),
            torch.randn(2, 4, 3),
            expert_owner=torch.zeros(2, dtype=torch.long),
            topk=1,
        )
