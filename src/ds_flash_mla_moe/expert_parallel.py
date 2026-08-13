"""Correctness-first Expert Parallel MoE over ``torch.distributed``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple

import torch
import torch.distributed as dist
from torch import Tensor

from .expert_ops import expert_major_pack, swiglu_experts_expert_major
from .moe import (
    swiglu_expert,
    swiglu_experts_expert_major_reference,
    swiglu_experts_padded_reference,
    to_expert_major_reference,
)
from .route_ops import RouteBackend, RoutePackResult, route_combine, route_pack
from .router_ops import RouterBackend, grouped_topk

ExpertComputeBackend = Literal["loop", "padded", "cuda"]


class ExpertParallelTrace(NamedTuple):
    """Communication metadata for inspecting an Expert Parallel forward pass."""

    packed: RoutePackResult
    recv_counts: Tensor
    received_expert_indices: Tensor
    local_expert_ids: Tensor


class _RankMajorChunkPlan(NamedTuple):
    """Per-chunk rank splits and source-buffer ranges."""

    splits: tuple[tuple[int, ...], ...]
    ranges: tuple[tuple[tuple[int, int], ...], ...]


class _PendingCollective:
    """Keep an async collective and its input storage alive until stream wait."""

    def __init__(self) -> None:
        self._work: dist.Work | None = None
        self._input_buffer: Tensor | None = None
        self._output_buffer: Tensor | None = None

    def attach(
        self,
        work: dist.Work | None,
        input_buffer: Tensor,
        output_buffer: Tensor,
    ) -> None:
        if work is None:
            raise RuntimeError("an asynchronous all-to-all did not return a Work handle")
        if self._work is not None:
            raise RuntimeError("a pending collective cannot be reused")
        self._work = work
        self._input_buffer = input_buffer
        self._output_buffer = output_buffer

    def wait(self) -> None:
        if self._work is None:
            raise RuntimeError("the pending collective has no unfinished work")
        self._work.wait()
        self._work = None
        self._input_buffer = None
        self._output_buffer = None


def _split_tuple(counts: Tensor, world_size: int, name: str) -> tuple[int, ...]:
    if counts.shape != (world_size,):
        raise ValueError(f"{name} must have shape [world_size]")
    values = tuple(int(value) for value in counts.detach().cpu().tolist())
    if any(value < 0 for value in values):
        raise ValueError(f"{name} cannot contain a negative split")
    return values


def _rank_major_chunk_plan(
    splits: tuple[int, ...],
    chunks: int,
) -> _RankMajorChunkPlan:
    """Split every rank segment evenly while preserving rank-major ordering."""

    if chunks <= 0:
        raise ValueError("chunks must be positive")
    if any(count < 0 for count in splits):
        raise ValueError("rank splits cannot be negative")
    rank_offsets = [0]
    for count in splits:
        rank_offsets.append(rank_offsets[-1] + count)

    chunk_splits = [[] for _ in range(chunks)]
    chunk_ranges = [[] for _ in range(chunks)]
    for rank, count in enumerate(splits):
        quotient, remainder = divmod(count, chunks)
        cursor = rank_offsets[rank]
        for chunk in range(chunks):
            chunk_count = quotient + int(chunk < remainder)
            chunk_splits[chunk].append(chunk_count)
            chunk_ranges[chunk].append((cursor, cursor + chunk_count))
            cursor += chunk_count
        if cursor != rank_offsets[rank + 1]:
            raise RuntimeError("rank-major chunk planning did not cover a segment")
    return _RankMajorChunkPlan(
        splits=tuple(tuple(values) for values in chunk_splits),
        ranges=tuple(tuple(values) for values in chunk_ranges),
    )


def _gather_rank_major_chunk(
    tensor: Tensor,
    ranges: tuple[tuple[int, int], ...],
) -> Tensor:
    """Gather one slice from every rank segment into a contiguous chunk."""

    pieces = [tensor[start:end] for start, end in ranges if end > start]
    if not pieces:
        return tensor[:0].contiguous()
    if len(pieces) == 1:
        return pieces[0].contiguous()
    return torch.cat(pieces, dim=0).contiguous()


def _launch_all_to_all_single_raw(
    tensor: Tensor,
    send_splits: tuple[int, ...],
    recv_splits: tuple[int, ...],
    group: dist.ProcessGroup | None,
) -> tuple[Tensor, dist.Work | None, Tensor]:
    if tensor.ndim < 1:
        raise ValueError("all-to-all input must have at least one dimension")
    if sum(send_splits) != tensor.shape[0]:
        raise ValueError("send splits do not cover the all-to-all input")

    input_buffer = tensor.contiguous()
    output = tensor.new_empty((sum(recv_splits), *tensor.shape[1:]))
    work = dist.all_to_all_single(
        output,
        input_buffer,
        output_split_sizes=list(recv_splits),
        input_split_sizes=list(send_splits),
        group=group,
        async_op=True,
    )
    return output, work, input_buffer


def _all_to_all_single_raw(
    tensor: Tensor,
    send_splits: tuple[int, ...],
    recv_splits: tuple[int, ...],
    group: dist.ProcessGroup | None,
) -> Tensor:
    if tensor.ndim < 1:
        raise ValueError("all-to-all input must have at least one dimension")
    if sum(send_splits) != tensor.shape[0]:
        raise ValueError("send splits do not cover the all-to-all input")

    output = tensor.new_empty((sum(recv_splits), *tensor.shape[1:]))
    dist.all_to_all_single(
        output,
        tensor.contiguous(),
        output_split_sizes=list(recv_splits),
        input_split_sizes=list(send_splits),
        group=group,
    )
    return output


class _VariableAllToAll(torch.autograd.Function):
    """Variable-size all-to-all whose backward performs the inverse exchange."""

    @staticmethod
    def forward(
        ctx: object,
        tensor: Tensor,
        _autograd_anchor: Tensor,
        send_splits: tuple[int, ...],
        recv_splits: tuple[int, ...],
        group: dist.ProcessGroup | None,
    ) -> Tensor:
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        ctx.group = group
        return _all_to_all_single_raw(tensor, send_splits, recv_splits, group)

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, None, None, None, None]:
        grad_input = _all_to_all_single_raw(
            grad_output,
            ctx.recv_splits,
            ctx.send_splits,
            ctx.group,
        )
        return grad_input, None, None, None, None


class _AsyncVariableAllToAll(torch.autograd.Function):
    """Async forward all-to-all with an ordered inverse collective backward."""

    @staticmethod
    def forward(
        ctx: object,
        tensor: Tensor,
        order_token: Tensor,
        send_splits: tuple[int, ...],
        recv_splits: tuple[int, ...],
        group: dist.ProcessGroup | None,
        pending: _PendingCollective,
    ) -> tuple[Tensor, Tensor]:
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        ctx.group = group
        output, work, input_buffer = _launch_all_to_all_single_raw(
            tensor,
            send_splits,
            recv_splits,
            group,
        )
        pending.attach(work, input_buffer, output)
        return output, order_token.clone()

    @staticmethod
    def backward(
        ctx: object,
        grad_output: Tensor,
        grad_order_token: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, None, None, None, None]:
        grad_input = _all_to_all_single_raw(
            grad_output,
            ctx.recv_splits,
            ctx.send_splits,
            ctx.group,
        )
        return grad_input, grad_order_token, None, None, None, None


def _collective_autograd_anchor(
    reference: Tensor,
    differentiable_inputs: tuple[Tensor | None, ...],
    group: dist.ProcessGroup | None,
) -> Tensor:
    """Keep every rank in collective backward even when its local shard is empty."""

    local_enabled = torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in differentiable_inputs
    )
    enabled = torch.tensor(int(local_enabled), device=reference.device, dtype=torch.int32)
    dist.all_reduce(enabled, op=dist.ReduceOp.MAX, group=group)
    return reference.new_zeros((), requires_grad=bool(enabled.item()))


def _validate_local_experts(
    x: Tensor,
    gate_weight: Tensor,
    local_expert_w1: Tensor,
    local_expert_w2: Tensor,
    local_expert_w3: Tensor,
    expert_owner: Tensor,
    *,
    rank: int,
    world_size: int,
) -> Tensor:
    integer_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    if x.ndim < 2:
        raise ValueError("x must have shape [..., tokens, model_dim]")
    if gate_weight.ndim != 2:
        raise ValueError("gate_weight must have shape [experts, model_dim]")
    experts, model_dim = gate_weight.shape
    if x.shape[-1] != model_dim:
        raise ValueError("x model dimension does not match gate_weight")
    if gate_weight.device != x.device:
        raise ValueError("x and gate_weight must be on the same device")
    if expert_owner.shape != (experts,) or expert_owner.device != x.device:
        raise ValueError("expert_owner must be [experts] on the same device as x")
    if expert_owner.dtype not in integer_dtypes:
        raise TypeError("expert_owner must use an integer dtype")
    if torch.any(expert_owner < 0) or torch.any(expert_owner >= world_size):
        raise ValueError("expert_owner contains a rank outside the process group")

    local_expert_ids = torch.where(expert_owner.to(torch.long) == rank)[0]
    local_experts = local_expert_ids.numel()
    if local_expert_w1.ndim != 3 or local_expert_w2.ndim != 3 or local_expert_w3.ndim != 3:
        raise ValueError("local expert weights must be rank-3 tensors")
    if local_expert_w1.shape[0] != local_experts:
        raise ValueError("local_expert_w1 must contain exactly the experts owned by this rank")
    hidden = local_expert_w1.shape[1]
    if hidden <= 0 or local_expert_w1.shape[2] != model_dim:
        raise ValueError("local_expert_w1 must have shape [local_experts, hidden, model_dim]")
    if local_expert_w3.shape != (local_experts, hidden, model_dim):
        raise ValueError("local_expert_w3 shape must match local_expert_w1")
    if local_expert_w2.shape != (local_experts, model_dim, hidden):
        raise ValueError("local_expert_w2 must have shape [local_experts, model_dim, hidden]")
    if any(
        weight.device != x.device for weight in (local_expert_w1, local_expert_w2, local_expert_w3)
    ):
        raise ValueError("x and local expert weights must be on the same device")
    return local_expert_ids


def _compute_received_experts(
    received_activations: Tensor,
    received_expert_indices: Tensor,
    *,
    expert_owner: Tensor,
    local_expert_ids: Tensor,
    local_expert_w1: Tensor,
    local_expert_w2: Tensor,
    local_expert_w3: Tensor,
    expert_backend: ExpertComputeBackend,
    rank: int,
) -> Tensor:
    """Regroup one received chunk, run local experts, and restore source order."""

    if expert_backend != "cuda" and received_expert_indices.numel() > 0:
        received_owners = expert_owner.to(torch.long)[received_expert_indices]
        if torch.any(received_owners != rank):
            raise RuntimeError("received a route for an expert not owned by this rank")

    if expert_backend == "cuda":
        expert_major_activations, local_offsets, inverse_permutation = expert_major_pack(
            received_activations,
            received_expert_indices,
            local_expert_ids,
            backend="cuda",
        )
    else:
        expert_major = to_expert_major_reference(
            received_activations,
            received_expert_indices,
            n_experts=expert_owner.numel(),
        )
        expert_major_activations = expert_major.activations
        inverse_permutation = expert_major.inverse_permutation
        local_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=received_activations.device),
                expert_major.counts_per_expert[local_expert_ids].cumsum(0),
            )
        )

    if expert_backend == "loop":
        expert_major_contributions = swiglu_experts_expert_major_reference(
            expert_major_activations,
            local_offsets,
            local_expert_ids,
            local_expert_w1,
            local_expert_w2,
            local_expert_w3,
        )
    elif expert_backend == "padded":
        expert_major_contributions = swiglu_experts_padded_reference(
            expert_major_activations,
            local_offsets,
            local_expert_w1,
            local_expert_w2,
            local_expert_w3,
        )
    else:
        expert_major_contributions = swiglu_experts_expert_major(
            expert_major_activations,
            local_offsets,
            local_expert_w1,
            local_expert_w2,
            local_expert_w3,
            backend="cuda",
        )
    return expert_major_contributions.index_select(0, inverse_permutation)


def _chunked_dispatch_compute_restore(
    packed: RoutePackResult,
    *,
    send_splits: tuple[int, ...],
    recv_splits: tuple[int, ...],
    chunks: int,
    anchor: Tensor,
    expert_owner: Tensor,
    local_expert_ids: Tensor,
    local_expert_w1: Tensor,
    local_expert_w2: Tensor,
    local_expert_w3: Tensor,
    expert_backend: ExpertComputeBackend,
    rank: int,
    group: dist.ProcessGroup | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pipeline chunk dispatch/compute/restore with ordered autograd collectives."""

    send_plan = _rank_major_chunk_plan(send_splits, chunks)
    recv_plan = _rank_major_chunk_plan(recv_splits, chunks)
    order_token = anchor
    source_positions = torch.arange(
        packed.activations.shape[0],
        dtype=torch.long,
        device=packed.activations.device,
    )
    received_positions = torch.arange(
        sum(recv_splits),
        dtype=torch.long,
        device=packed.activations.device,
    )

    def launch_dispatch(
        chunk: int,
        token: Tensor,
    ) -> tuple[
        tuple[Tensor, Tensor, _PendingCollective, _PendingCollective, Tensor],
        Tensor,
    ]:
        activation_input = _gather_rank_major_chunk(
            packed.activations,
            send_plan.ranges[chunk],
        )
        activation_pending = _PendingCollective()
        received_activations, next_token = _AsyncVariableAllToAll.apply(
            activation_input,
            token,
            send_plan.splits[chunk],
            recv_plan.splits[chunk],
            group,
            activation_pending,
        )

        index_input = _gather_rank_major_chunk(
            packed.expert_indices,
            send_plan.ranges[chunk],
        )
        received_indices, index_work, index_buffer = _launch_all_to_all_single_raw(
            index_input,
            send_plan.splits[chunk],
            recv_plan.splits[chunk],
            group,
        )
        index_pending = _PendingCollective()
        index_pending.attach(index_work, index_buffer, received_indices)
        source_indices = _gather_rank_major_chunk(
            source_positions,
            send_plan.ranges[chunk],
        )
        return (
            (
                received_activations,
                received_indices,
                activation_pending,
                index_pending,
                source_indices,
            ),
            next_token,
        )

    pending_dispatch, order_token = launch_dispatch(0, order_token)
    restored_chunks: list[Tensor] = []
    restored_indices: list[Tensor] = []
    restore_pending: list[_PendingCollective] = []
    received_index_chunks: list[Tensor] = []
    received_position_chunks: list[Tensor] = []
    for chunk in range(chunks):
        (
            received_activations,
            received_indices,
            activation_pending,
            index_pending,
            source_indices,
        ) = pending_dispatch
        activation_pending.wait()
        index_pending.wait()
        received_index_chunks.append(received_indices)
        received_position_chunks.append(
            _gather_rank_major_chunk(
                received_positions,
                recv_plan.ranges[chunk],
            )
        )

        next_dispatch = None
        if chunk + 1 < chunks:
            next_dispatch, order_token = launch_dispatch(chunk + 1, order_token)
        received_contributions = _compute_received_experts(
            received_activations,
            received_indices,
            expert_owner=expert_owner,
            local_expert_ids=local_expert_ids,
            local_expert_w1=local_expert_w1,
            local_expert_w2=local_expert_w2,
            local_expert_w3=local_expert_w3,
            expert_backend=expert_backend,
            rank=rank,
        )

        pending = _PendingCollective()
        returned, order_token = _AsyncVariableAllToAll.apply(
            received_contributions,
            order_token,
            recv_plan.splits[chunk],
            send_plan.splits[chunk],
            group,
            pending,
        )
        restored_chunks.append(returned)
        restored_indices.append(source_indices)
        restore_pending.append(pending)
        if next_dispatch is not None:
            pending_dispatch = next_dispatch

    for pending in restore_pending:
        pending.wait()
    route_count = packed.activations.shape[0]
    returned_values = torch.cat(restored_chunks, dim=0)
    returned_indices = torch.cat(restored_indices, dim=0)
    returned_contributions = packed.activations.new_zeros(
        (route_count, packed.activations.shape[1])
    ).index_copy(0, returned_indices, returned_values)
    returned_contributions = (
        returned_contributions + order_token.to(returned_contributions.dtype) * 0
    )
    received_expert_indices = packed.expert_indices.new_empty(sum(recv_splits)).index_copy(
        0,
        torch.cat(received_position_chunks, dim=0),
        torch.cat(received_index_chunks, dim=0),
    )
    return returned_contributions, received_expert_indices, order_token


def deepseek_moe_expert_parallel_reference(
    x: Tensor,
    gate_weight: Tensor,
    local_expert_w1: Tensor,
    local_expert_w2: Tensor,
    local_expert_w3: Tensor,
    *,
    expert_owner: Tensor,
    topk: int,
    n_groups: int = 1,
    topk_groups: int | None = None,
    score_func: str = "sigmoid",
    score_bias: Tensor | None = None,
    route_scale: float = 1.0,
    shared_w1: Tensor | None = None,
    shared_w2: Tensor | None = None,
    shared_w3: Tensor | None = None,
    group: dist.ProcessGroup | None = None,
    router_backend: RouterBackend = "auto",
    route_backend: RouteBackend = "auto",
    expert_backend: ExpertComputeBackend = "loop",
    pipeline_chunks: int = 1,
    return_trace: bool = False,
    stage_observer: Callable[[str], None] | None = None,
) -> Tensor | tuple[Tensor, ExpertParallelTrace]:
    """Run a rank-local DeepSeek-style MoE through two variable all-to-alls.

    ``expert_owner`` contains process-group-local ranks. Local expert weights
    correspond to this rank's global expert ids in ascending order. The router
    is replicated, while ``x`` contains only this rank's token shard.

    Every rank in ``group`` must call forward, and—when gradients are needed—
    backward in the same order. Routing weights stay on the source rank and are
    applied only after nonlinear expert outputs return.
    """

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before Expert Parallel MoE")
    if expert_backend not in {"loop", "padded", "cuda"}:
        raise ValueError("expert_backend must be loop, padded, or cuda")
    if not isinstance(pipeline_chunks, int) or isinstance(pipeline_chunks, bool):
        raise TypeError("pipeline_chunks must be an integer")
    if pipeline_chunks <= 0:
        raise ValueError("pipeline_chunks must be positive")
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    local_expert_ids = _validate_local_experts(
        x,
        gate_weight,
        local_expert_w1,
        local_expert_w2,
        local_expert_w3,
        expert_owner,
        rank=rank,
        world_size=world_size,
    )

    backend = str(dist.get_backend(group=group)).lower()
    if backend == "gloo" and x.device.type != "cpu":
        raise ValueError("the Gloo reference path requires CPU tensors")
    if backend == "nccl" and x.device.type != "cuda":
        raise ValueError("an NCCL process group requires CUDA tensors")
    if pipeline_chunks > 1 and backend != "nccl":
        raise ValueError("pipeline_chunks > 1 requires an NCCL process group")
    if score_bias is not None and score_bias.device != x.device:
        raise ValueError("score_bias must be on the same device as x")

    shared = (shared_w1, shared_w2, shared_w3)
    if any(weight is None for weight in shared) and not all(weight is None for weight in shared):
        raise ValueError("provide all three shared expert weights or none of them")
    if any(weight is not None and weight.device != x.device for weight in shared):
        raise ValueError("shared expert weights must be on the same device as x")

    routing = grouped_topk(
        x,
        gate_weight,
        topk=topk,
        n_groups=n_groups,
        topk_groups=topk_groups,
        score_func=score_func,
        score_bias=score_bias,
        route_scale=route_scale,
        backend=router_backend,
    )
    flat_x = x.reshape(-1, x.shape[-1])
    flat_route_weights = routing.weights.reshape(-1, topk)
    flat_expert_indices = routing.indices.reshape(-1, topk)
    packed = route_pack(
        flat_x,
        flat_route_weights,
        flat_expert_indices,
        expert_owner.to(torch.long),
        world_size=world_size,
        backend=route_backend,
    )
    if stage_observer is not None:
        stage_observer("route_and_pack")

    send_counts = packed.rank_counts.to(torch.long).contiguous()
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    send_splits = _split_tuple(send_counts, world_size, "send_counts")
    recv_splits = _split_tuple(recv_counts, world_size, "recv_counts")
    if stage_observer is not None:
        stage_observer("exchange_counts")

    anchor = _collective_autograd_anchor(
        x,
        (
            x,
            gate_weight,
            local_expert_w1,
            local_expert_w2,
            local_expert_w3,
            score_bias,
            shared_w1,
            shared_w2,
            shared_w3,
        ),
        group,
    )
    if pipeline_chunks > 1:
        returned_contributions, received_expert_indices, _order_token = (
            _chunked_dispatch_compute_restore(
                packed,
                send_splits=send_splits,
                recv_splits=recv_splits,
                chunks=pipeline_chunks,
                anchor=anchor,
                expert_owner=expert_owner,
                local_expert_ids=local_expert_ids,
                local_expert_w1=local_expert_w1,
                local_expert_w2=local_expert_w2,
                local_expert_w3=local_expert_w3,
                expert_backend=expert_backend,
                rank=rank,
                group=group,
            )
        )
        if stage_observer is not None:
            stage_observer("pipelined_core")
    else:
        received_activations = _VariableAllToAll.apply(
            packed.activations,
            anchor,
            send_splits,
            recv_splits,
            group,
        )
        received_expert_indices = _all_to_all_single_raw(
            packed.expert_indices,
            send_splits,
            recv_splits,
            group,
        )
        if stage_observer is not None:
            stage_observer("dispatch")

        received_contributions = _compute_received_experts(
            received_activations,
            received_expert_indices,
            expert_owner=expert_owner,
            local_expert_ids=local_expert_ids,
            local_expert_w1=local_expert_w1,
            local_expert_w2=local_expert_w2,
            local_expert_w3=local_expert_w3,
            expert_backend=expert_backend,
            rank=rank,
        )
        if stage_observer is not None:
            stage_observer("expert_compute")

        returned_contributions = _VariableAllToAll.apply(
            received_contributions,
            anchor,
            recv_splits,
            send_splits,
            group,
        )
        if stage_observer is not None:
            stage_observer("restore")
    output = route_combine(
        returned_contributions,
        packed.route_weights,
        packed.token_indices,
        token_count=flat_x.shape[0],
        backend=route_backend,
    ).reshape_as(x)
    if stage_observer is not None:
        stage_observer("combine")
    if shared_w1 is not None and shared_w2 is not None and shared_w3 is not None:
        output = output + swiglu_expert(x, shared_w1, shared_w2, shared_w3)
        if stage_observer is not None:
            stage_observer("shared_expert")

    if return_trace:
        return output, ExpertParallelTrace(
            packed=packed,
            recv_counts=recv_counts,
            received_expert_indices=received_expert_indices,
            local_expert_ids=local_expert_ids,
        )
    return output
