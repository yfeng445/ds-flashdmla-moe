"""Analytical symmetric-buffer layouts for one-sided Expert Parallel MoE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SymmetricMoEBufferLayout:
    """One PE's row-major ``[peer, round, buffer, expert, row, feature]`` layout.

    The two buffer roles are outgoing staging and incoming delivery. This class
    models storage only; it does not allocate symmetric memory or perform
    one-sided communication.
    """

    world_size: int
    rounds: int
    local_expert_slots: int
    cell_capacity: int
    model_dim: int
    element_size_bytes: int

    def validate(self) -> None:
        values = (
            self.world_size,
            self.rounds,
            self.local_expert_slots,
            self.cell_capacity,
            self.model_dim,
            self.element_size_bytes,
        )
        if any(value <= 0 for value in values):
            raise ValueError("symmetric-buffer dimensions and element size must be positive")

    @property
    def tensor_shape_per_rank(self) -> tuple[int, int, int, int, int, int]:
        self.validate()
        return (
            self.world_size,
            self.rounds,
            2,
            self.local_expert_slots,
            self.cell_capacity,
            self.model_dim,
        )


def symmetric_moe_buffer_offset(
    layout: SymmetricMoEBufferLayout,
    *,
    peer: int,
    round_index: int,
    buffer_index: int,
    local_expert: int,
    row: int,
    feature: int,
) -> int:
    """Return the row-major element offset for one symmetric tensor coordinate."""

    shape = layout.tensor_shape_per_rank
    indices = (peer, round_index, buffer_index, local_expert, row, feature)
    labels = ("peer", "round_index", "buffer_index", "local_expert", "row", "feature")
    for index, extent, label in zip(indices, shape, labels, strict=True):
        if not 0 <= index < extent:
            raise IndexError(f"{label} must be in [0, {extent})")

    offset = 0
    for index, extent in zip(indices, shape, strict=True):
        offset = offset * extent + index
    return offset


def symmetric_moe_buffer_estimate(
    layout: SymmetricMoEBufferLayout,
    *,
    aggregate_active_route_rows: int,
) -> dict[str, Any]:
    """Estimate storage and utilization for a two-role symmetric MoE buffer."""

    shape = layout.tensor_shape_per_rank
    if aggregate_active_route_rows < 0:
        raise ValueError("aggregate_active_route_rows must be non-negative")
    route_capacity_per_round_per_rank = (
        layout.world_size * layout.local_expert_slots * layout.cell_capacity
    )
    aggregate_route_capacity_per_round = layout.world_size * route_capacity_per_round_per_rank
    if aggregate_active_route_rows > aggregate_route_capacity_per_round:
        raise ValueError("active routes exceed the aggregate peer/expert cell capacity")

    elements_per_rank = math.prod(shape)
    row_slots_per_rank = elements_per_rank // layout.model_dim
    aggregate_elements = layout.world_size * elements_per_rank
    aggregate_row_slots = layout.world_size * row_slots_per_rank
    active_row_placements = aggregate_active_route_rows * layout.rounds * 2
    return {
        "tensor_shape_per_rank": list(shape),
        "buffer_roles": ["outgoing", "incoming"],
        "route_capacity_per_round_per_rank": route_capacity_per_round_per_rank,
        "aggregate_route_capacity_per_round": aggregate_route_capacity_per_round,
        "aggregate_active_route_rows_per_round": aggregate_active_route_rows,
        "row_slots_per_rank": row_slots_per_rank,
        "elements_per_rank": elements_per_rank,
        "bytes_per_rank": elements_per_rank * layout.element_size_bytes,
        "aggregate_row_slots": aggregate_row_slots,
        "aggregate_elements": aggregate_elements,
        "aggregate_bytes": aggregate_elements * layout.element_size_bytes,
        "active_buffer_row_placements": active_row_placements,
        "storage_utilization": (
            active_row_placements / aggregate_row_slots if aggregate_row_slots else 1.0
        ),
        "signal_and_allocator_metadata_bytes_included": False,
        "analytical_only": True,
    }


def symmetric_moe_buffer_model_from_routes(
    source_expert_counts: list[list[int]],
    *,
    expert_owner: list[int],
    model_dim: int,
    element_size_bytes: int,
    rounds: int = 2,
    cell_capacity: int | None = None,
) -> dict[str, Any]:
    """Build a symmetric-buffer model from source-rank by expert route counts."""

    world_size = len(source_expert_counts)
    if world_size == 0:
        raise ValueError("source_expert_counts must contain at least one source rank")
    experts = len(expert_owner)
    if experts == 0:
        raise ValueError("expert_owner must contain at least one expert")
    if any(len(row) != experts for row in source_expert_counts):
        raise ValueError("source_expert_counts must have shape [world_size, experts]")
    if any(value < 0 for row in source_expert_counts for value in row):
        raise ValueError("source_expert_counts cannot contain negative counts")
    if any(owner < 0 or owner >= world_size for owner in expert_owner):
        raise ValueError("expert_owner contains a rank outside world_size")

    local_experts_per_rank = [expert_owner.count(rank) for rank in range(world_size)]
    local_expert_slots = max(local_experts_per_rank)
    observed_capacity = max(value for row in source_expert_counts for value in row)
    if observed_capacity == 0 and cell_capacity is None:
        raise ValueError("an all-zero route matrix requires an explicit positive cell_capacity")
    selected_capacity = observed_capacity if cell_capacity is None else cell_capacity
    if selected_capacity <= 0:
        raise ValueError("cell_capacity must be positive")

    total_routes = sum(sum(row) for row in source_expert_counts)
    accepted_routes = sum(
        min(value, selected_capacity) for row in source_expert_counts for value in row
    )
    layout = SymmetricMoEBufferLayout(
        world_size=world_size,
        rounds=rounds,
        local_expert_slots=local_expert_slots,
        cell_capacity=selected_capacity,
        model_dim=model_dim,
        element_size_bytes=element_size_bytes,
    )
    estimate = symmetric_moe_buffer_estimate(
        layout,
        aggregate_active_route_rows=accepted_routes,
    )
    return {
        "source_expert_counts_matrix": source_expert_counts,
        "expert_owner": expert_owner,
        "local_experts_per_rank": local_experts_per_rank,
        "unassigned_expert_slots_per_rank": [
            local_expert_slots - count for count in local_experts_per_rank
        ],
        "observed_cell_capacity_required_without_drop": observed_capacity,
        "modeled_cell_capacity": selected_capacity,
        "capacity_basis": (
            "observed_max_source_expert_route_rows" if cell_capacity is None else "explicit"
        ),
        "total_route_rows": total_routes,
        "accepted_route_rows": accepted_routes,
        "dropped_route_rows": total_routes - accepted_routes,
        "dropped_route_fraction": (
            (total_routes - accepted_routes) / total_routes if total_routes else 0.0
        ),
        **estimate,
    }
