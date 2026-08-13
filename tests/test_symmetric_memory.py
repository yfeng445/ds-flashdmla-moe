from __future__ import annotations

import pytest

from ds_flash_mla_moe.symmetric_memory import (
    SymmetricMoEBufferLayout,
    symmetric_moe_buffer_estimate,
    symmetric_moe_buffer_model_from_routes,
    symmetric_moe_buffer_offset,
)


def test_symmetric_buffer_model_derives_observed_no_drop_capacity() -> None:
    model = symmetric_moe_buffer_model_from_routes(
        [[3, 0, 0, 0], [4, 0, 0, 0]],
        expert_owner=[0, 1, 0, 1],
        model_dim=4,
        element_size_bytes=8,
    )

    assert model["tensor_shape_per_rank"] == [2, 2, 2, 2, 4, 4]
    assert model["local_experts_per_rank"] == [2, 2]
    assert model["observed_cell_capacity_required_without_drop"] == 4
    assert model["modeled_cell_capacity"] == 4
    assert model["accepted_route_rows"] == 7
    assert model["dropped_route_rows"] == 0
    assert model["row_slots_per_rank"] == 64
    assert model["bytes_per_rank"] == 64 * 4 * 8
    assert model["aggregate_bytes"] == 2 * 64 * 4 * 8
    assert model["active_buffer_row_placements"] == 7 * 2 * 2
    assert model["storage_utilization"] == pytest.approx(28 / 128)


def test_symmetric_buffer_model_reports_explicit_cell_overflow() -> None:
    model = symmetric_moe_buffer_model_from_routes(
        [[3, 0, 0, 0], [4, 0, 0, 0]],
        expert_owner=[0, 1, 0, 1],
        model_dim=4,
        element_size_bytes=4,
        cell_capacity=2,
    )

    assert model["capacity_basis"] == "explicit"
    assert model["accepted_route_rows"] == 4
    assert model["dropped_route_rows"] == 3
    assert model["dropped_route_fraction"] == pytest.approx(3 / 7)
    assert model["storage_utilization"] == pytest.approx(16 / 64)


def test_symmetric_buffer_layout_uses_row_major_offsets() -> None:
    layout = SymmetricMoEBufferLayout(
        world_size=2,
        rounds=2,
        local_expert_slots=2,
        cell_capacity=3,
        model_dim=5,
        element_size_bytes=4,
    )

    assert (
        symmetric_moe_buffer_offset(
            layout,
            peer=0,
            round_index=0,
            buffer_index=0,
            local_expert=0,
            row=0,
            feature=0,
        )
        == 0
    )
    assert (
        symmetric_moe_buffer_offset(
            layout,
            peer=1,
            round_index=1,
            buffer_index=1,
            local_expert=1,
            row=2,
            feature=4,
        )
        == 2 * 2 * 2 * 2 * 3 * 5 - 1
    )


def test_symmetric_buffer_estimate_rejects_more_routes_than_cells() -> None:
    layout = SymmetricMoEBufferLayout(
        world_size=2,
        rounds=2,
        local_expert_slots=1,
        cell_capacity=3,
        model_dim=4,
        element_size_bytes=4,
    )
    with pytest.raises(ValueError, match="exceed"):
        symmetric_moe_buffer_estimate(layout, aggregate_active_route_rows=13)


@pytest.mark.parametrize(
    ("counts", "owners", "kwargs", "message"),
    [
        ([], [0], {}, "source rank"),
        ([[1]], [], {}, "at least one expert"),
        ([[1, 0], [0]], [0, 1], {}, "shape"),
        ([[1, -1]], [0, 0], {}, "negative"),
        ([[1]], [1], {}, "outside world_size"),
        ([[0]], [0], {}, "all-zero"),
        ([[1]], [0], {"cell_capacity": 0}, "positive"),
    ],
)
def test_invalid_symmetric_buffer_models_are_rejected(
    counts: list[list[int]],
    owners: list[int],
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        symmetric_moe_buffer_model_from_routes(
            counts,
            expert_owner=owners,
            model_dim=4,
            element_size_bytes=4,
            **kwargs,
        )


def test_symmetric_buffer_offset_rejects_an_out_of_range_coordinate() -> None:
    layout = SymmetricMoEBufferLayout(2, 2, 1, 3, 4, 4)
    with pytest.raises(IndexError, match="peer"):
        symmetric_moe_buffer_offset(
            layout,
            peer=2,
            round_index=0,
            buffer_index=0,
            local_expert=0,
            row=0,
            feature=0,
        )
