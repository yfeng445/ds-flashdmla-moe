from __future__ import annotations

import pytest

from ds_flash_mla_moe.parallel_topology import (
    ExpertPlacement,
    ParallelCoordinate,
    ParallelMesh,
)


def test_parallel_mesh_uses_tp_fastest_rank_bijection() -> None:
    mesh = ParallelMesh(dp_size=2, ep_size=3, tp_size=4)

    assert mesh.world_size == 24
    for rank in range(mesh.world_size):
        coordinate = mesh.coordinate(rank)
        assert mesh.rank(coordinate) == rank

    assert mesh.rank(ParallelCoordinate(dp=1, ep=2, tp=3)) == 23
    assert mesh.coordinate(17) == ParallelCoordinate(dp=1, ep=1, tp=1)


def test_parallel_mesh_builds_deterministic_axis_groups() -> None:
    mesh = ParallelMesh(dp_size=2, ep_size=3, tp_size=4)

    assert mesh.tp_group(dp=1, ep=2) == (20, 21, 22, 23)
    assert mesh.ep_group(dp=1, tp=2) == (14, 18, 22)
    assert mesh.dp_group(ep=1, tp=3) == (7, 19)
    assert mesh.tp_groups() == tuple(
        (
            mesh.rank(ParallelCoordinate(dp, ep, 0)) + 0,
            mesh.rank(ParallelCoordinate(dp, ep, 0)) + 1,
            mesh.rank(ParallelCoordinate(dp, ep, 0)) + 2,
            mesh.rank(ParallelCoordinate(dp, ep, 0)) + 3,
        )
        for dp in range(2)
        for ep in range(3)
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dp_size": 0, "ep_size": 1, "tp_size": 1}, "positive integer"),
        ({"dp_size": 1, "ep_size": -1, "tp_size": 1}, "positive integer"),
        ({"dp_size": 1, "ep_size": 1, "tp_size": True}, "positive integer"),
    ],
)
def test_parallel_mesh_rejects_invalid_axis_sizes(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ParallelMesh(**kwargs)


def test_parallel_mesh_rejects_invalid_coordinates_and_ranks() -> None:
    mesh = ParallelMesh(dp_size=2, ep_size=2, tp_size=2)

    with pytest.raises(ValueError, match="dp coordinate"):
        mesh.rank(ParallelCoordinate(dp=2, ep=0, tp=0))
    with pytest.raises(ValueError, match="ep coordinate"):
        mesh.rank(ParallelCoordinate(dp=0, ep=-1, tp=0))
    with pytest.raises(ValueError, match="tp coordinate"):
        mesh.rank(ParallelCoordinate(dp=0, ep=0, tp=2))
    with pytest.raises(ValueError, match="rank"):
        mesh.coordinate(mesh.world_size)
    with pytest.raises(TypeError, match="ParallelCoordinate"):
        mesh.rank((0, 0, 0))  # type: ignore[arg-type]


def test_expert_placement_assigns_stable_owner_local_slots() -> None:
    mesh = ParallelMesh(dp_size=2, ep_size=3, tp_size=2)
    placement = ExpertPlacement(mesh, expert_owner=(2, 0, 2, 1, 0, 2))

    assert placement.num_experts == 6
    assert placement.local_experts(ep=0) == (1, 4)
    assert placement.local_experts(ep=1) == (3,)
    assert placement.local_experts(ep=2) == (0, 2, 5)
    assert [placement.local_slot(expert) for expert in range(6)] == [0, 0, 1, 0, 1, 2]
    assert placement.owner(5) == 2
    assert placement.expert_ranks(global_expert_id=2, dp=1) == (10, 11)


@pytest.mark.parametrize(
    ("owners", "message"),
    [
        ((), "at least one"),
        ((0, 3), "outside the EP axis"),
        ((0, -1), "outside the EP axis"),
        ((0, True), "integers"),
    ],
)
def test_expert_placement_rejects_invalid_owner_tables(
    owners: tuple[int, ...], message: str
) -> None:
    mesh = ParallelMesh(dp_size=1, ep_size=3, tp_size=2)
    with pytest.raises((TypeError, ValueError), match=message):
        ExpertPlacement(mesh, expert_owner=owners)


def test_expert_placement_rejects_invalid_queries() -> None:
    placement = ExpertPlacement(
        ParallelMesh(dp_size=2, ep_size=2, tp_size=2),
        expert_owner=(0, 1),
    )

    with pytest.raises(ValueError, match="global_expert_id"):
        placement.owner(2)
    with pytest.raises(ValueError, match="ep coordinate"):
        placement.local_experts(ep=2)
    with pytest.raises(ValueError, match="dp coordinate"):
        placement.expert_ranks(global_expert_id=0, dp=2)
