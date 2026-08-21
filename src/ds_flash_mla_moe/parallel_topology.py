"""Deterministic DP/EP/TP rank and expert-placement semantics.

This module describes logical topology only.  It does not create process groups
or perform communication.
"""

from __future__ import annotations

from dataclasses import dataclass


def _require_positive_axis(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_coordinate(value: int, size: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < size:
        raise ValueError(f"{name} coordinate must be in [0, {size})")


@dataclass(frozen=True)
class ParallelCoordinate:
    """One coordinate in a DP x EP x TP mesh."""

    dp: int
    ep: int
    tp: int


@dataclass(frozen=True)
class ParallelMesh:
    """Validated mesh whose flattened ranks use TP as the fastest axis."""

    dp_size: int
    ep_size: int
    tp_size: int

    def __post_init__(self) -> None:
        _require_positive_axis(self.dp_size, "dp_size")
        _require_positive_axis(self.ep_size, "ep_size")
        _require_positive_axis(self.tp_size, "tp_size")

    @property
    def world_size(self) -> int:
        return self.dp_size * self.ep_size * self.tp_size

    def rank(self, coordinate: ParallelCoordinate) -> int:
        """Flatten ``coordinate`` as ``((dp * EP) + ep) * TP + tp``."""

        if not isinstance(coordinate, ParallelCoordinate):
            raise TypeError("coordinate must be a ParallelCoordinate")
        _require_coordinate(coordinate.dp, self.dp_size, "dp")
        _require_coordinate(coordinate.ep, self.ep_size, "ep")
        _require_coordinate(coordinate.tp, self.tp_size, "tp")
        return ((coordinate.dp * self.ep_size) + coordinate.ep) * self.tp_size + coordinate.tp

    def coordinate(self, rank: int) -> ParallelCoordinate:
        """Invert :meth:`rank` exactly."""

        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size})")
        dp_ep, tp = divmod(rank, self.tp_size)
        dp, ep = divmod(dp_ep, self.ep_size)
        return ParallelCoordinate(dp=dp, ep=ep, tp=tp)

    def tp_group(self, *, dp: int, ep: int) -> tuple[int, ...]:
        _require_coordinate(dp, self.dp_size, "dp")
        _require_coordinate(ep, self.ep_size, "ep")
        return tuple(self.rank(ParallelCoordinate(dp, ep, tp)) for tp in range(self.tp_size))

    def ep_group(self, *, dp: int, tp: int) -> tuple[int, ...]:
        _require_coordinate(dp, self.dp_size, "dp")
        _require_coordinate(tp, self.tp_size, "tp")
        return tuple(self.rank(ParallelCoordinate(dp, ep, tp)) for ep in range(self.ep_size))

    def dp_group(self, *, ep: int, tp: int) -> tuple[int, ...]:
        _require_coordinate(ep, self.ep_size, "ep")
        _require_coordinate(tp, self.tp_size, "tp")
        return tuple(self.rank(ParallelCoordinate(dp, ep, tp)) for dp in range(self.dp_size))

    def tp_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            self.tp_group(dp=dp, ep=ep) for dp in range(self.dp_size) for ep in range(self.ep_size)
        )

    def ep_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            self.ep_group(dp=dp, tp=tp) for dp in range(self.dp_size) for tp in range(self.tp_size)
        )

    def dp_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            self.dp_group(ep=ep, tp=tp) for ep in range(self.ep_size) for tp in range(self.tp_size)
        )


@dataclass(frozen=True)
class ExpertPlacement:
    """Map global experts to EP coordinates and stable owner-local slots.

    An expert owner is an EP coordinate, not one flattened rank.  Consequently
    every expert is represented by the owner's entire TP group within a DP
    replica.  Local slots follow ascending global expert id.
    """

    mesh: ParallelMesh
    expert_owner: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, ParallelMesh):
            raise TypeError("mesh must be a ParallelMesh")
        owners = tuple(self.expert_owner)
        if not owners:
            raise ValueError("expert_owner must contain at least one expert")
        if any(isinstance(owner, bool) or not isinstance(owner, int) for owner in owners):
            raise TypeError("expert_owner values must be integers")
        if any(not 0 <= owner < self.mesh.ep_size for owner in owners):
            raise ValueError("expert_owner contains an owner outside the EP axis")
        object.__setattr__(self, "expert_owner", owners)

    @property
    def num_experts(self) -> int:
        return len(self.expert_owner)

    def _validate_expert(self, global_expert_id: int) -> None:
        if (
            isinstance(global_expert_id, bool)
            or not isinstance(global_expert_id, int)
            or not 0 <= global_expert_id < self.num_experts
        ):
            raise ValueError(f"global_expert_id must be in [0, {self.num_experts})")

    def owner(self, global_expert_id: int) -> int:
        self._validate_expert(global_expert_id)
        return self.expert_owner[global_expert_id]

    def local_experts(self, ep: int) -> tuple[int, ...]:
        _require_coordinate(ep, self.mesh.ep_size, "ep")
        return tuple(expert for expert, owner in enumerate(self.expert_owner) if owner == ep)

    def local_slot(self, global_expert_id: int) -> int:
        owner = self.owner(global_expert_id)
        return self.local_experts(owner).index(global_expert_id)

    def expert_ranks(self, *, global_expert_id: int, dp: int) -> tuple[int, ...]:
        owner = self.owner(global_expert_id)
        _require_coordinate(dp, self.mesh.dp_size, "dp")
        return self.mesh.tp_group(dp=dp, ep=owner)
