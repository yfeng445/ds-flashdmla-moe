"""Deterministic logical-PE dispatch/return simulator.

The simulator executes the protocol state machine on one process and one
device.  It is useful for route-identity and ordering tests, but it deliberately
does not model remote visibility, a transport, or multi-GPU overlap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from torch import Tensor

from .one_sided_protocol import (
    CellKey,
    OneSidedCell,
    OneSidedProtocol,
    RouteIdentity,
)


class SimulationError(RuntimeError):
    """Invalid simulator input detected before an exchange is executed."""


@dataclass(frozen=True)
class LogicalRoute:
    identity: RouteIdentity
    global_expert_id: int
    payload: Tensor


@dataclass(frozen=True)
class ReturnedRoute:
    identity: RouteIdentity
    global_expert_id: int
    payload: Tensor


@dataclass(frozen=True)
class SimulationReport:
    route_count: int
    dispatch_cell_count: int
    return_cell_count: int
    simulated: bool = field(default=True, init=False)
    remote_visibility_verified: bool = field(default=False, init=False)
    transport_performed: bool = field(default=False, init=False)
    multi_gpu_verified: bool = field(default=False, init=False)

    def to_dict(self) -> dict[str, bool | int]:
        return {
            "simulated": self.simulated,
            "remote_visibility_verified": self.remote_visibility_verified,
            "transport_performed": self.transport_performed,
            "multi_gpu_verified": self.multi_gpu_verified,
            "route_count": self.route_count,
            "dispatch_cell_count": self.dispatch_cell_count,
            "return_cell_count": self.return_cell_count,
        }


@dataclass(frozen=True)
class SimulationResult:
    routes: tuple[ReturnedRoute, ...]
    report: SimulationReport

    @property
    def by_identity(self) -> dict[RouteIdentity, Tensor]:
        return {route.identity: route.payload for route in self.routes}


@dataclass(frozen=True)
class _CellRoute:
    route: LogicalRoute | ReturnedRoute
    row_index: int


class FakeDistributedMoE:
    """Run an expert dispatch/return exchange through logical protocol cells."""

    def __init__(
        self,
        *,
        pe_count: int,
        expert_owner: Sequence[int],
        cell_capacity: int,
    ) -> None:
        if isinstance(pe_count, bool) or not isinstance(pe_count, int) or pe_count <= 0:
            raise ValueError("pe_count must be a positive integer")
        if (
            isinstance(cell_capacity, bool)
            or not isinstance(cell_capacity, int)
            or cell_capacity <= 0
        ):
            raise ValueError("cell_capacity must be a positive integer")
        owners = tuple(expert_owner)
        if not owners:
            raise ValueError("expert_owner must contain at least one expert")
        if any(isinstance(owner, bool) or not isinstance(owner, int) for owner in owners):
            raise TypeError("expert_owner values must be integers")
        if any(not 0 <= owner < pe_count for owner in owners):
            raise ValueError("expert_owner contains an owner outside the logical PE range")
        self._pe_count = pe_count
        self._expert_owner = owners
        self._cell_capacity = cell_capacity
        self._completed_rounds = 0
        self._local_experts = tuple(
            tuple(expert for expert, owner in enumerate(owners) if owner == pe)
            for pe in range(pe_count)
        )
        self._local_slot = {
            expert: slot for experts in self._local_experts for slot, expert in enumerate(experts)
        }

    @property
    def completed_rounds(self) -> int:
        return self._completed_rounds

    def local_experts(self, pe: int) -> tuple[int, ...]:
        self._validate_pe(pe)
        return self._local_experts[pe]

    def local_slot(self, global_expert_id: int) -> int:
        self._validate_expert(global_expert_id)
        return self._local_slot[global_expert_id]

    def _validate_pe(self, pe: int) -> None:
        if isinstance(pe, bool) or not isinstance(pe, int) or not 0 <= pe < self._pe_count:
            raise SimulationError(f"PE must be in [0, {self._pe_count})")

    def _validate_expert(self, expert: int) -> None:
        if (
            isinstance(expert, bool)
            or not isinstance(expert, int)
            or not 0 <= expert < len(self._expert_owner)
        ):
            raise SimulationError(f"global expert id must be in [0, {len(self._expert_owner)})")

    @staticmethod
    def _validate_nonnegative(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SimulationError(f"{name} must be a non-negative integer")

    def _validate_routes(
        self,
        routes: tuple[LogicalRoute, ...],
        generation: int,
    ) -> dict[RouteIdentity, LogicalRoute]:
        indexed: dict[RouteIdentity, LogicalRoute] = {}
        for route in routes:
            if not isinstance(route, LogicalRoute):
                raise SimulationError("routes must contain LogicalRoute values")
            identity = route.identity
            if not isinstance(identity, RouteIdentity):
                raise SimulationError("route identity must be a RouteIdentity")
            try:
                self._validate_pe(identity.source_pe)
            except SimulationError as error:
                raise SimulationError(f"source PE must be in [0, {self._pe_count})") from error
            self._validate_nonnegative(identity.route_id, "route_id")
            self._validate_nonnegative(identity.generation, "route generation")
            if identity.generation != generation:
                raise SimulationError(
                    f"route identity generation must equal exchange generation {generation}"
                )
            self._validate_expert(route.global_expert_id)
            if not isinstance(route.payload, Tensor):
                raise SimulationError("route payload must be a torch.Tensor")
            if identity in indexed:
                raise SimulationError(f"duplicate route identity {identity}")
            indexed[identity] = route
        return indexed

    @staticmethod
    def _validated_order(
        order: Sequence[RouteIdentity] | None,
        identities: tuple[RouteIdentity, ...],
        name: str,
    ) -> tuple[RouteIdentity, ...]:
        selected = identities if order is None else tuple(order)
        if len(selected) != len(identities) or set(selected) != set(identities):
            raise SimulationError(f"{name} must be an exact permutation of route identities")
        return selected

    def _cell_key(
        self,
        route: LogicalRoute | ReturnedRoute,
        *,
        returning: bool,
        round_id: int,
        buffer_slot: int,
    ) -> CellKey:
        owner = self._expert_owner[route.global_expert_id]
        producer = owner if returning else route.identity.source_pe
        consumer = route.identity.source_pe if returning else owner
        return CellKey(
            producer_pe=producer,
            consumer_pe=consumer,
            round_id=round_id,
            buffer_slot=buffer_slot,
            local_expert_slot=self._local_slot[route.global_expert_id],
        )

    def _group_routes(
        self,
        routes: Sequence[LogicalRoute | ReturnedRoute],
        *,
        returning: bool,
        round_id: int,
        buffer_slot: int,
    ) -> tuple[
        dict[CellKey, tuple[_CellRoute, ...]],
        dict[RouteIdentity, tuple[CellKey, int]],
    ]:
        grouped_lists: dict[CellKey, list[LogicalRoute | ReturnedRoute]] = {}
        for route in routes:
            key = self._cell_key(
                route,
                returning=returning,
                round_id=round_id,
                buffer_slot=buffer_slot,
            )
            grouped_lists.setdefault(key, []).append(route)
        groups: dict[CellKey, tuple[_CellRoute, ...]] = {}
        positions: dict[RouteIdentity, tuple[CellKey, int]] = {}
        for key, values in grouped_lists.items():
            ordered = sorted(values, key=lambda value: value.identity)
            if len(ordered) > self._cell_capacity:
                raise SimulationError(
                    f"cell {key} route count {len(ordered)} exceeds capacity {self._cell_capacity}"
                )
            rows = tuple(_CellRoute(route, row) for row, route in enumerate(ordered))
            groups[key] = rows
            for row in rows:
                positions[row.route.identity] = (key, row.row_index)
        return groups, positions

    def _execute_cells(
        self,
        groups: dict[CellKey, tuple[_CellRoute, ...]],
        positions: dict[RouteIdentity, tuple[CellKey, int]],
        route_lookup: dict[RouteIdentity, LogicalRoute | ReturnedRoute],
        delivery_order: tuple[RouteIdentity, ...],
        *,
        generation: int,
    ) -> tuple[tuple[OneSidedCell, tuple[_CellRoute, ...]], ...]:
        protocol = OneSidedProtocol(
            pe_count=self._pe_count,
            cell_capacity=self._cell_capacity,
        )
        cells: dict[CellKey, OneSidedCell] = {}
        for key, rows in groups.items():
            cell = protocol.open_cell(key, initial_generation=generation)
            cell.begin_write(
                actor_pe=key.producer_pe,
                generation=generation,
                count=len(rows),
            )
            cells[key] = cell
        for identity in delivery_order:
            key, row_index = positions[identity]
            cell = cells[key]
            route = route_lookup[identity]
            cell.write_payload(
                actor_pe=key.producer_pe,
                generation=generation,
                row_index=row_index,
                identity=identity,
                payload=route.payload,
            )
        for key, cell in cells.items():
            cell.signal_ready(
                actor_pe=key.producer_pe,
                generation=generation,
                count=len(groups[key]),
            )
        return tuple((cells[key], groups[key]) for key in sorted(groups))

    def dispatch_and_return(
        self,
        routes: Sequence[LogicalRoute],
        *,
        expert_fn: Callable[[int, Tensor], Tensor] | None = None,
        generation: int | None = None,
        round_id: int | None = None,
        buffer_slot: int = 0,
        delivery_order: Sequence[RouteIdentity] | None = None,
        return_order: Sequence[RouteIdentity] | None = None,
    ) -> SimulationResult:
        """Dispatch, compute, and restore routes by identity, not arrival order."""

        selected_generation = self._completed_rounds if generation is None else generation
        self._validate_nonnegative(selected_generation, "generation")
        if selected_generation != self._completed_rounds:
            direction = "stale" if selected_generation < self._completed_rounds else "future"
            raise SimulationError(
                f"{direction} generation {selected_generation}; expected {self._completed_rounds}"
            )
        selected_round = selected_generation if round_id is None else round_id
        self._validate_nonnegative(selected_round, "round_id")
        self._validate_nonnegative(buffer_slot, "buffer_slot")
        route_tuple = tuple(routes)
        indexed = self._validate_routes(route_tuple, selected_generation)
        identities = tuple(indexed)
        dispatch_delivery = self._validated_order(delivery_order, identities, "delivery_order")
        return_delivery = self._validated_order(return_order, identities, "return_order")
        dispatch_groups, dispatch_positions = self._group_routes(
            route_tuple,
            returning=False,
            round_id=selected_round,
            buffer_slot=buffer_slot,
        )
        dispatch_cells = self._execute_cells(
            dispatch_groups,
            dispatch_positions,
            indexed,
            dispatch_delivery,
            generation=selected_generation,
        )

        compute = expert_fn if expert_fn is not None else lambda _expert, payload: payload
        computed: dict[RouteIdentity, ReturnedRoute] = {}
        for cell, rows in dispatch_cells:
            payload_rows = cell.begin_read(
                actor_pe=cell.key.consumer_pe,
                generation=selected_generation,
            )
            route_by_row = {row.row_index: row.route for row in rows}
            for payload_row in payload_rows:
                dispatched = route_by_row[payload_row.row_index]
                value = compute(dispatched.global_expert_id, payload_row.payload)
                if not isinstance(value, Tensor):
                    raise SimulationError("expert_fn must return a torch.Tensor")
                computed[payload_row.identity] = ReturnedRoute(
                    identity=payload_row.identity,
                    global_expert_id=dispatched.global_expert_id,
                    payload=value.detach().clone(),
                )
            cell.ack_consumed(
                actor_pe=cell.key.consumer_pe,
                generation=selected_generation,
            )
            cell.recycle(actor_pe=cell.key.producer_pe, generation=selected_generation)

        returned_values = tuple(computed[identity] for identity in identities)
        return_groups, return_positions = self._group_routes(
            returned_values,
            returning=True,
            round_id=selected_round,
            buffer_slot=buffer_slot,
        )
        return_cells = self._execute_cells(
            return_groups,
            return_positions,
            computed,
            return_delivery,
            generation=selected_generation,
        )
        restored: dict[RouteIdentity, ReturnedRoute] = {}
        for cell, rows in return_cells:
            payload_rows = cell.begin_read(
                actor_pe=cell.key.consumer_pe,
                generation=selected_generation,
            )
            route_by_row = {row.row_index: row.route for row in rows}
            for payload_row in payload_rows:
                returned = route_by_row[payload_row.row_index]
                restored[payload_row.identity] = ReturnedRoute(
                    identity=payload_row.identity,
                    global_expert_id=returned.global_expert_id,
                    payload=payload_row.payload,
                )
            cell.ack_consumed(
                actor_pe=cell.key.consumer_pe,
                generation=selected_generation,
            )
            cell.recycle(actor_pe=cell.key.producer_pe, generation=selected_generation)

        restored_routes = tuple(restored[identity] for identity in sorted(restored))
        self._completed_rounds += 1
        return SimulationResult(
            routes=restored_routes,
            report=SimulationReport(
                route_count=len(restored_routes),
                dispatch_cell_count=len(dispatch_groups),
                return_cell_count=len(return_groups),
            ),
        )
