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
    CellState,
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

    _DISPATCH_ROUND_ID = 0
    _RETURN_ROUND_ID = 1

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
        self._protocol = OneSidedProtocol(
            pe_count=pe_count,
            cell_capacity=cell_capacity,
        )
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

    @property
    def protocol(self) -> OneSidedProtocol:
        """Return the persistent registry used by both protocol phases."""

        return self._protocol

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
    ) -> CellKey:
        owner = self._expert_owner[route.global_expert_id]
        producer = owner if returning else route.identity.source_pe
        consumer = route.identity.source_pe if returning else owner
        return CellKey(
            producer_pe=producer,
            consumer_pe=consumer,
            round_id=self._RETURN_ROUND_ID if returning else self._DISPATCH_ROUND_ID,
            buffer_slot=0,
            local_expert_slot=self._local_slot[route.global_expert_id],
        )

    def _group_routes(
        self,
        routes: Sequence[LogicalRoute | ReturnedRoute],
        *,
        returning: bool,
    ) -> tuple[
        dict[CellKey, tuple[_CellRoute, ...]],
        dict[RouteIdentity, tuple[CellKey, int]],
    ]:
        grouped_lists: dict[CellKey, list[LogicalRoute | ReturnedRoute]] = {}
        for route in routes:
            key = self._cell_key(
                route,
                returning=returning,
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
        phase_round_id: int,
    ) -> tuple[tuple[OneSidedCell, tuple[_CellRoute, ...]], ...]:
        all_keys, phase_keys = self._preflight_cells(
            groups,
            generation=generation,
            phase_round_id=phase_round_id,
        )
        cells: dict[CellKey, OneSidedCell] = {}
        for key in all_keys:
            rows = groups.get(key, ())
            if key in phase_keys:
                cell = self._protocol.cell(key)
            else:
                cell = self._protocol.open_cell(key, initial_generation=generation)
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
                count=len(groups.get(key, ())),
            )
        return tuple((cells[key], groups.get(key, ())) for key in all_keys)

    def _preflight_cells(
        self,
        groups: dict[CellKey, tuple[_CellRoute, ...]],
        *,
        generation: int,
        phase_round_id: int,
    ) -> tuple[tuple[CellKey, ...], set[CellKey]]:
        phase_keys = {
            key
            for key in self._protocol.cell_keys
            if key.round_id == phase_round_id and key.buffer_slot == 0
        }
        all_keys = tuple(sorted(phase_keys | groups.keys()))
        for key in all_keys:
            self._protocol.validate_cell_key(key)
            if key in phase_keys:
                cell = self._protocol.cell(key)
                if cell.state is not CellState.EMPTY:
                    raise SimulationError(f"protocol cell {key} must be EMPTY before an iteration")
                if cell.generation != generation:
                    direction = "stale" if generation < cell.generation else "future"
                    raise SimulationError(
                        f"{direction} generation {generation} for protocol cell {key}; "
                        f"expected {cell.generation}"
                    )
        return all_keys, phase_keys

    @staticmethod
    def _consume_cells(
        cells: tuple[tuple[OneSidedCell, tuple[_CellRoute, ...]], ...],
        *,
        generation: int,
    ) -> dict[RouteIdentity, ReturnedRoute | LogicalRoute]:
        consumed: dict[RouteIdentity, ReturnedRoute | LogicalRoute] = {}
        for cell, rows in cells:
            payload_rows = cell.begin_read(
                actor_pe=cell.key.consumer_pe,
                generation=generation,
            )
            route_by_row = {row.row_index: row.route for row in rows}
            for payload_row in payload_rows:
                route = route_by_row[payload_row.row_index]
                route_type = LogicalRoute if isinstance(route, LogicalRoute) else ReturnedRoute
                consumed[payload_row.identity] = route_type(
                    payload_row.identity, route.global_expert_id, payload_row.payload
                )
            cell.ack_consumed(
                actor_pe=cell.key.consumer_pe,
                generation=generation,
            )
            cell.recycle(actor_pe=cell.key.producer_pe, generation=generation)
        return consumed

    def dispatch_and_return(
        self,
        routes: Sequence[LogicalRoute],
        *,
        expert_fn: Callable[[int, Tensor], Tensor] | None = None,
        generation: int | None = None,
        delivery_order: Sequence[RouteIdentity] | None = None,
        return_order: Sequence[RouteIdentity] | None = None,
    ) -> SimulationResult:
        """Dispatch, compute, and restore routes by identity, not arrival order.

        Structural and protocol preflight completes before ``expert_fn`` is
        called. Expert computation consumes the payload cloned out of the
        dispatch cell. Any exception restores the persistent protocol to its
        pre-iteration state.
        """

        selected_generation = self._completed_rounds if generation is None else generation
        self._validate_nonnegative(selected_generation, "generation")
        if selected_generation != self._completed_rounds:
            direction = "stale" if selected_generation < self._completed_rounds else "future"
            raise SimulationError(
                f"{direction} generation {selected_generation}; expected {self._completed_rounds}"
            )
        route_tuple = tuple(routes)
        indexed = self._validate_routes(route_tuple, selected_generation)
        identities = tuple(indexed)
        dispatch_delivery = self._validated_order(delivery_order, identities, "delivery_order")
        return_delivery = self._validated_order(return_order, identities, "return_order")
        dispatch_groups, dispatch_positions = self._group_routes(
            route_tuple,
            returning=False,
        )
        return_placeholders = tuple(
            ReturnedRoute(
                identity=identity,
                global_expert_id=route.global_expert_id,
                payload=route.payload,
            )
            for identity, route in indexed.items()
        )
        return_groups, return_positions = self._group_routes(
            return_placeholders,
            returning=True,
        )
        self._preflight_cells(
            dispatch_groups,
            generation=selected_generation,
            phase_round_id=self._DISPATCH_ROUND_ID,
        )
        self._preflight_cells(
            return_groups,
            generation=selected_generation,
            phase_round_id=self._RETURN_ROUND_ID,
        )
        checkpoint = self._protocol._checkpoint()
        try:
            dispatch_cells = self._execute_cells(
                dispatch_groups,
                dispatch_positions,
                indexed,
                dispatch_delivery,
                generation=selected_generation,
                phase_round_id=self._DISPATCH_ROUND_ID,
            )
            dispatched = self._consume_cells(
                dispatch_cells,
                generation=selected_generation,
            )
            if dispatched.keys() != indexed.keys() or not all(
                isinstance(route, LogicalRoute) for route in dispatched.values()
            ):
                raise SimulationError("dispatch protocol did not preserve every route identity")

            compute = expert_fn if expert_fn is not None else lambda _expert, payload: payload
            computed: dict[RouteIdentity, ReturnedRoute] = {}
            for identity in identities:
                route = dispatched[identity]
                value = compute(route.global_expert_id, route.payload)
                if not isinstance(value, Tensor):
                    raise SimulationError("expert_fn must return a torch.Tensor")
                computed[identity] = ReturnedRoute(
                    identity=identity,
                    global_expert_id=route.global_expert_id,
                    payload=value.detach().clone(),
                )

            return_cells = self._execute_cells(
                return_groups,
                return_positions,
                computed,
                return_delivery,
                generation=selected_generation,
                phase_round_id=self._RETURN_ROUND_ID,
            )
            consumed_returns = self._consume_cells(
                return_cells,
                generation=selected_generation,
            )
            restored = {
                identity: route
                for identity, route in consumed_returns.items()
                if isinstance(route, ReturnedRoute)
            }
            if restored.keys() != computed.keys():
                raise SimulationError("return protocol did not preserve every route identity")
        except BaseException:
            self._protocol._restore(checkpoint)
            raise

        restored_routes = tuple(restored[identity] for identity in sorted(restored))
        self._completed_rounds += 1
        return SimulationResult(
            routes=restored_routes,
            report=SimulationReport(
                route_count=len(restored_routes),
                dispatch_cell_count=len(dispatch_cells),
                return_cell_count=len(return_cells),
            ),
        )
