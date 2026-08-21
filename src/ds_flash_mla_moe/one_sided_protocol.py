"""Vendor-neutral payload/signal protocol for logical one-sided cells.

The state machine is executable specification code.  It neither performs a
remote write nor verifies remote-memory visibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from torch import Tensor


class ProtocolError(RuntimeError):
    """A deterministic protocol-state or ownership violation."""


class CellState(str, Enum):
    EMPTY = "empty"
    WRITING = "writing"
    READY = "ready"
    READING = "reading"
    CONSUMED = "consumed"


@dataclass(frozen=True, order=True)
class RouteIdentity:
    """Globally unique logical route identity within a generation."""

    source_pe: int
    generation: int
    route_id: int


@dataclass(frozen=True, order=True)
class CellKey:
    """Identity of a time-buffered producer/consumer expert cell."""

    producer_pe: int
    consumer_pe: int
    round_id: int
    buffer_slot: int
    local_expert_slot: int


@dataclass(frozen=True)
class PayloadRow:
    identity: RouteIdentity
    row_index: int
    payload: Tensor


@dataclass(frozen=True)
class CellSnapshot:
    state: CellState
    generation: int
    count: int
    row_indices: tuple[int, ...]
    route_identities: tuple[RouteIdentity, ...]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class OneSidedCell:
    """One payload-before-signal cell with consumed-generation acknowledgement."""

    def __init__(
        self,
        key: CellKey,
        *,
        capacity: int,
        initial_generation: int = 0,
        max_generation: int = (1 << 63) - 1,
    ) -> None:
        if not isinstance(key, CellKey):
            raise TypeError("key must be a CellKey")
        self._validate_key_fields(key)
        self._capacity = _positive_integer(capacity, "capacity")
        if (
            isinstance(initial_generation, bool)
            or not isinstance(initial_generation, int)
            or initial_generation < 0
        ):
            raise ValueError("initial_generation must be a non-negative integer")
        if (
            isinstance(max_generation, bool)
            or not isinstance(max_generation, int)
            or max_generation < initial_generation
        ):
            raise ValueError("max_generation must be an integer at least initial_generation")
        self._key = key
        self._max_generation = max_generation
        self._generation = initial_generation
        self._state = CellState.EMPTY
        self._count = 0
        self._rows: dict[int, PayloadRow] = {}

    @staticmethod
    def _validate_key_fields(key: CellKey) -> None:
        for name, value in (
            ("producer_pe", key.producer_pe),
            ("consumer_pe", key.consumer_pe),
            ("round_id", key.round_id),
            ("buffer_slot", key.buffer_slot),
            ("local_expert_slot", key.local_expert_slot),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProtocolError(f"{name} must be a non-negative integer")

    @property
    def key(self) -> CellKey:
        return self._key

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def state(self) -> CellState:
        return self._state

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def count(self) -> int:
        return self._count

    def snapshot(self) -> CellSnapshot:
        rows = tuple(self._rows[index] for index in sorted(self._rows))
        return CellSnapshot(
            state=self._state,
            generation=self._generation,
            count=self._count,
            row_indices=tuple(row.row_index for row in rows),
            route_identities=tuple(row.identity for row in rows),
        )

    def _require_actor(self, actor_pe: int, *, producer: bool) -> None:
        expected = self._key.producer_pe if producer else self._key.consumer_pe
        role = "producer PE" if producer else "consumer PE"
        if isinstance(actor_pe, bool) or not isinstance(actor_pe, int) or actor_pe != expected:
            raise ProtocolError(f"event must be issued by {role} {expected}")

    def _require_generation(self, generation: int) -> None:
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise ProtocolError("generation must be an integer")
        if generation < self._generation:
            raise ProtocolError(
                f"stale generation {generation}; current generation is {self._generation}"
            )
        if generation > self._generation:
            raise ProtocolError(
                f"future generation {generation}; current generation is {self._generation}"
            )

    def _require_state(self, expected: CellState) -> None:
        if self._state is not expected:
            raise ProtocolError(
                f"cell must be {expected.name}; current state is {self._state.name}"
            )

    def begin_write(self, *, actor_pe: int, generation: int, count: int) -> None:
        self._require_actor(actor_pe, producer=True)
        self._require_generation(generation)
        self._require_state(CellState.EMPTY)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ProtocolError("count must be a non-negative integer")
        if count > self._capacity:
            raise ProtocolError(f"count {count} exceeds cell capacity {self._capacity}")
        self._count = count
        self._state = CellState.WRITING

    def write_payload(
        self,
        *,
        actor_pe: int,
        generation: int,
        row_index: int,
        identity: RouteIdentity,
        payload: Tensor,
    ) -> None:
        self._require_actor(actor_pe, producer=True)
        self._require_generation(generation)
        self._require_state(CellState.WRITING)
        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or not 0 <= row_index < self._count
        ):
            raise ProtocolError(f"row_index must be in [0, {self._count})")
        if not isinstance(identity, RouteIdentity):
            raise ProtocolError("identity must be a RouteIdentity")
        for name, value in (
            ("source_pe", identity.source_pe),
            ("generation", identity.generation),
            ("route_id", identity.route_id),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProtocolError(f"route identity {name} must be a non-negative integer")
        if identity.generation != generation:
            raise ProtocolError("route identity generation does not match the cell generation")
        if row_index in self._rows:
            raise ProtocolError(f"duplicate payload row {row_index}")
        if any(row.identity == identity for row in self._rows.values()):
            raise ProtocolError(f"duplicate route identity {identity}")
        if not isinstance(payload, Tensor):
            raise ProtocolError("payload must be a torch.Tensor")
        cloned_payload = payload.detach().clone()
        self._rows[row_index] = PayloadRow(identity, row_index, cloned_payload)

    def signal_ready(self, *, actor_pe: int, generation: int, count: int) -> None:
        self._require_actor(actor_pe, producer=True)
        self._require_generation(generation)
        self._require_state(CellState.WRITING)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ProtocolError("ready signal count must be a non-negative integer")
        if count != self._count:
            raise ProtocolError(f"ready signal count does not match reserved count {self._count}")
        missing = tuple(index for index in range(self._count) if index not in self._rows)
        if missing:
            raise ProtocolError(f"ready signal has missing payload rows {missing}")
        self._state = CellState.READY

    def begin_read(self, *, actor_pe: int, generation: int) -> tuple[PayloadRow, ...]:
        self._require_actor(actor_pe, producer=False)
        self._require_generation(generation)
        self._require_state(CellState.READY)
        rows = tuple(
            PayloadRow(row.identity, row.row_index, row.payload.detach().clone())
            for row in (self._rows[index] for index in range(self._count))
        )
        self._state = CellState.READING
        return rows

    def ack_consumed(self, *, actor_pe: int, generation: int) -> None:
        self._require_actor(actor_pe, producer=False)
        self._require_generation(generation)
        self._require_state(CellState.READING)
        self._state = CellState.CONSUMED

    def recycle(self, *, actor_pe: int, generation: int) -> None:
        self._require_actor(actor_pe, producer=True)
        self._require_generation(generation)
        self._require_state(CellState.CONSUMED)
        if self._generation == self._max_generation:
            raise ProtocolError("generation wrap is forbidden")
        self._rows.clear()
        self._count = 0
        self._generation += 1
        self._state = CellState.EMPTY


class OneSidedProtocol:
    """Registry that validates logical PEs before creating protocol cells."""

    def __init__(
        self,
        *,
        pe_count: int,
        cell_capacity: int,
        max_generation: int = (1 << 63) - 1,
    ) -> None:
        self._pe_count = _positive_integer(pe_count, "pe_count")
        self._cell_capacity = _positive_integer(cell_capacity, "cell_capacity")
        self._max_generation = max_generation
        self._cells: dict[CellKey, OneSidedCell] = {}

    @property
    def pe_count(self) -> int:
        return self._pe_count

    @property
    def cell_capacity(self) -> int:
        return self._cell_capacity

    def _validate_key(self, key: CellKey) -> None:
        if not isinstance(key, CellKey):
            raise TypeError("key must be a CellKey")
        OneSidedCell._validate_key_fields(key)
        if key.producer_pe >= self._pe_count:
            raise ProtocolError(f"producer_pe must be in [0, {self._pe_count})")
        if key.consumer_pe >= self._pe_count:
            raise ProtocolError(f"consumer_pe must be in [0, {self._pe_count})")

    def open_cell(self, key: CellKey, *, initial_generation: int = 0) -> OneSidedCell:
        self._validate_key(key)
        if key in self._cells:
            raise ProtocolError(f"cell {key} already exists")
        cell = OneSidedCell(
            key,
            capacity=self._cell_capacity,
            initial_generation=initial_generation,
            max_generation=self._max_generation,
        )
        self._cells[key] = cell
        return cell

    def cell(self, key: CellKey) -> OneSidedCell:
        self._validate_key(key)
        try:
            return self._cells[key]
        except KeyError as error:
            raise ProtocolError(f"unknown cell {key}") from error
