from __future__ import annotations

import pytest
import torch

from ds_flash_mla_moe.one_sided_protocol import (
    CellKey,
    CellState,
    OneSidedCell,
    OneSidedProtocol,
    ProtocolError,
    RouteIdentity,
)


def _key() -> CellKey:
    return CellKey(
        producer_pe=0,
        consumer_pe=1,
        round_id=7,
        buffer_slot=1,
        local_expert_slot=3,
    )


def _identity(route_id: int, *, generation: int = 0) -> RouteIdentity:
    return RouteIdentity(source_pe=0, generation=generation, route_id=route_id)


def _ready_cell(*, capacity: int = 2, count: int = 2) -> OneSidedCell:
    cell = OneSidedCell(_key(), capacity=capacity)
    cell.begin_write(actor_pe=0, generation=0, count=count)
    for row_index in reversed(range(count)):
        cell.write_payload(
            actor_pe=0,
            generation=0,
            row_index=row_index,
            identity=_identity(10 + row_index),
            payload=torch.tensor([float(row_index)]),
        )
    cell.signal_ready(actor_pe=0, generation=0, count=count)
    return cell


def test_cell_models_payload_before_signal_and_consumed_ack_lifecycle() -> None:
    cell = _ready_cell()

    assert cell.state is CellState.READY
    rows = cell.begin_read(actor_pe=1, generation=0)
    assert cell.state is CellState.READING
    assert [row.row_index for row in rows] == [0, 1]
    assert [row.identity.route_id for row in rows] == [10, 11]

    cell.ack_consumed(actor_pe=1, generation=0)
    assert cell.state is CellState.CONSUMED
    cell.recycle(actor_pe=0, generation=0)
    assert cell.state is CellState.EMPTY
    assert cell.generation == 1
    assert cell.count == 0


def test_cell_clones_payload_when_it_is_enqueued() -> None:
    cell = OneSidedCell(_key(), capacity=1)
    payload = torch.tensor([3.0])
    cell.begin_write(actor_pe=0, generation=0, count=1)
    cell.write_payload(
        actor_pe=0,
        generation=0,
        row_index=0,
        identity=_identity(4),
        payload=payload,
    )
    payload.add_(100)
    cell.signal_ready(actor_pe=0, generation=0, count=1)

    rows = cell.begin_read(actor_pe=1, generation=0)
    torch.testing.assert_close(rows[0].payload, torch.tensor([3.0]))


def test_cell_rejects_route_generation_mismatch_without_mutation() -> None:
    cell = OneSidedCell(_key(), capacity=1)
    cell.begin_write(actor_pe=0, generation=0, count=1)
    before = cell.snapshot()

    with pytest.raises(ProtocolError, match="identity generation"):
        cell.write_payload(
            actor_pe=0,
            generation=0,
            row_index=0,
            identity=_identity(1, generation=1),
            payload=torch.tensor([1.0]),
        )

    assert cell.snapshot() == before


def test_zero_count_cell_still_requires_full_signal_and_ack_lifecycle() -> None:
    cell = OneSidedCell(_key(), capacity=1)

    cell.begin_write(actor_pe=0, generation=0, count=0)
    cell.signal_ready(actor_pe=0, generation=0, count=0)
    assert cell.begin_read(actor_pe=1, generation=0) == ()
    cell.ack_consumed(actor_pe=1, generation=0)
    cell.recycle(actor_pe=0, generation=0)

    assert cell.generation == 1
    assert cell.state is CellState.EMPTY


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ("missing_payload", "missing payload rows"),
        ("count_mismatch", "count does not match"),
        ("duplicate_payload", "duplicate payload row"),
        ("duplicate_identity", "duplicate route identity"),
        ("duplicate_signal", "READY"),
        ("read_before_ready", "WRITING"),
        ("ack_before_read", "READY"),
        ("recycle_before_ack", "READING"),
    ],
)
def test_invalid_or_out_of_order_cell_transitions_are_atomic(operation: str, message: str) -> None:
    cell = OneSidedCell(_key(), capacity=2)
    cell.begin_write(actor_pe=0, generation=0, count=2)
    cell.write_payload(
        actor_pe=0,
        generation=0,
        row_index=0,
        identity=_identity(10),
        payload=torch.tensor([1.0]),
    )

    if operation == "missing_payload":
        before = cell.snapshot()
        with pytest.raises(ProtocolError, match=message):
            cell.signal_ready(actor_pe=0, generation=0, count=2)
    elif operation == "count_mismatch":
        before = cell.snapshot()
        with pytest.raises(ProtocolError, match=message):
            cell.signal_ready(actor_pe=0, generation=0, count=1)
    elif operation == "duplicate_payload":
        before = cell.snapshot()
        with pytest.raises(ProtocolError, match=message):
            cell.write_payload(
                actor_pe=0,
                generation=0,
                row_index=0,
                identity=_identity(11),
                payload=torch.tensor([2.0]),
            )
    elif operation == "duplicate_identity":
        before = cell.snapshot()
        with pytest.raises(ProtocolError, match=message):
            cell.write_payload(
                actor_pe=0,
                generation=0,
                row_index=1,
                identity=_identity(10),
                payload=torch.tensor([2.0]),
            )
    elif operation == "read_before_ready":
        before = cell.snapshot()
        with pytest.raises(ProtocolError, match=message):
            cell.begin_read(actor_pe=1, generation=0)
    else:
        cell.write_payload(
            actor_pe=0,
            generation=0,
            row_index=1,
            identity=_identity(11),
            payload=torch.tensor([2.0]),
        )
        cell.signal_ready(actor_pe=0, generation=0, count=2)
        if operation == "duplicate_signal":
            before = cell.snapshot()
            with pytest.raises(ProtocolError, match=message):
                cell.signal_ready(actor_pe=0, generation=0, count=2)
        elif operation == "ack_before_read":
            before = cell.snapshot()
            with pytest.raises(ProtocolError, match=message):
                cell.ack_consumed(actor_pe=1, generation=0)
        else:
            cell.begin_read(actor_pe=1, generation=0)
            before = cell.snapshot()
            with pytest.raises(ProtocolError, match=message):
                cell.recycle(actor_pe=0, generation=0)

    assert cell.snapshot() == before


def test_cell_rejects_overflow_before_mutating_state() -> None:
    cell = OneSidedCell(_key(), capacity=2)
    before = cell.snapshot()

    with pytest.raises(ProtocolError, match="capacity"):
        cell.begin_write(actor_pe=0, generation=0, count=3)

    assert cell.snapshot() == before


def test_cell_rejects_boolean_signal_count_and_actor_ids() -> None:
    cell = OneSidedCell(_key(), capacity=1)
    cell.begin_write(actor_pe=0, generation=0, count=0)
    before = cell.snapshot()

    with pytest.raises(ProtocolError, match="count"):
        cell.signal_ready(actor_pe=0, generation=0, count=False)
    assert cell.snapshot() == before

    with pytest.raises(ProtocolError, match="producer PE"):
        cell.signal_ready(actor_pe=False, generation=0, count=0)
    assert cell.snapshot() == before


@pytest.mark.parametrize(
    ("generation", "message"),
    [(-1, "stale generation"), (1, "future generation")],
)
def test_cell_rejects_stale_and_future_generation_atomically(generation: int, message: str) -> None:
    cell = OneSidedCell(_key(), capacity=1)
    before = cell.snapshot()

    with pytest.raises(ProtocolError, match=message):
        cell.begin_write(actor_pe=0, generation=generation, count=0)

    assert cell.snapshot() == before


def test_cell_rejects_stale_duplicate_ack_and_recycle() -> None:
    cell = _ready_cell(capacity=1, count=1)
    cell.begin_read(actor_pe=1, generation=0)
    cell.ack_consumed(actor_pe=1, generation=0)

    before = cell.snapshot()
    with pytest.raises(ProtocolError, match="CONSUMED"):
        cell.ack_consumed(actor_pe=1, generation=0)
    assert cell.snapshot() == before

    cell.recycle(actor_pe=0, generation=0)
    before = cell.snapshot()
    with pytest.raises(ProtocolError, match="stale generation"):
        cell.recycle(actor_pe=0, generation=0)
    assert cell.snapshot() == before


@pytest.mark.parametrize(
    ("method", "actor", "message"),
    [
        ("begin_write", 1, "producer PE"),
        ("begin_read", 0, "consumer PE"),
        ("ack_consumed", 0, "consumer PE"),
        ("recycle", 1, "producer PE"),
    ],
)
def test_cell_rejects_events_from_the_wrong_owner(method: str, actor: int, message: str) -> None:
    cell = _ready_cell(capacity=1, count=1)
    if method == "begin_write":
        cell = OneSidedCell(_key(), capacity=1)
        args = {"actor_pe": actor, "generation": 0, "count": 0}
    elif method == "begin_read":
        args = {"actor_pe": actor, "generation": 0}
    else:
        cell.begin_read(actor_pe=1, generation=0)
        if method == "recycle":
            cell.ack_consumed(actor_pe=1, generation=0)
        args = {"actor_pe": actor, "generation": 0}
    before = cell.snapshot()

    with pytest.raises(ProtocolError, match=message):
        getattr(cell, method)(**args)

    assert cell.snapshot() == before


def test_generation_wrap_is_rejected_before_recycle() -> None:
    cell = OneSidedCell(_key(), capacity=1, initial_generation=3, max_generation=3)
    cell.begin_write(actor_pe=0, generation=3, count=0)
    cell.signal_ready(actor_pe=0, generation=3, count=0)
    cell.begin_read(actor_pe=1, generation=3)
    cell.ack_consumed(actor_pe=1, generation=3)
    before = cell.snapshot()

    with pytest.raises(ProtocolError, match="generation wrap"):
        cell.recycle(actor_pe=0, generation=3)

    assert cell.snapshot() == before


def test_protocol_registry_validates_pe_ids_keys_and_duplicate_cells() -> None:
    protocol = OneSidedProtocol(pe_count=2, cell_capacity=4)
    cell = protocol.open_cell(_key())
    assert protocol.cell(_key()) is cell

    with pytest.raises(ProtocolError, match="already exists"):
        protocol.open_cell(_key())
    with pytest.raises(ProtocolError, match="producer_pe"):
        protocol.open_cell(CellKey(2, 0, 0, 0, 0))
    with pytest.raises(ProtocolError, match="consumer_pe"):
        protocol.open_cell(CellKey(0, -1, 0, 0, 0))
    with pytest.raises(ProtocolError, match="non-negative"):
        protocol.open_cell(CellKey(0, 1, -1, 0, 0))
    with pytest.raises(ProtocolError, match="unknown cell"):
        protocol.cell(CellKey(1, 0, 0, 0, 0))


def test_protocol_rejects_route_source_outside_pe_count_atomically() -> None:
    protocol = OneSidedProtocol(pe_count=2, cell_capacity=1)
    cell = protocol.open_cell(_key())
    cell.begin_write(actor_pe=0, generation=0, count=1)
    before = cell.snapshot()

    with pytest.raises(ProtocolError, match="source_pe must be in"):
        cell.write_payload(
            actor_pe=0,
            generation=0,
            row_index=0,
            identity=RouteIdentity(source_pe=2, generation=0, route_id=0),
            payload=torch.tensor([1.0]),
        )

    assert cell.snapshot() == before
