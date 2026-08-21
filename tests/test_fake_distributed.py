from __future__ import annotations

import pytest
import torch

from ds_flash_mla_moe.fake_distributed import (
    FakeDistributedMoE,
    LogicalRoute,
    SimulationError,
    SimulationReport,
)
from ds_flash_mla_moe.one_sided_protocol import RouteIdentity


def _route(
    source_pe: int,
    route_id: int,
    expert: int,
    value: float,
    *,
    generation: int = 0,
) -> LogicalRoute:
    return LogicalRoute(
        identity=RouteIdentity(source_pe, generation, route_id),
        global_expert_id=expert,
        payload=torch.tensor([value], dtype=torch.float32),
    )


def test_simulator_dispatches_and_returns_by_route_identity_despite_reordering() -> None:
    routes = (
        _route(1, 9, 2, 3.0),
        _route(0, 4, 1, 2.0),
        _route(0, 1, 0, 5.0),
        _route(1, 2, 3, 7.0),
    )
    delivery_order = tuple(reversed([route.identity for route in routes]))
    simulator = FakeDistributedMoE(
        pe_count=2,
        expert_owner=(1, 0, 1, 0),
        cell_capacity=2,
    )

    result = simulator.dispatch_and_return(
        routes,
        expert_fn=lambda expert, payload: payload * (expert + 1),
        delivery_order=delivery_order,
        return_order=tuple(reversed(delivery_order)),
    )

    assert [route.identity for route in result.routes] == [
        RouteIdentity(0, 0, 1),
        RouteIdentity(0, 0, 4),
        RouteIdentity(1, 0, 2),
        RouteIdentity(1, 0, 9),
    ]
    expected = {
        RouteIdentity(0, 0, 1): torch.tensor([5.0]),
        RouteIdentity(0, 0, 4): torch.tensor([4.0]),
        RouteIdentity(1, 0, 2): torch.tensor([28.0]),
        RouteIdentity(1, 0, 9): torch.tensor([9.0]),
    }
    assert result.by_identity.keys() == expected.keys()
    for identity, value in expected.items():
        torch.testing.assert_close(result.by_identity[identity], value)

    assert result.report.to_dict() == {
        "simulated": True,
        "remote_visibility_verified": False,
        "transport_performed": False,
        "multi_gpu_verified": False,
        "route_count": 4,
        "dispatch_cell_count": 4,
        "return_cell_count": 4,
    }


def test_simulator_uses_stable_owner_local_expert_slots() -> None:
    simulator = FakeDistributedMoE(
        pe_count=3,
        expert_owner=(2, 0, 2, 1, 0, 2),
        cell_capacity=4,
    )

    assert simulator.local_experts(0) == (1, 4)
    assert simulator.local_experts(1) == (3,)
    assert simulator.local_experts(2) == (0, 2, 5)
    assert [simulator.local_slot(i) for i in range(6)] == [0, 0, 1, 0, 1, 2]


def test_simulator_accepts_empty_round_and_reports_zero_cells() -> None:
    result = FakeDistributedMoE(
        pe_count=2,
        expert_owner=(0, 1),
        cell_capacity=1,
    ).dispatch_and_return(())

    assert result.routes == ()
    assert result.by_identity == {}
    assert result.report.route_count == 0
    assert result.report.dispatch_cell_count == 0
    assert result.report.return_cell_count == 0


def test_simulation_evidence_flags_cannot_be_overridden() -> None:
    with pytest.raises(TypeError, match="simulated"):
        SimulationReport(0, 0, 0, simulated=False)  # type: ignore[call-arg]


def test_simulator_requires_monotonic_exchange_generations() -> None:
    simulator = FakeDistributedMoE(
        pe_count=1,
        expert_owner=(0,),
        cell_capacity=1,
    )
    simulator.dispatch_and_return((_route(0, 0, 0, 1.0),))

    with pytest.raises(SimulationError, match="stale generation"):
        simulator.dispatch_and_return((_route(0, 0, 0, 1.0),), generation=0)
    with pytest.raises(SimulationError, match="future generation"):
        simulator.dispatch_and_return(
            (_route(0, 0, 0, 1.0, generation=2),),
            generation=2,
        )

    result = simulator.dispatch_and_return(
        (_route(0, 0, 0, 2.0, generation=1),),
        generation=1,
    )
    assert result.routes[0].identity.generation == 1
    assert simulator.completed_rounds == 2


@pytest.mark.parametrize(
    ("routes", "message"),
    [
        ((_route(0, 1, 0, 1.0), _route(0, 1, 1, 2.0)), "duplicate route identity"),
        ((_route(2, 1, 0, 1.0),), "source PE"),
        ((_route(0, 1, 2, 1.0),), "global expert"),
        ((_route(0, 1, 0, 1.0, generation=1),), "generation"),
    ],
)
def test_simulator_rejects_invalid_routes_atomically(
    routes: tuple[LogicalRoute, ...], message: str
) -> None:
    simulator = FakeDistributedMoE(
        pe_count=2,
        expert_owner=(0, 1),
        cell_capacity=2,
    )

    with pytest.raises(SimulationError, match=message):
        simulator.dispatch_and_return(routes, generation=0)

    assert simulator.completed_rounds == 0


def test_simulator_rejects_cell_overflow_atomically() -> None:
    simulator = FakeDistributedMoE(
        pe_count=2,
        expert_owner=(1,),
        cell_capacity=1,
    )
    routes = (_route(0, 0, 0, 1.0), _route(0, 1, 0, 2.0))

    with pytest.raises(SimulationError, match="capacity"):
        simulator.dispatch_and_return(routes)

    assert simulator.completed_rounds == 0


@pytest.mark.parametrize(
    "order",
    [
        (RouteIdentity(0, 0, 0),),
        (RouteIdentity(0, 0, 0), RouteIdentity(0, 0, 0)),
        (RouteIdentity(0, 0, 9), RouteIdentity(0, 0, 0)),
    ],
)
def test_simulator_rejects_non_permutation_delivery_orders(
    order: tuple[RouteIdentity, ...],
) -> None:
    routes = (_route(0, 0, 0, 1.0), _route(1, 1, 1, 2.0))
    simulator = FakeDistributedMoE(
        pe_count=2,
        expert_owner=(0, 1),
        cell_capacity=2,
    )

    with pytest.raises(SimulationError, match="exact permutation"):
        simulator.dispatch_and_return(routes, delivery_order=order)

    assert simulator.completed_rounds == 0


def test_simulator_clones_routes_before_expert_execution() -> None:
    payload = torch.tensor([2.0])
    route = LogicalRoute(RouteIdentity(0, 0, 0), 0, payload)
    simulator = FakeDistributedMoE(pe_count=1, expert_owner=(0,), cell_capacity=1)

    result = simulator.dispatch_and_return((route,), expert_fn=lambda _expert, value: value)
    payload.fill_(99)

    torch.testing.assert_close(result.by_identity[route.identity], torch.tensor([2.0]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pe_count": 2, "expert_owner": (), "cell_capacity": 1}, "at least one"),
        ({"pe_count": 2, "expert_owner": (0, 2), "cell_capacity": 1}, "owner"),
        ({"pe_count": 2, "expert_owner": (0, True), "cell_capacity": 1}, "integers"),
    ],
)
def test_simulator_rejects_invalid_owner_tables(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        FakeDistributedMoE(**kwargs)  # type: ignore[arg-type]


def test_simulator_rejects_invalid_expert_result_without_completing_round() -> None:
    simulator = FakeDistributedMoE(pe_count=1, expert_owner=(0,), cell_capacity=1)

    with pytest.raises(SimulationError, match="expert_fn"):
        simulator.dispatch_and_return(
            (_route(0, 0, 0, 1.0),),
            expert_fn=lambda _expert, _payload: "not a tensor",  # type: ignore[return-value]
        )

    assert simulator.completed_rounds == 0
