"""Emit structured single-process evidence for logical EP protocol and TP SwiGLU."""

from __future__ import annotations

import argparse

import torch

from ds_flash_mla_moe import (
    FakeDistributedMoE,
    LogicalRoute,
    RouteIdentity,
    tensor_parallel_swiglu_forward,
)
from ds_flash_mla_moe.benchmarking import write_benchmark_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pes", type=int, default=2)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--routes-per-pe", type=int, default=4)
    parser.add_argument("--cell-capacity", type=int, default=4)
    parser.add_argument("--model-dim", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="write JSON to this path instead of stdout")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    positive = (
        arguments.pes,
        arguments.experts,
        arguments.routes_per_pe,
        arguments.cell_capacity,
        arguments.model_dim,
        arguments.hidden,
    )
    if any(value <= 0 for value in positive):
        raise SystemExit("all logical-distributed dimensions must be positive")
    if arguments.hidden % arguments.tp_size:
        raise SystemExit("hidden must be divisible by tp-size")

    generator = torch.Generator().manual_seed(arguments.seed)
    owners = tuple(expert % arguments.pes for expert in range(arguments.experts))
    routes = tuple(
        LogicalRoute(
            identity=RouteIdentity(source_pe=source, generation=0, route_id=route_id),
            global_expert_id=(source + route_id) % arguments.experts,
            payload=torch.randn(arguments.model_dim, generator=generator),
        )
        for source in range(arguments.pes)
        for route_id in range(arguments.routes_per_pe)
    )
    simulator = FakeDistributedMoE(
        pe_count=arguments.pes,
        expert_owner=owners,
        cell_capacity=arguments.cell_capacity,
    )
    reverse_order = tuple(route.identity for route in reversed(routes))
    simulated = simulator.dispatch_and_return(
        routes,
        expert_fn=lambda expert, payload: payload * float(expert + 1),
        delivery_order=reverse_order,
        return_order=tuple(reversed(reverse_order)),
    )

    x = torch.randn(2, arguments.model_dim, generator=generator)
    w1 = torch.randn(arguments.hidden, arguments.model_dim, generator=generator)
    w2 = torch.randn(arguments.model_dim, arguments.hidden, generator=generator)
    w3 = torch.randn(arguments.hidden, arguments.model_dim, generator=generator)
    tp_output, tp_report = tensor_parallel_swiglu_forward(
        x,
        w1,
        w2,
        w3,
        tp_size=arguments.tp_size,
        return_report=True,
    )

    report = {
        "schema_version": 1,
        "benchmark": "logical_ep_tp_single_process",
        "config": {
            "pes": arguments.pes,
            "experts": arguments.experts,
            "routes_per_pe": arguments.routes_per_pe,
            "cell_capacity": arguments.cell_capacity,
            "model_dim": arguments.model_dim,
            "hidden": arguments.hidden,
            "tp_size": arguments.tp_size,
            "seed": arguments.seed,
        },
        "protocol": simulated.report.to_dict(),
        "tensor_parallel": tp_report.to_dict(),
        "facts": {
            "restored_route_identities": [
                {
                    "source_pe": route.identity.source_pe,
                    "generation": route.identity.generation,
                    "route_id": route.identity.route_id,
                }
                for route in simulated.routes
            ],
            "restored_checksum": sum(
                float(route.payload.double().sum().item()) for route in simulated.routes
            ),
            "tp_output_checksum": float(tp_output.double().sum().item()),
        },
        "claims": {
            "real_transport": False,
            "communication_overlap": False,
            "performance_speedup": False,
        },
    }
    write_benchmark_report(report, arguments.output)


if __name__ == "__main__":
    main()
