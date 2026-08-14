from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from ds_flash_mla_moe import profiling
from ds_flash_mla_moe.profiling import (
    OperatorProfileConfig,
    profile_operator_case,
    run_nvtx_operator_case,
)


class FakeEvent:
    def __init__(
        self,
        key: str,
        *,
        count: int = 1,
        self_cpu_us: float = 0.0,
        cpu_total_us: float = 0.0,
        self_device_us: float = 0.0,
        device_total_us: float = 0.0,
    ) -> None:
        self.key = key
        self.count = count
        self.self_cpu_time_total = self_cpu_us
        self.cpu_time_total = cpu_total_us
        self.self_device_time_total = self_device_us
        self.device_time_total = device_total_us


class FakeProfiler:
    def __init__(self, events: list[FakeEvent]) -> None:
        self.events = events
        self.exported_trace: str | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def key_averages(self) -> list[FakeEvent]:
        return self.events

    def export_chrome_trace(self, path: str) -> None:
        self.exported_trace = path


def _benchmark_report() -> dict:
    return {
        "environment": {"device": "cuda", "device_name": "test GPU"},
        "latency": {"median_ms": 1.25},
    }


def test_benchmark_script_directory_does_not_shadow_stdlib_profile() -> None:
    benchmark_directory = Path(__file__).resolve().parents[1] / "benchmarks"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, profile; "
                "assert hasattr(profile, 'run'); "
                "print(pathlib.Path(profile.__file__).resolve())"
            ),
        ],
        cwd=benchmark_directory,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).parent != benchmark_directory


@pytest.mark.parametrize(
    "config,match",
    [
        (OperatorProfileConfig(case="mla_prefill_regular", device="cpu"), "CUDA"),
        (OperatorProfileConfig(case=""), "case name"),
        (OperatorProfileConfig(case="mla_prefill_regular", side="invalid"), "side"),
        (OperatorProfileConfig(case="mla_prefill_regular", warmup=-1), "warmup"),
        (OperatorProfileConfig(case="mla_prefill_regular", iterations=0), "iterations"),
        (OperatorProfileConfig(case="mla_prefill_regular", row_limit=0), "row_limit"),
    ],
)
def test_invalid_profile_configuration_is_rejected(
    config: OperatorProfileConfig,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        config.validate()


def test_unknown_profile_case_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        profiling._selected_case(OperatorProfileConfig(case="missing"))


def test_flash_attn_4_matrix_case_can_be_selected_for_profiling() -> None:
    case = profiling._selected_case(
        OperatorProfileConfig(
            case="attention_fa4_decode_bfloat16",
            warmup=0,
            iterations=1,
        )
    )

    assert case.native_config.backend == "cuda"
    assert case.baseline_config.backend == "flash-attn-4"
    assert case.native_config.dtype == case.baseline_config.dtype == "bfloat16"


def test_torch_profile_report_classifies_events_and_exports_trace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_profiler = FakeProfiler(
        [
            FakeEvent(
                "ds_flash_mla_moe::mla_absorbed_attention",
                count=3,
                self_cpu_us=200.0,
                cpu_total_us=400.0,
                self_device_us=1_500.0,
                device_total_us=2_000.0,
            ),
            FakeEvent(
                "mla_absorbed_attention_cuda",
                count=3,
                self_device_us=3_000.0,
                device_total_us=3_000.0,
            ),
            FakeEvent("aten::_local_scalar_dense", count=7, cpu_total_us=350.0),
            FakeEvent("cudaStreamSynchronize", count=5, cpu_total_us=600.0),
        ]
    )
    profile_arguments = {}

    def make_profiler(**kwargs):
        profile_arguments.update(kwargs)
        return fake_profiler

    executed = []
    monkeypatch.setattr(profiling.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profiling.torch.profiler, "profile", make_profiler)
    monkeypatch.setattr(
        profiling,
        "benchmark_matrix_case_side",
        lambda case, side: executed.append((case.name, side)) or _benchmark_report(),
    )

    trace_path = tmp_path / "nested" / "mla.json"
    report = profile_operator_case(
        OperatorProfileConfig(
            case="mla_prefill_regular",
            iterations=1,
            record_shapes=False,
            row_limit=1,
        ),
        trace_path=trace_path,
    )

    assert executed == [
        ("mla_prefill_regular", "native"),
        ("mla_prefill_regular", "native"),
    ]
    assert profile_arguments["record_shapes"] is False
    assert profile_arguments["acc_events"] is True
    assert report["profile"] == "single_gpu_operator_torch_profiler"
    assert report["environment"]["device_name"] == "test GPU"
    assert report["trace_path"] == str(trace_path)
    assert fake_profiler.exported_trace == str(trace_path)
    assert trace_path.parent.is_dir()
    assert [row["name"] for row in report["events"]["top_self_device_events"]] == [
        "mla_absorbed_attention_cuda"
    ]
    assert report["events"]["custom_operator_events"][0]["device_total_ms"] == 2.0
    synchronization = report["events"]["synchronization_events"]
    assert [row["category"] for row in synchronization] == [
        "device_to_host_scalar",
        "stream_synchronize",
    ]
    assert [row["count"] for row in synchronization] == [7, 5]


def test_torch_profile_requires_an_available_cuda_device(monkeypatch) -> None:
    monkeypatch.setattr(profiling.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="not available"):
        profile_operator_case(OperatorProfileConfig(case="mla_prefill_regular", iterations=1))


def test_nvtx_profile_wraps_the_exact_matrix_side(monkeypatch) -> None:
    ranges = []
    emit_arguments = []
    executed = []

    @contextmanager
    def nvtx_range(label: str):
        ranges.append(label)
        yield

    @contextmanager
    def emit_nvtx(*, record_shapes: bool):
        emit_arguments.append(record_shapes)
        yield

    monkeypatch.setattr(profiling.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profiling.torch.cuda.nvtx, "range", nvtx_range)
    monkeypatch.setattr(profiling.torch.autograd.profiler, "emit_nvtx", emit_nvtx)
    monkeypatch.setattr(
        profiling,
        "benchmark_matrix_case_side",
        lambda case, side: executed.append((case.name, side)) or _benchmark_report(),
    )

    report = run_nvtx_operator_case(
        OperatorProfileConfig(
            case="mla_decode_regular",
            side="baseline",
            iterations=1,
            record_shapes=False,
        )
    )

    label = "ds_flash_mla_moe::mla_decode_regular::baseline"
    assert ranges == [label]
    assert emit_arguments == [False]
    assert executed == [
        ("mla_decode_regular", "baseline"),
        ("mla_decode_regular", "baseline"),
    ]
    assert report["nvtx_range"] == label
    assert report["case"]["side"] == "baseline"
