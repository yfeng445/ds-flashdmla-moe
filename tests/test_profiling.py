from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from ds_flash_mla_moe import profiling
from ds_flash_mla_moe.moe_benchmarking import MoEForwardBenchmarkConfig
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


def _moe_config() -> MoEForwardBenchmarkConfig:
    return MoEForwardBenchmarkConfig(
        tokens=4,
        model_dim=8,
        hidden_dim=16,
        experts=2,
        topk=1,
        dtype="float32",
        device="cuda",
        backend="reference",
        warmup=0,
        iterations=1,
        verify=False,
    )


def _stub_cuda_memory(
    monkeypatch,
    *,
    allocated_before: int,
    allocated_after: int,
    peak_allocated: int,
) -> list[object]:
    allocated = iter((allocated_before, allocated_after))
    reset_devices = []
    monkeypatch.setattr(
        profiling.torch.cuda,
        "memory_allocated",
        lambda _device: next(allocated),
    )
    monkeypatch.setattr(
        profiling.torch.cuda,
        "reset_peak_memory_stats",
        lambda device: reset_devices.append(device),
    )
    monkeypatch.setattr(
        profiling.torch.cuda,
        "max_memory_allocated",
        lambda _device: peak_allocated,
    )
    return reset_devices


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


def test_mla_low_precision_matrix_case_can_be_selected_for_profiling() -> None:
    case = profiling._selected_case(
        OperatorProfileConfig(
            case="mla_low_precision_decode_bfloat16",
            warmup=0,
            iterations=1,
        )
    )

    assert case.native_config.implementation == "cuda"
    assert case.baseline_config.implementation == "absorbed"
    assert case.native_config.dtype == case.baseline_config.dtype == "bfloat16"


def test_mla_paged_matrix_case_can_be_selected_for_profiling() -> None:
    case = profiling._selected_case(
        OperatorProfileConfig(
            case="mla_paged_decode_bfloat16_long",
            warmup=0,
            iterations=1,
        )
    )

    assert case.native_config.implementation == "cuda"
    assert case.baseline_config.implementation == "absorbed"
    assert case.native_config.workload == "decode_with_paged_write"
    assert case.native_config.sequence_length == 257


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
                "mla_absorbed_attention_generic_kernel",
                count=3,
                self_device_us=3_000.0,
                device_total_us=3_000.0,
            ),
            FakeEvent("aten::_local_scalar_dense", count=7, cpu_total_us=350.0),
            FakeEvent("cudaStreamSynchronize", count=5, cpu_total_us=600.0),
        ]
    )
    profile_arguments = {}
    reset_devices = _stub_cuda_memory(
        monkeypatch,
        allocated_before=1_024,
        allocated_after=1_536,
        peak_allocated=4_096,
    )

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
        "mla_absorbed_attention_generic_kernel"
    ]
    assert report["events"]["observed_device_activity_count"] == 6
    assert report["events"]["observed_custom_kernel_count"] == 3
    assert [row["name"] for row in report["events"]["custom_kernel_events"]] == [
        "mla_absorbed_attention_generic_kernel"
    ]
    assert report["events"]["custom_operator_events"][0]["device_total_ms"] == 2.0
    synchronization = report["events"]["synchronization_events"]
    assert [row["category"] for row in synchronization] == [
        "device_to_host_scalar",
        "stream_synchronize",
    ]
    assert [row["count"] for row in synchronization] == [7, 5]
    assert report["activities"] == ["cpu", "cuda"]
    assert report["memory"] == {
        "allocated_before_bytes": 1_024,
        "allocated_after_bytes": 1_536,
        "allocated_delta_bytes": 512,
        "peak_allocated_bytes": 4_096,
        "peak_delta_bytes": 3_072,
    }
    assert reset_devices == [profiling.torch.device("cuda")]


def test_profiler_classifies_real_project_kernels_without_framework_false_positives() -> None:
    summary = profiling._summarize_events(
        [
            FakeEvent(
                "void at::native::vectorized_elementwise_kernel<4>()",
                count=7,
                self_device_us=5_000.0,
            ),
            FakeEvent("grouped_topk_select_kernel", count=2, self_device_us=4_000.0),
            FakeEvent("count_route_keys_kernel", count=3, self_device_us=3_000.0),
            FakeEvent(
                "swiglu_hidden_grouped_tiled_float_kernel",
                count=5,
                self_device_us=2_000.0,
            ),
            FakeEvent(
                "pack_single_device_routes_float_kernel",
                count=1,
                self_device_us=1_500.0,
            ),
            FakeEvent("fused_down_atomic_float_kernel", count=2, self_device_us=1_250.0),
            FakeEvent(
                "persistent_down_atomic_float_kernel",
                count=3,
                self_device_us=1_000.0,
            ),
            FakeEvent("ampere_sgemm_128x128_nn", count=11, self_device_us=1_000.0),
        ],
        row_limit=10,
    )

    assert summary["observed_device_activity_count"] == 34
    assert summary["observed_custom_kernel_count"] == 16
    assert [row["name"] for row in summary["custom_kernel_events"]] == [
        "grouped_topk_select_kernel",
        "count_route_keys_kernel",
        "swiglu_hidden_grouped_tiled_float_kernel",
        "pack_single_device_routes_float_kernel",
        "fused_down_atomic_float_kernel",
        "persistent_down_atomic_float_kernel",
    ]


def test_torch_profile_requires_an_available_cuda_device(monkeypatch) -> None:
    monkeypatch.setattr(profiling.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="not available"):
        profile_operator_case(OperatorProfileConfig(case="mla_prefill_regular", iterations=1))


def test_moe_kineto_profile_captures_the_whole_layer_report(monkeypatch, tmp_path: Path) -> None:
    fake_profiler = FakeProfiler(
        [
            FakeEvent(
                "grouped_topk_select_kernel",
                count=4,
                self_device_us=2_000.0,
                device_total_us=2_000.0,
            ),
            FakeEvent("cudaDeviceSynchronize", count=2, cpu_total_us=500.0),
        ]
    )
    profile_arguments = {}
    executed = []
    reset_devices = _stub_cuda_memory(
        monkeypatch,
        allocated_before=8_192,
        allocated_after=9_216,
        peak_allocated=12_288,
    )

    def make_profiler(**kwargs):
        profile_arguments.update(kwargs)
        return fake_profiler

    def run_moe(config: MoEForwardBenchmarkConfig) -> dict:
        executed.append(config)
        return _benchmark_report()

    monkeypatch.setattr(profiling.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profiling.torch.profiler, "profile", make_profiler)
    monkeypatch.setattr(profiling, "benchmark_moe_forward", run_moe, raising=False)

    trace_path = tmp_path / "moe.json"
    report = profiling.profile_moe_forward(
        _moe_config(),
        trace_path=trace_path,
        record_shapes=False,
        row_limit=3,
    )

    assert executed == [_moe_config(), _moe_config()]
    assert profile_arguments["record_shapes"] is False
    assert report["profile"] == "single_gpu_moe_kineto"
    assert report["profile_configuration"]["benchmark"]["backend"] == "reference"
    assert report["profile_configuration"]["row_limit"] == 3
    assert report["activities"] == ["cpu", "cuda"]
    assert report["events"]["observed_device_activity_count"] == 4
    assert report["events"]["observed_custom_kernel_count"] == 4
    assert report["events"]["synchronization_events"][0]["category"] == ("device_synchronize")
    assert report["memory"]["allocated_delta_bytes"] == 1_024
    assert report["memory"]["peak_allocated_bytes"] == 12_288
    assert report["trace_path"] == str(trace_path)
    assert fake_profiler.exported_trace == str(trace_path)
    assert reset_devices == [profiling.torch.device("cuda")]
    assert any(
        "complete benchmark harness" in note and "not one isolated forward" in note
        for note in report["notes"]
    )


def test_moe_nvtx_profile_marks_counts_for_external_observation(monkeypatch) -> None:
    ranges = []
    emit_arguments = []
    executed = []
    _stub_cuda_memory(
        monkeypatch,
        allocated_before=4_096,
        allocated_after=4_096,
        peak_allocated=6_144,
    )

    @contextmanager
    def nvtx_range(label: str):
        ranges.append(label)
        yield

    @contextmanager
    def emit_nvtx(*, record_shapes: bool):
        emit_arguments.append(record_shapes)
        yield

    def run_moe(config: MoEForwardBenchmarkConfig) -> dict:
        executed.append(config)
        return _benchmark_report()

    monkeypatch.setattr(profiling.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profiling.torch.cuda.nvtx, "range", nvtx_range)
    monkeypatch.setattr(profiling.torch.autograd.profiler, "emit_nvtx", emit_nvtx)
    monkeypatch.setattr(profiling, "benchmark_moe_forward", run_moe, raising=False)

    report = profiling.run_nvtx_moe_forward(_moe_config(), record_shapes=False)

    label = "ds_flash_mla_moe::deepseek_moe_forward::reference"
    assert ranges == [label]
    assert emit_arguments == [False]
    assert executed == [_moe_config(), _moe_config()]
    assert report["profile"] == "single_gpu_moe_nvtx_run"
    assert report["nvtx_range"] == label
    assert report["events"]["observation"] == "external_profiler_required"
    assert report["events"]["observed_device_activity_count"] is None
    assert report["events"]["observed_custom_kernel_count"] is None
    assert report["events"]["synchronization_events"] is None
    assert report["memory"]["peak_delta_bytes"] == 2_048
    assert any(
        "complete benchmark harness" in note and "not one isolated forward" in note
        for note in report["notes"]
    )


def test_nvtx_profile_wraps_the_exact_matrix_side(monkeypatch) -> None:
    ranges = []
    emit_arguments = []
    executed = []
    reset_devices = _stub_cuda_memory(
        monkeypatch,
        allocated_before=2_048,
        allocated_after=2_560,
        peak_allocated=4_096,
    )

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
    assert report["events"] == {
        "observation": "external_profiler_required",
        "observed_device_activity_count": None,
        "observed_custom_kernel_count": None,
        "custom_kernel_events": [],
        "synchronization_events": None,
    }
    assert report["memory"] == {
        "allocated_before_bytes": 2_048,
        "allocated_after_bytes": 2_560,
        "allocated_delta_bytes": 512,
        "peak_allocated_bytes": 4_096,
        "peak_delta_bytes": 2_048,
    }
    assert reset_devices == [profiling.torch.device("cuda")]
