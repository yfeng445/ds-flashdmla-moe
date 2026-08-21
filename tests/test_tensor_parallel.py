from __future__ import annotations

import pytest
import torch

from ds_flash_mla_moe.moe import swiglu_expert
from ds_flash_mla_moe.tensor_parallel import (
    TensorParallelReport,
    tensor_parallel_swiglu_forward,
)


def _inputs(*, dtype: torch.dtype = torch.float64, hidden: int = 8) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(202)
    x = torch.randn(2, 3, 4, dtype=dtype)
    w1 = torch.randn(hidden, 4, dtype=dtype)
    w2 = torch.randn(4, hidden, dtype=dtype)
    w3 = torch.randn(hidden, 4, dtype=dtype)
    return x, w1, w2, w3


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float16, 1e-3, 1e-5),
        (torch.bfloat16, 1.6e-2, 1e-5),
        (torch.float32, 1e-5, 1e-6),
        (torch.float64, 1e-12, 1e-12),
    ],
)
def test_logical_tensor_parallel_swiglu_matches_existing_expert_oracle(
    tp_size: int,
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    x, w1, w2, w3 = _inputs(dtype=dtype)
    expected = swiglu_expert(x, w1, w2, w3)

    actual, report = tensor_parallel_swiglu_forward(
        x,
        w1,
        w2,
        w3,
        tp_size=tp_size,
        return_report=True,
    )

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert actual.dtype == dtype
    assert not actual.requires_grad
    assert report.to_dict() == {
        "simulated": True,
        "remote_visibility_verified": False,
        "transport_performed": False,
        "multi_gpu_verified": False,
        "tp_size": tp_size,
        "hidden_size": 8,
        "shard_hidden_size": 8 // tp_size,
        "accumulation_dtype": "float64" if dtype == torch.float64 else "float32",
    }


@pytest.mark.parametrize("tp_size", [0, 3, 8, True])
def test_tensor_parallel_rejects_unsupported_tp_sizes(tp_size: int) -> None:
    with pytest.raises(ValueError, match="1, 2, or 4"):
        tensor_parallel_swiglu_forward(*_inputs(), tp_size=tp_size)


def test_tensor_parallel_rejects_hidden_sizes_not_divisible_by_tp() -> None:
    with pytest.raises(ValueError, match="divisible"):
        tensor_parallel_swiglu_forward(*_inputs(hidden=6), tp_size=4)


def test_tensor_parallel_rejects_zero_model_dimension() -> None:
    values = (
        torch.empty(2, 0),
        torch.empty(8, 0),
        torch.empty(0, 8),
        torch.empty(8, 0),
    )
    with pytest.raises(ValueError, match="model dimension"):
        tensor_parallel_swiglu_forward(*values, tp_size=2)


@pytest.mark.parametrize("index", range(4))
def test_tensor_parallel_is_forward_only_even_inside_no_grad(index: int) -> None:
    tensors = list(_inputs())
    tensors[index].requires_grad_()

    with torch.no_grad(), pytest.raises(RuntimeError, match="forward-only"):
        tensor_parallel_swiglu_forward(*tensors, tp_size=2)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda values: (values[0][..., :3], *values[1:]), "model dimension"),
        (lambda values: (values[0], values[1][:-1], values[2], values[3]), "W1 and W3"),
        (lambda values: (values[0], values[1], values[2][:, :-1], values[3]), "W2"),
        (
            lambda values: (values[0], values[1].float(), values[2], values[3]),
            "same dtype",
        ),
    ],
)
def test_tensor_parallel_validates_shapes_and_dtypes(mutator: object, message: str) -> None:
    values = _inputs()
    changed = mutator(values)  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError), match=message):
        tensor_parallel_swiglu_forward(*changed, tp_size=2)


def test_tensor_parallel_returns_plain_tensor_without_report() -> None:
    output = tensor_parallel_swiglu_forward(*_inputs(), tp_size=2)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 3, 4)


def test_tensor_parallel_evidence_flags_cannot_be_overridden() -> None:
    with pytest.raises(TypeError, match="multi_gpu_verified"):
        TensorParallelReport(2, 8, 4, "float32", multi_gpu_verified=True)  # type: ignore[call-arg]
