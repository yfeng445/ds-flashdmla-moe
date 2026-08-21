from __future__ import annotations

import pytest
import torch

import ds_flash_mla_moe.ops as attention_ops
from ds_flash_mla_moe import flash_attention_forward


def _cpu_inputs(*, requires_grad: bool = False):
    q = torch.randn(1, 2, 3, 5, requires_grad=requires_grad)
    k = torch.randn(1, 2, 7, 5, requires_grad=requires_grad)
    v = torch.randn(1, 2, 7, 4, requires_grad=requires_grad)
    return q, k, v


def test_reference_and_blockwise_are_distinct_explicit_branches(monkeypatch) -> None:
    q, k, v = _cpu_inputs()
    materialized = torch.full((1, 2, 3, 4), 1.0)
    blocked = torch.full((1, 2, 3, 4), 2.0)
    monkeypatch.setattr(
        attention_ops,
        "scaled_dot_product_attention_reference",
        lambda *args, **kwargs: materialized,
    )
    monkeypatch.setattr(
        attention_ops,
        "blockwise_attention",
        lambda *args, **kwargs: blocked,
    )

    assert flash_attention_forward(q, k, v, backend="reference") is materialized
    assert flash_attention_forward(q, k, v, backend="blockwise") is blocked


@pytest.mark.parametrize("backend", ["fa1", "fa2"])
def test_formal_fa_backends_reject_autograd_before_device_dispatch(backend: str) -> None:
    q, k, v = _cpu_inputs(requires_grad=True)
    with pytest.raises(RuntimeError, match="forward-only"):
        flash_attention_forward(q, k, v, backend=backend)  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["fa1", "fa2"])
def test_formal_fa_backends_never_fall_back_on_cpu(backend: str) -> None:
    q, k, v = _cpu_inputs()
    with pytest.raises(RuntimeError, match=rf"{backend} attention is unavailable"):
        flash_attention_forward(q, k, v, backend=backend)  # type: ignore[arg-type]


def test_cuda_alias_warns_and_uses_rowwise_contract() -> None:
    q, k, v = _cpu_inputs()
    with (
        pytest.warns(FutureWarning, match="cuda_rowwise"),
        pytest.raises(RuntimeError, match="cuda_rowwise attention is unavailable"),
    ):
        flash_attention_forward(q, k, v, backend="cuda")


def test_formal_operator_schemas_exist_without_native_extension() -> None:
    assert attention_ops._operator_is_defined("attention_fa1_forward")
    assert attention_ops._operator_is_defined("attention_fa2_forward")
