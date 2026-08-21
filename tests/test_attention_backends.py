from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import ds_flash_mla_moe.ops as attention_ops
from ds_flash_mla_moe import (
    blockwise_attention,
    cuda_attention_backend_available,
    flash_attention_forward,
)


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


@pytest.mark.parametrize(
    "operator_name", ["attention_fa1_forward", "attention_fa2_forward"]
)
def test_formal_operator_fake_rejects_zero_key_length(operator_name: str) -> None:
    with FakeTensorMode():
        q = torch.randn(1, 2, 3, 5, dtype=torch.float16)
        k = torch.randn(1, 2, 0, 5, dtype=torch.float16)
        v = torch.randn(1, 2, 0, 4, dtype=torch.float16)
        operator = getattr(torch.ops.ds_flash_mla_moe, operator_name).default

        with pytest.raises(RuntimeError):
            operator(q, k, v, False, 0.5)


def _fa_tolerances() -> tuple[float, float]:
    return 1e-2, 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    ("query_length", "key_length", "head_dim", "value_dim"),
    [(1, 7, 8, 5), (4, 4, 32, 32), (7, 11, 65, 33), (9, 17, 128, 127)],
)
def test_fa1_forward_matches_reference(
    causal: bool,
    query_length: int,
    key_length: int,
    head_dim: int,
    value_dim: int,
) -> None:
    torch.manual_seed(101)
    q = torch.randn(2, 3, query_length, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 3, key_length, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 3, key_length, value_dim, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        actual = flash_attention_forward(q, k, v, causal=causal, backend="fa1")
        expected = blockwise_attention(q, k, v, causal=causal, block_size=3)
    rtol, atol = _fa_tolerances()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert actual.shape == (2, 3, query_length, value_dim)
    assert actual.dtype == torch.float16
    assert actual.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    ("query_length", "key_length", "head_dim", "value_dim"),
    [(1, 7, 8, 5), (4, 4, 32, 32), (7, 11, 65, 33), (9, 17, 128, 127)],
)
def test_fa2_forward_matches_reference(
    causal: bool,
    query_length: int,
    key_length: int,
    head_dim: int,
    value_dim: int,
) -> None:
    torch.manual_seed(101)
    q = torch.randn(2, 3, query_length, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 3, key_length, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 3, key_length, value_dim, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        actual = flash_attention_forward(q, k, v, causal=causal, backend="fa2")
        expected = blockwise_attention(q, k, v, causal=causal, block_size=3)
    rtol, atol = _fa_tolerances()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert actual.shape == (2, 3, query_length, value_dim)
    assert actual.dtype == torch.float16
    assert actual.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.cuda
def test_fa1_forward_uses_current_stream() -> None:
    torch.manual_seed(103)
    q = torch.empty(1, 2, 7, 65, device="cuda", dtype=torch.float16)
    k = torch.empty(1, 2, 11, 65, device="cuda", dtype=torch.float16)
    v = torch.empty(1, 2, 11, 33, device="cuda", dtype=torch.float16)
    stream = torch.cuda.Stream()

    with torch.no_grad(), torch.cuda.stream(stream):
        q.normal_()
        k.normal_()
        v.normal_()
        actual = flash_attention_forward(q, k, v, causal=True, backend="fa1")
        actual.record_stream(stream)
    stream.synchronize()

    with torch.no_grad():
        expected = blockwise_attention(q, k, v, causal=True, block_size=3)
    rtol, atol = _fa_tolerances()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert cuda_attention_backend_available("fa1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.cuda
def test_fa2_forward_uses_current_stream() -> None:
    torch.manual_seed(103)
    q = torch.empty(1, 2, 7, 65, device="cuda", dtype=torch.float16)
    k = torch.empty(1, 2, 11, 65, device="cuda", dtype=torch.float16)
    v = torch.empty(1, 2, 11, 33, device="cuda", dtype=torch.float16)
    stream = torch.cuda.Stream()

    with torch.no_grad(), torch.cuda.stream(stream):
        q.normal_()
        k.normal_()
        v.normal_()
        actual = flash_attention_forward(q, k, v, causal=True, backend="fa2")
        actual.record_stream(stream)
    stream.synchronize()

    with torch.no_grad():
        expected = blockwise_attention(q, k, v, causal=True, block_size=3)
    rtol, atol = _fa_tolerances()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert cuda_attention_backend_available("fa2")


@pytest.mark.skipif(
    not (
        cuda_attention_backend_available("fa1")
        and cuda_attention_backend_available("fa2")
    ),
    reason="requires built FA1 and FA2 CUDA kernels",
)
@pytest.mark.cuda
def test_fa1_and_fa2_match_the_same_reference_on_identical_inputs() -> None:
    torch.manual_seed(103)
    q = torch.randn(1, 2, 9, 65, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 17, 65, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 17, 33, device="cuda", dtype=torch.float16)
    expected = blockwise_attention(q, k, v, causal=True, block_size=5)
    rtol, atol = _fa_tolerances()
    for backend in ("fa1", "fa2"):
        actual = flash_attention_forward(q, k, v, causal=True, backend=backend)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
