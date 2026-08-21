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

_TEACHING_BACKENDS = ("fa1", "fa2", "fa3")


@pytest.fixture
def _require_formal_cuda_kernels() -> None:
    unavailable = [
        backend
        for backend in _TEACHING_BACKENDS
        if not cuda_attention_backend_available(backend)  # type: ignore[arg-type]
    ]
    if unavailable:
        pytest.skip(
            "requires built teaching FA1, FA2, and FA3 CUDA kernels; unavailable: "
            + ", ".join(unavailable)
        )


_REQUIRES_FORMAL_CUDA = pytest.mark.usefixtures("_require_formal_cuda_kernels")


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


@pytest.mark.parametrize("backend", _TEACHING_BACKENDS)
def test_formal_backends_reject_non_fp16_before_dispatch(backend: str) -> None:
    q = torch.randn(1, 1, 3, 8)
    with pytest.raises(RuntimeError, match=rf"{backend} attention.*supports float16"):
        flash_attention_forward(q, q, q, backend=backend)  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", _TEACHING_BACKENDS)
def test_formal_fa_backends_reject_autograd_before_device_dispatch(backend: str) -> None:
    q, k, v = _cpu_inputs(requires_grad=True)
    with pytest.raises(RuntimeError, match="forward-only"):
        flash_attention_forward(q, k, v, backend=backend)  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", _TEACHING_BACKENDS)
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
    assert attention_ops._operator_is_defined("attention_fa3_forward")


@pytest.mark.parametrize(
    "operator_name",
    ["attention_fa1_forward", "attention_fa2_forward", "attention_fa3_forward"],
)
@pytest.mark.parametrize(
    "dispatch_key",
    ["AutogradCUDA", "CompositeExplicitAutograd", "CompositeImplicitAutograd"],
)
def test_formal_operators_have_no_autograd_registration(
    operator_name: str, dispatch_key: str
) -> None:
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(  # type: ignore[attr-defined]
        f"ds_flash_mla_moe::{operator_name}", dispatch_key
    )


@pytest.mark.parametrize(
    "operator_name",
    ["attention_fa1_forward", "attention_fa2_forward", "attention_fa3_forward"],
)
def test_formal_operator_fake_rejects_zero_key_length(operator_name: str) -> None:
    with FakeTensorMode():
        q = torch.randn(1, 2, 3, 5, dtype=torch.float16)
        k = torch.randn(1, 2, 0, 5, dtype=torch.float16)
        v = torch.randn(1, 2, 0, 4, dtype=torch.float16)
        operator = getattr(torch.ops.ds_flash_mla_moe, operator_name).default

        with pytest.raises(RuntimeError):
            operator(q, k, v, False, 0.5)


@pytest.mark.parametrize(
    "operator_name",
    ["attention_fa1_forward", "attention_fa2_forward", "attention_fa3_forward"],
)
@pytest.mark.parametrize("grad_enabled", [True, False])
def test_formal_operator_fake_rejects_requires_grad_inputs(
    operator_name: str, grad_enabled: bool
) -> None:
    with FakeTensorMode():
        q = torch.randn(1, 2, 3, 5, dtype=torch.float16, requires_grad=True)
        k = torch.randn(1, 2, 7, 5, dtype=torch.float16, requires_grad=True)
        v = torch.randn(1, 2, 7, 4, dtype=torch.float16, requires_grad=True)
        operator = getattr(torch.ops.ds_flash_mla_moe, operator_name).default
        grad_context = torch.enable_grad() if grad_enabled else torch.no_grad()

        with (
            grad_context,
            pytest.raises(RuntimeError, match="teaching FA1/FA2/FA3.*forward-only.*requires_grad"),
        ):
            operator(q, k, v, False, 0.5)


def test_fa3_operator_fake_preserves_output_contract() -> None:
    with FakeTensorMode():
        q = torch.randn(2, 3, 7, 65, dtype=torch.float16)
        k = torch.randn(2, 3, 11, 65, dtype=torch.float16)
        v = torch.randn(2, 3, 11, 33, dtype=torch.float16)

        output = torch.ops.ds_flash_mla_moe.attention_fa3_forward.default(q, k, v, True, 65**-0.5)

        assert output.shape == (2, 3, 7, 33)
        assert output.dtype == torch.float16
        assert output.device == q.device


@pytest.mark.parametrize("contract_break", ["rank", "dtype"])
def test_fa3_operator_fake_rejects_unsupported_shape_or_dtype(contract_break: str) -> None:
    with FakeTensorMode():
        if contract_break == "rank":
            q = torch.randn(2, 7, 65, dtype=torch.float16)
            k = torch.randn(2, 11, 65, dtype=torch.float16)
            v = torch.randn(2, 11, 33, dtype=torch.float16)
        else:
            q = torch.randn(1, 2, 7, 65, dtype=torch.float32)
            k = torch.randn(1, 2, 11, 65, dtype=torch.float32)
            v = torch.randn(1, 2, 11, 33, dtype=torch.float32)

        with pytest.raises(RuntimeError):
            torch.ops.ds_flash_mla_moe.attention_fa3_forward.default(q, k, v, False, 65**-0.5)


@pytest.mark.parametrize(
    ("backend", "expected_operator"),
    [
        ("cuda_rowwise", "attention_forward"),
        ("fa1", "attention_fa1_forward"),
        ("fa2", "attention_fa2_forward"),
        ("fa3", "attention_fa3_forward"),
    ],
)
def test_cuda_attention_backend_availability_uses_backend_operator_mapping(
    monkeypatch, backend: str, expected_operator: str
) -> None:
    monkeypatch.setattr(attention_ops, "_NATIVE_EXTENSION_LOADED", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        attention_ops,
        "_operator_has_cuda_kernel",
        lambda operator: operator == expected_operator,
    )

    assert cuda_attention_backend_available(backend)  # type: ignore[arg-type]


def test_cuda_attention_backend_availability_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="cuda_rowwise, fa1, fa2, or fa3"):
        cuda_attention_backend_available("unknown")  # type: ignore[arg-type]


def _fa_tolerances() -> tuple[float, float]:
    return 1e-2, 1e-2


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("backend", _TEACHING_BACKENDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_formal_backends_reject_unsupported_cuda_dtype(backend: str, dtype: torch.dtype) -> None:
    q = torch.randn(1, 2, 7, 65, device="cuda", dtype=dtype)
    k = torch.randn(1, 2, 11, 65, device="cuda", dtype=dtype)
    v = torch.randn(1, 2, 11, 33, device="cuda", dtype=dtype)

    with pytest.raises(RuntimeError, match=rf"{backend} attention.*supports float16"):
        flash_attention_forward(q, k, v, backend=backend)  # type: ignore[arg-type]


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("backend", _TEACHING_BACKENDS)
def test_formal_backends_reject_mixed_cuda_dtypes(backend: str) -> None:
    q = torch.randn(1, 2, 7, 65, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 11, 65, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 2, 11, 33, device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match=rf"{backend} attention.*same dtype"):
        flash_attention_forward(q, k, v, backend=backend)  # type: ignore[arg-type]


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("backend", _TEACHING_BACKENDS)
def test_formal_backends_reject_noncontiguous_cuda_storage(backend: str) -> None:
    q = torch.randn(1, 2, 7, 130, device="cuda", dtype=torch.float16)[..., ::2]
    k = torch.randn(1, 2, 11, 130, device="cuda", dtype=torch.float16)[..., ::2]
    v = torch.randn(1, 2, 11, 66, device="cuda", dtype=torch.float16)[..., ::2]
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()

    with pytest.raises(RuntimeError, match=rf"{backend} attention.*contiguous"):
        flash_attention_forward(q, k, v, backend=backend)  # type: ignore[arg-type]


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("backend", _TEACHING_BACKENDS)
def test_formal_backends_reject_head_dim_above_128(backend: str) -> None:
    q = torch.randn(1, 2, 7, 129, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 11, 129, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 11, 33, device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match=rf"{backend} attention.*head_dim"):
        flash_attention_forward(q, k, v, backend=backend)  # type: ignore[arg-type]


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("backend", _TEACHING_BACKENDS)
def test_formal_backends_reject_value_dim_above_128(backend: str) -> None:
    q = torch.randn(1, 2, 7, 65, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 11, 65, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 11, 129, device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match=rf"{backend} attention.*value_dim"):
        flash_attention_forward(q, k, v, backend=backend)  # type: ignore[arg-type]


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("backend", _TEACHING_BACKENDS)
def test_formal_backends_reject_explicit_boolean_mask(backend: str) -> None:
    q = torch.randn(1, 2, 7, 65, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 11, 65, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 11, 33, device="cuda", dtype=torch.float16)
    attn_mask = torch.ones(7, 11, device="cuda", dtype=torch.bool)

    with pytest.raises(RuntimeError, match=rf"{backend} attention.*explicit attention mask"):
        flash_attention_forward(  # type: ignore[arg-type]
            q, k, v, attn_mask=attn_mask, backend=backend
        )


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("backend", _TEACHING_BACKENDS)
def test_formal_backends_reject_cuda_requires_grad(backend: str) -> None:
    q = torch.randn(1, 2, 7, 65, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 2, 11, 65, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 2, 11, 33, device="cuda", dtype=torch.float16, requires_grad=True)

    with pytest.raises(RuntimeError, match=rf"{backend} attention.*forward-only.*requires_grad"):
        flash_attention_forward(q, k, v, backend=backend)  # type: ignore[arg-type]


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
@pytest.mark.parametrize(
    "operator_name",
    ["attention_fa1_forward", "attention_fa2_forward", "attention_fa3_forward"],
)
@pytest.mark.parametrize("grad_enabled", [True, False])
def test_formal_cuda_operator_rejects_requires_grad_inputs_directly(
    operator_name: str, grad_enabled: bool
) -> None:
    q = torch.randn(1, 2, 7, 65, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 2, 11, 65, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 2, 11, 33, device="cuda", dtype=torch.float16, requires_grad=True)
    operator = getattr(torch.ops.ds_flash_mla_moe, operator_name).default
    grad_context = torch.enable_grad() if grad_enabled else torch.no_grad()

    with (
        grad_context,
        pytest.raises(RuntimeError, match="teaching FA1/FA2/FA3.*forward-only.*requires_grad"),
    ):
        operator(q, k, v, False, 0.5)


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("backend", _TEACHING_BACKENDS)
@pytest.mark.parametrize(
    ("batch", "heads", "query_length", "value_dim"),
    [(0, 3, 7, 33), (2, 0, 7, 33), (2, 3, 0, 33), (2, 3, 7, 0)],
)
def test_formal_backends_return_exact_empty_output(
    backend: str,
    batch: int,
    heads: int,
    query_length: int,
    value_dim: int,
) -> None:
    q = torch.empty(batch, heads, query_length, 65, device="cuda", dtype=torch.float16)
    k = torch.empty(batch, heads, 11, 65, device="cuda", dtype=torch.float16)
    v = torch.empty(batch, heads, 11, value_dim, device="cuda", dtype=torch.float16)

    actual = flash_attention_forward(q, k, v, backend=backend)  # type: ignore[arg-type]

    assert actual.shape == (batch, heads, query_length, value_dim)
    assert actual.dtype == torch.float16
    assert actual.device == q.device
    assert actual.numel() == 0


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("backend", _TEACHING_BACKENDS)
def test_formal_backends_reject_empty_key_before_dispatch(backend: str) -> None:
    q = torch.empty(1, 2, 7, 65, device="cuda", dtype=torch.float16)
    k = torch.empty(1, 2, 0, 65, device="cuda", dtype=torch.float16)
    v = torch.empty(1, 2, 0, 33, device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="key sequence length must be positive"):
        flash_attention_forward(q, k, v, backend=backend)  # type: ignore[arg-type]


@_REQUIRES_FORMAL_CUDA
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


@_REQUIRES_FORMAL_CUDA
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


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    ("query_length", "key_length", "head_dim", "value_dim"),
    [(1, 7, 8, 5), (4, 4, 32, 32), (7, 11, 65, 33), (9, 17, 128, 127)],
)
def test_fa3_forward_matches_reference(
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
        actual = flash_attention_forward(q, k, v, causal=causal, backend="fa3")
        expected = blockwise_attention(q, k, v, causal=causal, block_size=3)
    rtol, atol = _fa_tolerances()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert actual.shape == (2, 3, query_length, value_dim)
    assert actual.dtype == torch.float16
    assert actual.is_contiguous()


@_REQUIRES_FORMAL_CUDA
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


@_REQUIRES_FORMAL_CUDA
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


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
def test_fa3_forward_uses_current_stream() -> None:
    torch.manual_seed(103)
    q = torch.empty(1, 2, 7, 65, device="cuda", dtype=torch.float16)
    k = torch.empty(1, 2, 11, 65, device="cuda", dtype=torch.float16)
    v = torch.empty(1, 2, 11, 33, device="cuda", dtype=torch.float16)
    stream = torch.cuda.Stream()

    with torch.no_grad(), torch.cuda.stream(stream):
        q.normal_()
        k.normal_()
        v.normal_()
        actual = flash_attention_forward(q, k, v, causal=True, backend="fa3")
        actual.record_stream(stream)
    stream.synchronize()

    with torch.no_grad():
        expected = blockwise_attention(q, k, v, causal=True, block_size=3)
    rtol, atol = _fa_tolerances()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert cuda_attention_backend_available("fa3")


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("backend", _TEACHING_BACKENDS)
def test_formal_backends_right_aligned_single_query_sees_full_history(
    backend: str,
) -> None:
    q = torch.zeros(1, 2, 1, 65, device="cuda", dtype=torch.float16)
    k = torch.zeros(1, 2, 17, 65, device="cuda", dtype=torch.float16)
    v = torch.zeros(1, 2, 17, 33, device="cuda", dtype=torch.float16)
    v[:, :, -1, :] = 17.0

    actual = flash_attention_forward(  # type: ignore[arg-type]
        q, k, v, causal=True, backend=backend
    )

    torch.testing.assert_close(actual, torch.ones_like(actual))


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("backend", _TEACHING_BACKENDS)
def test_formal_backends_right_aligned_partial_causal_boundary(backend: str) -> None:
    torch.manual_seed(107)
    q = torch.randn(1, 2, 7, 65, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 11, 65, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 11, 33, device="cuda", dtype=torch.float16)

    actual = flash_attention_forward(  # type: ignore[arg-type]
        q, k, v, causal=True, backend=backend
    )
    expected = blockwise_attention(q, k, v, causal=True, block_size=3)

    rtol, atol = _fa_tolerances()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("backend", _TEACHING_BACKENDS)
def test_formal_backends_stable_softmax_with_large_logits(backend: str) -> None:
    torch.manual_seed(109)
    q = torch.randn(1, 2, 7, 65, device="cuda", dtype=torch.float16) * 20
    k = torch.randn(1, 2, 17, 65, device="cuda", dtype=torch.float16) * 20
    v = torch.randn(1, 2, 17, 33, device="cuda", dtype=torch.float16)

    actual = flash_attention_forward(  # type: ignore[arg-type]
        q, k, v, causal=True, backend=backend
    )
    expected = blockwise_attention(q, k, v, causal=True, block_size=3)

    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    rtol, atol = _fa_tolerances()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@_REQUIRES_FORMAL_CUDA
@pytest.mark.cuda
def test_fa1_fa2_and_fa3_match_the_same_reference_on_identical_inputs() -> None:
    torch.manual_seed(103)
    q = torch.randn(1, 2, 9, 65, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 17, 65, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 17, 33, device="cuda", dtype=torch.float16)
    expected = blockwise_attention(q, k, v, causal=True, block_size=5)
    rtol, atol = _fa_tolerances()
    for backend in _TEACHING_BACKENDS:
        actual = flash_attention_forward(  # type: ignore[arg-type]
            q, k, v, causal=True, backend=backend
        )
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
