from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensorMode

from ds_flash_mla_moe import (
    cuda_moe_available,
    deepseek_moe_forward,
    deepseek_moe_reference,
)
from ds_flash_mla_moe import moe_ops as facade_ops
from ds_flash_mla_moe import ops as moe_ops

TOKENS = 7
MODEL_DIM = 5
HIDDEN = 9
EXPERTS = 4
TOPK = 2
GROUPS = 2
TOPK_GROUPS = 1


def _raw_moe_inputs(
    *,
    tokens: int = TOKENS,
    model_dim: int = MODEL_DIM,
    hidden: int = HIDDEN,
    experts: int = EXPERTS,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    options = {"dtype": dtype, "device": device}
    return (
        torch.empty(tokens, model_dim, **options),
        torch.empty(experts, model_dim, **options),
        torch.empty(experts, hidden, model_dim, **options),
        torch.empty(experts, model_dim, hidden, **options),
        torch.empty(experts, hidden, model_dim, **options),
    )


def _call_raw_moe(
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    *,
    topk: int = TOPK,
    n_groups: int = GROUPS,
    topk_groups: int = TOPK_GROUPS,
    score_bias: Tensor | None = None,
    route_scale: float = 1.0,
) -> Tensor:
    return torch.ops.ds_flash_mla_moe.deepseek_moe_forward.default(
        *inputs,
        topk,
        n_groups,
        topk_groups,
        score_bias,
        route_scale,
    )


def _noncontiguous_empty_like(tensor: Tensor) -> Tensor:
    return torch.empty(
        (*tensor.shape, 2),
        dtype=tensor.dtype,
        device=tensor.device,
    )[..., 0]


def test_raw_moe_operator_schema_is_always_defined() -> None:
    assert moe_ops._operator_is_defined("deepseek_moe_forward")


@pytest.mark.parametrize(
    "dispatch_key",
    ("CPU", "AutogradCUDA", "CompositeExplicitAutograd", "CompositeImplicitAutograd"),
)
def test_raw_moe_operator_has_only_forward_native_dispatch_policy(
    dispatch_key: str,
) -> None:
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(  # type: ignore[attr-defined]
        "ds_flash_mla_moe::deepseek_moe_forward",
        dispatch_key,
    )


@pytest.mark.parametrize("tokens", [0, TOKENS])
@pytest.mark.parametrize("with_bias", [False, True])
def test_raw_moe_fake_propagates_flattened_output_metadata(
    tokens: int,
    with_bias: bool,
) -> None:
    with FakeTensorMode():
        inputs = _raw_moe_inputs(tokens=tokens)
        score_bias = (
            torch.empty(EXPERTS, dtype=inputs[0].dtype, device=inputs[0].device)
            if with_bias
            else None
        )

        output = _call_raw_moe(inputs, score_bias=score_bias)

        assert output.shape == (tokens, MODEL_DIM)
        assert output.dtype == inputs[0].dtype
        assert output.device == inputs[0].device
        assert output.stride() == (MODEL_DIM, 1)
        assert output.is_contiguous()


@pytest.mark.parametrize(
    ("index", "shape"),
    [
        (0, (1, TOKENS, MODEL_DIM)),
        (1, (EXPERTS, 1, MODEL_DIM)),
        (2, (EXPERTS, HIDDEN)),
        (3, (EXPERTS, MODEL_DIM)),
        (4, (EXPERTS, HIDDEN, MODEL_DIM, 1)),
    ],
)
def test_raw_moe_fake_rejects_invalid_input_ranks(
    index: int,
    shape: tuple[int, ...],
) -> None:
    with FakeTensorMode():
        inputs = list(_raw_moe_inputs())
        inputs[index] = torch.empty(shape, dtype=torch.float32)

        with pytest.raises(RuntimeError):
            _call_raw_moe(tuple(inputs))


@pytest.mark.parametrize(
    ("index", "shape"),
    [
        (0, (TOKENS, MODEL_DIM + 1)),
        (1, (EXPERTS + 1, MODEL_DIM)),
        (1, (EXPERTS, MODEL_DIM + 1)),
        (2, (EXPERTS + 1, HIDDEN, MODEL_DIM)),
        (2, (EXPERTS, HIDDEN + 1, MODEL_DIM)),
        (2, (EXPERTS, HIDDEN, MODEL_DIM + 1)),
        (3, (EXPERTS + 1, MODEL_DIM, HIDDEN)),
        (3, (EXPERTS, MODEL_DIM + 1, HIDDEN)),
        (3, (EXPERTS, MODEL_DIM, HIDDEN + 1)),
        (4, (EXPERTS + 1, HIDDEN, MODEL_DIM)),
        (4, (EXPERTS, HIDDEN + 1, MODEL_DIM)),
        (4, (EXPERTS, HIDDEN, MODEL_DIM + 1)),
    ],
)
def test_raw_moe_fake_rejects_inconsistent_dimensions(
    index: int,
    shape: tuple[int, ...],
) -> None:
    with FakeTensorMode():
        inputs = list(_raw_moe_inputs())
        inputs[index] = torch.empty(shape, dtype=torch.float32)

        with pytest.raises(RuntimeError):
            _call_raw_moe(tuple(inputs))


@pytest.mark.parametrize(
    ("experts", "hidden", "model_dim"),
    [(0, HIDDEN, MODEL_DIM), (EXPERTS, 0, MODEL_DIM), (EXPERTS, HIDDEN, 0)],
)
def test_raw_moe_fake_rejects_nonpositive_model_dimensions(
    experts: int,
    hidden: int,
    model_dim: int,
) -> None:
    with FakeTensorMode(), pytest.raises(RuntimeError):
        _call_raw_moe(
            _raw_moe_inputs(
                experts=experts,
                hidden=hidden,
                model_dim=model_dim,
            )
        )


def test_raw_moe_fake_rejects_nonfloating_inputs() -> None:
    with FakeTensorMode(), pytest.raises(RuntimeError):
        _call_raw_moe(_raw_moe_inputs(dtype=torch.int64))


@pytest.mark.parametrize("index", range(5))
def test_raw_moe_fake_rejects_mismatched_input_dtypes(index: int) -> None:
    with FakeTensorMode():
        inputs = list(_raw_moe_inputs())
        inputs[index] = torch.empty(inputs[index].shape, dtype=torch.float64)

        with pytest.raises(RuntimeError):
            _call_raw_moe(tuple(inputs))


def test_raw_moe_fake_rejects_mismatched_input_devices() -> None:
    with FakeTensorMode():
        inputs = list(_raw_moe_inputs())
        inputs[1] = torch.empty(inputs[1].shape, dtype=torch.float32, device="meta")

        with pytest.raises(RuntimeError):
            _call_raw_moe(tuple(inputs))


@pytest.mark.parametrize("index", range(6))
def test_raw_moe_fake_rejects_noncontiguous_inputs(index: int) -> None:
    with FakeTensorMode():
        inputs = list(_raw_moe_inputs())
        score_bias = torch.empty(EXPERTS, dtype=torch.float32)
        if index < len(inputs):
            inputs[index] = _noncontiguous_empty_like(inputs[index])
        else:
            score_bias = _noncontiguous_empty_like(score_bias)

        with pytest.raises(RuntimeError):
            _call_raw_moe(tuple(inputs), score_bias=score_bias)


@pytest.mark.parametrize("failure", ["shape", "dtype", "device"])
def test_raw_moe_fake_rejects_invalid_score_bias(failure: str) -> None:
    with FakeTensorMode():
        inputs = _raw_moe_inputs()
        if failure == "shape":
            score_bias = torch.empty(EXPERTS, 1, dtype=torch.float32)
        elif failure == "dtype":
            score_bias = torch.empty(EXPERTS, dtype=torch.float64)
        else:
            score_bias = torch.empty(EXPERTS, dtype=torch.float32, device="meta")

        with pytest.raises(RuntimeError):
            _call_raw_moe(inputs, score_bias=score_bias)


@pytest.mark.parametrize(
    "overrides",
    [
        {"topk": 0},
        {"topk": EXPERTS + 1},
        {"topk": 3, "n_groups": GROUPS, "topk_groups": TOPK_GROUPS},
        {"n_groups": 0},
        {"n_groups": 3},
        {"n_groups": GROUPS, "topk_groups": 0},
        {"n_groups": GROUPS, "topk_groups": GROUPS + 1},
        {"route_scale": float("nan")},
        {"route_scale": float("inf")},
        {"route_scale": float("-inf")},
    ],
)
def test_raw_moe_fake_rejects_invalid_routing_configuration(
    overrides: dict[str, int | float],
) -> None:
    with FakeTensorMode(), pytest.raises(RuntimeError):
        _call_raw_moe(_raw_moe_inputs(), **overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize("index", range(6))
def test_raw_moe_fake_rejects_requires_grad_inputs(index: int) -> None:
    with FakeTensorMode():
        inputs = list(_raw_moe_inputs())
        score_bias = torch.empty(EXPERTS, dtype=torch.float32)
        if index < len(inputs):
            inputs[index].requires_grad_(True)
        else:
            score_bias.requires_grad_(True)

        with pytest.raises(RuntimeError, match="forward-only"):
            _call_raw_moe(tuple(inputs), score_bias=score_bias)


def _moe_inputs(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    rank: int = 2,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(421)
    if rank == 2:
        token_shape = (TOKENS,)
    elif rank == 3:
        token_shape = (1, TOKENS)
    elif rank == 4:
        token_shape = (1, 1, TOKENS)
    else:
        raise ValueError("test inputs support rank 2, 3, or 4")
    return (
        torch.randn(*token_shape, MODEL_DIM, dtype=dtype, device=device),
        torch.randn(EXPERTS, MODEL_DIM, dtype=dtype, device=device),
        torch.randn(EXPERTS, HIDDEN, MODEL_DIM, dtype=dtype, device=device),
        torch.randn(EXPERTS, MODEL_DIM, HIDDEN, dtype=dtype, device=device),
        torch.randn(EXPERTS, HIDDEN, MODEL_DIM, dtype=dtype, device=device),
    )


def _fresh_grad_inputs(inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return tuple(tensor.detach().clone().requires_grad_(True) for tensor in inputs)


def _assert_reference_and_gradient_parity(
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    *,
    score_bias: Tensor | None = None,
    **kwargs: object,
) -> tuple[tuple[Tensor | None, ...], tuple[Tensor | None, ...]]:
    facade_inputs = _fresh_grad_inputs(inputs)
    oracle_inputs = _fresh_grad_inputs(inputs)
    facade_bias = None if score_bias is None else score_bias.detach().clone().requires_grad_(True)
    oracle_bias = None if score_bias is None else score_bias.detach().clone().requires_grad_(True)

    actual = deepseek_moe_forward(
        *facade_inputs,
        score_bias=facade_bias,
        backend="reference",
        **kwargs,
    )
    expected = deepseek_moe_reference(
        *oracle_inputs,
        score_bias=oracle_bias,
        **kwargs,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert actual.is_contiguous()
    upstream = torch.linspace(
        -0.75,
        1.25,
        actual.numel(),
        dtype=actual.dtype,
        device=actual.device,
    ).reshape_as(actual)
    facade_parameters = (*facade_inputs,) if facade_bias is None else (*facade_inputs, facade_bias)
    oracle_parameters = (*oracle_inputs,) if oracle_bias is None else (*oracle_inputs, oracle_bias)
    actual_grads = torch.autograd.grad(
        actual,
        facade_parameters,
        upstream,
        allow_unused=True,
    )
    expected_grads = torch.autograd.grad(
        expected,
        oracle_parameters,
        upstream,
        allow_unused=True,
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        if expected_grad is None:
            assert actual_grad is None
        else:
            assert actual_grad is not None
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-9)
    return actual_grads, expected_grads


@pytest.mark.parametrize(
    ("rank", "score_func", "with_bias", "n_groups", "topk_groups", "route_scale"),
    [
        (2, "sigmoid", False, 1, None, 0.625),
        (3, "softmax", True, GROUPS, TOPK_GROUPS, 1.75),
    ],
)
def test_reference_facade_matches_fresh_direct_oracle_outputs_and_gradients(
    rank: int,
    score_func: str,
    with_bias: bool,
    n_groups: int,
    topk_groups: int | None,
    route_scale: float,
) -> None:
    inputs = _moe_inputs(rank=rank)
    score_bias = torch.tensor([0.8, -0.3, 0.4, -0.7], dtype=torch.float64)

    _assert_reference_and_gradient_parity(
        inputs,
        topk=TOPK,
        n_groups=n_groups,
        topk_groups=topk_groups,
        score_func=score_func,
        score_bias=score_bias if with_bias else None,
        route_scale=route_scale,
    )


def test_reference_facade_supports_rank_four_output_and_gradients() -> None:
    _assert_reference_and_gradient_parity(
        _moe_inputs(rank=4),
        topk=TOPK,
        n_groups=GROUPS,
        topk_groups=TOPK_GROUPS,
        score_func="sigmoid",
        route_scale=1.25,
    )


def test_reference_facade_preserves_exact_tie_routing() -> None:
    inputs = list(_moe_inputs())
    inputs[1] = torch.zeros_like(inputs[1])
    typed_inputs = tuple(inputs)

    _assert_reference_and_gradient_parity(
        typed_inputs,
        topk=TOPK,
        n_groups=GROUPS,
        topk_groups=TOPK_GROUPS,
        score_func="sigmoid",
        route_scale=1.0,
    )
    _, routing = deepseek_moe_reference(
        *typed_inputs,
        topk=TOPK,
        n_groups=GROUPS,
        topk_groups=TOPK_GROUPS,
        return_routing=True,
    )
    assert routing.indices.tolist() == [[0, 1]] * TOKENS


def test_topk_one_leaves_unselected_expert_gradients_inactive() -> None:
    inputs = list(_moe_inputs())
    inputs[0] = torch.ones_like(inputs[0])
    inputs[1] = -torch.ones_like(inputs[1])
    inputs[1][0] = 1.0
    typed_inputs = tuple(inputs)

    actual_grads, _ = _assert_reference_and_gradient_parity(
        typed_inputs,
        topk=1,
        n_groups=1,
        score_func="sigmoid",
        route_scale=1.0,
    )
    _, routing = deepseek_moe_reference(
        *typed_inputs,
        topk=1,
        return_routing=True,
    )
    assert routing.indices.tolist() == [[0]] * TOKENS
    for expert_grad in actual_grads[2:5]:
        assert expert_grad is not None
        assert torch.count_nonzero(expert_grad[1:]).item() == 0


@pytest.mark.parametrize("rank", [2, 3])
def test_reference_facade_supports_zero_tokens(rank: int) -> None:
    inputs = list(_moe_inputs(rank=rank))
    inputs[0] = (
        torch.empty(0, MODEL_DIM, dtype=torch.float64)
        if rank == 2
        else torch.empty(2, 0, MODEL_DIM, dtype=torch.float64)
    )

    actual = deepseek_moe_forward(
        *inputs,
        topk=TOPK,
        n_groups=GROUPS,
        topk_groups=TOPK_GROUPS,
        backend="reference",
    )
    expected = deepseek_moe_reference(
        *inputs,
        topk=TOPK,
        n_groups=GROUPS,
        topk_groups=TOPK_GROUPS,
    )

    torch.testing.assert_close(actual, expected)
    assert actual.shape == inputs[0].shape
    assert actual.numel() == 0
    assert actual.is_contiguous()


def test_invalid_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        deepseek_moe_forward(*_moe_inputs(), topk=TOPK, backend="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor(1.0, dtype=torch.float64),
        torch.randn(MODEL_DIM, dtype=torch.float64),
    ],
)
def test_invalid_input_rank_is_rejected(x: Tensor) -> None:
    inputs = list(_moe_inputs())
    inputs[0] = x
    with pytest.raises(ValueError, match="rank at least 2"):
        deepseek_moe_forward(*inputs, topk=TOPK)


@pytest.mark.parametrize(
    ("model_dim", "hidden", "message"),
    [
        (0, HIDDEN, "model_dim must be positive"),
        (MODEL_DIM, 0, "hidden must be positive"),
    ],
)
def test_zero_model_or_hidden_dimension_is_rejected(
    model_dim: int,
    hidden: int,
    message: str,
) -> None:
    inputs = (
        torch.empty(TOKENS, model_dim, dtype=torch.float64),
        torch.empty(EXPERTS, model_dim, dtype=torch.float64),
        torch.empty(EXPERTS, hidden, model_dim, dtype=torch.float64),
        torch.empty(EXPERTS, model_dim, hidden, dtype=torch.float64),
        torch.empty(EXPERTS, hidden, model_dim, dtype=torch.float64),
    )

    with pytest.raises(ValueError, match=message):
        deepseek_moe_forward(*inputs, topk=TOPK, backend="reference")


@pytest.mark.parametrize(
    ("index", "shape"),
    [
        (0, (TOKENS, MODEL_DIM + 1)),
        (1, (EXPERTS, MODEL_DIM + 1)),
        (2, (EXPERTS, HIDDEN, MODEL_DIM + 1)),
        (3, (EXPERTS, MODEL_DIM, HIDDEN + 1)),
        (4, (EXPERTS, HIDDEN + 1, MODEL_DIM)),
    ],
)
def test_inconsistent_shapes_are_rejected(index: int, shape: tuple[int, ...]) -> None:
    inputs = list(_moe_inputs())
    inputs[index] = torch.randn(*shape, dtype=torch.float64)
    with pytest.raises(ValueError):
        deepseek_moe_forward(*inputs, topk=TOPK)


def test_floating_inputs_must_share_dtype() -> None:
    inputs = list(_moe_inputs())
    inputs[3] = inputs[3].float()
    with pytest.raises(ValueError, match="dtype"):
        deepseek_moe_forward(*inputs, topk=TOPK)


def test_floating_inputs_must_share_device() -> None:
    inputs = list(_moe_inputs())
    inputs[1] = torch.empty(inputs[1].shape, dtype=torch.float64, device="meta")
    with pytest.raises(ValueError, match="device"):
        deepseek_moe_forward(*inputs, topk=TOPK)


def test_nonfloating_inputs_are_rejected() -> None:
    inputs = tuple(torch.zeros_like(tensor, dtype=torch.int64) for tensor in _moe_inputs())
    with pytest.raises(TypeError, match="floating-point"):
        deepseek_moe_forward(*inputs, topk=TOPK)


@pytest.mark.parametrize(
    "overrides",
    [
        {"topk": 0},
        {"topk": EXPERTS + 1},
        {"topk": 3, "n_groups": GROUPS, "topk_groups": TOPK_GROUPS},
        {"n_groups": 0},
        {"n_groups": 3},
        {"n_groups": GROUPS, "topk_groups": 0},
        {"n_groups": GROUPS, "topk_groups": GROUPS + 1},
    ],
)
def test_invalid_routing_configuration_is_rejected(overrides: dict[str, int]) -> None:
    kwargs = {"topk": TOPK, **overrides}
    with pytest.raises(ValueError):
        deepseek_moe_forward(*_moe_inputs(), **kwargs)


def test_invalid_score_function_is_rejected() -> None:
    with pytest.raises(ValueError, match="score_func"):
        deepseek_moe_forward(
            *_moe_inputs(),
            topk=TOPK,
            score_func="relu",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("route_scale", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_route_scale_is_rejected(route_scale: float) -> None:
    with pytest.raises(ValueError, match="route_scale"):
        deepseek_moe_forward(*_moe_inputs(), topk=TOPK, route_scale=route_scale)


def test_malformed_score_bias_is_rejected() -> None:
    with pytest.raises(ValueError, match="score_bias"):
        deepseek_moe_forward(
            *_moe_inputs(),
            topk=TOPK,
            score_bias=torch.randn(EXPERTS, 1, dtype=torch.float64),
        )


@pytest.mark.parametrize(
    "make_bias",
    [
        lambda: torch.randn(EXPERTS, dtype=torch.float32),
        lambda: torch.empty(EXPERTS, dtype=torch.float64, device="meta"),
    ],
)
def test_score_bias_must_share_dtype_and_device(make_bias: Callable[[], Tensor]) -> None:
    with pytest.raises(ValueError, match="score_bias"):
        deepseek_moe_forward(
            *_moe_inputs(),
            topk=TOPK,
            score_bias=make_bias(),
        )


def test_explicit_cuda_rejects_cpu_request_with_required_prefix() -> None:
    with pytest.raises(RuntimeError, match=r"^CUDA DeepSeek MoE is unavailable:"):
        deepseek_moe_forward(
            *_moe_inputs(dtype=torch.float32),
            topk=TOPK,
            backend="cuda",
        )


def test_cuda_moe_available_returns_bool() -> None:
    assert isinstance(cuda_moe_available(), bool)


def test_cuda_moe_available_queries_whole_layer_operator(monkeypatch) -> None:
    operators: list[str] = []

    def has_cuda_kernel(operator: str) -> bool:
        operators.append(operator)
        return False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(facade_ops, "_operator_has_cuda_kernel", has_cuda_kernel)

    assert not cuda_moe_available()
    assert operators == ["deepseek_moe_forward"]


def test_eligible_auto_calls_native_once(monkeypatch) -> None:
    inputs = _moe_inputs(dtype=torch.float32)
    expected = torch.full_like(inputs[0], 3.0)
    native_calls = 0

    def call_native(*args: object, **kwargs: object) -> Tensor:
        nonlocal native_calls
        native_calls += 1
        return expected

    monkeypatch.setattr(facade_ops, "_cuda_moe_ineligibility_reason", lambda *a, **k: None)
    monkeypatch.setattr(facade_ops, "_call_cuda_moe", call_native)
    monkeypatch.setattr(
        facade_ops,
        "deepseek_moe_packed_reference",
        lambda *a, **k: pytest.fail("reference fallback"),
    )

    actual = deepseek_moe_forward(*inputs, topk=TOPK, backend="auto")

    assert actual is expected
    assert native_calls == 1


def test_ineligible_auto_calls_packed_reference_once(monkeypatch) -> None:
    inputs = _moe_inputs(dtype=torch.float32)
    expected = torch.full_like(inputs[0], 5.0)
    reference_calls: list[dict[str, object]] = []

    def call_reference(*args: object, **kwargs: object) -> Tensor:
        reference_calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        facade_ops,
        "_cuda_moe_ineligibility_reason",
        lambda *a, **k: "not eligible",
    )
    monkeypatch.setattr(
        facade_ops,
        "_call_cuda_moe",
        lambda *a, **k: pytest.fail("native dispatch"),
    )
    monkeypatch.setattr(facade_ops, "deepseek_moe_packed_reference", call_reference)

    actual = deepseek_moe_forward(*inputs, topk=TOPK, backend="auto")

    assert actual is expected
    assert len(reference_calls) == 1
    assert reference_calls[0]["topk_groups"] == 1


def test_selected_cuda_failure_is_not_retried(monkeypatch) -> None:
    inputs = _moe_inputs(dtype=torch.float32)
    monkeypatch.setattr(facade_ops, "_cuda_moe_ineligibility_reason", lambda *a, **k: None)
    monkeypatch.setattr(
        facade_ops,
        "_call_cuda_moe",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("launch failed")),
    )
    monkeypatch.setattr(
        facade_ops,
        "deepseek_moe_packed_reference",
        lambda *a, **k: pytest.fail("fallback"),
    )
    with pytest.raises(RuntimeError, match="launch failed"):
        deepseek_moe_forward(*inputs, topk=TOPK, backend="auto")


def test_reference_backend_ignores_native_eligibility(monkeypatch) -> None:
    inputs = _moe_inputs(dtype=torch.float32)
    expected = torch.full_like(inputs[0], 7.0)
    monkeypatch.setattr(facade_ops, "_cuda_moe_ineligibility_reason", lambda *a, **k: None)
    monkeypatch.setattr(
        facade_ops,
        "_call_cuda_moe",
        lambda *a, **k: pytest.fail("native dispatch"),
    )
    monkeypatch.setattr(
        facade_ops,
        "deepseek_moe_packed_reference",
        lambda *a, **k: expected,
    )

    actual = deepseek_moe_forward(*inputs, topk=TOPK, backend="reference")

    assert actual is expected


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_reference_backend_never_invokes_native_for_cuda_tensors(monkeypatch) -> None:
    inputs = _moe_inputs(dtype=torch.float32, device="cuda")
    monkeypatch.setattr(
        facade_ops,
        "_call_cuda_moe",
        lambda *a, **k: pytest.fail("native dispatch"),
    )

    actual = deepseek_moe_forward(*inputs, topk=TOPK, backend="reference")
    expected = deepseek_moe_reference(*inputs, topk=TOPK)

    torch.testing.assert_close(actual, expected)
