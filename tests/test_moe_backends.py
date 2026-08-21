from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_MOE_OPERATORS = (
    "deepseek_moe_forward",
    "deepseek_moe_forward_fused",
    "deepseek_moe_forward_persistent",
)
EXPLICIT_NATIVE_BACKENDS = {
    "cuda": "deepseek_moe_forward_fused",
    "cuda_staged": "deepseek_moe_forward",
    "cuda_fused": "deepseek_moe_forward_fused",
    "cuda_persistent": "deepseek_moe_forward_persistent",
}


def test_reader_docs_describe_the_whole_layer_moe_backends_honestly() -> None:
    reader_docs = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "chapters" / "04-deepseek-moe.md",
        REPO_ROOT / "docs" / "chapters" / "06-pytorch-custom-operators.md",
        REPO_ROOT / "docs" / "chapters" / "07-benchmarking-and-roofline.md",
    )

    for document in reader_docs:
        source = document.read_text(encoding="utf-8")
        assert "deepseek_moe_forward" in source, document
        assert "single-device" in source, document
        assert "staged" in source, document
        assert "correctness-first" in source, document
        assert "cuda_fused" in source, document
        assert "cuda_persistent" in source, document

    scope_statement = (
        "The persistent variant is a single-device expert core, not a distributed "
        "FlashMoE megakernel"
    )
    scope_docs = reader_docs[:2]
    missing_statement = [
        document
        for document in scope_docs
        if scope_statement not in " ".join(document.read_text(encoding="utf-8").split())
    ]
    assert not missing_statement, missing_statement


def test_readme_status_table_lists_every_single_device_whole_layer_moe_path() -> None:
    readme_lines = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    status_row = next(
        line for line in readme_lines if line.startswith("| DeepSeek whole-layer MoE forward |")
    )

    for contract_term in (
        "single-device",
        "staged",
        "fused",
        "persistent",
        "correctness-first",
        "FP32",
    ):
        assert contract_term in status_row
    assert "FlashMoE" not in status_row


def test_cuda_build_gate_includes_formal_attention_and_whole_layer_moe_ops() -> None:
    workflow_source = (REPO_ROOT / ".github" / "workflows" / "cuda-build.yml").read_text(
        encoding="utf-8"
    )

    for operator in (
        "attention_fa1_forward",
        "attention_fa2_forward",
        "attention_fa3_forward",
        "deepseek_moe_forward",
        "deepseek_moe_forward_fused",
        "deepseek_moe_forward_persistent",
    ):
        assert f'"{operator}"' in workflow_source


def test_whole_layer_cuda_sources_are_packaged_and_registered() -> None:
    cuda_sources = (
        REPO_ROOT / "csrc" / "moe" / "deepseek_moe_forward_cuda.cu",
        REPO_ROOT / "csrc" / "moe" / "deepseek_moe_forward_fused_cuda.cu",
    )
    host_header = REPO_ROOT / "csrc" / "moe" / "moe_cuda_ops.h"
    assert all(cuda_source.is_file() for cuda_source in cuda_sources)
    assert host_header.is_file()

    setup_source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    for cuda_source in cuda_sources:
        relative = cuda_source.relative_to(REPO_ROOT).as_posix()
        assert f'"{relative}"' in setup_source

    ops_source = (REPO_ROOT / "csrc" / "ops.cpp").read_text(encoding="utf-8")
    for operator in RAW_MOE_OPERATORS:
        operator_schema = (
            f"{operator}(Tensor x, Tensor gate_weight, Tensor expert_w1, "
            "Tensor expert_w2, Tensor expert_w3, int topk, int n_groups, "
            "int topk_groups, Tensor? score_bias, float route_scale) -> Tensor"
        )
        assert operator_schema in ops_source

    manifest_lines = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    csrc_patterns = {
        pattern
        for line in manifest_lines
        if line.startswith("recursive-include csrc ")
        for pattern in line.split()[2:]
    }
    assert {"*.h", "*.cu", "*.cpp"} <= csrc_patterns


def test_private_single_device_pack_contract_omits_distributed_route_metadata() -> None:
    route_source = (REPO_ROOT / "csrc" / "moe" / "route_ops_cuda.cu").read_text(encoding="utf-8")
    header_source = (REPO_ROOT / "csrc" / "moe" / "moe_cuda_ops.h").read_text(encoding="utf-8")

    assert "single_device_route_pack_cuda_entry(" in header_source
    implementation = route_source.split("single_device_route_pack_cuda(", 1)[1].split(
        "at::Tensor route_combine_cuda(", 1
    )[0]
    for output in (
        "packed_activations",
        "packed_weights",
        "token_indices",
        "expert_offsets",
    ):
        assert output in implementation
    for distributed_artifact in (
        "expert_owner",
        "rank_counts",
        "packed_expert_indices",
        "counts_per_expert",
        "at::cat",
        "at::cumsum",
        "at::floor_divide",
    ):
        assert distributed_artifact not in implementation


def test_private_generated_offsets_have_host_structure_guards_without_value_sync() -> None:
    route_source = (REPO_ROOT / "csrc" / "moe" / "route_ops_cuda.cu").read_text(encoding="utf-8")
    expert_source = (REPO_ROOT / "csrc" / "moe" / "swiglu_experts_cuda.cu").read_text(
        encoding="utf-8"
    )
    private_pack = route_source.split("single_device_route_pack_cuda(", 1)[1].split(
        "at::Tensor route_combine_cuda(", 1
    )[0]
    fused_experts = expert_source.split("at::Tensor swiglu_experts_fused_cuda(", 1)[1].split(
        "at::Tensor swiglu_experts_cuda(", 1
    )[0]

    for implementation in (private_pack, fused_experts):
        assert "TORCH_CHECK(experts < INT64_MAX" in implementation
        assert "experts + 1 overflows int64" in implementation
        assert "without device-to-host synchronization" in implementation
        for host_sync in ("cudaMemcpy", ".cpu(", ".item("):
            assert host_sync not in implementation


def test_fused_expert_epilogue_and_persistent_scheduler_have_distinct_contracts() -> None:
    expert_source = (REPO_ROOT / "csrc" / "moe" / "swiglu_experts_cuda.cu").read_text(
        encoding="utf-8"
    )
    header_source = (REPO_ROOT / "csrc" / "moe" / "moe_cuda_ops.h").read_text(encoding="utf-8")

    for entry in (
        "swiglu_experts_fused_cuda_entry(",
        "swiglu_experts_persistent_cuda_entry(",
    ):
        assert entry in header_source
        assert entry in expert_source
    assert "fused_down_atomic_float_kernel" in expert_source
    assert "persistent_hidden_float_kernel" in expert_source
    assert "persistent_down_atomic_float_kernel" in expert_source
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in expert_source
    assert "persistent_task_queue" in expert_source
    assert "atomicAdd" in expert_source
    assert "kPersistentSmallWorkRoutes" in expert_source
    normalized_expert_source = " ".join(expert_source.replace("//", "").split())
    assert "single-device expert core" in normalized_expert_source
    assert "not a distributed FlashMoE megakernel" in normalized_expert_source


def test_fused_whole_layer_source_uses_only_private_single_device_entries() -> None:
    fused_source = (REPO_ROOT / "csrc" / "moe" / "deepseek_moe_forward_fused_cuda.cu").read_text(
        encoding="utf-8"
    )

    assert "grouped_topk_cuda_entry(" in fused_source
    assert "single_device_route_pack_cuda_entry(" in fused_source
    assert "swiglu_experts_fused_cuda_entry(" in fused_source
    assert "swiglu_experts_persistent_cuda_entry(" in fused_source
    fused_source_without_private_pack = fused_source.replace(
        "single_device_route_pack_cuda_entry(", ""
    )
    for staged_artifact in (
        "expert_owner",
        "route_pack_cuda_entry(",
        "route_combine_cuda_entry(",
        "swiglu_experts_cuda_entry(",
        "at::cat",
        "at::cumsum",
        "at::floor_divide",
        "contributions",
    ):
        assert staged_artifact not in fused_source_without_private_pack


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
    operator: str = "deepseek_moe_forward",
) -> Tensor:
    return getattr(torch.ops.ds_flash_mla_moe, operator).default(
        *inputs,
        topk,
        n_groups,
        topk_groups,
        score_bias,
        route_scale,
    )


def _noncontiguous_view(tensor: Tensor) -> Tensor:
    return tensor.as_strided(tensor.shape, (0, *tensor.stride()[1:]))


@pytest.mark.parametrize("operator", RAW_MOE_OPERATORS)
def test_raw_moe_operator_schema_is_always_defined(operator: str) -> None:
    assert moe_ops._operator_is_defined(operator)


@pytest.mark.parametrize(
    "dispatch_key",
    ("CPU", "AutogradCUDA", "CompositeExplicitAutograd", "CompositeImplicitAutograd"),
)
@pytest.mark.parametrize("operator", RAW_MOE_OPERATORS)
def test_raw_moe_operator_has_only_forward_native_dispatch_policy(
    dispatch_key: str,
    operator: str,
) -> None:
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(  # type: ignore[attr-defined]
        f"ds_flash_mla_moe::{operator}",
        dispatch_key,
    )


@pytest.mark.parametrize("tokens", [0, TOKENS])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("operator", RAW_MOE_OPERATORS)
def test_raw_moe_fake_propagates_flattened_output_metadata(
    tokens: int,
    with_bias: bool,
    operator: str,
) -> None:
    with FakeTensorMode():
        inputs = _raw_moe_inputs(tokens=tokens)
        score_bias = (
            torch.empty(EXPERTS, dtype=inputs[0].dtype, device=inputs[0].device)
            if with_bias
            else None
        )

        output = _call_raw_moe(inputs, score_bias=score_bias, operator=operator)

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


@pytest.mark.parametrize("operator", RAW_MOE_OPERATORS)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "bias"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_raw_moe_fake_rejects_uniform_unsupported_floating_dtypes(
    dtype: torch.dtype,
    with_bias: bool,
    operator: str,
) -> None:
    with FakeTensorMode():
        inputs = _raw_moe_inputs(dtype=dtype)
        score_bias = torch.empty(EXPERTS, dtype=dtype) if with_bias else None

        with pytest.raises(RuntimeError, match="float32"):
            _call_raw_moe(inputs, score_bias=score_bias, operator=operator)


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
            inputs[index] = _noncontiguous_view(inputs[index])
        else:
            score_bias = _noncontiguous_view(score_bias)

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
@pytest.mark.parametrize("operator", RAW_MOE_OPERATORS)
def test_raw_moe_fake_rejects_requires_grad_inputs(index: int, operator: str) -> None:
    with FakeTensorMode():
        inputs = list(_raw_moe_inputs())
        score_bias = torch.empty(EXPERTS, dtype=torch.float32)
        if index < len(inputs):
            inputs[index].requires_grad_(True)
        else:
            score_bias.requires_grad_(True)

        with pytest.raises(RuntimeError, match="forward-only"):
            _call_raw_moe(tuple(inputs), score_bias=score_bias, operator=operator)


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


@pytest.mark.parametrize("backend", EXPLICIT_NATIVE_BACKENDS)
def test_explicit_cuda_rejects_cpu_request_with_required_prefix(backend: str) -> None:
    with pytest.raises(RuntimeError, match=r"^CUDA DeepSeek MoE is unavailable:"):
        deepseek_moe_forward(
            *_moe_inputs(dtype=torch.float32),
            topk=TOPK,
            backend=backend,  # type: ignore[arg-type]
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
    assert operators == ["deepseek_moe_forward_fused"]


@pytest.mark.parametrize(
    ("backend", "expected_operator"),
    EXPLICIT_NATIVE_BACKENDS.items(),
)
def test_explicit_native_backend_selects_exactly_one_whole_layer_operator(
    backend: str,
    expected_operator: str,
    monkeypatch,
) -> None:
    inputs = _moe_inputs(dtype=torch.float32)
    expected = torch.full_like(inputs[0], 2.0)
    eligibility_queries: list[str] = []
    native_calls: list[str] = []

    def eligible(*args: object, operator: str, **kwargs: object) -> None:
        eligibility_queries.append(operator)

    def call_native(*args: object, operator: str, **kwargs: object) -> Tensor:
        native_calls.append(operator)
        return expected

    monkeypatch.setattr(facade_ops, "_cuda_moe_ineligibility_reason", eligible)
    monkeypatch.setattr(facade_ops, "_call_cuda_moe", call_native)
    monkeypatch.setattr(
        facade_ops,
        "deepseek_moe_packed_reference",
        lambda *a, **k: pytest.fail("reference fallback"),
    )

    actual = deepseek_moe_forward(
        *inputs,
        topk=TOPK,
        backend=backend,  # type: ignore[arg-type]
    )

    assert actual is expected
    assert eligibility_queries == [expected_operator]
    assert native_calls == [expected_operator]


def test_eligible_auto_calls_native_once(monkeypatch) -> None:
    inputs = _moe_inputs(dtype=torch.float32)
    expected = torch.full_like(inputs[0], 3.0)
    native_calls: list[str] = []

    def call_native(*args: object, operator: str, **kwargs: object) -> Tensor:
        native_calls.append(operator)
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
    assert native_calls == ["deepseek_moe_forward_fused"]


def test_auto_falls_back_to_staged_without_considering_persistent(monkeypatch) -> None:
    inputs = _moe_inputs(dtype=torch.float32)
    expected = torch.full_like(inputs[0], 4.0)
    eligibility_queries: list[str] = []
    native_calls: list[str] = []

    def eligibility(*args: object, operator: str, **kwargs: object) -> str | None:
        eligibility_queries.append(operator)
        if operator == "deepseek_moe_forward_fused":
            return "fused operator is unavailable"
        return None

    def call_native(*args: object, operator: str, **kwargs: object) -> Tensor:
        native_calls.append(operator)
        return expected

    monkeypatch.setattr(facade_ops, "_cuda_moe_ineligibility_reason", eligibility)
    monkeypatch.setattr(facade_ops, "_call_cuda_moe", call_native)
    monkeypatch.setattr(
        facade_ops,
        "deepseek_moe_packed_reference",
        lambda *a, **k: pytest.fail("reference fallback"),
    )

    actual = deepseek_moe_forward(*inputs, topk=TOPK, backend="auto")

    assert actual is expected
    assert eligibility_queries == [
        "deepseek_moe_forward_fused",
        "deepseek_moe_forward",
    ]
    assert "deepseek_moe_forward_persistent" not in eligibility_queries
    assert native_calls == ["deepseek_moe_forward"]


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


CUDA_MOE_SHAPES = (
    (0, 5, 7, 4, 2, 2, 1),
    (1, 1, 1, 1, 1, 1, 1),
    (7, 15, 17, 4, 2, 2, 1),
    (17, 33, 65, 8, 3, 4, 2),
    (31, 65, 33, 9, 4, 3, 2),
)
NATIVE_MOE_SKIP_REASON = "native CUDA DeepSeek MoE operator is unavailable"


def _numerical_moe_inputs(
    tokens: int,
    model_dim: int,
    hidden: int,
    experts: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    token_shape: tuple[int, ...] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1907 + tokens + 3 * model_dim + 5 * hidden + 7 * experts)

    def scaled_randn(*shape: int) -> Tensor:
        return (
            torch.randn(*shape, generator=generator, dtype=torch.float32)
            .mul_(0.125)
            .to(device=device, dtype=dtype)
            .detach()
            .contiguous()
        )

    x_shape = (tokens,) if token_shape is None else token_shape
    return (
        scaled_randn(*x_shape, model_dim),
        scaled_randn(experts, model_dim),
        scaled_randn(experts, hidden, model_dim),
        scaled_randn(experts, model_dim, hidden),
        scaled_randn(experts, hidden, model_dim),
    )


def _numerical_score_bias(
    experts: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    return (
        torch.linspace(-0.2, 0.2, experts, dtype=torch.float32)
        .to(device=device, dtype=dtype)
        .detach()
        .contiguous()
    )


def _assert_native_output(
    actual: Tensor,
    expected: Tensor,
    x: Tensor,
) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.device == x.device
    assert actual.stride() == x.stride()
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_moe_available(), reason=NATIVE_MOE_SKIP_REASON)
@pytest.mark.parametrize(
    ("tokens", "model_dim", "hidden", "experts", "topk", "n_groups", "topk_groups"),
    CUDA_MOE_SHAPES,
)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "bias"])
@pytest.mark.parametrize("backend", [*EXPLICIT_NATIVE_BACKENDS, "auto"])
def test_native_cuda_matches_independent_reference_across_shape_matrix(
    tokens: int,
    model_dim: int,
    hidden: int,
    experts: int,
    topk: int,
    n_groups: int,
    topk_groups: int,
    with_bias: bool,
    backend: str,
) -> None:
    inputs = _numerical_moe_inputs(
        tokens,
        model_dim,
        hidden,
        experts,
        device="cuda",
    )
    score_bias = _numerical_score_bias(experts, device="cuda") if with_bias else None
    kwargs = {
        "topk": topk,
        "n_groups": n_groups,
        "topk_groups": topk_groups,
        "score_bias": score_bias,
        "route_scale": 0.75,
    }
    assert (
        facade_ops._cuda_moe_ineligibility_reason(
            *inputs,
            "sigmoid",
            score_bias,
        )
        is None
    )

    actual = deepseek_moe_forward(*inputs, backend=backend, **kwargs)  # type: ignore[arg-type]
    expected = deepseek_moe_reference(*inputs, **kwargs)

    _assert_native_output(actual, expected, inputs[0])


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_moe_available(), reason=NATIVE_MOE_SKIP_REASON)
@pytest.mark.parametrize("backend", [*EXPLICIT_NATIVE_BACKENDS, "auto"])
def test_native_cuda_preserves_exact_ties(
    backend: str,
) -> None:
    inputs = list(_numerical_moe_inputs(7, 15, 17, 4, device="cuda"))
    inputs[1] = torch.zeros_like(inputs[1]).detach().contiguous()
    typed_inputs = tuple(inputs)
    kwargs = {"topk": 2, "n_groups": 2, "topk_groups": 1}

    actual = deepseek_moe_forward(*typed_inputs, backend=backend, **kwargs)  # type: ignore[arg-type]
    expected, routing = deepseek_moe_reference(
        *typed_inputs,
        return_routing=True,
        **kwargs,
    )

    assert routing.indices.tolist() == [[0, 1]] * 7
    _assert_native_output(actual, expected, typed_inputs[0])


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_moe_available(), reason=NATIVE_MOE_SKIP_REASON)
@pytest.mark.parametrize("backend", [*EXPLICIT_NATIVE_BACKENDS, "auto"])
def test_native_cuda_handles_hot_and_inactive_experts(
    backend: str,
) -> None:
    inputs = list(_numerical_moe_inputs(7, 15, 17, 4, device="cuda"))
    inputs[0] = torch.ones_like(inputs[0]).mul_(0.25)
    inputs[1] = -torch.ones_like(inputs[1])
    inputs[1][0].fill_(1.0)
    for weights in inputs[2:]:
        weights[1:].fill_(8.0)
    typed_inputs = tuple(tensor.detach().contiguous() for tensor in inputs)
    kwargs = {"topk": 1, "n_groups": 1, "topk_groups": 1}

    actual = deepseek_moe_forward(*typed_inputs, backend=backend, **kwargs)  # type: ignore[arg-type]
    expected, routing = deepseek_moe_reference(
        *typed_inputs,
        return_routing=True,
        **kwargs,
    )

    assert routing.indices.tolist() == [[0]] * 7
    _assert_native_output(actual, expected, typed_inputs[0])


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_moe_available(), reason=NATIVE_MOE_SKIP_REASON)
def test_persistent_cuda_repeats_skewed_routes_without_queue_state_leakage() -> None:
    inputs = list(_numerical_moe_inputs(17, 15, 17, 4, device="cuda"))
    inputs[0] = torch.ones_like(inputs[0]).mul_(0.25)
    inputs[1] = -torch.ones_like(inputs[1])
    inputs[1][0].fill_(1.0)
    typed_inputs = tuple(tensor.detach().contiguous() for tensor in inputs)
    kwargs = {"topk": 1, "n_groups": 1, "topk_groups": 1}
    expected = deepseek_moe_reference(*typed_inputs, **kwargs)

    for _ in range(25):
        actual = deepseek_moe_forward(
            *typed_inputs,
            backend="cuda_persistent",
            **kwargs,
        )
        _assert_native_output(actual, expected, typed_inputs[0])


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_moe_available(), reason=NATIVE_MOE_SKIP_REASON)
@pytest.mark.parametrize("backend", [*EXPLICIT_NATIVE_BACKENDS, "auto"])
def test_native_cuda_restores_rank_three_facade_output(
    backend: str,
) -> None:
    inputs = _numerical_moe_inputs(
        6,
        15,
        17,
        4,
        device="cuda",
        token_shape=(2, 3),
    )
    score_bias = _numerical_score_bias(4, device="cuda")
    kwargs = {
        "topk": 2,
        "n_groups": 2,
        "topk_groups": 1,
        "score_bias": score_bias,
        "route_scale": 1.25,
    }

    actual = deepseek_moe_forward(*inputs, backend=backend, **kwargs)  # type: ignore[arg-type]
    expected = deepseek_moe_reference(*inputs, **kwargs)

    _assert_native_output(actual, expected, inputs[0])


CUDA_INELIGIBILITY_CASES = (
    ("float16", "float32 only"),
    ("bfloat16", "float32 only"),
    ("softmax", "sigmoid scores only"),
    ("noncontiguous", "requires contiguous tensors"),
    ("requires_grad", "forward-only for requires_grad tensors"),
    ("deterministic", "deterministic algorithms are enabled"),
    ("missing_native", "does not register a CUDA DeepSeek MoE forward"),
)


@pytest.mark.parametrize(("case", "message"), CUDA_INELIGIBILITY_CASES)
def test_explicit_cuda_rejects_every_ineligible_policy_case_without_native(
    case: str,
    message: str,
    monkeypatch,
) -> None:
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(case == "deterministic")
    try:
        with FakeTensorMode():
            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }.get(case, torch.float32)
            inputs = list(_raw_moe_inputs(dtype=dtype, device="cuda"))
            score_func = "softmax" if case == "softmax" else "sigmoid"
            if case == "noncontiguous":
                inputs[0] = _noncontiguous_view(inputs[0])
            elif case == "requires_grad":
                inputs[0].requires_grad_(True)
            elif case == "missing_native":
                monkeypatch.setattr(facade_ops, "_operator_has_cuda_kernel", lambda _: False)

            with pytest.raises(RuntimeError, match=message):
                deepseek_moe_forward(
                    *inputs,
                    topk=TOPK,
                    n_groups=GROUPS,
                    topk_groups=TOPK_GROUPS,
                    score_func=score_func,  # type: ignore[arg-type]
                    backend="cuda",
                )
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _noncontiguous_clone(tensor: Tensor) -> Tensor:
    return torch.stack((tensor, tensor), dim=-1)[..., 0]


@pytest.mark.parametrize("case", [item[0] for item in CUDA_INELIGIBILITY_CASES])
def test_ineligible_auto_completes_reference_fallback(
    case: str,
    monkeypatch,
) -> None:
    real_reference = facade_ops.deepseek_moe_packed_reference
    reference_calls = 0

    def call_real_reference(*args: object, **kwargs: object) -> Tensor:
        nonlocal reference_calls
        reference_calls += 1
        return real_reference(*args, **kwargs)

    monkeypatch.setattr(
        facade_ops,
        "_call_cuda_moe",
        lambda *args, **kwargs: pytest.fail("ineligible auto must not call native"),
    )
    monkeypatch.setattr(
        facade_ops,
        "deepseek_moe_packed_reference",
        call_real_reference,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(case, torch.float32)
    inputs = list(_numerical_moe_inputs(7, 5, 9, 4, device=device, dtype=dtype))
    score_func = "softmax" if case == "softmax" else "sigmoid"
    if case == "noncontiguous":
        inputs[0] = _noncontiguous_clone(inputs[0])
        assert not inputs[0].is_contiguous()
    elif case == "requires_grad":
        inputs[0].requires_grad_(True)
    elif case == "missing_native":
        monkeypatch.setattr(facade_ops, "_operator_has_cuda_kernel", lambda _: False)

    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(case == "deterministic")
    try:
        kwargs = {
            "topk": TOPK,
            "n_groups": GROUPS,
            "topk_groups": TOPK_GROUPS,
            "score_func": score_func,
            "route_scale": 0.75,
        }
        if inputs[0].device.type == "cuda":
            reason = facade_ops._cuda_moe_ineligibility_reason(
                *inputs,
                score_func,
                None,
            )
            assert reason is not None
            assert dict(CUDA_INELIGIBILITY_CASES)[case] in reason
        actual = deepseek_moe_forward(*inputs, backend="auto", **kwargs)  # type: ignore[arg-type]
        expected = deepseek_moe_reference(*inputs, **kwargs)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    assert actual.shape == inputs[0].shape
    assert actual.dtype == inputs[0].dtype
    assert actual.device == inputs[0].device
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    assert reference_calls == 1


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_moe_available(), reason=NATIVE_MOE_SKIP_REASON)
@pytest.mark.parametrize("backend", [*EXPLICIT_NATIVE_BACKENDS, "auto"])
def test_native_cuda_uses_current_nondefault_stream(backend: str) -> None:
    inputs = list(_numerical_moe_inputs(7, 15, 17, 4, device="cuda"))
    score_bias = _numerical_score_bias(4, device="cuda")
    producer_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(producer_stream)
    kwargs = {
        "topk": 2,
        "n_groups": 2,
        "topk_groups": 1,
        "score_bias": score_bias,
        "route_scale": 0.75,
    }

    with torch.cuda.stream(stream):
        inputs[0].add_(0.03125)
        actual = deepseek_moe_forward(*inputs, backend=backend, **kwargs)  # type: ignore[arg-type]
        actual.record_stream(stream)
    stream.synchronize()
    expected = deepseek_moe_reference(*inputs, **kwargs)

    _assert_native_output(actual, expected, inputs[0])


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_moe_available(), reason=NATIVE_MOE_SKIP_REASON)
@pytest.mark.parametrize("operator", RAW_MOE_OPERATORS)
def test_raw_native_cuda_operator_passes_opcheck(operator: str) -> None:
    inputs = tuple(
        tensor.detach().contiguous()
        for tensor in _numerical_moe_inputs(7, 15, 17, 4, device="cuda")
    )
    score_bias = _numerical_score_bias(4, device="cuda").detach().contiguous()
    result = torch.library.opcheck(
        getattr(torch.ops.ds_flash_mla_moe, operator).default,
        (*inputs, 2, 2, 1, score_bias, 0.75),
    )

    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_moe_available(), reason=NATIVE_MOE_SKIP_REASON)
@pytest.mark.parametrize("operator", RAW_MOE_OPERATORS)
def test_raw_native_cuda_operator_supports_fullgraph_compile(operator: str) -> None:
    inputs = tuple(
        tensor.detach().contiguous()
        for tensor in _numerical_moe_inputs(7, 15, 17, 4, device="cuda")
    )
    score_bias = _numerical_score_bias(4, device="cuda").detach().contiguous()

    @torch.compile(fullgraph=True, backend="eager")
    def compiled_raw_moe(
        x: Tensor,
        gate_weight: Tensor,
        expert_w1: Tensor,
        expert_w2: Tensor,
        expert_w3: Tensor,
        bias: Tensor,
    ) -> Tensor:
        return getattr(torch.ops.ds_flash_mla_moe, operator).default(
            x,
            gate_weight,
            expert_w1,
            expert_w2,
            expert_w3,
            2,
            2,
            1,
            bias,
            0.75,
        )

    actual = compiled_raw_moe(*inputs, score_bias)
    expected = deepseek_moe_reference(
        *inputs,
        topk=2,
        n_groups=2,
        topk_groups=1,
        score_bias=score_bias,
        route_scale=0.75,
    )

    _assert_native_output(actual, expected, inputs[0])


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_moe_available(), reason=NATIVE_MOE_SKIP_REASON)
@pytest.mark.parametrize("inside_no_grad", [False, True], ids=["normal", "no_grad"])
@pytest.mark.parametrize("operator", RAW_MOE_OPERATORS)
def test_raw_native_cuda_operator_rejects_requires_grad_even_in_no_grad(
    inside_no_grad: bool,
    operator: str,
) -> None:
    inputs = [
        tensor.detach().contiguous()
        for tensor in _numerical_moe_inputs(7, 15, 17, 4, device="cuda")
    ]
    inputs[0] = inputs[0].requires_grad_(True)

    def call_raw() -> Tensor:
        return _call_raw_moe(
            tuple(inputs),
            topk=2,
            n_groups=2,
            topk_groups=1,
            operator=operator,
        )

    with pytest.raises(RuntimeError, match="forward-only|requires_grad"):
        if inside_no_grad:
            with torch.no_grad():
                call_raw()
        else:
            call_raw()
