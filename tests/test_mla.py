from dataclasses import fields

import pytest
import torch

from ds_flash_mla_moe import (
    MLAConfig,
    MLAPagedCache,
    MLAStaticCache,
    MLAWeights,
    allocate_mla_paged_cache,
    allocate_mla_static_cache,
    append_mla_cache,
    build_mla_cache,
    cuda_mla_available,
    cuda_paged_mla_available,
    materialize_mla_paged_cache,
    mla_absorbed_attention,
    mla_absorbed_attention_reference,
    mla_naive_attention_reference,
    mla_paged_attention,
    write_mla_paged_cache,
    write_mla_static_cache,
)
from ds_flash_mla_moe.ops import (
    mla_absorbed_attention as dispatch_mla_absorbed_attention,
)
from ds_flash_mla_moe.ops import (
    mla_cache_projection,
    mla_output_projection,
    mla_query_lora_projection,
    mla_query_projection,
)


def make_fixture(*, direct_query: bool = False):
    torch.manual_seed(31)
    dtype = torch.float64
    batch, sequence, model_dim = 2, 6, 8
    config = MLAConfig(
        n_heads=3,
        q_lora_rank=0 if direct_query else 5,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=3,
    )
    common = {
        "wkv_a": torch.randn(config.kv_lora_rank + config.qk_rope_head_dim, model_dim, dtype=dtype),
        "kv_norm_weight": torch.randn(config.kv_lora_rank, dtype=dtype),
        "wkv_b": torch.randn(
            config.n_heads * (config.qk_nope_head_dim + config.v_head_dim),
            config.kv_lora_rank,
            dtype=dtype,
        ),
        "wo": torch.randn(model_dim, config.n_heads * config.v_head_dim, dtype=dtype),
    }
    if direct_query:
        weights = MLAWeights(
            **common,
            wq=torch.randn(config.n_heads * config.qk_head_dim, model_dim, dtype=dtype),
        )
    else:
        weights = MLAWeights(
            **common,
            wq_a=torch.randn(config.q_lora_rank, model_dim, dtype=dtype),
            q_norm_weight=torch.randn(config.q_lora_rank, dtype=dtype),
            wq_b=torch.randn(
                config.n_heads * config.qk_head_dim,
                config.q_lora_rank,
                dtype=dtype,
            ),
        )
    x = torch.randn(batch, sequence, model_dim, dtype=dtype)
    return x, config, weights


def physical_slots(pages: list[int], length: int, page_size: int) -> torch.Tensor:
    return torch.tensor(
        [pages[index // page_size] * page_size + index % page_size for index in range(length)],
        dtype=torch.long,
    )


def cuda_weights(
    weights: MLAWeights,
    *,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> MLAWeights:
    return MLAWeights(
        **{
            field.name: (
                getattr(weights, field.name)
                .to(device="cuda", dtype=dtype)
                .detach()
                .requires_grad_(requires_grad)
                if getattr(weights, field.name) is not None
                else None
            )
            for field in fields(MLAWeights)
        }
    )


def _mla_cuda_tolerances(
    dtype: torch.dtype,
    *,
    backward: bool = False,
) -> tuple[float, float]:
    if dtype == torch.float32:
        return (1e-3, 1e-3) if backward else (5e-5, 5e-5)
    if dtype == torch.float16:
        return (8e-2, 8e-2) if backward else (3e-2, 3e-2)
    return (3e-1, 3e-1) if backward else (1e-1, 1e-1)


def _mla_cuda_stage_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 5e-5, 5e-5
    if dtype == torch.float16:
        return 5e-3, 5e-3
    return 2e-2, 2e-2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mla_reference_projection_stages_preserve_storage_dtype(dtype: torch.dtype) -> None:
    x, config, weights = make_fixture()
    x = x.to(dtype)
    weights = MLAWeights(
        **{
            field.name: (
                getattr(weights, field.name).to(dtype)
                if getattr(weights, field.name) is not None
                else None
            )
            for field in fields(MLAWeights)
        }
    )
    positions = torch.arange(x.shape[1])

    cache = build_mla_cache(x, config, weights, positions=positions, backend="reference")
    q_nope, q_pe = torch.ops.ds_flash_mla_moe.mla_query_lora_projection.default(
        x,
        weights.wq_a,
        weights.q_norm_weight,
        weights.wq_b,
        positions,
        config.n_heads,
        config.qk_nope_head_dim,
        config.qk_rope_head_dim,
        config.rope_theta,
        config.rms_norm_eps,
    )

    assert cache.kv.dtype == cache.pe.dtype == dtype
    assert q_nope.dtype == q_pe.dtype == dtype


@pytest.mark.parametrize("direct_query", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_naive_and_absorbed_paths_are_equivalent(direct_query: bool, causal: bool) -> None:
    x, config, weights = make_fixture(direct_query=direct_query)
    positions = torch.arange(x.shape[1])
    cache = build_mla_cache(x, config, weights, positions=positions)

    naive = mla_naive_attention_reference(
        x,
        cache,
        config,
        weights,
        query_positions=positions,
        causal=causal,
    )
    absorbed = mla_absorbed_attention_reference(
        x,
        cache,
        config,
        weights,
        query_positions=positions,
        causal=causal,
    )

    torch.testing.assert_close(absorbed, naive, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("direct_query", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_dispatchable_absorbed_reference_matches_specification(
    direct_query: bool, causal: bool
) -> None:
    x, config, weights = make_fixture(direct_query=direct_query)
    positions = torch.arange(x.shape[1])
    cache = build_mla_cache(x, config, weights, positions=positions)

    actual = mla_absorbed_attention(
        x,
        cache,
        config,
        weights,
        query_positions=positions,
        causal=causal,
        backend="reference",
    )
    expected = mla_absorbed_attention_reference(
        x,
        cache,
        config,
        weights,
        query_positions=positions,
        causal=causal,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_dispatchable_absorbed_reference_preserves_gradients() -> None:
    x, config, weights = make_fixture()
    actual_x = x.detach().clone().requires_grad_(True)
    expected_x = x.detach().clone().requires_grad_(True)
    positions = torch.arange(x.shape[1])
    actual_cache = build_mla_cache(actual_x, config, weights, positions=positions)
    expected_cache = build_mla_cache(expected_x, config, weights, positions=positions)
    upstream = torch.randn_like(x)

    actual = mla_absorbed_attention(
        actual_x,
        actual_cache,
        config,
        weights,
        query_positions=positions,
        backend="reference",
    )
    expected = mla_absorbed_attention_reference(
        expected_x,
        expected_cache,
        config,
        weights,
        query_positions=positions,
    )
    with torch.autograd.set_multithreading_enabled(False):
        actual.backward(upstream)
    expected.backward(upstream)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(actual_x.grad, expected_x.grad, rtol=1e-9, atol=1e-9)


def test_raw_absorbed_operator_passes_opcheck() -> None:
    torch.manual_seed(173)
    inputs = (
        torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(2, 3, 4, 2, dtype=torch.float64, requires_grad=True),
        torch.randn(2, 6, 7, dtype=torch.float64, requires_grad=True),
        torch.randn(2, 6, 2, dtype=torch.float64, requires_grad=True),
        torch.randn(4, 5, 7, dtype=torch.float64, requires_grad=True),
        torch.randn(4, 3, 7, dtype=torch.float64, requires_grad=True),
        torch.arange(3),
        torch.arange(6),
        True,
        0.25,
    )
    torch.library.opcheck(
        torch.ops.ds_flash_mla_moe.mla_absorbed_attention.default,
        inputs,
    )
    output = torch.ops.ds_flash_mla_moe.mla_absorbed_attention.default(*inputs)
    assert output.is_contiguous()
    assert output.stride() == torch.empty_like(output).stride()


def test_raw_mla_projection_operators_pass_opcheck() -> None:
    torch.manual_seed(191)
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    positions = torch.arange(3)
    direct_weight = torch.randn(18, 8, dtype=torch.float64, requires_grad=True)
    q_a = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
    q_norm = torch.randn(5, dtype=torch.float64, requires_grad=True)
    q_b = torch.randn(18, 5, dtype=torch.float64, requires_grad=True)
    kv_a = torch.randn(6, 8, dtype=torch.float64, requires_grad=True)
    kv_norm = torch.randn(4, dtype=torch.float64, requires_grad=True)
    heads = torch.randn(2, 3, 3, 3, dtype=torch.float64, requires_grad=True)
    output_weight = torch.randn(8, 9, dtype=torch.float64, requires_grad=True)
    cases = (
        (
            torch.ops.ds_flash_mla_moe.mla_query_projection.default,
            (x, direct_weight, positions, 3, 4, 2, 10_000.0),
        ),
        (
            torch.ops.ds_flash_mla_moe.mla_query_lora_projection.default,
            (x, q_a, q_norm, q_b, positions, 3, 4, 2, 10_000.0, 1e-6),
        ),
        (
            torch.ops.ds_flash_mla_moe.mla_cache_projection.default,
            (x, kv_a, kv_norm, positions, 4, 10_000.0, 1e-6),
        ),
        (
            torch.ops.ds_flash_mla_moe.mla_output_projection.default,
            (heads, output_weight),
        ),
    )

    for operator, inputs in cases:
        torch.library.opcheck(operator, inputs)
        outputs = operator(*inputs)
        if isinstance(outputs, tuple):
            assert all(output.is_contiguous() for output in outputs)
        else:
            assert outputs.is_contiguous()


def test_raw_mla_static_cache_write_passes_opcheck_and_mutates_only_target_slice() -> None:
    x, config, weights = make_fixture()
    positions = torch.arange(7, 7 + x.shape[1])
    kv_storage = torch.full(
        (x.shape[0], x.shape[1] + 2, config.kv_lora_rank),
        torch.nan,
        dtype=x.dtype,
    )
    pe_storage = torch.full(
        (x.shape[0], x.shape[1] + 2, config.qk_rope_head_dim),
        torch.nan,
        dtype=x.dtype,
    )
    position_storage = torch.full((x.shape[1] + 2,), -1, dtype=torch.long)
    inputs = (
        x,
        weights.wkv_a,
        weights.kv_norm_weight,
        positions,
        kv_storage,
        pe_storage,
        position_storage,
        1,
        config.rope_theta,
        config.rms_norm_eps,
    )
    torch.library.opcheck(
        torch.ops.ds_flash_mla_moe.mla_cache_projection_write.default,
        inputs,
    )

    torch.ops.ds_flash_mla_moe.mla_cache_projection_write.default(*inputs)
    expected = build_mla_cache(
        x,
        config,
        weights,
        positions=positions,
        backend="reference",
    )
    torch.testing.assert_close(kv_storage[:, 1:-1], expected.kv)
    torch.testing.assert_close(pe_storage[:, 1:-1], expected.pe)
    torch.testing.assert_close(position_storage[1:-1], positions)
    assert torch.isnan(kv_storage[:, :1]).all() and torch.isnan(kv_storage[:, -1:]).all()
    assert torch.isnan(pe_storage[:, :1]).all() and torch.isnan(pe_storage[:, -1:]).all()
    assert position_storage[[0, -1]].tolist() == [-1, -1]


def test_raw_absorbed_operator_runs_through_torch_compile() -> None:
    @torch.compile(fullgraph=True, backend="eager")
    def compiled(
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv: torch.Tensor,
        pe: torch.Tensor,
        key_up: torch.Tensor,
        value_up: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.ds_flash_mla_moe.mla_absorbed_attention.default(
            q_nope,
            q_pe,
            kv,
            pe,
            key_up,
            value_up,
            query_positions,
            key_positions,
            True,
            0.5,
        )

    tensors = (
        torch.randn(1, 2, 3, 4),
        torch.randn(1, 2, 3, 2),
        torch.randn(1, 5, 6),
        torch.randn(1, 5, 2),
        torch.randn(3, 4, 6),
        torch.randn(3, 7, 6),
        torch.arange(2),
        torch.arange(5),
    )
    actual = compiled(*tensors)
    expected = torch.ops.ds_flash_mla_moe.mla_absorbed_attention.default(*tensors, True, 0.5)
    torch.testing.assert_close(actual, expected)


def test_explicit_cuda_mla_fails_loudly_without_native_inputs() -> None:
    x, config, weights = make_fixture()
    positions = torch.arange(x.shape[1])
    cache = build_mla_cache(x, config, weights, positions=positions)
    with pytest.raises(RuntimeError, match="CUDA MLA is unavailable"):
        mla_absorbed_attention(
            x,
            cache,
            config,
            weights,
            query_positions=positions,
            backend="cuda",
        )


def test_explicit_cuda_mla_rejects_attention_mask() -> None:
    x, config, weights = make_fixture()
    positions = torch.arange(x.shape[1])
    cache = build_mla_cache(x, config, weights, positions=positions)
    with pytest.raises(RuntimeError, match="explicit attention masks"):
        mla_absorbed_attention(
            x,
            cache,
            config,
            weights,
            query_positions=positions,
            attn_mask=torch.ones(x.shape[1], x.shape[1], dtype=torch.bool),
            backend="cuda",
        )


@pytest.mark.skipif(
    not cuda_mla_available(),
    reason="requires a built native extension and a CUDA device",
)
@pytest.mark.cuda
@pytest.mark.parametrize("query_length", [1, 3, 6])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("direct_query", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_absorbed_mla_matches_reference(
    query_length: int,
    causal: bool,
    direct_query: bool,
    dtype: torch.dtype,
) -> None:
    x, config, weights = make_fixture(direct_query=direct_query)
    x = x.to(device="cuda", dtype=dtype)
    weights = cuda_weights(weights, dtype=dtype)
    positions = torch.arange(x.shape[1], device="cuda")
    cache = build_mla_cache(x, config, weights, positions=positions, backend="cuda")
    reference_cache = build_mla_cache(
        x,
        config,
        weights,
        positions=positions,
        backend="reference",
    )
    query = x[:, -query_length:]
    query_positions = positions[-query_length:]

    with torch.no_grad():
        actual = mla_absorbed_attention(
            query,
            reference_cache,
            config,
            weights,
            query_positions=query_positions,
            causal=causal,
            backend="cuda",
        )
        expected = mla_absorbed_attention_reference(
            query,
            cache,
            config,
            weights,
            query_positions=query_positions,
            causal=causal,
        )
    rtol, atol = _mla_cuda_tolerances(dtype)
    assert actual.dtype == cache.kv.dtype == reference_cache.kv.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not cuda_mla_available(),
    reason="requires a built native extension and a CUDA device",
)
@pytest.mark.cuda
@pytest.mark.parametrize("direct_query", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_mla_projection_stages_match_reference(
    direct_query: bool,
    dtype: torch.dtype,
) -> None:
    x, config, weights = make_fixture(direct_query=direct_query)
    x = x.to(device="cuda", dtype=dtype)
    weights = cuda_weights(weights, dtype=dtype)
    positions = torch.arange(x.shape[1], device="cuda")
    rtol, atol = _mla_cuda_stage_tolerances(dtype)

    if direct_query:
        assert weights.wq is not None
        actual_query = mla_query_projection(
            x,
            weights.wq,
            positions,
            n_heads=config.n_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            rope_theta=config.rope_theta,
            backend="cuda",
        )
        expected_query = mla_query_projection(
            x,
            weights.wq,
            positions,
            n_heads=config.n_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            rope_theta=config.rope_theta,
            backend="reference",
        )
    else:
        assert weights.wq_a is not None
        assert weights.q_norm_weight is not None
        assert weights.wq_b is not None
        query_arguments = (
            x,
            weights.wq_a,
            weights.q_norm_weight,
            weights.wq_b,
            positions,
        )
        query_keywords = {
            "n_heads": config.n_heads,
            "qk_nope_head_dim": config.qk_nope_head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "rope_theta": config.rope_theta,
            "rms_norm_eps": config.rms_norm_eps,
        }
        actual_query = mla_query_lora_projection(
            *query_arguments,
            **query_keywords,
            backend="cuda",
        )
        expected_query = mla_query_lora_projection(
            *query_arguments,
            **query_keywords,
            backend="reference",
        )

    actual_cache = mla_cache_projection(
        x,
        weights.wkv_a,
        weights.kv_norm_weight,
        positions,
        kv_lora_rank=config.kv_lora_rank,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        backend="cuda",
    )
    expected_cache = mla_cache_projection(
        x,
        weights.wkv_a,
        weights.kv_norm_weight,
        positions,
        kv_lora_rank=config.kv_lora_rank,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        backend="reference",
    )
    heads = torch.randn(
        x.shape[0],
        x.shape[1],
        config.n_heads,
        config.v_head_dim,
        device="cuda",
        dtype=dtype,
    )
    actual_output = mla_output_projection(heads, weights.wo, backend="cuda")
    expected_output = mla_output_projection(heads, weights.wo, backend="reference")

    for actual, expected in (
        *zip(actual_query, expected_query),
        *zip(actual_cache, expected_cache),
    ):
        assert actual.dtype == expected.dtype == dtype
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_output, expected_output, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not cuda_mla_available(),
    reason="requires a built native extension and a CUDA device",
)
@pytest.mark.cuda
def test_cuda_mla_rejects_mixed_storage_dtypes() -> None:
    x, config, weights = make_fixture()
    x = x.to(device="cuda", dtype=torch.float16)
    weights = cuda_weights(weights, dtype=torch.float16)
    weights = MLAWeights(
        **{
            field.name: (
                getattr(weights, field.name).to(torch.bfloat16)
                if field.name == "wkv_a"
                else getattr(weights, field.name)
            )
            for field in fields(MLAWeights)
        }
    )

    with pytest.raises(RuntimeError, match="same dtype"):
        build_mla_cache(x, config, weights, backend="cuda")


@pytest.mark.skipif(
    not cuda_mla_available(),
    reason="requires a built native extension and a CUDA device",
)
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("latent_dim", "rope_dim", "value_dim", "strided"),
    [
        pytest.param(20, 10, 24, True, id="specialized-tail-strided"),
        pytest.param(32, 32, 32, False, id="specialized-boundary"),
        pytest.param(33, 10, 24, True, id="generic-latent"),
        pytest.param(20, 34, 24, True, id="generic-rope"),
        pytest.param(20, 10, 35, True, id="generic-value"),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_absorbed_attention_dimension_dispatch_matches_reference(
    latent_dim: int,
    rope_dim: int,
    value_dim: int,
    strided: bool,
    causal: bool,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260814)
    batch, query_length, heads, key_length, nope_dim = 2, 5, 3, 17, 19

    def make_tensor(*shape: int) -> torch.Tensor:
        if not strided:
            return torch.randn(*shape, device="cuda", dtype=dtype)
        storage = torch.randn(*shape[:-1], shape[-1] * 2, device="cuda", dtype=dtype)
        return storage[..., ::2]

    q_nope = make_tensor(batch, query_length, heads, nope_dim)
    q_pe = make_tensor(batch, query_length, heads, rope_dim)
    kv = make_tensor(batch, key_length, latent_dim)
    pe = make_tensor(batch, key_length, rope_dim)
    key_up = make_tensor(heads, nope_dim, latent_dim)
    value_up = make_tensor(heads, value_dim, latent_dim)
    key_positions = torch.arange(2 * key_length, device="cuda", dtype=torch.long)[::2]
    query_positions = key_positions[-query_length:]
    scale = float((nope_dim + rope_dim) ** -0.5)
    arguments = (q_nope, q_pe, kv, pe, key_up, value_up)

    with torch.no_grad():
        expected = dispatch_mla_absorbed_attention(
            *arguments,
            query_positions=query_positions,
            key_positions=key_positions,
            causal=causal,
            scale=scale,
            backend="reference",
        )
        actual = dispatch_mla_absorbed_attention(
            *arguments,
            query_positions=query_positions,
            key_positions=key_positions,
            causal=causal,
            scale=scale,
            backend="cuda",
        )

    assert actual.is_contiguous()
    rtol, atol = _mla_cuda_stage_tolerances(dtype)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not cuda_mla_available(),
    reason="requires a built native extension and a CUDA device",
)
@pytest.mark.filterwarnings(
    "ignore:Attempting to run cuBLAS, but there was no current CUDA context!:UserWarning"
)
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_absorbed_mla_backward_matches_reference(dtype: torch.dtype) -> None:
    x, config, weights = make_fixture()
    actual_x = x.to(device="cuda", dtype=dtype).requires_grad_(True)
    expected_x = actual_x.detach().clone().requires_grad_(True)
    actual_weights = cuda_weights(weights, dtype=dtype, requires_grad=True)
    expected_weights = MLAWeights(
        **{
            field.name: (
                getattr(actual_weights, field.name).detach().clone().requires_grad_(True)
                if getattr(actual_weights, field.name) is not None
                else None
            )
            for field in fields(MLAWeights)
        }
    )
    positions = torch.arange(x.shape[1], device="cuda")
    actual_cache = build_mla_cache(
        actual_x,
        config,
        actual_weights,
        positions=positions,
        backend="cuda",
    )
    expected_cache = build_mla_cache(
        expected_x,
        config,
        expected_weights,
        positions=positions,
        backend="reference",
    )
    upstream = torch.randn_like(actual_x)

    actual = mla_absorbed_attention(
        actual_x,
        actual_cache,
        config,
        actual_weights,
        query_positions=positions,
        backend="cuda",
    )
    expected = mla_absorbed_attention_reference(
        expected_x,
        expected_cache,
        config,
        expected_weights,
        query_positions=positions,
    )
    actual.backward(upstream)
    expected.backward(upstream)

    forward_rtol, forward_atol = _mla_cuda_tolerances(dtype)
    backward_rtol, backward_atol = _mla_cuda_tolerances(dtype, backward=True)
    torch.testing.assert_close(actual, expected, rtol=forward_rtol, atol=forward_atol)
    torch.testing.assert_close(
        actual_x.grad,
        expected_x.grad,
        rtol=backward_rtol,
        atol=backward_atol,
    )
    for field in fields(MLAWeights):
        actual_weight = getattr(actual_weights, field.name)
        expected_weight = getattr(expected_weights, field.name)
        if actual_weight is not None and expected_weight is not None:
            torch.testing.assert_close(
                actual_weight.grad,
                expected_weight.grad,
                rtol=backward_rtol,
                atol=backward_atol,
            )


@pytest.mark.skipif(
    not cuda_mla_available(),
    reason="requires a built native extension and a CUDA device",
)
@pytest.mark.cuda
def test_cuda_absorbed_mla_uses_current_stream() -> None:
    x, config, weights = make_fixture()
    x = x.float().cuda()
    weights = cuda_weights(weights)
    positions = torch.arange(x.shape[1], device="cuda")
    cache = build_mla_cache(x, config, weights, positions=positions, backend="cuda")
    stream = torch.cuda.Stream()

    with torch.no_grad(), torch.cuda.stream(stream):
        x.fill_(0.125)
        actual = mla_absorbed_attention(
            x,
            cache,
            config,
            weights,
            query_positions=positions,
            backend="cuda",
        )
        actual.record_stream(stream)
    stream.synchronize()
    with torch.no_grad():
        expected = mla_absorbed_attention_reference(
            x,
            cache,
            config,
            weights,
            query_positions=positions,
        )
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)


@pytest.mark.skipif(
    not cuda_mla_available(),
    reason="requires a built native extension and a CUDA device",
)
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_static_cache_projection_write_matches_reference_on_current_stream(
    dtype: torch.dtype,
) -> None:
    x, config, weights = make_fixture()
    x = x.to(device="cuda", dtype=dtype)
    weights = cuda_weights(weights, dtype=dtype)
    positions = torch.arange(11, 11 + x.shape[1], device="cuda")
    static = allocate_mla_static_cache(
        batch_size=x.shape[0],
        capacity=x.shape[1] + 2,
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    pointers = (
        static.kv_storage.data_ptr(),
        static.pe_storage.data_ptr(),
        static.position_storage.data_ptr(),
    )
    stream = torch.cuda.Stream()

    with torch.inference_mode(), torch.cuda.stream(stream):
        x.fill_(0.1875)
        cache = write_mla_static_cache(
            static,
            x,
            config,
            weights,
            positions=positions,
            backend="cuda",
        )
        cache.kv.record_stream(stream)
        cache.pe.record_stream(stream)
        cache.positions.record_stream(stream)
    stream.synchronize()
    expected = build_mla_cache(
        x,
        config,
        weights,
        positions=positions,
        backend="reference",
    )

    rtol, atol = _mla_cuda_stage_tolerances(dtype)
    torch.testing.assert_close(cache.kv, expected.kv, rtol=rtol, atol=atol)
    torch.testing.assert_close(cache.pe, expected.pe, rtol=rtol, atol=atol)
    torch.testing.assert_close(cache.positions, positions)
    assert static.valid_length == x.shape[1]
    assert pointers == (
        static.kv_storage.data_ptr(),
        static.pe_storage.data_ptr(),
        static.position_storage.data_ptr(),
    )


def test_incremental_latent_cache_decode_matches_causal_prefill() -> None:
    x, config, weights = make_fixture()
    positions = torch.arange(x.shape[1])
    full_cache = build_mla_cache(x, config, weights, positions=positions)
    full = mla_absorbed_attention_reference(
        x,
        full_cache,
        config,
        weights,
        query_positions=positions,
        causal=True,
    )

    cache = None
    decoded = []
    for position in range(x.shape[1]):
        query = x[:, position : position + 1]
        position_tensor = torch.tensor([position])
        cache = append_mla_cache(
            cache,
            query,
            config,
            weights,
            positions=position_tensor,
        )
        decoded.append(
            mla_absorbed_attention_reference(
                query,
                cache,
                config,
                weights,
                query_positions=position_tensor,
                causal=True,
            )
        )

    torch.testing.assert_close(torch.cat(decoded, dim=1), full, rtol=1e-10, atol=1e-10)


def test_naive_and_absorbed_gradients_are_equivalent() -> None:
    x, config, weights = make_fixture()
    x.requires_grad_(True)
    differentiable_weights = MLAWeights(
        **{
            field.name: (
                getattr(weights, field.name).requires_grad_(True)
                if getattr(weights, field.name) is not None
                else None
            )
            for field in fields(MLAWeights)
        }
    )
    positions = torch.arange(x.shape[1])
    cache = build_mla_cache(x, config, differentiable_weights, positions=positions)
    upstream = torch.randn_like(x)

    naive = mla_naive_attention_reference(
        x,
        cache,
        config,
        differentiable_weights,
        query_positions=positions,
    )
    absorbed = mla_absorbed_attention_reference(
        x,
        cache,
        config,
        differentiable_weights,
        query_positions=positions,
    )
    variables = [x] + [
        getattr(differentiable_weights, field.name)
        for field in fields(MLAWeights)
        if getattr(differentiable_weights, field.name) is not None
    ]
    naive_gradients = torch.autograd.grad((naive * upstream).sum(), variables, retain_graph=True)
    absorbed_gradients = torch.autograd.grad((absorbed * upstream).sum(), variables)

    for absorbed_gradient, naive_gradient in zip(absorbed_gradients, naive_gradients):
        torch.testing.assert_close(absorbed_gradient, naive_gradient, rtol=1e-9, atol=1e-9)


def test_cache_append_rejects_overlapping_positions() -> None:
    x, config, weights = make_fixture()
    cache = build_mla_cache(x[:, :2], config, weights, positions=torch.tensor([3, 4]))
    with pytest.raises(ValueError, match="follow"):
        append_mla_cache(
            cache,
            x[:, 2:3],
            config,
            weights,
            positions=torch.tensor([4]),
        )


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_chunked_cache_and_decode_match_full_causal_prefill(chunk_size: int) -> None:
    x, config, weights = make_fixture()
    positions = torch.arange(10, 10 + 2 * x.shape[1], 2)
    full_cache = build_mla_cache(x, config, weights, positions=positions)
    expected = mla_absorbed_attention_reference(
        x,
        full_cache,
        config,
        weights,
        query_positions=positions,
        causal=True,
    )

    cache = None
    chunks = []
    for start in range(0, x.shape[1], chunk_size):
        end = min(start + chunk_size, x.shape[1])
        chunk = x[:, start:end]
        chunk_positions = positions[start:end]
        cache = append_mla_cache(
            cache,
            chunk,
            config,
            weights,
            positions=chunk_positions,
        )
        chunks.append(
            mla_absorbed_attention_reference(
                chunk,
                cache,
                config,
                weights,
                query_positions=chunk_positions,
                causal=True,
            )
        )

    assert cache is not None
    assert cache.sequence_length == x.shape[1]
    assert cache.positions.tolist() == positions.tolist()
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, rtol=1e-10, atol=1e-10)


def test_default_cache_append_positions_continue_after_sparse_absolute_position() -> None:
    x, config, weights = make_fixture()
    cache = build_mla_cache(x[:, :2], config, weights, positions=torch.tensor([7, 11]))

    cache = append_mla_cache(cache, x[:, 2:4], config, weights)

    assert cache.positions.tolist() == [7, 11, 12, 13]
    assert cache.sequence_length == 4


def test_cache_append_rejects_batch_size_change() -> None:
    x, config, weights = make_fixture()
    cache = build_mla_cache(x[:, :2], config, weights)
    with pytest.raises(ValueError, match="batch"):
        append_mla_cache(cache, x[:1, 2:3], config, weights)


def test_cache_build_and_attention_support_empty_batch() -> None:
    x, config, weights = make_fixture()
    empty = x[:0]
    positions = torch.arange(x.shape[1])

    cache = build_mla_cache(empty, config, weights, positions=positions)
    output = mla_absorbed_attention_reference(
        empty,
        cache,
        config,
        weights,
        query_positions=positions,
    )

    assert cache.kv.shape == (0, x.shape[1], config.kv_lora_rank)
    assert output.shape == empty.shape


def test_positions_must_be_strictly_increasing_and_nonnegative() -> None:
    x, config, weights = make_fixture()
    with pytest.raises(ValueError, match="strictly"):
        build_mla_cache(x[:, :3], config, weights, positions=torch.tensor([0, 2, 2]))
    with pytest.raises(ValueError, match="non-negative"):
        build_mla_cache(x[:, :2], config, weights, positions=torch.tensor([-1, 0]))


def test_cached_position_validation_is_invalidated_by_tensor_mutation() -> None:
    x, config, weights = make_fixture()
    positions = torch.arange(x.shape[1])
    cache = build_mla_cache(x, config, weights, positions=positions)

    cache.positions[2] = cache.positions[1]

    with pytest.raises(ValueError, match="strictly"):
        mla_absorbed_attention_reference(
            x,
            cache,
            config,
            weights,
            query_positions=positions,
        )


def test_cache_append_revalidates_a_mutated_prefix() -> None:
    x, config, weights = make_fixture()
    cache = build_mla_cache(x[:, :3], config, weights, positions=torch.arange(3))
    cache.positions[1] = cache.positions[0]

    with pytest.raises(ValueError, match="strictly"):
        append_mla_cache(
            cache,
            x[:, 3:4],
            config,
            weights,
            positions=torch.tensor([3]),
        )


def test_recent_query_position_validation_is_invalidated_by_tensor_mutation() -> None:
    x, config, weights = make_fixture()
    cache = build_mla_cache(x, config, weights)
    query_positions = torch.arange(x.shape[1])
    mla_absorbed_attention_reference(
        x,
        cache,
        config,
        weights,
        query_positions=query_positions,
    )

    query_positions[3] = query_positions[2]

    with pytest.raises(ValueError, match="strictly"):
        mla_absorbed_attention_reference(
            x,
            cache,
            config,
            weights,
            query_positions=query_positions,
        )


def test_static_position_validation_is_invalidated_by_storage_mutation() -> None:
    x, config, weights = make_fixture()
    static = allocate_mla_static_cache(
        batch_size=x.shape[0],
        capacity=x.shape[1],
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    with torch.inference_mode():
        write_mla_static_cache(
            static,
            x[:, :3],
            config,
            weights,
            positions=torch.arange(3),
        )
        static.position_storage[1] = static.position_storage[0]

        with pytest.raises(ValueError, match="strictly"):
            write_mla_static_cache(static, x[:, 3:4], config, weights)


def test_static_cache_view_does_not_trust_mutated_position_storage() -> None:
    x, config, weights = make_fixture()
    static = allocate_mla_static_cache(
        batch_size=x.shape[0],
        capacity=x.shape[1],
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    with torch.inference_mode():
        write_mla_static_cache(static, x, config, weights, positions=torch.arange(x.shape[1]))
        static.position_storage[2] = static.position_storage[1]
        cache = static.as_latent_cache()

        with pytest.raises(ValueError, match="strictly"):
            mla_absorbed_attention_reference(
                x,
                cache,
                config,
                weights,
                query_positions=cache.positions,
            )


def test_static_cache_chunk_writes_match_functional_cache_without_reallocation() -> None:
    x, config, weights = make_fixture()
    positions = torch.arange(20, 20 + x.shape[1])
    static = allocate_mla_static_cache(
        batch_size=x.shape[0],
        capacity=x.shape[1] + 2,
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    pointers = (
        static.kv_storage.data_ptr(),
        static.pe_storage.data_ptr(),
        static.position_storage.data_ptr(),
    )
    functional = None

    with torch.inference_mode():
        for start, end in ((0, 2), (2, 5), (5, 6)):
            functional = append_mla_cache(
                functional,
                x[:, start:end],
                config,
                weights,
                positions=positions[start:end],
            )
            view = write_mla_static_cache(
                static,
                x[:, start:end],
                config,
                weights,
                positions=positions[start:end],
            )
            torch.testing.assert_close(view.kv, functional.kv)
            torch.testing.assert_close(view.pe, functional.pe)
            torch.testing.assert_close(view.positions, functional.positions)

    assert static.valid_length == x.shape[1]
    assert static.capacity == x.shape[1] + 2
    assert pointers == (
        static.kv_storage.data_ptr(),
        static.pe_storage.data_ptr(),
        static.position_storage.data_ptr(),
    )


def test_static_cache_incremental_decode_matches_causal_prefill() -> None:
    x, config, weights = make_fixture()
    positions = torch.arange(10, 10 + 2 * x.shape[1], 2)
    full_cache = build_mla_cache(x, config, weights, positions=positions)
    expected = mla_absorbed_attention_reference(
        x,
        full_cache,
        config,
        weights,
        query_positions=positions,
        causal=True,
    )
    static = allocate_mla_static_cache(
        batch_size=x.shape[0],
        capacity=x.shape[1],
        config=config,
        device=x.device,
        dtype=x.dtype,
    )

    decoded = []
    with torch.inference_mode():
        for index in range(x.shape[1]):
            query = x[:, index : index + 1]
            query_positions = positions[index : index + 1]
            cache = write_mla_static_cache(
                static,
                query,
                config,
                weights,
                positions=query_positions,
            )
            decoded.append(
                mla_absorbed_attention_reference(
                    query,
                    cache,
                    config,
                    weights,
                    query_positions=query_positions,
                    causal=True,
                )
            )

    torch.testing.assert_close(torch.cat(decoded, dim=1), expected, rtol=1e-10, atol=1e-10)


def test_static_cache_truncate_and_overwrite_preserve_storage() -> None:
    x, config, weights = make_fixture()
    static = allocate_mla_static_cache(
        batch_size=x.shape[0],
        capacity=4,
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    with torch.inference_mode():
        write_mla_static_cache(static, x[:, :3], config, weights)
        pointer = static.kv_storage.data_ptr()
        static.truncate(2)
        view = write_mla_static_cache(
            static,
            x[:, 4:5],
            config,
            weights,
            positions=torch.tensor([2]),
        )

    assert static.valid_length == 3
    assert pointer == static.kv_storage.data_ptr()
    expected_last = build_mla_cache(x[:, 4:5], config, weights, positions=torch.tensor([2]))
    torch.testing.assert_close(view.kv[:, -1:], expected_last.kv)


def test_static_cache_rejects_capacity_position_and_batch_errors() -> None:
    x, config, weights = make_fixture()
    static = allocate_mla_static_cache(
        batch_size=x.shape[0],
        capacity=2,
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    with torch.inference_mode():
        write_mla_static_cache(static, x[:, :2], config, weights, positions=torch.tensor([3, 4]))
        with pytest.raises(ValueError, match="capacity"):
            write_mla_static_cache(static, x[:, 2:3], config, weights)
        static.truncate(1)
        with pytest.raises(ValueError, match="follow"):
            write_mla_static_cache(
                static,
                x[:, 2:3],
                config,
                weights,
                positions=torch.tensor([3]),
            )
        with pytest.raises(ValueError, match="batch"):
            write_mla_static_cache(static, x[:1, 2:3], config, weights)
    with pytest.raises(ValueError, match="truncate"):
        static.truncate(static.valid_length + 1)


def test_static_cache_is_explicitly_inference_only() -> None:
    x, config, weights = make_fixture()
    static: MLAStaticCache = allocate_mla_static_cache(
        batch_size=x.shape[0],
        capacity=x.shape[1],
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    with pytest.raises(RuntimeError, match="inference-only"):
        write_mla_static_cache(static, x.requires_grad_(), config, weights)


def test_paged_cache_allocation_layout_and_clear_contract() -> None:
    _, config, _ = make_fixture()
    cache = allocate_mla_paged_cache(
        num_pages=5,
        page_size=3,
        config=config,
        device="cpu",
        dtype=torch.float32,
    )

    assert cache.kv_storage.shape == (5, 3, config.kv_lora_rank)
    assert cache.pe_storage.shape == (5, 3, config.qk_rope_head_dim)
    assert cache.position_storage.shape == (5, 3)
    assert cache.num_pages == 5 and cache.page_size == 3 and cache.capacity == 15
    assert torch.all(cache.position_storage == -1)
    cache.position_storage[2, 1] = 17
    cache.clear()
    assert torch.all(cache.position_storage == -1)


def test_paged_cache_slot_writes_materialize_variable_logical_lengths() -> None:
    x, config, weights = make_fixture()
    page_size = 2
    cache = allocate_mla_paged_cache(
        num_pages=6,
        page_size=page_size,
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    page_rows = ([3, 0, 5], [2, 4])
    lengths = torch.tensor([6, 4])
    block_table = torch.tensor([[3, 0, 5], [2, 4, -1]])
    position_rows = (torch.arange(10, 16), torch.arange(20, 24))
    expected = []

    with torch.inference_mode():
        for batch_index, (pages, length, positions) in enumerate(
            zip(page_rows, lengths.tolist(), position_rows, strict=True)
        ):
            write_mla_paged_cache(
                cache,
                x[batch_index : batch_index + 1, :length],
                config,
                weights,
                positions=positions,
                slot_mapping=physical_slots(list(pages), length, page_size).unsqueeze(0),
                backend="reference",
            )
            expected.append(
                build_mla_cache(
                    x[batch_index : batch_index + 1, :length],
                    config,
                    weights,
                    positions=positions,
                    backend="reference",
                )
            )
        view = materialize_mla_paged_cache(cache, block_table, lengths)

    assert view.kv.shape == (2, 6, config.kv_lora_rank)
    assert view.pe.shape == (2, 6, config.qk_rope_head_dim)
    assert view.positions.tolist() == [[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, -1, -1]]
    assert view.valid_mask.tolist() == [
        [True, True, True, True, True, True],
        [True, True, True, True, False, False],
    ]
    for batch_index, length in enumerate(lengths.tolist()):
        torch.testing.assert_close(
            view.kv[batch_index : batch_index + 1, :length], expected[batch_index].kv
        )
        torch.testing.assert_close(
            view.pe[batch_index : batch_index + 1, :length], expected[batch_index].pe
        )
    assert torch.count_nonzero(view.kv[1, 4:]) == 0
    assert torch.count_nonzero(view.pe[1, 4:]) == 0


def test_paged_attention_matches_per_sequence_contiguous_reference() -> None:
    x, config, weights = make_fixture()
    cache = allocate_mla_paged_cache(
        num_pages=6,
        page_size=2,
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    page_rows = ([3, 0, 5], [2, 4])
    lengths = torch.tensor([6, 4])
    block_table = torch.tensor([[3, 0, 5], [2, 4, -1]])
    position_rows = (torch.arange(10, 16), torch.arange(20, 24))
    query_x = torch.cat((x[0:1, 4:6], x[1:2, 2:4]), dim=0)
    query_positions = torch.tensor([[14, 15], [22, 23]])
    expected_outputs = []

    with torch.inference_mode():
        for batch_index, (pages, length, positions) in enumerate(
            zip(page_rows, lengths.tolist(), position_rows, strict=True)
        ):
            row_x = x[batch_index : batch_index + 1, :length]
            write_mla_paged_cache(
                cache,
                row_x,
                config,
                weights,
                positions=positions,
                slot_mapping=physical_slots(list(pages), length, cache.page_size).unsqueeze(0),
                backend="reference",
            )
            contiguous = build_mla_cache(
                row_x,
                config,
                weights,
                positions=positions,
                backend="reference",
            )
            expected_outputs.append(
                mla_absorbed_attention_reference(
                    query_x[batch_index : batch_index + 1],
                    contiguous,
                    config,
                    weights,
                    query_positions=query_positions[batch_index],
                    causal=True,
                )
            )
        actual = mla_paged_attention(
            query_x,
            cache,
            block_table,
            lengths,
            config,
            weights,
            query_positions=query_positions,
            causal=True,
            backend="reference",
        )

    torch.testing.assert_close(actual, torch.cat(expected_outputs, dim=0), rtol=1e-10, atol=1e-10)


def test_paged_attention_long_context_tail_page_matches_contiguous_reference() -> None:
    _, config, weights = make_fixture()
    sequence_length = 257
    page_size = 8
    required_pages = (sequence_length + page_size - 1) // page_size
    logical_pages = list(range(0, required_pages, 2)) + list(range(1, required_pages, 2))
    x = torch.randn(1, sequence_length, weights.wkv_a.shape[1], dtype=weights.wkv_a.dtype)
    positions = torch.arange(100, 100 + sequence_length * 2, 2)
    cache = allocate_mla_paged_cache(
        num_pages=required_pages + 3,
        page_size=page_size,
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    block_table = torch.tensor([logical_pages])
    lengths = torch.tensor([sequence_length])

    with torch.inference_mode():
        write_mla_paged_cache(
            cache,
            x,
            config,
            weights,
            positions=positions,
            slot_mapping=physical_slots(logical_pages, sequence_length, page_size).unsqueeze(0),
            backend="reference",
        )
        contiguous = build_mla_cache(
            x,
            config,
            weights,
            positions=positions,
            backend="reference",
        )
        actual = mla_paged_attention(
            x[:, -1:],
            cache,
            block_table,
            lengths,
            config,
            weights,
            query_positions=positions[-1:].unsqueeze(0),
            backend="reference",
        )
        expected = mla_absorbed_attention_reference(
            x[:, -1:],
            contiguous,
            config,
            weights,
            query_positions=positions[-1:],
        )

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_paged_cache_rejects_duplicate_and_out_of_range_slot_writes() -> None:
    x, config, weights = make_fixture()
    cache = allocate_mla_paged_cache(
        num_pages=2,
        page_size=2,
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    positions = torch.tensor([0, 1])

    with torch.inference_mode(), pytest.raises(ValueError, match="duplicate"):
        write_mla_paged_cache(
            cache,
            x[:1, :2],
            config,
            weights,
            positions=positions,
            slot_mapping=torch.tensor([[1, 1]]),
        )
    with torch.inference_mode(), pytest.raises(ValueError, match="out-of-range"):
        write_mla_paged_cache(
            cache,
            x[:1, :2],
            config,
            weights,
            positions=positions,
            slot_mapping=torch.tensor([[0, cache.capacity]]),
        )


def test_paged_cache_allows_complete_slot_overwrite_across_calls() -> None:
    x, config, weights = make_fixture()
    cache = allocate_mla_paged_cache(
        num_pages=2,
        page_size=2,
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    slot = torch.tensor([[3]])

    with torch.inference_mode():
        write_mla_paged_cache(
            cache,
            x[:1, :1],
            config,
            weights,
            positions=torch.tensor([5]),
            slot_mapping=slot,
            backend="reference",
        )
        write_mla_paged_cache(
            cache,
            x[:1, 4:5],
            config,
            weights,
            positions=torch.tensor([9]),
            slot_mapping=slot,
            backend="reference",
        )
        expected = build_mla_cache(
            x[:1, 4:5],
            config,
            weights,
            positions=torch.tensor([9]),
            backend="reference",
        )

    torch.testing.assert_close(cache.kv_storage.view(-1, config.kv_lora_rank)[3], expected.kv[0, 0])
    torch.testing.assert_close(
        cache.pe_storage.view(-1, config.qk_rope_head_dim)[3], expected.pe[0, 0]
    )
    assert cache.position_storage.view(-1)[3].item() == 9


def test_paged_slot_validation_cache_invalidates_after_inplace_mutation() -> None:
    x, config, weights = make_fixture()
    cache = allocate_mla_paged_cache(
        num_pages=2,
        page_size=2,
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    positions = torch.tensor([0, 1])
    slots = torch.tensor([[0, 1]])

    with torch.inference_mode():
        write_mla_paged_cache(
            cache,
            x[:1, :2],
            config,
            weights,
            positions=positions,
            slot_mapping=slots,
            backend="reference",
        )
        slots[0, 1] = 0
        with pytest.raises(ValueError, match="duplicate"):
            write_mla_paged_cache(
                cache,
                x[:1, :2],
                config,
                weights,
                positions=positions,
                slot_mapping=slots,
                backend="reference",
            )


@pytest.mark.parametrize(
    ("block_table", "lengths", "message"),
    [
        (torch.tensor([[3]]), torch.tensor([1]), "out-of-range"),
        (torch.tensor([[0, 0]]), torch.tensor([3]), "repeats"),
        (torch.tensor([[0, 1]]), torch.tensor([1]), "unused"),
        (torch.tensor([[0]]), torch.tensor([3]), "capacity"),
    ],
)
def test_paged_cache_rejects_invalid_block_table_contract(
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    message: str,
) -> None:
    _, config, _ = make_fixture()
    cache = allocate_mla_paged_cache(
        num_pages=3,
        page_size=2,
        config=config,
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match=message):
        materialize_mla_paged_cache(cache, block_table, lengths)


def test_paged_metadata_validation_cache_invalidates_after_inplace_mutation() -> None:
    x, config, weights = make_fixture()
    cache = allocate_mla_paged_cache(
        num_pages=2,
        page_size=2,
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    table = torch.tensor([[0, -1]])
    lengths = torch.tensor([2])
    with torch.inference_mode():
        write_mla_paged_cache(
            cache,
            x[:1, :2],
            config,
            weights,
            positions=torch.tensor([0, 1]),
            slot_mapping=torch.tensor([[0, 1]]),
            backend="reference",
        )
        materialize_mla_paged_cache(cache, table, lengths)
        table[0, 1] = 1
    with pytest.raises(ValueError, match="unused"):
        materialize_mla_paged_cache(cache, table, lengths)


def test_paged_cache_rejects_unwritten_and_nonmonotonic_logical_slots() -> None:
    x, config, weights = make_fixture()
    cache = allocate_mla_paged_cache(
        num_pages=2,
        page_size=2,
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    table = torch.tensor([[0]])
    lengths = torch.tensor([2])
    with pytest.raises(ValueError, match="unwritten"):
        materialize_mla_paged_cache(cache, table, lengths)

    with torch.inference_mode():
        write_mla_paged_cache(
            cache,
            x[:1, :2],
            config,
            weights,
            positions=torch.tensor([3, 4]),
            slot_mapping=torch.tensor([[0, 1]]),
            backend="reference",
        )
        materialize_mla_paged_cache(cache, table, lengths)
        cache.position_storage[0, 1] = 3
    with pytest.raises(ValueError, match="increase"):
        materialize_mla_paged_cache(cache, table, lengths)


def test_paged_cache_is_explicitly_inference_only() -> None:
    x, config, weights = make_fixture()
    cache: MLAPagedCache = allocate_mla_paged_cache(
        num_pages=2,
        page_size=4,
        config=config,
        device=x.device,
        dtype=x.dtype,
    )
    with pytest.raises(RuntimeError, match="inference-only"):
        write_mla_paged_cache(
            cache,
            x[:1, :1].requires_grad_(),
            config,
            weights,
            positions=torch.tensor([0]),
            slot_mapping=torch.tensor([[0]]),
        )


def test_raw_mla_paged_cache_write_passes_opcheck() -> None:
    x, config, weights = make_fixture()
    positions = torch.stack((torch.arange(3), torch.arange(10, 13)))
    slot_mapping = torch.tensor([[0, 3, 4], [1, 6, 7]])
    kv_storage = torch.full((2, 4, config.kv_lora_rank), torch.nan, dtype=x.dtype)
    pe_storage = torch.full((2, 4, config.qk_rope_head_dim), torch.nan, dtype=x.dtype)
    position_storage = torch.full((2, 4), -1, dtype=torch.long)
    inputs = (
        x[:, :3],
        weights.wkv_a,
        weights.kv_norm_weight,
        positions,
        slot_mapping,
        kv_storage,
        pe_storage,
        position_storage,
        False,
        config.rope_theta,
        config.rms_norm_eps,
    )

    torch.library.opcheck(
        torch.ops.ds_flash_mla_moe.mla_cache_projection_write_slots.default,
        inputs,
    )
    torch.ops.ds_flash_mla_moe.mla_cache_projection_write_slots.default(*inputs)
    for batch_index in range(x.shape[0]):
        expected = build_mla_cache(
            x[batch_index : batch_index + 1, :3],
            config,
            weights,
            positions=positions[batch_index],
            backend="reference",
        )
        slots = slot_mapping[batch_index]
        torch.testing.assert_close(
            kv_storage.view(-1, config.kv_lora_rank)[slots],
            expected.kv[0],
        )
        torch.testing.assert_close(
            pe_storage.view(-1, config.qk_rope_head_dim)[slots],
            expected.pe[0],
        )


def test_raw_mla_paged_attention_runs_through_torch_compile() -> None:
    torch.manual_seed(20260814)
    q_nope = torch.randn(2, 2, 3, 4, dtype=torch.float64)
    q_pe = torch.randn(2, 2, 3, 2, dtype=torch.float64)
    kv_storage = torch.randn(3, 2, 4, dtype=torch.float64)
    pe_storage = torch.randn(3, 2, 2, dtype=torch.float64)
    position_storage = torch.tensor([[2, 3], [0, 1], [10, 11]])
    block_table = torch.tensor([[1, 0], [2, -1]])
    sequence_lengths = torch.tensor([4, 2])
    key_up = torch.randn(3, 4, 4, dtype=torch.float64)
    value_up = torch.randn(3, 3, 4, dtype=torch.float64)
    query_positions = torch.tensor([[2, 3], [10, 11]])

    @torch.compile(fullgraph=True, backend="eager")
    def compiled(*inputs: torch.Tensor) -> torch.Tensor:
        return torch.ops.ds_flash_mla_moe.mla_paged_absorbed_attention.default(
            *inputs,
            False,
            True,
            0.25,
        )

    inputs = (
        q_nope,
        q_pe,
        kv_storage,
        pe_storage,
        position_storage,
        block_table,
        sequence_lengths,
        key_up,
        value_up,
        query_positions,
    )
    expected = torch.ops.ds_flash_mla_moe.mla_paged_absorbed_attention.default(
        *inputs,
        False,
        True,
        0.25,
    )
    torch.testing.assert_close(compiled(*inputs), expected)


@pytest.mark.skipif(
    not cuda_paged_mla_available(),
    reason="requires native paged MLA kernels and a CUDA device",
)
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_paged_cache_write_and_attention_match_reference(dtype: torch.dtype) -> None:
    x, config, weights = make_fixture()
    x = x.to(device="cuda", dtype=dtype)
    weights = cuda_weights(weights, dtype=dtype)
    native_cache = allocate_mla_paged_cache(
        num_pages=6,
        page_size=2,
        config=config,
        device="cuda",
        dtype=dtype,
    )
    reference_cache = allocate_mla_paged_cache(
        num_pages=6,
        page_size=2,
        config=config,
        device="cuda",
        dtype=dtype,
    )
    page_rows = ([3, 0, 5], [2, 4])
    lengths = torch.tensor([6, 4], device="cuda")
    block_table = torch.tensor([[3, 0, 5], [2, 4, -1]], device="cuda")
    position_rows = (
        torch.arange(10, 16, device="cuda"),
        torch.arange(20, 24, device="cuda"),
    )

    with torch.inference_mode():
        for batch_index, (pages, length, positions) in enumerate(
            zip(page_rows, lengths.cpu().tolist(), position_rows, strict=True)
        ):
            slots = (
                physical_slots(list(pages), length, native_cache.page_size).to("cuda").unsqueeze(0)
            )
            row_x = x[batch_index : batch_index + 1, :length]
            write_mla_paged_cache(
                native_cache,
                row_x,
                config,
                weights,
                positions=positions,
                slot_mapping=slots,
                backend="cuda",
            )
            write_mla_paged_cache(
                reference_cache,
                row_x,
                config,
                weights,
                positions=positions,
                slot_mapping=slots,
                backend="reference",
            )

        native_view = materialize_mla_paged_cache(native_cache, block_table, lengths)
        reference_view = materialize_mla_paged_cache(reference_cache, block_table, lengths)
        query_x = torch.cat((x[0:1, 4:6], x[1:2, 2:4]), dim=0)
        query_positions = torch.tensor([[14, 15], [22, 23]], device="cuda")
        actual = mla_paged_attention(
            query_x,
            native_cache,
            block_table,
            lengths,
            config,
            weights,
            query_positions=query_positions,
            backend="cuda",
        )
        expected = mla_paged_attention(
            query_x,
            reference_cache,
            block_table,
            lengths,
            config,
            weights,
            query_positions=query_positions,
            backend="reference",
        )

    stage_rtol, stage_atol = _mla_cuda_stage_tolerances(dtype)
    torch.testing.assert_close(native_view.kv, reference_view.kv, rtol=stage_rtol, atol=stage_atol)
    torch.testing.assert_close(native_view.pe, reference_view.pe, rtol=stage_rtol, atol=stage_atol)
    rtol, atol = _mla_cuda_tolerances(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not cuda_paged_mla_available(),
    reason="requires native paged MLA kernels and a CUDA device",
)
@pytest.mark.cuda
def test_cuda_paged_attention_long_context_tail_page_matches_reference() -> None:
    _, config, weights = make_fixture()
    dtype = torch.bfloat16
    weights = cuda_weights(weights, dtype=dtype)
    sequence_length = 257
    page_size = 16
    required_pages = (sequence_length + page_size - 1) // page_size
    logical_pages = list(range(1, required_pages + 1))
    x = torch.randn(1, sequence_length, weights.wkv_a.shape[1], device="cuda", dtype=dtype)
    positions = torch.arange(50, 50 + sequence_length, device="cuda")
    native_cache = allocate_mla_paged_cache(
        num_pages=required_pages + 1,
        page_size=page_size,
        config=config,
        device="cuda",
        dtype=dtype,
    )
    reference_cache = allocate_mla_paged_cache(
        num_pages=required_pages + 1,
        page_size=page_size,
        config=config,
        device="cuda",
        dtype=dtype,
    )
    block_table = torch.tensor([logical_pages], device="cuda")
    lengths = torch.tensor([sequence_length], device="cuda")
    slots = physical_slots(logical_pages, sequence_length, page_size).to("cuda").unsqueeze(0)

    with torch.inference_mode():
        write_mla_paged_cache(
            native_cache,
            x,
            config,
            weights,
            positions=positions,
            slot_mapping=slots,
            backend="cuda",
        )
        write_mla_paged_cache(
            reference_cache,
            x,
            config,
            weights,
            positions=positions,
            slot_mapping=slots,
            backend="reference",
        )
        query_positions = positions[-1:].unsqueeze(0)
        actual = mla_paged_attention(
            x[:, -1:],
            native_cache,
            block_table,
            lengths,
            config,
            weights,
            query_positions=query_positions,
            backend="cuda",
        )
        expected = mla_paged_attention(
            x[:, -1:],
            reference_cache,
            block_table,
            lengths,
            config,
            weights,
            query_positions=query_positions,
            backend="reference",
        )

    rtol, atol = _mla_cuda_tolerances(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not cuda_paged_mla_available(),
    reason="requires native paged MLA kernels and a CUDA device",
)
@pytest.mark.cuda
@pytest.mark.parametrize("slot_mapping", [torch.tensor([[0, 0]]), torch.tensor([[0, 4]])])
def test_raw_cuda_paged_cache_write_rejects_unsafe_slots(slot_mapping: torch.Tensor) -> None:
    x, config, weights = make_fixture()
    x = x[:1, :2].to(device="cuda", dtype=torch.float32)
    weights = cuda_weights(weights)
    cache = allocate_mla_paged_cache(
        num_pages=2,
        page_size=2,
        config=config,
        device="cuda",
        dtype=torch.float32,
    )
    positions = torch.tensor([[0, 1]], device="cuda")

    with pytest.raises(RuntimeError, match="slot_mapping"):
        torch.ops.ds_flash_mla_moe.mla_cache_projection_write_slots.default(
            x,
            weights.wkv_a,
            weights.kv_norm_weight,
            positions,
            slot_mapping.to("cuda"),
            cache.kv_storage,
            cache.pe_storage,
            cache.position_storage,
            False,
            config.rope_theta,
            config.rms_norm_eps,
        )
