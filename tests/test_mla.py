from dataclasses import fields

import pytest
import torch

from ds_flash_mla_moe import (
    MLAConfig,
    MLAStaticCache,
    MLAWeights,
    allocate_mla_static_cache,
    append_mla_cache,
    build_mla_cache,
    mla_absorbed_attention_reference,
    mla_naive_attention_reference,
    write_mla_static_cache,
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
