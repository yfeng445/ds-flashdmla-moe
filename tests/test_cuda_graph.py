from __future__ import annotations

import pytest
import torch

from ds_flash_mla_moe.cuda_graph import (
    MLAPagedDecodeGraphBucket,
    MLAPagedDecodeGraphRunner,
    SingleOutputCUDAGraphRunner,
    StaticTensorSpec,
)
from ds_flash_mla_moe.mla import (
    MLAConfig,
    MLAWeights,
    allocate_mla_paged_cache,
    mla_paged_attention,
    write_mla_paged_cache,
)
from ds_flash_mla_moe.ops import cuda_paged_mla_available


def test_static_tensor_spec_accepts_new_address_but_rejects_shape_dtype_and_device() -> None:
    example = torch.randn(2, 3)
    spec = StaticTensorSpec.from_tensor(example)

    assert spec.is_compatible(example.clone())
    assert not spec.is_compatible(torch.randn(3, 2))
    assert not spec.is_compatible(torch.randn(2, 3, dtype=torch.float64))
    assert not spec.is_compatible(torch.empty(2, 3, device="meta"))


def test_graph_and_scheduler_types_are_public_package_exports() -> None:
    from ds_flash_mla_moe import (
        ContinuousBatchingScheduler,
        FixedPageAllocator,
        MLAPagedDecodeGraphBucket,
        MLAPagedDecodeGraphRunner,
        ScheduledBatch,
        SequenceState,
        SequenceStatus,
        SingleOutputCUDAGraphRunner,
        StaticTensorSpec,
    )

    assert all(
        value is not None
        for value in (
            ContinuousBatchingScheduler,
            FixedPageAllocator,
            MLAPagedDecodeGraphBucket,
            MLAPagedDecodeGraphRunner,
            ScheduledBatch,
            SequenceState,
            SequenceStatus,
            SingleOutputCUDAGraphRunner,
            StaticTensorSpec,
        )
    )


def test_static_tensor_spec_validation_reports_named_input() -> None:
    spec = StaticTensorSpec.from_tensor(torch.randn(2, 3))

    with pytest.raises(ValueError, match=r"query.*shape"):
        spec.validate(torch.randn(2, 4), name="query")
    with pytest.raises(TypeError, match=r"query.*dtype"):
        spec.validate(torch.randn(2, 3, dtype=torch.float64), name="query")


def test_graph_capture_rejects_cpu_and_requires_at_least_one_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SingleOutputCUDAGraphRunner.capture(lambda: torch.ones(1), ())
    with pytest.raises(ValueError, match="CUDA"):
        SingleOutputCUDAGraphRunner.capture(lambda x: x + 1, (torch.ones(1),))


def test_mla_decode_bucket_requires_exact_static_shapes() -> None:
    bucket = MLAPagedDecodeGraphBucket(batch_size=2, max_logical_pages=3, model_dim=8)

    assert bucket.is_compatible(
        torch.empty(2, 1, 8),
        torch.empty(2, 3, dtype=torch.long),
        torch.empty(2, dtype=torch.long),
        torch.empty(2, 1, dtype=torch.long),
    )
    assert not bucket.is_compatible(
        torch.empty(1, 1, 8),
        torch.empty(1, 3, dtype=torch.long),
        torch.empty(1, dtype=torch.long),
        torch.empty(1, 1, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="positive"):
        MLAPagedDecodeGraphBucket(batch_size=0, max_logical_pages=3, model_dim=8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph support")
@pytest.mark.cuda
def test_single_output_graph_replay_copies_values_and_keeps_output_address_stable() -> None:
    example = torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3)
    runner = SingleOutputCUDAGraphRunner.capture(lambda x: x.square() + 1, (example,))
    pointer = runner.output.data_ptr()

    first_input = torch.full_like(example, 2)
    first = runner.replay(first_input)
    torch.cuda.synchronize()
    torch.testing.assert_close(first, torch.full_like(example, 5))

    first_input.fill_(9)
    torch.testing.assert_close(first, torch.full_like(example, 5))

    second = runner(torch.full_like(example, 3))
    torch.cuda.synchronize()
    assert second.data_ptr() == pointer == first.data_ptr()
    torch.testing.assert_close(second, torch.full_like(example, 10))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph support")
@pytest.mark.cuda
def test_graph_replay_rejects_incompatible_bucket_before_overwriting_static_input() -> None:
    example = torch.ones(2, 3, device="cuda")
    runner = SingleOutputCUDAGraphRunner.capture(lambda x: x + 4, (example,))
    before = runner.static_inputs[0].clone()

    with pytest.raises(ValueError, match="shape"):
        runner.replay(torch.ones(3, 2, device="cuda"))

    torch.testing.assert_close(runner.static_inputs[0], before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph support")
@pytest.mark.cuda
def test_graph_capture_rejects_multiple_outputs_and_requires_grad_inputs() -> None:
    example = torch.ones(2, device="cuda")
    with pytest.raises(TypeError, match="single Tensor"):
        SingleOutputCUDAGraphRunner.capture(lambda x: (x, x + 1), (example,))
    with pytest.raises(RuntimeError, match="forward-only"):
        SingleOutputCUDAGraphRunner.capture(lambda x: x + 1, (example.requires_grad_(),))


def _make_cuda_mla_graph_fixture():
    torch.manual_seed(20260822)
    config = MLAConfig(
        n_heads=2,
        q_lora_rank=0,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
    )
    model_dim = 8
    dtype = torch.float32
    device = torch.device("cuda")
    weights = MLAWeights(
        wkv_a=torch.randn(8, model_dim, device=device, dtype=dtype),
        kv_norm_weight=torch.randn(4, device=device, dtype=dtype),
        wkv_b=torch.randn(16, 4, device=device, dtype=dtype),
        wo=torch.randn(model_dim, 8, device=device, dtype=dtype),
        wq=torch.randn(16, model_dim, device=device, dtype=dtype),
    )
    cache = allocate_mla_paged_cache(
        num_pages=4,
        page_size=2,
        config=config,
        device=device,
        dtype=dtype,
    )
    cache_x = torch.randn(2, 3, model_dim, device=device, dtype=dtype)
    positions = torch.tensor([[0, 1, 2], [0, 1, 2]], device=device)
    slots = torch.tensor([[0, 1, 2], [4, 5, 6]], device=device)
    write_mla_paged_cache(
        cache,
        cache_x,
        config,
        weights,
        positions=positions,
        slot_mapping=slots,
        backend="cuda",
    )
    block_table = torch.tensor([[0, 1], [2, 3]], device=device)
    lengths = torch.tensor([3, 3], device=device)
    query_positions = torch.tensor([[2], [2]], device=device)
    query = torch.randn(2, 1, model_dim, device=device, dtype=dtype)
    return query, cache, block_table, lengths, config, weights, query_positions


@pytest.mark.skipif(
    not cuda_paged_mla_available(),
    reason="requires built paged MLA CUDA operators",
)
@pytest.mark.cuda
def test_mla_paged_decode_graph_replays_raw_pipeline_and_validates_metadata_first() -> None:
    query, cache, block_table, lengths, config, weights, query_positions = (
        _make_cuda_mla_graph_fixture()
    )
    runner = MLAPagedDecodeGraphRunner.capture(
        query,
        cache,
        block_table,
        lengths,
        config,
        weights,
        query_positions=query_positions,
    )
    new_query = torch.randn_like(query)
    expected = mla_paged_attention(
        new_query,
        cache,
        block_table,
        lengths,
        config,
        weights,
        query_positions=query_positions,
        backend="cuda",
    )

    actual = runner.replay(
        new_query,
        block_table,
        lengths,
        query_positions=query_positions,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)

    static_table_before = runner.static_inputs[1].clone()
    invalid_table = block_table.clone()
    invalid_table[0, 0] = cache.num_pages
    with pytest.raises(ValueError, match="out-of-range"):
        runner.replay(
            new_query,
            invalid_table,
            lengths,
            query_positions=query_positions,
        )
    torch.testing.assert_close(runner.static_inputs[1], static_table_before)

    original_kv = cache.kv_storage
    cache.kv_storage = original_kv.clone()
    with pytest.raises(RuntimeError, match="cache.*replaced"):
        runner.replay(
            new_query,
            block_table,
            lengths,
            query_positions=query_positions,
        )
    cache.kv_storage = original_kv
