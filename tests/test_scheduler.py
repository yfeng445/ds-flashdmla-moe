from __future__ import annotations

import pytest

from ds_flash_mla_moe.scheduler import (
    ContinuousBatchingScheduler,
    FixedPageAllocator,
    SequenceStatus,
)


def test_fixed_page_allocator_failure_is_atomic_and_release_reuses_pages_fifo() -> None:
    allocator = FixedPageAllocator(num_pages=3)

    first = allocator.allocate(2)
    before_failure = allocator.snapshot()

    with pytest.raises(RuntimeError, match="insufficient free pages"):
        allocator.allocate(2)

    assert first == (0, 1)
    assert allocator.snapshot() == before_failure
    allocator.release(first)
    assert allocator.allocate(3) == (2, 0, 1)


def test_scheduler_admits_prefill_fifo_and_builds_fixed_page_metadata() -> None:
    scheduler = ContinuousBatchingScheduler(page_size=4, num_pages=8, max_batch_size=2)
    scheduler.submit("first", prompt_length=5, max_new_tokens=2)
    scheduler.submit("second", prompt_length=3, max_new_tokens=1)
    scheduler.submit("third", prompt_length=1, max_new_tokens=1)

    batch = scheduler.schedule()

    assert batch is not None
    assert batch.phase == "prefill"
    assert batch.request_ids == ("first", "second")
    assert batch.token_counts == (5, 3)
    assert batch.start_positions == (0, 0)
    assert batch.sequence_lengths == (5, 3)
    assert batch.block_tables == ((0, 1), (2, -1))
    assert batch.slot_mappings == ((0, 1, 2, 3, 4), (8, 9, 10))
    assert scheduler.queued_request_ids == ("third",)


def test_admission_selects_largest_fundable_fifo_prefix_without_skipping() -> None:
    scheduler = ContinuousBatchingScheduler(page_size=2, num_pages=3, max_batch_size=3)
    scheduler.submit("first", prompt_length=3, max_new_tokens=1)
    scheduler.submit("blocked-second", prompt_length=3, max_new_tokens=1)
    scheduler.submit("small-third", prompt_length=1, max_new_tokens=1)

    batch = scheduler.schedule()

    assert batch is not None
    assert batch.request_ids == ("first",)
    assert scheduler.queued_request_ids == ("blocked-second", "small-third")
    assert scheduler.state("blocked-second").status is SequenceStatus.QUEUED
    assert scheduler.state("small-third").status is SequenceStatus.QUEUED
    assert scheduler.allocator.free_page_count == 1


def test_admission_reserves_worst_case_pages_and_decode_crosses_boundary_without_allocating() -> (
    None
):
    scheduler = ContinuousBatchingScheduler(page_size=2, num_pages=4, max_batch_size=2)
    scheduler.submit("a", prompt_length=2, max_new_tokens=2)
    scheduler.submit("b", prompt_length=1, max_new_tokens=3)
    prefill = scheduler.schedule()
    assert prefill is not None

    assert scheduler.allocator.free_page_count == 0
    assert prefill.block_tables == ((0, -1), (2, -1))
    scheduler.complete(prefill)
    a = scheduler.state("a")
    b = scheduler.state("b")
    assert a.used_page_ids == (0,)
    assert a.page_ids == a.used_page_ids
    assert a.reserved_page_ids == (0, 1)
    assert a.used_page_count == 1
    assert a.reserved_page_count == 2
    assert b.used_page_ids == (2,)
    assert b.reserved_page_ids == (2, 3)

    before_decode_allocator = scheduler.allocator.snapshot()
    decode = scheduler.schedule()

    assert decode is not None
    assert decode.phase == "decode"
    assert decode.block_tables == ((0, 1), (2, -1))
    assert decode.slot_mappings == ((2,), (5,))
    assert scheduler.allocator.snapshot() == before_decode_allocator


def test_failed_multi_request_schedule_leaves_every_state_unchanged() -> None:
    scheduler = ContinuousBatchingScheduler(page_size=2, num_pages=3, max_batch_size=2)
    scheduler.allocator.allocate(2)
    scheduler.submit("a", prompt_length=2, max_new_tokens=1)
    scheduler.submit("b", prompt_length=1, max_new_tokens=1)
    before = scheduler.snapshot()

    with pytest.raises(RuntimeError, match="insufficient free pages"):
        scheduler.schedule()

    assert scheduler.snapshot() == before
    assert scheduler.inflight_batch is None


def test_abort_restores_prefill_admission_pages_and_fifo_order_exactly() -> None:
    scheduler = ContinuousBatchingScheduler(page_size=2, num_pages=5, max_batch_size=2)
    scheduler.submit("a", prompt_length=3, max_new_tokens=2)
    scheduler.submit("b", prompt_length=1, max_new_tokens=2)
    before = scheduler.snapshot()

    batch = scheduler.schedule()
    assert batch is not None
    scheduler.abort(batch)

    assert scheduler.snapshot() == before
    replayed = scheduler.schedule()
    assert replayed is not None
    assert replayed.request_ids == batch.request_ids
    assert replayed.block_tables == batch.block_tables
    assert replayed.slot_mappings == batch.slot_mappings


def test_decode_reserves_one_token_per_request_and_abort_restores_boundary_page() -> None:
    scheduler = ContinuousBatchingScheduler(page_size=2, num_pages=6, max_batch_size=2)
    scheduler.submit("a", prompt_length=2, max_new_tokens=2)
    scheduler.submit("b", prompt_length=1, max_new_tokens=2)
    prefill = scheduler.schedule()
    assert prefill is not None
    scheduler.complete(prefill)
    before_decode = scheduler.snapshot()

    decode = scheduler.schedule()

    assert decode is not None
    assert decode.phase == "decode"
    assert decode.token_counts == (1, 1)
    assert decode.start_positions == (2, 1)
    assert decode.sequence_lengths == (3, 2)
    assert decode.block_tables == ((0, 1), (2, -1))
    assert decode.slot_mappings == ((2,), (5,))

    scheduler.abort(decode)
    assert scheduler.snapshot() == before_decode


def test_complete_commits_lengths_and_admits_fifo_replacement_at_next_boundary() -> None:
    scheduler = ContinuousBatchingScheduler(page_size=2, num_pages=6, max_batch_size=2)
    scheduler.submit("a", prompt_length=1, max_new_tokens=1)
    scheduler.submit("b", prompt_length=1, max_new_tokens=2)
    scheduler.submit("c", prompt_length=2, max_new_tokens=1)
    prefill = scheduler.schedule()
    assert prefill is not None
    scheduler.complete(prefill)

    decode = scheduler.schedule()
    assert decode is not None
    scheduler.complete(decode)

    a = scheduler.state("a")
    b = scheduler.state("b")
    assert a.status is SequenceStatus.FINISHED
    assert a.committed_length == 2
    assert a.generated_tokens == 1
    assert a.page_ids == ()
    assert b.status is SequenceStatus.ACTIVE
    assert b.committed_length == 2
    assert b.generated_tokens == 1

    replacement = scheduler.schedule()
    assert replacement is not None
    assert replacement.phase == "prefill"
    assert replacement.request_ids == ("c",)


def test_complete_can_finish_decode_early_and_release_pages() -> None:
    scheduler = ContinuousBatchingScheduler(page_size=2, num_pages=3, max_batch_size=1)
    scheduler.submit("a", prompt_length=2, max_new_tokens=4)
    prefill = scheduler.schedule()
    assert prefill is not None
    scheduler.complete(prefill)
    decode = scheduler.schedule()
    assert decode is not None

    scheduler.complete(decode, finished_request_ids=("a",))

    state = scheduler.state("a")
    assert state.status is SequenceStatus.FINISHED
    assert state.committed_length == 3
    assert state.generated_tokens == 1
    assert scheduler.allocator.free_page_count == 3


def test_cancel_queued_or_idle_active_request_releases_resources() -> None:
    scheduler = ContinuousBatchingScheduler(page_size=2, num_pages=3, max_batch_size=1)
    scheduler.submit("queued", prompt_length=1, max_new_tokens=1)
    assert scheduler.cancel("queued") is True
    assert scheduler.state("queued").status is SequenceStatus.CANCELLED

    scheduler.submit("active", prompt_length=2, max_new_tokens=2)
    prefill = scheduler.schedule()
    assert prefill is not None
    scheduler.complete(prefill)
    assert scheduler.cancel("active") is True
    assert scheduler.state("active").status is SequenceStatus.CANCELLED
    assert scheduler.allocator.free_page_count == 3
    assert scheduler.cancel("active") is False


def test_cancel_rejects_inflight_request_until_batch_is_aborted() -> None:
    scheduler = ContinuousBatchingScheduler(page_size=2, num_pages=2, max_batch_size=1)
    scheduler.submit("a", prompt_length=1, max_new_tokens=1)
    batch = scheduler.schedule()
    assert batch is not None

    with pytest.raises(RuntimeError, match="iteration is in flight"):
        scheduler.cancel("a")

    scheduler.abort(batch)
    assert scheduler.cancel("a") is True


def test_cancel_rejects_any_active_request_while_an_iteration_is_inflight() -> None:
    scheduler = ContinuousBatchingScheduler(page_size=2, num_pages=4, max_batch_size=2)
    scheduler.submit("decoding", prompt_length=1, max_new_tokens=1)
    first_prefill = scheduler.schedule()
    assert first_prefill is not None
    scheduler.complete(first_prefill)
    scheduler.submit("replacement", prompt_length=1, max_new_tokens=1)
    replacement_prefill = scheduler.schedule()
    assert replacement_prefill is not None
    assert replacement_prefill.request_ids == ("replacement",)

    with pytest.raises(RuntimeError, match="iteration is in flight"):
        scheduler.cancel("decoding")

    scheduler.abort(replacement_prefill)
    assert scheduler.cancel("decoding") is True


def test_scheduler_rejects_stale_batches_and_duplicate_requests() -> None:
    scheduler = ContinuousBatchingScheduler(page_size=2, num_pages=2, max_batch_size=1)
    scheduler.submit("a", prompt_length=1, max_new_tokens=1)
    with pytest.raises(ValueError, match="already exists"):
        scheduler.submit("a", prompt_length=1, max_new_tokens=1)
    batch = scheduler.schedule()
    assert batch is not None
    with pytest.raises(RuntimeError, match="in flight"):
        scheduler.schedule()
    scheduler.complete(batch)

    with pytest.raises(RuntimeError, match="stale"):
        scheduler.abort(batch)
    with pytest.raises(ValueError, match="unknown request"):
        scheduler.cancel("missing")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"page_size": 0, "num_pages": 1, "max_batch_size": 1}, "positive"),
        ({"page_size": 1, "num_pages": 0, "max_batch_size": 1}, "positive"),
        ({"page_size": 1, "num_pages": 1, "max_batch_size": 0}, "positive"),
    ],
)
def test_scheduler_rejects_invalid_capacity(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ContinuousBatchingScheduler(**kwargs)


@pytest.mark.parametrize(
    ("prompt_length", "max_new_tokens"),
    [(0, 1), (-1, 1), (1, 0), (1, -1)],
)
def test_submit_rejects_non_positive_lengths(prompt_length: int, max_new_tokens: int) -> None:
    scheduler = ContinuousBatchingScheduler(page_size=2, num_pages=2, max_batch_size=1)
    with pytest.raises(ValueError, match="positive"):
        scheduler.submit("bad", prompt_length=prompt_length, max_new_tokens=max_new_tokens)
