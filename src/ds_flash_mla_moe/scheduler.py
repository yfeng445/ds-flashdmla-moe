"""A small transactional FIFO control plane for paged continuous batching.

This module models request admission and fixed-page ownership.  It deliberately
does not execute a model, manage token payloads, evict pages, or overlap prefill
and decode.  At most one homogeneous batch may be in flight, so request
replacement happens only between iterations.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class SequenceStatus(str, Enum):
    """Externally visible lifecycle of one submitted sequence."""

    QUEUED = "queued"
    ACTIVE = "active"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SequenceState:
    """Read-only snapshot of one scheduler sequence."""

    request_id: str
    prompt_length: int
    max_new_tokens: int
    status: SequenceStatus
    committed_length: int
    generated_tokens: int
    page_ids: tuple[int, ...]
    reserved_length: int | None


@dataclass
class _SequenceRecord:
    request_id: str
    prompt_length: int
    max_new_tokens: int
    status: SequenceStatus = SequenceStatus.QUEUED
    committed_length: int = 0
    generated_tokens: int = 0
    page_ids: list[int] = field(default_factory=list)
    reserved_length: int | None = None

    def snapshot(self) -> SequenceState:
        return SequenceState(
            request_id=self.request_id,
            prompt_length=self.prompt_length,
            max_new_tokens=self.max_new_tokens,
            status=self.status,
            committed_length=self.committed_length,
            generated_tokens=self.generated_tokens,
            page_ids=tuple(self.page_ids),
            reserved_length=self.reserved_length,
        )


class FixedPageAllocator:
    """Deterministic fixed-page allocator with atomic multi-page allocation."""

    def __init__(self, num_pages: int) -> None:
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        self._num_pages = num_pages
        self._free = deque(range(num_pages))
        self._allocated: set[int] = set()

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @property
    def free_page_count(self) -> int:
        return len(self._free)

    @property
    def allocated_page_count(self) -> int:
        return len(self._allocated)

    def can_allocate(self, count: int) -> bool:
        if count < 0:
            raise ValueError("page count must be non-negative")
        return count <= len(self._free)

    def allocate(self, count: int) -> tuple[int, ...]:
        """Allocate ``count`` pages or leave allocator state unchanged."""

        if count < 0:
            raise ValueError("page count must be non-negative")
        if count > len(self._free):
            raise RuntimeError(
                f"insufficient free pages: requested {count}, available {len(self._free)}"
            )
        pages = tuple(self._free.popleft() for _ in range(count))
        self._allocated.update(pages)
        return pages

    def release(self, pages: Iterable[int]) -> None:
        """Release owned pages to the end of the FIFO free list atomically."""

        page_tuple = self._validate_release(pages)
        self._allocated.difference_update(page_tuple)
        self._free.extend(page_tuple)

    def _restore(self, pages: Iterable[int]) -> None:
        """Undo the most recent transactional allocation, preserving FIFO order."""

        page_tuple = self._validate_release(pages)
        self._allocated.difference_update(page_tuple)
        for page in reversed(page_tuple):
            self._free.appendleft(page)

    def _validate_release(self, pages: Iterable[int]) -> tuple[int, ...]:
        page_tuple = tuple(pages)
        if len(set(page_tuple)) != len(page_tuple):
            raise ValueError("released pages must be unique")
        if any(page < 0 or page >= self._num_pages for page in page_tuple):
            raise ValueError("released page is out of range")
        if any(page not in self._allocated for page in page_tuple):
            raise ValueError("cannot release a page that is not allocated")
        return page_tuple

    def snapshot(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return deterministic allocator state for tests and reports."""

        return tuple(self._free), tuple(sorted(self._allocated))


@dataclass(frozen=True)
class ScheduledBatch:
    """One homogeneous transactional prefill or decode iteration.

    ``block_tables`` is rectangular and padded with ``-1``.  ``slot_mappings``
    contains only the tokens scheduled in this iteration, so prefill rows may
    have different lengths while decode rows always contain one slot.
    """

    batch_id: int
    phase: Literal["prefill", "decode"]
    request_ids: tuple[str, ...]
    token_counts: tuple[int, ...]
    start_positions: tuple[int, ...]
    sequence_lengths: tuple[int, ...]
    block_tables: tuple[tuple[int, ...], ...]
    slot_mappings: tuple[tuple[int, ...], ...]
    _prior_pages: tuple[tuple[int, ...], ...] = field(repr=False, compare=False)
    _allocated_pages: tuple[int, ...] = field(repr=False, compare=False)
    _was_queued: tuple[bool, ...] = field(repr=False, compare=False)


class ContinuousBatchingScheduler:
    """Transactional FIFO scheduler for fixed-page inference experiments.

    Prefill consumes a whole prompt in one iteration.  Decode emits exactly one
    token for every active request.  New requests are admitted only when no
    batch is in flight; a replacement prefill batch is preferred over decode
    when active capacity is available.
    """

    def __init__(self, *, page_size: int, num_pages: int, max_batch_size: int) -> None:
        if page_size <= 0 or num_pages <= 0 or max_batch_size <= 0:
            raise ValueError("page_size, num_pages, and max_batch_size must be positive")
        self.page_size = page_size
        self.max_batch_size = max_batch_size
        self.allocator = FixedPageAllocator(num_pages)
        self._records: dict[str, _SequenceRecord] = {}
        self._queued: deque[str] = deque()
        self._inflight: ScheduledBatch | None = None
        self._next_batch_id = 0

    @property
    def inflight_batch(self) -> ScheduledBatch | None:
        return self._inflight

    @property
    def queued_request_ids(self) -> tuple[str, ...]:
        return tuple(self._queued)

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(
            request_id
            for request_id, record in self._records.items()
            if record.status is SequenceStatus.ACTIVE
        )

    def submit(self, request_id: str, *, prompt_length: int, max_new_tokens: int) -> None:
        """Append a new request to the admission FIFO."""

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if request_id in self._records:
            raise ValueError(f"request {request_id!r} already exists")
        if prompt_length <= 0 or max_new_tokens <= 0:
            raise ValueError("prompt_length and max_new_tokens must be positive")
        maximum_pages = self._pages_for_length(prompt_length + max_new_tokens)
        if maximum_pages > self.allocator.num_pages:
            raise ValueError("one request cannot exceed the scheduler page capacity")
        self._records[request_id] = _SequenceRecord(
            request_id=request_id,
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
        )
        self._queued.append(request_id)

    def state(self, request_id: str) -> SequenceState:
        try:
            return self._records[request_id].snapshot()
        except KeyError as error:
            raise ValueError(f"unknown request {request_id!r}") from error

    def snapshot(self) -> tuple[object, ...]:
        """Return the externally meaningful state, excluding monotonic batch ids."""

        return (
            self.allocator.snapshot(),
            tuple(self._queued),
            tuple(record.snapshot() for record in self._records.values()),
            None if self._inflight is None else self._inflight.batch_id,
        )

    def schedule(self) -> ScheduledBatch | None:
        """Reserve one homogeneous iteration without committing sequence lengths."""

        if self._inflight is not None:
            raise RuntimeError("a scheduled batch is already in flight")

        active = [
            record for record in self._records.values() if record.status is SequenceStatus.ACTIVE
        ]
        available_slots = self.max_batch_size - len(active)
        queued = [self._records[request_id] for request_id in list(self._queued)[:available_slots]]
        if queued:
            needed = sum(self._pages_for_length(record.prompt_length) for record in queued)
            if self.allocator.can_allocate(needed):
                return self._reserve(queued, phase="prefill")
            if not active:
                self.allocator.allocate(needed)  # raises without mutation and supplies detail
        if active:
            return self._reserve(active, phase="decode")
        return None

    def complete(
        self,
        batch: ScheduledBatch,
        *,
        finished_request_ids: Iterable[str] = (),
    ) -> tuple[str, ...]:
        """Commit a batch and release requests that reached EOS or token limit."""

        self._validate_current_batch(batch)
        early_finished = tuple(finished_request_ids)
        if len(set(early_finished)) != len(early_finished):
            raise ValueError("finished_request_ids must be unique")
        if batch.phase != "decode" and early_finished:
            raise ValueError("only decode batches can report early completion")
        unknown = set(early_finished).difference(batch.request_ids)
        if unknown:
            raise ValueError("finished_request_ids must belong to the in-flight batch")

        finished: list[str] = []
        for request_id, committed_length in zip(
            batch.request_ids,
            batch.sequence_lengths,
            strict=True,
        ):
            record = self._records[request_id]
            record.committed_length = committed_length
            record.reserved_length = None
            if batch.phase == "decode":
                record.generated_tokens += 1
                reached_limit = record.generated_tokens >= record.max_new_tokens
                if request_id in early_finished or reached_limit:
                    record.status = SequenceStatus.FINISHED
                    self.allocator.release(record.page_ids)
                    record.page_ids.clear()
                    finished.append(request_id)
        self._inflight = None
        return tuple(finished)

    def abort(self, batch: ScheduledBatch) -> None:
        """Roll back every page and lifecycle reservation made by ``schedule``."""

        self._validate_current_batch(batch)
        self.allocator._restore(batch._allocated_pages)
        queued_to_restore: list[str] = []
        for request_id, prior_pages, was_queued in zip(
            batch.request_ids,
            batch._prior_pages,
            batch._was_queued,
            strict=True,
        ):
            record = self._records[request_id]
            record.page_ids[:] = prior_pages
            record.reserved_length = None
            if was_queued:
                record.status = SequenceStatus.QUEUED
                queued_to_restore.append(request_id)
        for request_id in reversed(queued_to_restore):
            self._queued.appendleft(request_id)
        self._inflight = None

    def cancel(self, request_id: str) -> bool:
        """Cancel queued or idle active work; in-flight work must resolve first."""

        try:
            record = self._records[request_id]
        except KeyError as error:
            raise ValueError(f"unknown request {request_id!r}") from error
        if record.status in {SequenceStatus.FINISHED, SequenceStatus.CANCELLED}:
            return False
        if self._inflight is not None and request_id in self._inflight.request_ids:
            raise RuntimeError(
                "abort or complete the in-flight batch before cancelling its request"
            )
        if record.status is SequenceStatus.QUEUED:
            self._queued.remove(request_id)
        else:
            self.allocator.release(record.page_ids)
            record.page_ids.clear()
        record.status = SequenceStatus.CANCELLED
        record.reserved_length = None
        return True

    def _reserve(
        self,
        records: list[_SequenceRecord],
        *,
        phase: Literal["prefill", "decode"],
    ) -> ScheduledBatch:
        token_counts = tuple(
            record.prompt_length if phase == "prefill" else 1 for record in records
        )
        start_positions = tuple(record.committed_length for record in records)
        sequence_lengths = tuple(
            start + count for start, count in zip(start_positions, token_counts, strict=True)
        )
        prior_pages = tuple(tuple(record.page_ids) for record in records)
        needed_per_record = tuple(
            self._pages_for_length(length) - len(record.page_ids)
            for record, length in zip(
                records,
                sequence_lengths,
                strict=True,
            )
        )
        allocated_pages = self.allocator.allocate(sum(needed_per_record))

        cursor = 0
        was_queued: list[bool] = []
        for record, needed, reserved_length in zip(
            records,
            needed_per_record,
            sequence_lengths,
            strict=True,
        ):
            record.page_ids.extend(allocated_pages[cursor : cursor + needed])
            cursor += needed
            queued = record.status is SequenceStatus.QUEUED
            was_queued.append(queued)
            if queued:
                popped = self._queued.popleft()
                if popped != record.request_id:
                    raise AssertionError("FIFO admission invariant violated")
                record.status = SequenceStatus.ACTIVE
            record.reserved_length = reserved_length

        max_pages = max((len(record.page_ids) for record in records), default=0)
        block_tables = tuple(
            tuple(record.page_ids) + (-1,) * (max_pages - len(record.page_ids))
            for record in records
        )
        slot_mappings = tuple(
            tuple(self._physical_slot(record.page_ids, position) for position in range(start, end))
            for record, start, end in zip(
                records,
                start_positions,
                sequence_lengths,
                strict=True,
            )
        )
        batch = ScheduledBatch(
            batch_id=self._next_batch_id,
            phase=phase,
            request_ids=tuple(record.request_id for record in records),
            token_counts=token_counts,
            start_positions=start_positions,
            sequence_lengths=sequence_lengths,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            _prior_pages=prior_pages,
            _allocated_pages=allocated_pages,
            _was_queued=tuple(was_queued),
        )
        self._next_batch_id += 1
        self._inflight = batch
        return batch

    def _validate_current_batch(self, batch: ScheduledBatch) -> None:
        if self._inflight is not batch:
            raise RuntimeError("batch is stale or is not the current in-flight transaction")

    def _pages_for_length(self, length: int) -> int:
        return (length + self.page_size - 1) // self.page_size

    def _physical_slot(self, page_ids: list[int], logical_position: int) -> int:
        logical_page, offset = divmod(logical_position, self.page_size)
        return page_ids[logical_page] * self.page_size + offset
