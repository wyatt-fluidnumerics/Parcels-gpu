"""Reading data ahead of time, keyed by opaque hashable keys.

A prefetcher is handed a ``read`` function that maps a key to an array. A consumer tells it which
keys it will want soon (``request``) and later asks for one of them (``take``). Prefetchers know
nothing about what the keys mean (time levels, chunks, ...) or where the arrays end up; the
consumer decides what to ask for, the prefetcher how and when to read it.

A backend is given a prefetcher *factory*: a callable that takes the ``read`` function and returns
a prefetcher, such as one of the classes below or a ``functools.partial`` of one with its options
set.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Protocol, Self, runtime_checkable

from parcels._core.utils.array_backend import nbytes_of

__all__ = ["NoPrefetcher", "Prefetcher", "PrefetcherFactory", "ReadFunction", "ThreadPrefetcher"]

ReadFunction = Callable[[Any], Any]
"""Maps a key to an array."""


@runtime_checkable
class Prefetcher(Protocol):
    """What a backend needs from a prefetcher.

    ``hits`` counts ``take`` calls served by a read that was started ahead of time, ``misses`` those
    that had to read in the calling thread, and ``wait_seconds`` the time ``take`` spent waiting
    for reads that were started ahead of time but had not finished.
    """

    hits: int
    misses: int
    wait_seconds: float

    def request(self, keys: Sequence[Hashable]) -> None:
        """Declare the keys wanted soon, most urgent first, replacing any earlier request."""
        ...

    def take(self, key: Hashable) -> Any:
        """Return the array for ``key`` and forget it, reading it now if it was not read ahead."""
        ...

    def close(self) -> None:
        """Stop reading ahead and release threads or processes. Safe to call more than once."""
        ...


PrefetcherFactory = Callable[[ReadFunction], Prefetcher]


class NoPrefetcher:
    """Reads every key on demand, in the calling thread. Switches reading ahead off."""

    def __init__(self, read: ReadFunction) -> None:
        self._read = read
        self.hits = 0
        self.misses = 0
        self.wait_seconds = 0.0

    def request(self, keys: Sequence[Hashable]) -> None:
        pass

    def take(self, key: Hashable) -> Any:
        self.misses += 1
        return self._read(key)

    def close(self) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class ThreadPrefetcher:
    """Reads the keys wanted next on a pool of threads, holding a bounded amount of data ready.

    Of the keys most recently passed to ``request``, the first ``max_keys_ahead`` that have not been
    taken yet are read ahead, in order. Keys dropped from a later request are cancelled if their read
    has not started, and their data is discarded otherwise. Taking a key frees its place, so the next
    wanted key starts reading.

    Parameters
    ----------
    read : callable
        Maps a key to an array. Called on the worker threads, and in the calling thread for keys
        that were not read ahead.
    max_workers : int, optional
        Number of threads reading at the same time. Defaults to two, so that a read that is no
        longer wanted but cannot be cancelled does not hold up the next one.
    max_keys_ahead : int, optional
        Most keys read ahead (in flight or ready) at any time. Defaults to two.
    max_bytes_ahead : int or None, optional
        Bound on the data held ahead, in bytes: ready results plus the expected size of reads in
        flight (the largest result seen so far). The first wanted key is always read ahead, so a
        single key larger than the bound is still prefetched; only further keys are held back.
        ``None`` removes the bound. Defaults to 512 MiB.
    """

    def __init__(
        self,
        read: ReadFunction,
        *,
        max_workers: int = 2,
        max_keys_ahead: int = 2,
        max_bytes_ahead: int | None = 512 * 2**20,
    ) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}.")
        if max_keys_ahead < 0:
            raise ValueError(f"max_keys_ahead must not be negative, got {max_keys_ahead}.")
        self._read = read
        self._max_workers = max_workers
        self._max_keys_ahead = max_keys_ahead
        self._max_bytes_ahead = max_bytes_ahead
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._wanted: deque[Hashable] = deque()
        self._reads: dict[Hashable, Future] = {}
        self._largest_result_bytes = 0
        self._closed = False
        self.hits = 0
        self.misses = 0
        self.wait_seconds = 0.0

    def request(self, keys: Sequence[Hashable]) -> None:
        with self._lock:
            if self._closed:
                return
            self._wanted = deque(keys)
            wanted = set(self._wanted)
            for key in [key for key in self._reads if key not in wanted]:
                self._reads.pop(key).cancel()
            self._start_reads()

    def take(self, key: Hashable) -> Any:
        with self._lock:
            read = self._reads.pop(key, None)
            try:
                self._wanted.remove(key)
            except ValueError:
                pass
            if not self._closed:
                self._start_reads()
        if read is None:
            self.misses += 1
            return self._read_and_measure(key)
        self.hits += 1
        started_waiting = time.perf_counter()
        try:
            return read.result()
        finally:
            self.wait_seconds += time.perf_counter() - started_waiting

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._wanted.clear()
            for read in self._reads.values():
                read.cancel()
            self._reads.clear()
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def keys_ahead(self) -> list[Hashable]:
        """Keys currently being read or held ready, in no particular order."""
        with self._lock:
            return list(self._reads)

    def _bytes_ahead(self) -> int:
        held = 0
        for read in self._reads.values():
            if read.done() and not read.cancelled() and read.exception() is None:
                held += nbytes_of(read.result())
            else:
                held += self._largest_result_bytes
        return held

    def _read_and_measure(self, key: Hashable) -> Any:
        value = self._read(key)
        nbytes = nbytes_of(value)
        with self._lock:
            self._largest_result_bytes = max(self._largest_result_bytes, nbytes)
        return value

    def _start_reads(self) -> None:
        """Start reads for the first wanted keys, within the bounds. Called with the lock held."""
        for position, key in enumerate(self._wanted):
            if position >= self._max_keys_ahead:
                return
            if key in self._reads:
                continue
            if (
                position > 0
                and self._max_bytes_ahead is not None
                and self._bytes_ahead() + self._largest_result_bytes > self._max_bytes_ahead
            ):
                return
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._max_workers, thread_name_prefix="parcels-prefetch"
                )
            self._reads[key] = self._executor.submit(self._read_and_measure, key)
