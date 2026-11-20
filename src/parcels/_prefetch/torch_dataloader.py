"""A prefetcher that reads ahead with a PyTorch ``DataLoader``, in worker processes.

PyTorch is imported only when a :class:`DataLoaderPrefetcher` is created.
"""

from __future__ import annotations

import multiprocessing
import time
from collections import deque
from collections.abc import Hashable, Iterator, Sequence
from types import ModuleType
from typing import Any, Self

from parcels._prefetch.core import ReadFunction

__all__ = ["DataLoaderPrefetcher"]


def _import_torch() -> ModuleType:
    try:
        import torch
        import torch.utils.data
    except ImportError as e:
        raise ImportError(
            "Reading data ahead with a PyTorch DataLoader requires PyTorch, which is not installed in this environment."
        ) from e
    return torch


class _KeyedDataset:
    """Map-style dataset whose item for a key is ``read(key)``.

    Keys are whatever the sampler yields, so the dataset has no length. It is sent to the
    DataLoader's worker processes, so ``read`` must be picklable.
    """

    def __init__(self, read: ReadFunction) -> None:
        self.read = read

    def __getitem__(self, key: Hashable) -> Any:
        return self.read(key)


class _KeyOrder:
    """Sampler yielding the keys of the latest request, in order. Restarted with each new iterator."""

    def __init__(self) -> None:
        self.keys: list[Hashable] = []

    def __iter__(self) -> Iterator[Hashable]:
        return iter(list(self.keys))

    def __len__(self) -> int:
        return len(self.keys)


def _use_synchronous_dask_scheduler(worker_id: int) -> None:
    """Make dask compute in the worker's own thread, so that workers do not each start a thread pool."""
    try:
        import dask
    except ImportError:
        return
    dask.config.set(scheduler="synchronous")


class DataLoaderPrefetcher:
    """Reads the keys wanted next with a ``torch.utils.data.DataLoader``.

    The DataLoader reads a map-style dataset whose item for a key is ``read(key)``, converted to a
    tensor, in the order of the latest ``request``. Its iterator is kept: ``take`` of the key the
    iterator yields next takes it from the loader (a hit), while a request that is not the iterator's
    remaining order (the first request, a jump, a change of direction) restarts the iterator on the
    new order. A key taken out of order is read in the calling thread (a miss).

    With ``num_workers > 0`` the reads run in persistent worker processes, each computing dask
    arrays with dask's synchronous scheduler. ``read`` is sent to those processes, so it must be
    picklable. With ``num_workers=0`` the loader reads in the calling thread when a key is taken,
    so nothing is read ahead; ``hits`` then counts the keys taken in the requested order.

    Parameters
    ----------
    read : callable
        Maps a key to a NumPy array (or a tensor).
    num_workers : int, optional
        Number of worker processes. Defaults to one.
    prefetch_factor : int, optional
        Keys read ahead per worker. Defaults to two. Ignored when ``num_workers`` is zero.
    pin_memory : bool, optional
        Return page-locked tensors, for faster copies to a CUDA device. Defaults to False.
    read_timeout_s : float, optional
        Longest time to wait for one key from the worker processes before raising. Defaults to zero,
        which waits without limit. A worker that has exited is noticed either way; the limit is for
        workers that are alive but stalled, for example when the node is out of memory.
    multiprocessing_context : str or multiprocessing context, optional
        How worker processes are started. Defaults to "forkserver" where the platform has it and
        "spawn" otherwise, never "fork": the parent runs threads (dask, PyTorch), and forked workers
        would share the file handles xarray has open. Workers then receive ``read`` by pickling, and
        a lazily opened NetCDF file is reopened in each worker.
    """

    def __init__(
        self,
        read: ReadFunction,
        *,
        num_workers: int = 1,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        read_timeout_s: float = 0.0,
        multiprocessing_context: Any = None,
    ) -> None:
        torch = _import_torch()
        self._torch = torch
        self._read = read
        self._key_order = _KeyOrder()
        uses_workers = num_workers > 0
        if multiprocessing_context is None:
            multiprocessing_context = (
                "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
            )
        self._loader = torch.utils.data.DataLoader(
            _KeyedDataset(read),
            batch_size=None,
            sampler=self._key_order,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if uses_workers else None,
            pin_memory=pin_memory,
            timeout=read_timeout_s if uses_workers else 0,
            persistent_workers=uses_workers,
            worker_init_fn=_use_synchronous_dask_scheduler if uses_workers else None,
            multiprocessing_context=multiprocessing_context if uses_workers else None,
        )
        self._iterator: Iterator[Any] | None = None
        self._keys_from_iterator: deque[Hashable] = deque()
        self._closed = False
        self.hits = 0
        self.misses = 0
        self.wait_seconds = 0.0
        self.restarts = 0
        """Number of times the loader's iterator was started on a new key order."""

    def request(self, keys: Sequence[Hashable]) -> None:
        if self._closed:
            return
        keys = list(keys)
        if keys == list(self._keys_from_iterator):
            return
        self._key_order.keys = keys
        self._keys_from_iterator = deque(keys)
        if keys:
            # With persistent workers this resets the existing iterator, which first drains the reads
            # still in flight for the previous order.
            self._iterator = iter(self._loader)
            self.restarts += 1
        else:
            self._iterator = None

    def take(self, key: Hashable) -> Any:
        if self._iterator is None or not self._keys_from_iterator or self._keys_from_iterator[0] != key:
            self.misses += 1
            return self._torch.utils.data.default_convert(self._read(key))
        self._keys_from_iterator.popleft()
        self.hits += 1
        started_waiting = time.perf_counter()
        try:
            return next(self._iterator)
        except RuntimeError as error:
            # PyTorch reports a worker process that exited, or a read that exceeded the time limit,
            # as a RuntimeError that names the DataLoader. The loader cannot continue after either, so
            # its workers are stopped. An error raised by ``read`` itself is passed on as it is.
            if "DataLoader" not in str(error):
                raise
            self.close()
            raise RuntimeError(
                f"The DataLoader prefetcher could not read key {key!r}: {error}. A worker process that exits "
                "unexpectedly has most often been killed for using too much memory; each worker holds its own "
                "copy of the reader and of the keys it has read ahead. Use fewer workers, a smaller "
                "prefetch_factor, or more memory."
            ) from error
        finally:
            self.wait_seconds += time.perf_counter() - started_waiting

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._keys_from_iterator.clear()
        self._iterator = None
        loader = self.__dict__.pop("_loader", None)
        persistent_iterator = getattr(loader, "_iterator", None)
        if persistent_iterator is not None:
            loader._iterator = None
            shutdown_workers = getattr(persistent_iterator, "_shutdown_workers", None)
            if shutdown_workers is not None:
                shutdown_workers()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
