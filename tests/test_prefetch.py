"""Tests of the prefetchers: the interface, the thread pool, the no-op reader and the PyTorch DataLoader.

Tests that start DataLoader worker processes are marked ``slow`` (deselect with ``-m "not slow"``), as each
worker process imports PyTorch and Parcels.
"""

import functools
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from parcels._prefetch import DataLoaderPrefetcher, NoPrefetcher, Prefetcher, ThreadPrefetcher
from parcels._windowed_array import WindowedArray
from parcels._windowed_array.core import _TimeLevelReader

TIMEOUT_S = 10


class RecordingRead:
    """Fake read function: returns ``np.full(size, key)``, records which keys it read, and can be held up per key."""

    def __init__(self, size: int = 4, held_keys=(), failing_keys=()):
        self.size = size
        self.read_keys = []
        self.started = {}
        self.release = {key: threading.Event() for key in held_keys}
        self.failing_keys = set(failing_keys)
        self._lock = threading.Lock()

    def started_event(self, key) -> threading.Event:
        with self._lock:
            return self.started.setdefault(key, threading.Event())

    def __call__(self, key):
        with self._lock:
            self.read_keys.append(key)
        self.started_event(key).set()
        if key in self.release:
            assert self.release[key].wait(TIMEOUT_S), f"read of {key!r} was never released"
        if key in self.failing_keys:
            raise ValueError(f"cannot read {key!r}")
        return np.full(self.size, key, dtype=np.float64)


def wait_until(condition, timeout_s: float = TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.001)


# ------------------------------------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------------------------------------
def test_implementations_satisfy_the_interface():
    read = RecordingRead()
    implementations = [ThreadPrefetcher(read), NoPrefetcher(read)]
    if _has_torch():
        implementations.append(DataLoaderPrefetcher(read, num_workers=0))
    for prefetcher in implementations:
        assert isinstance(prefetcher, Prefetcher)
        prefetcher.request([1])
        np.testing.assert_array_equal(np.asarray(prefetcher.take(1)), np.full(4, 1.0))
        assert prefetcher.hits + prefetcher.misses == 1
        prefetcher.close()
        prefetcher.close()  # closing twice is harmless


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


# ------------------------------------------------------------------------------------------------
# NoPrefetcher
# ------------------------------------------------------------------------------------------------
def test_no_prefetcher_reads_on_demand_only():
    read = RecordingRead()
    with NoPrefetcher(read) as prefetcher:
        prefetcher.request([1, 2, 3])
        assert read.read_keys == []
        np.testing.assert_array_equal(prefetcher.take(2), np.full(4, 2.0))
    assert read.read_keys == [2]
    assert (prefetcher.hits, prefetcher.misses, prefetcher.wait_seconds) == (0, 1, 0.0)


# ------------------------------------------------------------------------------------------------
# ThreadPrefetcher
# ------------------------------------------------------------------------------------------------
def test_thread_prefetcher_reads_the_first_wanted_keys_in_order():
    read = RecordingRead(held_keys=[0, 1, 2, 3])
    with ThreadPrefetcher(read, max_workers=1, max_keys_ahead=2, max_bytes_ahead=None) as prefetcher:
        prefetcher.request([0, 1, 2, 3])
        assert sorted(prefetcher.keys_ahead) == [0, 1]
        for key in [0, 1, 2, 3]:
            read.release[key].set()
            np.testing.assert_array_equal(prefetcher.take(key), np.full(4, float(key)))
        # one thread reads the wanted keys one after the other, in the order they were requested
        assert read.read_keys == [0, 1, 2, 3]
        assert (prefetcher.hits, prefetcher.misses) == (4, 0)


def test_thread_prefetcher_taking_a_key_starts_the_next_wanted_one():
    read = RecordingRead()
    with ThreadPrefetcher(read, max_keys_ahead=1, max_bytes_ahead=None) as prefetcher:
        prefetcher.request(range(5))
        assert prefetcher.keys_ahead == [0]
        prefetcher.take(0)
        assert prefetcher.keys_ahead == [1]
        assert read.started_event(1).wait(TIMEOUT_S)


def test_thread_prefetcher_drops_keys_no_longer_wanted():
    read = RecordingRead(held_keys=[1])
    with ThreadPrefetcher(read, max_workers=1, max_keys_ahead=3, max_bytes_ahead=None) as prefetcher:
        prefetcher.request([1, 2, 3])
        assert read.started_event(1).wait(TIMEOUT_S)  # 1 is being read, 2 and 3 wait for the thread
        prefetcher.request([5])
        assert prefetcher.keys_ahead == [5]
        read.release[1].set()
        np.testing.assert_array_equal(prefetcher.take(5), np.full(4, 5.0))
        assert prefetcher.hits == 1
        # the queued reads of 2 and 3 were cancelled; 1 was running and is discarded
        assert read.read_keys == [1, 5]
        prefetcher.take(1)
        assert prefetcher.misses == 1
        assert read.read_keys == [1, 5, 1]


def test_thread_prefetcher_bounds_the_bytes_held_ahead():
    key_bytes = 8 * 1000
    read = RecordingRead(size=1000)
    with ThreadPrefetcher(read, max_keys_ahead=10, max_bytes_ahead=int(2.5 * key_bytes)) as prefetcher:
        prefetcher.take(-1)  # a miss, which tells the prefetcher how large a result is
        prefetcher.request(range(10))
        assert sorted(prefetcher.keys_ahead) == [0, 1]
        prefetcher.take(0)
        assert sorted(prefetcher.keys_ahead) == [1, 2]

    # the first wanted key is read ahead even when it alone exceeds the bound
    with ThreadPrefetcher(read, max_keys_ahead=10, max_bytes_ahead=key_bytes // 2) as prefetcher:
        prefetcher.take(-1)
        prefetcher.request(range(10))
        assert prefetcher.keys_ahead == [0]


def test_thread_prefetcher_bounds_arrays_without_nbytes():
    xp = pytest.importorskip("array_api_strict")
    key_bytes = 8 * 1000

    def read(key):
        return xp.full(1000, float(key), dtype=xp.float64)  # array-api-strict arrays have no nbytes

    with ThreadPrefetcher(read, max_keys_ahead=10, max_bytes_ahead=int(2.5 * key_bytes)) as prefetcher:
        prefetcher.take(-1)
        prefetcher.request(range(10))
        assert sorted(prefetcher.keys_ahead) == [0, 1]


def test_thread_prefetcher_counts_hits_misses_and_waiting():
    read = RecordingRead(held_keys=[1])
    with ThreadPrefetcher(read, max_keys_ahead=2, max_bytes_ahead=None) as prefetcher:
        prefetcher.request([0, 1])
        assert read.started_event(1).wait(TIMEOUT_S)
        prefetcher.take(0)
        threading.Timer(0.05, read.release[1].set).start()
        prefetcher.take(1)  # in flight, so take waits for it
        prefetcher.take(7)  # never requested
        assert (prefetcher.hits, prefetcher.misses) == (2, 1)
        assert prefetcher.wait_seconds >= 0.04


def test_thread_prefetcher_raises_read_errors_in_take():
    read = RecordingRead(failing_keys=[2, 9])
    with ThreadPrefetcher(read, max_bytes_ahead=None) as prefetcher:
        prefetcher.request([1, 2])
        np.testing.assert_array_equal(prefetcher.take(1), np.full(4, 1.0))
        with pytest.raises(ValueError, match="cannot read 2"):
            prefetcher.take(2)
        with pytest.raises(ValueError, match="cannot read 9"):
            prefetcher.take(9)
        prefetcher.request([3])
        np.testing.assert_array_equal(prefetcher.take(3), np.full(4, 3.0))


def test_thread_prefetcher_reads_on_several_threads_at_once():
    n_workers = 3
    all_reading = threading.Barrier(n_workers, timeout=TIMEOUT_S)

    def read(key):
        all_reading.wait()  # passes only when n_workers reads run at the same time
        return np.full(2, key)

    with ThreadPrefetcher(read, max_workers=n_workers, max_keys_ahead=n_workers, max_bytes_ahead=None) as prefetcher:
        prefetcher.request(range(n_workers))
        for key in range(n_workers):
            np.testing.assert_array_equal(prefetcher.take(key), np.full(2, key))
        assert prefetcher.hits == n_workers


def test_thread_prefetcher_closes_without_waiting_for_reads():
    read = RecordingRead(held_keys=[0])
    prefetcher = ThreadPrefetcher(read, max_workers=1, max_keys_ahead=2, max_bytes_ahead=None)
    prefetcher.request([0, 1])
    assert read.started_event(0).wait(TIMEOUT_S)
    started_closing = time.perf_counter()
    prefetcher.close()
    assert time.perf_counter() - started_closing < 1.0
    read.release[0].set()
    # a closed prefetcher still works, reading on demand
    prefetcher.request([2])
    np.testing.assert_array_equal(prefetcher.take(2), np.full(4, 2.0))
    assert prefetcher.misses == 1
    assert 1 not in read.read_keys


# ------------------------------------------------------------------------------------------------
# DataLoaderPrefetcher
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def levels() -> np.ndarray:
    return np.arange(6 * 3 * 4, dtype=np.float32).reshape(6, 3, 4)


def test_dataloader_prefetcher_in_the_calling_process(torch, levels):
    read = _TimeLevelReader(da.from_array(levels, chunks=(1, 3, 4)))
    with DataLoaderPrefetcher(read, num_workers=0) as prefetcher:
        prefetcher.request(range(6))
        for level in range(3):
            taken = prefetcher.take(level)
            assert isinstance(taken, torch.Tensor)
            np.testing.assert_array_equal(taken.numpy(), levels[level])
        assert (prefetcher.hits, prefetcher.misses, prefetcher.restarts) == (3, 0, 1)

        # the same order continues the iterator; another order restarts it
        prefetcher.request(range(3, 6))
        assert prefetcher.restarts == 1
        prefetcher.request(range(4, -1, -1))
        assert prefetcher.restarts == 2
        np.testing.assert_array_equal(prefetcher.take(4).numpy(), levels[4])

        # a key out of order is read in the calling thread
        np.testing.assert_array_equal(prefetcher.take(0).numpy(), levels[0])
        assert (prefetcher.hits, prefetcher.misses) == (4, 1)

        # at the end of the time axis there is nothing left to request
        prefetcher.request([])
        np.testing.assert_array_equal(prefetcher.take(5).numpy(), levels[5])
        assert prefetcher.misses == 2


def test_dataloader_prefetcher_raises_read_errors_in_take(torch, levels):
    read = _TimeLevelReader(levels)
    with DataLoaderPrefetcher(read, num_workers=0) as prefetcher:
        prefetcher.request([0, 10])
        prefetcher.take(0)
        with pytest.raises(IndexError):
            prefetcher.take(10)


@pytest.mark.slow
def test_dataloader_prefetcher_worker_reads_levels_in_order_and_shuts_down(torch, levels):
    read = _TimeLevelReader(da.from_array(levels, chunks=(1, 3, 4)))
    prefetcher = DataLoaderPrefetcher(read, num_workers=1, prefetch_factor=2)
    prefetcher.request(range(6))
    taken = [prefetcher.take(level).numpy() for level in range(6)]
    np.testing.assert_array_equal(np.stack(taken), levels)
    assert (prefetcher.hits, prefetcher.misses, prefetcher.restarts) == (6, 0, 1)

    # a change of direction restarts the same worker on the new order
    prefetcher.request(range(4, -1, -1))
    taken = [prefetcher.take(level).numpy() for level in range(4, -1, -1)]
    np.testing.assert_array_equal(np.stack(taken), levels[4::-1])
    assert (prefetcher.hits, prefetcher.restarts) == (11, 2)

    # an error in the worker surfaces in take
    prefetcher.request([1, 10])
    prefetcher.take(1)
    with pytest.raises(IndexError):
        prefetcher.take(10)

    workers = list(prefetcher._loader._iterator._workers)
    prefetcher.close()
    assert not any(worker.is_alive() for worker in workers)


@pytest.mark.slow
def test_dataloader_prefetcher_reports_a_killed_worker_and_stops_the_others(torch, levels):
    read = _TimeLevelReader(da.from_array(levels, chunks=(1, 3, 4)))
    prefetcher = DataLoaderPrefetcher(read, num_workers=2, prefetch_factor=1)
    prefetcher.request(list(range(6)) * 4)
    prefetcher.take(0)
    workers = list(prefetcher._loader._iterator._workers)
    os.kill(workers[0].pid, signal.SIGKILL)

    with pytest.raises(RuntimeError, match="killed for using too much memory"):
        for level in list(range(1, 6)) + list(range(6)) * 3:
            prefetcher.take(level)
    assert not any(worker.is_alive() for worker in workers)
    prefetcher.close()


@pytest.mark.slow
def test_dataloader_prefetcher_worker_reads_lazily_opened_netcdf(torch, tmp_path, levels):
    path = tmp_path / "levels.nc"
    xr.Dataset({"U": (("time", "y", "x"), levels)}).to_netcdf(path)
    with xr.open_dataset(path, chunks={"time": 1}) as ds:
        lazy = ds["U"].data
        assert isinstance(lazy, da.Array)
        windowed = WindowedArray(lazy, windowed=True, prefetcher=functools.partial(DataLoaderPrefetcher, num_workers=1))
        spatial = (np.array([0, 2]), np.array([1, 3]))
        for lowest in range(5):
            time_index = np.array([lowest, lowest + 1])
            np.testing.assert_array_equal(windowed._raw_vindex(time_index, *spatial), levels[time_index, *spatial])
        assert windowed.loads == 6
        # The first request reads both its levels on demand; every later level comes from the worker.
        assert windowed.prefetch_hits == 4
        windowed.close()


@pytest.mark.slow
def test_dataloader_prefetcher_lets_the_interpreter_exit_without_close():
    script = textwrap.dedent(
        """
        import dask.array as da
        import numpy as np
        from parcels._prefetch import DataLoaderPrefetcher
        from parcels._windowed_array.core import _TimeLevelReader

        if __name__ == "__main__":
            prefetcher = DataLoaderPrefetcher(_TimeLevelReader(da.from_array(np.zeros((4, 2)), chunks=(1, 2))))
            prefetcher.request(range(4))
            prefetcher.take(0)
        """
    )
    completed = subprocess.run([sys.executable, "-c", script], timeout=120, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
