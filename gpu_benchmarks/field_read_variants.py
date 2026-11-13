"""Ways of reading one time level of a dask-backed field into host memory, timed on real data (SMOC).

Variants, each applied to the lazy (time, Z, Y, X) array a GpuArray wraps:
  threaded    : level.compute(), dask's default threaded scheduler
  synchronous : level.compute(scheduler="synchronous")
  per chunk   : every dask block of the level computed synchronously into a preallocated array

The variants take turns on each level in a rotating order, after every level has been read once, so
no variant systematically pays for a cold file cache.

Usage: python gpu_benchmarks/field_read_variants.py
"""

import time

import numpy as np
from smoc_gather_comparison import build_fieldset

TIMED_LEVELS = range(3, 27)


def read_threaded(lazy_field, level: int) -> np.ndarray:
    return lazy_field[level].compute()


def read_synchronous(lazy_field, level: int) -> np.ndarray:
    return lazy_field[level].compute(scheduler="synchronous")


def read_per_chunk(lazy_field, level: int) -> np.ndarray:
    lazy_level = lazy_field[level]
    host_values = np.empty(lazy_level.shape, dtype=lazy_level.dtype)
    chunk_starts = [np.concatenate(([0], np.cumsum(sizes))) for sizes in lazy_level.chunks]
    for block_index in np.ndindex(*lazy_level.numblocks):
        where = tuple(slice(starts[i], starts[i + 1]) for starts, i in zip(chunk_starts, block_index, strict=True))
        host_values[where] = lazy_level.blocks[block_index].compute(scheduler="synchronous")
    return host_values


VARIANTS = {"threaded": read_threaded, "synchronous": read_synchronous, "per chunk": read_per_chunk}

if __name__ == "__main__":
    fieldset = build_fieldset("gpu", backend="windowed")
    lazy_field = fieldset.models[0].data.variables["U"]._data.array
    for level in TIMED_LEVELS:
        read_threaded(lazy_field, level)

    seconds: dict[str, list[float]] = {name: [] for name in VARIANTS}
    names = list(VARIANTS)
    for position, level in enumerate(TIMED_LEVELS):
        reference = None
        for offset in range(len(names)):
            name = names[(position + offset) % len(names)]
            start = time.perf_counter()
            values = VARIANTS[name](lazy_field, level)
            seconds[name].append(time.perf_counter() - start)
            if reference is None:
                reference = values
            else:
                np.testing.assert_array_equal(values, reference)
    for name, values in seconds.items():
        print(
            f"{name:>12s}: median {1e3 * np.median(values):6.1f} ms  mean {1e3 * np.mean(values):6.1f}  min {1e3 * np.min(values):6.1f}  max {1e3 * np.max(values):6.1f}"
        )
