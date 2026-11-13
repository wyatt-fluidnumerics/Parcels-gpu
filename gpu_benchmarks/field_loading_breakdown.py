"""Where the time goes when a GpuArray loads one time level of real data (SMOC surface currents).

Stages timed for each of several time levels of the U field, in the order they happen:
  file read, raw     : the HDF5 read and decompression alone, through netCDF4 with masking switched off
  file read, decoded : the same read with netCDF4's fill-value masking, as xarray's decoding does
  dask compute       : `gpu_array.array[level].compute()`, what GpuArray._read_level runs: the task
                       graph xarray and Parcels built (open, decode, transpose, fillna(0))
  contiguous         : np.ascontiguousarray of that result
  copy to GPU        : `buffer[slot].set(host_values)` followed by a device synchronize

The levels are read once beforehand, so the file data is in the operating system's page cache and
the times are for decoding, not for the disk.

Usage: python gpu_benchmarks/field_loading_breakdown.py
"""

import glob
import time
from collections.abc import Callable

import cupy
import netCDF4
import numpy as np
from smoc_gather_comparison import SMOC_GLOB, build_fieldset, open_smoc

LEVELS_PER_FILE = 24
TIMED_LEVELS = range(3, 11)


def seconds_of(function: Callable[[], object]) -> tuple[float, object]:
    start = time.perf_counter()
    result = function()
    return time.perf_counter() - start, result


def read_level_from_file(level: int, decode: bool) -> np.ndarray:
    path = sorted(glob.glob(SMOC_GLOB))[level // LEVELS_PER_FILE]
    with netCDF4.Dataset(path) as dataset:
        variable = dataset["utotal"]
        variable.set_auto_maskandscale(decode)
        return variable[level % LEVELS_PER_FILE, 0]


if __name__ == "__main__":
    encoding = open_smoc()["utotal"].encoding
    print(
        "file encoding of utotal:",
        {key: encoding.get(key) for key in ("zlib", "complevel", "shuffle", "chunksizes", "dtype")},
    )

    fieldset = build_fieldset("gpu", backend="windowed")
    gpu_array = fieldset.models[0].data.variables["U"]._data
    lazy_field = gpu_array.array
    print(
        f"field shape {lazy_field.shape}, dtype {lazy_field.dtype}, dask chunks along (Y, X): {lazy_field.chunks[2:]}"
    )
    print(f"dask tasks behind one time level: {len(dict(lazy_field[TIMED_LEVELS[0]].__dask_graph__()))}")

    for level in TIMED_LEVELS:
        read_level_from_file(level, decode=False)

    stage_seconds: dict[str, list[float]] = {
        "file read, raw": [],
        "file read, decoded": [],
        "dask compute": [],
        "contiguous": [],
        "copy to GPU": [],
    }
    for level in TIMED_LEVELS:
        seconds, _ = seconds_of(lambda: read_level_from_file(level, decode=False))
        stage_seconds["file read, raw"].append(seconds)
        seconds, _ = seconds_of(lambda: read_level_from_file(level, decode=True))
        stage_seconds["file read, decoded"].append(seconds)
        seconds, host_values = seconds_of(lambda: lazy_field[level].compute())
        stage_seconds["dask compute"].append(seconds)
        seconds, contiguous_values = seconds_of(lambda: np.ascontiguousarray(host_values))
        stage_seconds["contiguous"].append(seconds)

        def copy_to_gpu():
            gpu_array._buffer[0].set(contiguous_values)
            cupy.cuda.runtime.deviceSynchronize()

        seconds, _ = seconds_of(copy_to_gpu)
        stage_seconds["copy to GPU"].append(seconds)

    level_megabytes = lazy_field[0].nbytes / 1e6
    print(f"one time level is {level_megabytes:.1f} MB; mean over {len(TIMED_LEVELS)} levels:")
    for stage, seconds in stage_seconds.items():
        print(
            f"  {stage:>20s}: {1e3 * np.mean(seconds):7.1f} ms  (min {1e3 * np.min(seconds):6.1f}, max {1e3 * np.max(seconds):6.1f})"
        )
