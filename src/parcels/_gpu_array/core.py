from __future__ import annotations

import functools
import warnings
from collections.abc import Hashable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from dask.base import is_dask_collection
from xarray.core.indexing import BasicIndexer, ExplicitlyIndexedNDArrayMixin, OuterIndexer, VectorizedIndexer

from parcels._chunk_cached_array import ChunkCachedArray
from parcels._core.utils.array_backend import import_cupy
from parcels._core.warnings import FieldSetWarning

if TYPE_CHECKING:
    import dask.array
    import xarray as xr

_DEFAULT_WINDOW_LEVELS = 2
_EMPTY_SLOT = -1


def wrap_dataset(ds: xr.Dataset, field_names: Iterable[Hashable], max_levels: int | None = None) -> xr.Dataset:
    """Replace the storage of the given field variables with GpuArray wrappers.

    Returns a shallow copy of the dataset. Each field variable's internal
    ``variable._data`` is swapped for a ``GpuArray`` that holds the data on the
    GPU. NumPy-backed variables are copied to the GPU in full. Dask-backed
    variables with a leading ``time`` dimension keep a rolling window of time
    levels on the GPU; other dask-backed variables are computed and copied in
    full. Coordinate variables are loaded eagerly into host memory.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset (not modified).
    field_names : iterable of hashable
        Names of the data variables to wrap.
    max_levels : int, optional
        Number of time levels kept on the GPU per windowed variable. Defaults to
        two.

    Returns
    -------
    xr.Dataset
        Copy with the field variables wrapped in GpuArray.
    """
    ds = ds.copy()
    for name in list(ds.coords):
        ds[name].load()
    for name in field_names:
        var = ds.variables[name]
        if isinstance(var._data, GpuArray):
            continue
        source = var._data.array if isinstance(var._data, ChunkCachedArray) else var._data
        if is_dask_collection(source):
            has_leading_time = var.ndim > 0 and var.dims[0] == "time"
            var._data = GpuArray(source, windowed=has_leading_time, max_levels=max_levels)  # type: ignore[assignment]
        else:
            var._data = GpuArray(np.asarray(var.values), windowed=False)  # type: ignore[assignment]
    return ds


@dataclass(frozen=True)
class BracketedBuffer:
    """A GpuArray's buffer and, for each point, the slots in it of two consecutive levels.

    ``buffer[lower_slot]`` and ``buffer[upper_slot]`` hold those levels. A slot is a plain
    int when every point shares the level, and a CuPy int64 array otherwise.
    """

    buffer: Any
    lower_slot: Any
    upper_slot: Any


def lowest_and_highest(index: Any) -> tuple[int, int]:
    """Lowest and highest value of a non-empty CuPy index array, fetched with one host round trip."""
    cupy = import_cupy()
    lowest, highest = cupy.stack([index.min(), index.max()]).get()
    return int(lowest), int(highest)


@functools.cache
def _gather_kernel(ndim: int):
    """Elementwise GPU kernel that gathers ``buffer[i0, i1, ...]`` for broadcast index arrays.

    Negative indices count from the end of their axis, as in NumPy. Indices that are
    still out of range after that are clamped to the axis, because raising would need a
    round trip to the host on every gather.
    """
    cupy = import_cupy()
    index_params = ", ".join(f"int64 i{axis}" for axis in range(ndim))
    size_params = ", ".join(f"int64 n{axis}" for axis in range(ndim))
    body = ["long long flat_index = 0;", "long long axis_index;"]
    for axis in range(ndim):
        body.append(f"axis_index = i{axis} < 0 ? i{axis} + n{axis} : i{axis};")
        body.append(f"axis_index = axis_index < 0 ? 0 : (axis_index >= n{axis} ? n{axis} - 1 : axis_index);")
        body.append(f"flat_index = flat_index * n{axis} + axis_index;")
    body.append("out = buffer[flat_index];")
    return cupy.ElementwiseKernel(
        f"raw T buffer, {index_params}, {size_params}", "T out", "\n".join(body), f"parcels_gather_{ndim}d"
    )


class GpuArray(ExplicitlyIndexedNDArrayMixin):
    """Array held on the GPU (as CuPy arrays) and indexed there.

    Implements xarray's ExplicitlyIndexed protocol so it can be used as
    a drop-in replacement for the array in ``da.data``. Xarray's ``.isel()``
    with vectorized indexers will route through ``_vindex_get``, which gathers
    on the GPU and returns a CuPy array. Index arrays may be NumPy or CuPy.

    The leading axis is the level axis (time, for field data). A windowed
    GpuArray keeps only ``max_levels`` levels on the GPU: each index operation
    loads the levels it asks for from the host array and evicts the others. A
    request that spans more levels than the window holds makes the window grow
    to that span. A GpuArray that is not windowed holds every level.

    Parameters
    ----------
    array : np.ndarray or dask.array.Array
        Host array holding the data.
    windowed : bool
        Whether to keep a rolling window of levels on the GPU instead of the
        whole array.
    max_levels : int, optional
        Number of levels a windowed GpuArray keeps on the GPU. Defaults to two,
        the levels bracketing the current time when all particles share a clock.
    """

    prefetch_next_level = True
    """Whether windowed GpuArrays read the next level ahead of time in a background thread."""

    def __init__(self, array: np.ndarray | dask.array.Array, *, windowed: bool, max_levels: int | None = None) -> None:
        if array.ndim < 1:
            raise ValueError("GpuArray needs an array with at least one dimension.")
        self._cupy = import_cupy()
        self.array = array
        self._is_windowed = windowed
        self._max = max_levels

        # diagnostics
        self.loads = 0
        self.bytes_read = 0
        self._level_bytes = int(np.prod(array.shape[1:])) * array.dtype.itemsize

        n_levels = array.shape[0]
        # Reading ahead: while the GPU works on the levels it holds, a background thread reads the
        # level the clock will reach next into host memory.
        self.prefetch_hits = 0
        self._prefetched: dict[int, Future] = {}
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._lowest_level_of_previous_request: int | None = None

        if windowed:
            n_slots = min(n_levels, max_levels if max_levels is not None else _DEFAULT_WINDOW_LEVELS)
            self._buffer = self._cupy.empty((n_slots, *array.shape[1:]), dtype=array.dtype)
            self._level_held_by_slot = np.full(n_slots, _EMPTY_SLOT, dtype=np.int64)
            self._slot_holding_level = self._cupy.full(n_levels, _EMPTY_SLOT, dtype=np.int64)
        else:
            host_values = array.compute() if is_dask_collection(array) else array
            self._buffer = self._cupy.asarray(host_values)
            self._level_held_by_slot = np.arange(n_levels, dtype=np.int64)
            self.loads = n_levels
            self.bytes_read = n_levels * self._level_bytes

    def __repr__(self) -> str:
        return (
            f"GpuArray(shape={self.shape}, dtype={self.dtype}, windowed={self._is_windowed}, "
            f"levels_on_gpu={self.levels_on_gpu}, loads={self.loads})"
        )

    @property
    def levels_on_gpu(self) -> list[int]:
        return sorted(int(level) for level in self._level_held_by_slot if level != _EMPTY_SLOT)

    def get_duck_array(self):
        if self._is_windowed:
            raise NotImplementedError(
                "Full materialisation of a windowed GpuArray is not supported, as only a window of its "
                "leading dimension is held on the GPU."
            )
        return self._buffer

    # --- window management ---

    def _read_level(self, level: int) -> np.ndarray:
        host_values = self.array[level]
        if is_dask_collection(host_values):
            host_values = host_values.compute()
        return np.ascontiguousarray(host_values)

    def _grow_window(self, n_slots: int) -> None:
        n_old_slots = self._buffer.shape[0]
        grown = self._cupy.empty((n_slots, *self._buffer.shape[1:]), dtype=self._buffer.dtype)
        grown[:n_old_slots] = self._buffer
        self._buffer = grown
        self._level_held_by_slot = np.concatenate(
            [self._level_held_by_slot, np.full(n_slots - n_old_slots, _EMPTY_SLOT, dtype=np.int64)]
        )

    def _ensure(self, lowest_level: int, highest_level: int) -> None:
        """Make every level in ``[lowest_level, highest_level]`` resident on the GPU."""
        requested = np.arange(lowest_level, highest_level + 1)
        missing = requested[~np.isin(requested, self._level_held_by_slot)]
        if missing.size == 0:
            return

        if requested.size > self._level_held_by_slot.size:
            if self._max is not None and requested.size > self._max:
                warnings.warn(
                    f"The GpuArray window is attempting to hold {requested.size} levels "
                    f"which exceeds max_levels={self._max}; the window will hold {requested.size} to maintain "
                    f"simulation accuracy. This may exhaust GPU memory. "
                    f"This most likely occurred due to non-synchronous particle clocks. Raise max_levels "
                    f"or narrow the spread of particle times.",
                    FieldSetWarning,
                    stacklevel=3,
                )
            self._grow_window(requested.size)

        reusable_slots = np.flatnonzero(~np.isin(self._level_held_by_slot, requested))
        for level, slot in zip(missing, reusable_slots[: missing.size], strict=True):
            evicted_level = int(self._level_held_by_slot[slot])
            if evicted_level != _EMPTY_SLOT:
                self._slot_holding_level[evicted_level] = _EMPTY_SLOT
            self._buffer[int(slot)].set(self._prefetched_or_read_level(int(level)))
            self._level_held_by_slot[slot] = level
            self._slot_holding_level[int(level)] = int(slot)
            self.loads += 1
            self.bytes_read += self._level_bytes

        self._prefetch_level_after(lowest_level, highest_level)

    def _prefetched_or_read_level(self, level: int) -> np.ndarray:
        prefetched = self._prefetched.pop(level, None)
        if prefetched is None:
            return self._read_level(level)
        self.prefetch_hits += 1
        return prefetched.result()

    def _prefetch_level_after(self, lowest_level: int, highest_level: int) -> None:
        """Start reading the level the clock reaches next, in the direction the requests are moving."""
        if not self.prefetch_next_level:
            return
        n_levels = self.array.shape[0]
        if self._lowest_level_of_previous_request is None:
            # The first request shows no direction yet. A run that starts on the last level can only go backward.
            moving_backward = highest_level + 1 >= n_levels
        else:
            moving_backward = lowest_level < self._lowest_level_of_previous_request
        self._lowest_level_of_previous_request = lowest_level
        next_level = lowest_level - 1 if moving_backward else highest_level + 1

        for level in [level for level in self._prefetched if level != next_level]:
            self._prefetched.pop(level).cancel()
        if not 0 <= next_level < n_levels or next_level in self._prefetched:
            return
        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parcels-gpu-prefetch")
        self._prefetched[next_level] = self._prefetch_executor.submit(self._read_level, next_level)

    def _level_extremes(self, level_indices: list) -> np.ndarray:
        """Lowest and highest index of each array of level indices, fetched with one host round trip."""
        extremes = []
        for level_index in level_indices:
            extremes += [level_index.min(), level_index.max()]
        return self._cupy.stack(extremes).get()

    def _window_slots(self, *level_indices) -> list:
        """Map each array of level indices to the window slots holding those levels, loading levels as needed."""
        cp = self._cupy
        n_levels = self.array.shape[0]
        level_indices = [cp.asarray(level_index) for level_index in level_indices]
        extremes = self._level_extremes(level_indices)
        lowest, highest = int(extremes.min()), int(extremes.max())
        if lowest < -n_levels or highest >= n_levels:
            out_of_bounds = lowest if lowest < -n_levels else highest
            raise IndexError(f"index {out_of_bounds} is out of bounds for axis 0 with size {n_levels}")
        if lowest < 0:
            level_indices = [cp.where(index < 0, index + n_levels, index) for index in level_indices]
            extremes = self._level_extremes(level_indices)
            lowest, highest = int(extremes.min()), int(extremes.max())

        self._ensure(lowest, highest)
        slots = []
        for position, level_index in enumerate(level_indices):
            lowest_in_array, highest_in_array = int(extremes[2 * position]), int(extremes[2 * position + 1])
            if lowest_in_array == highest_in_array:
                # every point asks for the same level, so its slot is a scalar
                slots.append(int(np.flatnonzero(self._level_held_by_slot == lowest_in_array)[0]))
            else:
                slots.append(self._slot_holding_level[level_index])
        return slots

    def _slot_holding(self, level: int) -> int:
        if not self._is_windowed:
            return level
        return int(np.flatnonzero(self._level_held_by_slot == level)[0])

    def bracketed_buffer(
        self,
        level_index,
        n_levels_clipped_to: int | None = None,
        level_extremes: tuple[int, int] | None = None,
    ) -> BracketedBuffer:
        """Give GPU kernels direct access to the data around each point's level.

        Parameters
        ----------
        level_index : np.ndarray or cupy.ndarray
            Non-negative indices along the leading (level) axis: the lower level of each point.
            The upper level is the next one.
        n_levels_clipped_to : int, optional
            The upper level is clipped to ``n_levels_clipped_to - 1``. Defaults to the number
            of levels of this array.
        level_extremes : tuple of int, optional
            Lowest and highest value in ``level_index``, from ``lowest_and_highest``. Passing
            it saves the host round trip that finds them, which matters when several arrays
            are sampled with the same ``level_index``.

        Returns
        -------
        BracketedBuffer
            The buffer and the slots in it of each point's lower and upper level.

        Notes
        -----
        The levels are loaded if needed and are all resident when this returns. A later
        request may evict them, so the slots are only valid until then.
        """
        cp = self._cupy
        n_levels = self.array.shape[0]
        last_upper_level = (n_levels if n_levels_clipped_to is None else n_levels_clipped_to) - 1

        level_index = cp.asarray(level_index)
        if level_index.size == 0:
            return BracketedBuffer(buffer=self._buffer, lower_slot=0, upper_slot=0)
        if level_extremes is None:
            level_extremes = (0, 0) if n_levels == 1 else lowest_and_highest(level_index)
        lowest, highest = level_extremes
        highest_requested = max(highest, min(highest + 1, last_upper_level))
        if lowest < 0 or highest_requested >= n_levels:
            out_of_bounds = lowest if lowest < 0 else highest_requested
            raise IndexError(f"index {out_of_bounds} is out of bounds for axis 0 with size {n_levels}")

        if self._is_windowed:
            self._ensure(lowest, highest_requested)
        if lowest == highest:
            # every point shares the two levels, so their slots are scalars
            return BracketedBuffer(
                buffer=self._buffer,
                lower_slot=self._slot_holding(lowest),
                upper_slot=self._slot_holding(min(lowest + 1, last_upper_level)),
            )

        lower_level = level_index.astype(np.int64, copy=False)
        upper_level = cp.minimum(lower_level + 1, last_upper_level)
        if not self._is_windowed:
            return BracketedBuffer(buffer=self._buffer, lower_slot=lower_level, upper_slot=upper_level)
        return BracketedBuffer(
            buffer=self._buffer,
            lower_slot=self._slot_holding_level[lower_level],
            upper_slot=self._slot_holding_level[upper_level],
        )

    # --- indexing ---

    def _raw_vindex(self, *indices) -> Any:
        """Vectorized indexing on the GPU.

        Parameters
        ----------
        *indices : np.ndarray or cupy.ndarray
            One integer index array per dimension. The arrays are broadcast against each other.

        Returns
        -------
        cupy.ndarray
            Array with the broadcast shape of ``indices`` holding the selected values.
        """
        cp = self._cupy
        assert len(indices) == self.ndim
        indices = tuple(cp.asarray(index).astype(np.int64, copy=False) for index in indices)
        out_shape = np.broadcast_shapes(*(index.shape for index in indices))
        if int(np.prod(out_shape)) == 0:
            return cp.empty(out_shape, dtype=self._buffer.dtype)

        leading_index = indices[0]
        if self._is_windowed:
            (leading_index,) = self._window_slots(leading_index)
            if isinstance(leading_index, int) and self.ndim == 1:
                leading_index = cp.full(out_shape, leading_index, dtype=np.int64)

        return _gather_kernel(self.ndim)(self._buffer, leading_index, *indices[1:], *self._buffer.shape)

    def _orthogonal_index(self, key: tuple) -> Any:
        """Index with one integer, slice or 1D integer array per dimension, each acting on its own axis."""
        cp = self._cupy
        key = tuple(key) + (slice(None),) * (self.ndim - len(key))
        per_axis = [
            cp.arange(n)[k] if isinstance(k, slice) else cp.asarray(k) for k, n in zip(key, self.shape, strict=True)
        ]
        n_kept_axes = sum(index.ndim for index in per_axis)
        grid = []
        kept_axis = 0
        for index in per_axis:
            shape = [1] * n_kept_axes
            if index.ndim == 1:
                shape[kept_axis] = index.size
                kept_axis += 1
            grid.append(index.reshape(shape))
        return self._raw_vindex(*grid)

    # --- ExplicitlyIndexed protocol ---

    def _vindex_get(self, indexer: VectorizedIndexer):
        # Dimensions not given an indexer in ``.isel()`` (e.g. a size-1 ``mockZ``)
        # arrive as slices. Under vectorized indexing semantics the output has the
        # broadcast dims of the array indexers first, then one dim per slice. Expand
        # each slice to an index array over its own trailing dim to match.
        cp = self._cupy
        key = indexer.tuple
        n_slices = sum(isinstance(k, slice) for k in key)
        trailing = (np.newaxis,) * n_slices
        indices = []
        slice_axis = 0
        for k, n in zip(key, self.shape, strict=True):
            if isinstance(k, slice):
                shape = [1] * n_slices
                shape[slice_axis] = -1
                slice_axis += 1
                indices.append(cp.arange(n)[k].reshape(shape))
            else:
                indices.append(cp.asarray(k)[(..., *trailing)])
        return self._raw_vindex(*indices)

    def _oindex_get(self, indexer: OuterIndexer):
        return self._orthogonal_index(indexer.tuple)

    def __getitem__(self, indexer):
        if isinstance(indexer, VectorizedIndexer):
            return self._vindex_get(indexer)
        if isinstance(indexer, (OuterIndexer, BasicIndexer)):
            return self._orthogonal_index(indexer.tuple)
        return self._orthogonal_index(indexer if isinstance(indexer, tuple) else (indexer,))
