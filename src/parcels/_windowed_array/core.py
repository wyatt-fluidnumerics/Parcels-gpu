from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Hashable, Iterable
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
from dask.base import is_dask_collection
from xarray.core.indexing import (
    BasicIndexer,
    ExplicitlyIndexedNDArrayMixin,
    OuterIndexer,
    VectorizedIndexer,
    as_indexable,
)

from parcels._chunk_cached_array import ChunkCachedArray
from parcels._core.utils.array_backend import (
    NUMPY_PLACEMENT,
    ArrayPlacement,
    host_array_for_conversion,
    is_array,
    placement_of_library,
    to_numpy,
)
from parcels._core.warnings import FieldSetWarning
from parcels._prefetch import Prefetcher, PrefetcherFactory, ThreadPrefetcher

if TYPE_CHECKING:
    import dask.array
    import xarray as xr

# Slots allocated up front for a window without max_levels: the two levels bracketing a shared clock.
# The window grows when a request needs more.
_DEFAULT_WINDOW_SLOTS = 2
_EMPTY_SLOT = -1


def wrap_dataset(
    ds: xr.Dataset,
    field_names: Iterable[Hashable],
    *,
    array_library: ModuleType | None = None,
    device: Any = None,
    max_levels: int | None = None,
    prefetcher: PrefetcherFactory | None = None,
) -> xr.Dataset:
    """Replace the storage of the given field variables with WindowedArray wrappers.

    Returns a shallow copy of the dataset. Dask-backed variables with a leading
    ``time`` dimension get a windowed ``WindowedArray`` that keeps a rolling window
    of time levels in the array library. Every other field is left as it is when
    its data is already returned in that library and device (with NumPy: eager
    NumPy fields, and lazy fields without a leading ``time`` dimension, which are
    computed to NumPy when gathered); otherwise it is converted to the library in
    full. Coordinate variables are loaded eagerly into host memory.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset (not modified).
    field_names : iterable of hashable
        Names of the data variables to wrap.
    array_library : module, optional
        Array library to hold the data in, e.g. ``numpy``, ``cupy`` or ``torch``. Defaults to NumPy.
    device : optional
        Device within ``array_library``, passed to it untouched.
    max_levels : int, optional
        Number of time levels kept in the window per windowed variable; see
        :class:`WindowedArray`.
    prefetcher : callable, optional
        Prefetcher factory; see :class:`WindowedArray`.

    Returns
    -------
    xr.Dataset
        Copy with the field variables wrapped in WindowedArray.
    """
    placement = placement_of_library(array_library, device)
    ds = ds.copy()
    for name in list(ds.coords):
        ds[name].load()
    for name in field_names:
        var = ds.variables[name]
        if isinstance(var._data, WindowedArray):
            if var._data.array_placement != placement:
                raise ValueError(
                    f"Field {name!r} is already held in a WindowedArray in {var._data.array_placement}, "
                    f"so it cannot be held in {placement} as well."
                )
            continue
        source = var._data.array if isinstance(var._data, ChunkCachedArray) else var._data
        windowed = is_dask_collection(source) and var.ndim > 0 and var.dims[0] == "time"
        if not windowed and _placement_of_field_data(source) == placement:
            continue
        var._data = WindowedArray(  # type: ignore[assignment]
            source if is_dask_collection(source) else to_numpy(var.data),
            windowed=windowed,
            array_library=array_library,
            device=device,
            max_levels=max_levels,
            prefetcher=prefetcher,
        )
    return ds


def _placement_of_field_data(data: Any) -> ArrayPlacement:
    """Where gathering from a variable's storage returns data, as ``ModelData.array_placement`` decides it."""
    placement = getattr(data, "array_placement", None)
    if placement is None and is_array(data) and not is_dask_collection(data):
        placement = ArrayPlacement.of(data)
    return NUMPY_PLACEMENT if placement is None else placement


class _TimeLevelReader:
    """Reads one level of the leading axis of a (lazy) array into host memory.

    A separate picklable object, so that prefetchers can read in other processes.
    """

    def __init__(self, array: np.ndarray | dask.array.Array) -> None:
        self.array = array

    def __call__(self, level: int) -> np.ndarray:
        values = self.array[level]
        if is_dask_collection(values):
            values = values.compute()
        return host_array_for_conversion(values)


class WindowedArray(ExplicitlyIndexedNDArrayMixin):
    """Array whose leading (time) axis is held as a rolling window of levels in an array library.

    Implements xarray's ExplicitlyIndexed protocol so it can be used as a
    drop-in replacement for the array in ``da.data``. Xarray's ``.isel()`` with
    vectorized indexers routes through ``_vindex_get``, which gathers from the
    window and returns an array of the library. ``_raw_vindex`` is the same gather
    for index arrays already in the library.

    A windowed WindowedArray holds a window of levels of the leading axis. On each
    gather it makes resident the distinct levels the gather asks for, and evicts
    every level outside the span between the lowest and the highest of them. With
    ``max_levels``, levels inside that span that were not asked for are evicted as
    well, lowest first, until at most ``max_levels`` remain; a gather that asks for
    more than ``max_levels`` levels gets them all, with a warning. After loading
    levels, the window asks its prefetcher to read the levels the clock reaches
    next: those after the highest requested level when the requests move forward,
    those before the lowest when they move backward.

    A WindowedArray that is not windowed holds every level.

    The gather maps level indices to window slots and then indexes the window
    buffer with integer arrays, so indices behave as the library's own integer
    array indexing does (in NumPy, negative indices count from the end and indices
    out of range raise ``IndexError``). Level indices of a windowed array are
    checked against the leading axis and may be negative.

    Parameters
    ----------
    array : np.ndarray or dask.array.Array
        Host array holding the data.
    windowed : bool
        Whether to keep a rolling window of levels instead of the whole array.
    array_library : module, optional
        Array library to hold and return the data in, e.g. ``numpy``, ``cupy``, ``torch`` or
        ``array_api_strict``. Defaults to NumPy.
    device : optional
        Device within ``array_library``, passed to it untouched.
    max_levels : int, optional
        Bound on the number of levels kept in the window. Defaults to no bound: each
        gather keeps the span of levels it asks for.
    prefetcher : callable, optional
        Prefetcher factory, called with the function that reads one level into host
        memory, e.g. :class:`~parcels._prefetch.ThreadPrefetcher` (the default),
        a ``functools.partial`` of it or of
        :class:`~parcels._prefetch.DataLoaderPrefetcher` with options set, or
        :class:`~parcels._prefetch.NoPrefetcher` to read levels on demand only.
    """

    def __init__(
        self,
        array: np.ndarray | dask.array.Array,
        *,
        windowed: bool,
        array_library: ModuleType | None = None,
        device: Any = None,
        max_levels: int | None = None,
        prefetcher: PrefetcherFactory | None = None,
    ) -> None:
        if array.ndim < 1:
            raise ValueError("WindowedArray needs an array with at least one dimension.")
        self.array = array
        self._is_windowed = windowed
        self._max = max_levels
        self._array_library = array_library
        self._device = device
        self._prefetcher_factory = prefetcher
        self._placement = placement_of_library(array_library, device)
        xp = self._placement.namespace
        self._read_level = _TimeLevelReader(array)

        # diagnostics
        self.loads = 0
        self.bytes_read = 0
        self._level_bytes = int(np.prod(array.shape[1:])) * array.dtype.itemsize

        n_levels = array.shape[0]
        self._lowest_level_of_previous_request: int | None = None
        self._prefetcher: Prefetcher | None = None

        if windowed:
            n_slots = min(n_levels, max_levels if max_levels is not None else _DEFAULT_WINDOW_SLOTS)
            dtype = self._placement.asarray(np.empty(0, dtype=np.dtype(array.dtype).newbyteorder("="))).dtype
            self._buffer = xp.empty((n_slots, *array.shape[1:]), dtype=dtype, device=self._placement.device)
            self._level_held_by_slot = np.full(n_slots, _EMPTY_SLOT, dtype=np.int64)
            self._slot_holding_level = xp.full(n_levels, _EMPTY_SLOT, dtype=xp.int64, device=self._placement.device)
            self._prefetcher = (ThreadPrefetcher if prefetcher is None else prefetcher)(self._read_level)
        else:
            host_values = array.compute() if is_dask_collection(array) else array
            self._buffer = self._placement.asarray(host_array_for_conversion(host_values))
            self._level_held_by_slot = np.arange(n_levels, dtype=np.int64)
            self.loads = n_levels
            self.bytes_read = n_levels * self._level_bytes

    @property
    def array_placement(self) -> ArrayPlacement:
        return self._placement

    @property
    def prefetch_hits(self) -> int:
        """Number of levels loaded into the window that the prefetcher had read ahead."""
        return 0 if self._prefetcher is None else self._prefetcher.hits

    @property
    def resident_levels(self) -> list[int]:
        """Levels of the leading axis currently held, in ascending order."""
        return sorted(int(level) for level in self._level_held_by_slot if level != _EMPTY_SLOT)

    def __repr__(self) -> str:
        return (
            f"WindowedArray(shape={self.shape}, dtype={self.dtype}, windowed={self._is_windowed}, "
            f"placement={self._placement}, resident_levels={self.resident_levels}, loads={self.loads})"
        )

    def close(self) -> None:
        """Stop reading levels ahead."""
        if self._prefetcher is not None:
            self._prefetcher.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __deepcopy__(self, memo: dict) -> WindowedArray:
        # The prefetcher holds threads or processes, so a copy starts from an empty window of its own.
        return type(self)(
            copy.deepcopy(self.array, memo),
            windowed=self._is_windowed,
            array_library=self._array_library,
            device=self._device,
            max_levels=self._max,
            prefetcher=self._prefetcher_factory,
        )

    def get_duck_array(self):
        if not self._is_windowed:
            return self._buffer
        host_values = self.array.compute() if is_dask_collection(self.array) else self.array
        return self._placement.asarray(host_array_for_conversion(host_values))

    # --- window management ---

    def _grow_window(self, n_slots: int) -> None:
        xp = self._placement.namespace
        n_old_slots = self._buffer.shape[0]
        grown = xp.empty((n_slots, *self._buffer.shape[1:]), dtype=self._buffer.dtype, device=self._placement.device)
        grown[:n_old_slots, ...] = self._buffer
        self._buffer = grown
        self._level_held_by_slot = np.concatenate(
            [self._level_held_by_slot, np.full(n_slots - n_old_slots, _EMPTY_SLOT, dtype=np.int64)]
        )

    def _moving_backward(self, lowest_level: int, highest_level: int) -> bool:
        """Whether the requests move toward lower levels, judged from the previous request that loaded levels."""
        if self._lowest_level_of_previous_request is None:
            # The first request shows no direction yet. A run that starts on the last level can only go backward.
            moving_backward = highest_level + 1 >= self.array.shape[0]
        else:
            moving_backward = lowest_level < self._lowest_level_of_previous_request
        self._lowest_level_of_previous_request = lowest_level
        return moving_backward

    def _ensure(self, requested: np.ndarray) -> None:
        """Make the sorted, distinct levels ``requested`` resident and evict the levels the policy drops."""
        if self._max is not None and requested.size > self._max:
            # The request must be granted, as the gather would otherwise index levels that are not
            # held, even though this can take far more memory than max_levels allows. The most likely
            # reason for this to occur is non-synchronous particle clocks.
            warnings.warn(
                f"The windowed array cache is attempting to hold {requested.size} time levels "
                f"which exceeds max_level={self._max}; the cache will hold {requested.size} to maintain "
                f"simulation accuracy. This may cause significant memory usage or an OOM error. "
                f"This most likely occured due to non-synchronous particle clockes. Raise max "
                f"levels or narrow the spread of particle times.",
                FieldSetWarning,
                stacklevel=4,
            )

        lowest_level, highest_level = int(requested[0]), int(requested[-1])
        held = self._level_held_by_slot
        keep = (held >= lowest_level) & (held <= highest_level)
        if self._max is not None:
            slots_in_span = np.flatnonzero(keep)
            slots_in_span = slots_in_span[np.argsort(held[slots_in_span])]
            slots_not_requested = slots_in_span[~np.isin(held[slots_in_span], requested)]
            spare = max(self._max - requested.size, 0)
            n_drop = max(slots_not_requested.size - spare, 0)
            keep[slots_not_requested[:n_drop]] = False
        for slot in np.flatnonzero((held != _EMPTY_SLOT) & ~keep):
            self._slot_holding_level[int(held[slot])] = _EMPTY_SLOT
            held[slot] = _EMPTY_SLOT

        missing = requested[~np.isin(requested, held)]
        if missing.size == 0:
            return
        free_slots = np.flatnonzero(held == _EMPTY_SLOT)
        if free_slots.size < missing.size:
            self._grow_window(held.size + missing.size - free_slots.size)
            free_slots = np.flatnonzero(self._level_held_by_slot == _EMPTY_SLOT)

        prefetcher = self._prefetcher
        assert prefetcher is not None
        moving_backward = self._moving_backward(lowest_level, highest_level)
        # Levels are taken in the order the clock reaches them, which is the order prefetchers read them in.
        load_order = missing[::-1] if moving_backward else missing
        for level, slot in zip(load_order, free_slots[: missing.size], strict=True):
            level, slot = int(level), int(slot)
            self._buffer[slot, ...] = self._placement.asarray(prefetcher.take(level))
            self._level_held_by_slot[slot] = level
            self._slot_holding_level[level] = slot
            self.loads += 1
            self.bytes_read += self._level_bytes

        if moving_backward:
            prefetcher.request(range(lowest_level - 1, -1, -1))
        else:
            prefetcher.request(range(highest_level + 1, self.array.shape[0]))

    def _level_extremes(self, level_index: Any) -> tuple[int, int]:
        """Lowest and highest level index, fetched with one host round trip."""
        xp = self._placement.namespace
        extremes = to_numpy(xp.stack([xp.min(level_index), xp.max(level_index)]))
        return int(extremes[0]), int(extremes[1])

    def _window_slots(self, level_index: Any) -> Any:
        """Map an array of level indices to the window slots holding those levels, loading levels as needed."""
        xp = self._placement.namespace
        n_levels = self.array.shape[0]
        lowest, highest = self._level_extremes(level_index)
        if lowest < -n_levels or highest >= n_levels:
            out_of_bounds = lowest if lowest < -n_levels else highest
            raise IndexError(f"index {out_of_bounds} is out of bounds for axis 0 with size {n_levels}")
        if lowest < 0:
            level_index = xp.where(level_index < 0, level_index + n_levels, level_index)
            lowest, highest = self._level_extremes(level_index)

        if highest - lowest <= 1:
            # the lowest and highest levels are the only ones asked for
            requested = np.unique(np.array([lowest, highest], dtype=np.int64))
        else:
            requested = np.unique(to_numpy(xp.unique_values(level_index))).astype(np.int64)
        self._ensure(requested)
        return self._slot_holding_level[level_index]

    # --- indexing ---

    def _raw_vindex(self, *indices) -> Any:
        """Vectorized indexing in the array library.

        Parameters
        ----------
        *indices : arrays of the array library
            One integer index array per dimension. The arrays are broadcast against each other.

        Returns
        -------
        array of the array library
            Array with the broadcast shape of ``indices`` holding the selected values.
        """
        assert len(indices) == self.ndim
        indices = tuple(self._placement.asarray(index) for index in indices)
        out_shape = np.broadcast_shapes(*(tuple(index.shape) for index in indices))
        if math.prod(out_shape) == 0:
            return self._placement.namespace.empty(out_shape, dtype=self._buffer.dtype, device=self._placement.device)

        leading_index = indices[0]
        if self._is_windowed:
            leading_index = self._window_slots(leading_index)
        return self._buffer[(leading_index, *indices[1:])]

    def _orthogonal_index(self, key: tuple) -> Any:
        """Index with one integer, slice or 1D integer array per dimension, each acting on its own axis."""
        key = tuple(key) + (slice(None),) * (self.ndim - len(key))
        if self._is_windowed and isinstance(key[0], slice):
            # A range of levels is not what the window holds, so it is read from the source, lazily.
            return as_indexable(self.array).oindex[OuterIndexer(key)]
        xp = self._placement.namespace
        per_axis = [
            self._placement.asarray(np.arange(n)[k] if isinstance(k, slice) else k)
            for k, n in zip(key, self.shape, strict=True)
        ]
        n_kept_axes = sum(index.ndim for index in per_axis)
        grid = []
        kept_axis = 0
        for index in per_axis:
            shape = [1] * n_kept_axes
            if index.ndim == 1:
                shape[kept_axis] = index.shape[0]
                kept_axis += 1
            grid.append(xp.reshape(index, tuple(shape)))
        return self._raw_vindex(*grid)

    # --- ExplicitlyIndexed protocol ---

    def _vindex_get(self, indexer: VectorizedIndexer):
        # Dimensions not given an indexer in ``.isel()`` (e.g. a size-1 ``mockZ``)
        # arrive as slices. Under vectorized indexing semantics the output has the
        # broadcast dims of the array indexers first, then one dim per slice. Expand
        # each slice to an index array over its own trailing dim to match.
        xp = self._placement.namespace
        key = indexer.tuple
        n_slices = sum(isinstance(k, slice) for k in key)
        indices = []
        slice_axis = 0
        for k, n in zip(key, self.shape, strict=True):
            if isinstance(k, slice):
                shape = [1] * n_slices
                shape[slice_axis] = -1
                slice_axis += 1
                indices.append(np.arange(n)[k].reshape(shape))
            else:
                index = self._placement.asarray(k)
                indices.append(xp.reshape(index, (*index.shape, *(1,) * n_slices)))
        return self._raw_vindex(*indices)

    def _oindex_get(self, indexer: OuterIndexer):
        return self._orthogonal_index(indexer.tuple)

    def __getitem__(self, indexer):
        if isinstance(indexer, VectorizedIndexer):
            return self._vindex_get(indexer)
        if isinstance(indexer, (OuterIndexer, BasicIndexer)):
            return self._orthogonal_index(indexer.tuple)
        return self._orthogonal_index(indexer if isinstance(indexer, tuple) else (indexer,))
