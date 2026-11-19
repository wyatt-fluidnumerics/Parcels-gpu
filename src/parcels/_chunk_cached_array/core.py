from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
from xarray.core.indexing import BasicIndexer, ExplicitlyIndexedNDArrayMixin, OuterIndexer, VectorizedIndexer
from xarray.namedarray.pycompat import is_duck_array

from parcels._core.utils.array_backend import (
    NUMPY_PLACEMENT,
    ArrayPlacement,
    flatnonzero,
    namespace_of_library,
    ravel_multi_index,
    to_numpy,
    unravel_index,
)

from .lru_cache import ByteBoundedLRUCache

if TYPE_CHECKING:
    import dask.array
    import xarray as xr


def _placement_of_library(array_library: ModuleType | None, device: Any) -> ArrayPlacement:
    """Where arrays created in ``array_library`` on ``device`` live, with the device named as the library names it.

    ``None`` means NumPy. Naming the device the library's way lets placements reported by different
    backends, and by arrays themselves, compare equal.
    """
    probe = ArrayPlacement(namespace_of_library(array_library), device).asarray(np.empty(0, dtype=np.float64))
    return ArrayPlacement.of(probe)


def _host_array(values: Any) -> np.ndarray:
    """``values`` as a C-contiguous, writable NumPy array in native byte order, ready to hand to any array library."""
    values = np.asarray(values)
    if not values.dtype.isnative:
        values = values.astype(values.dtype.newbyteorder("="))
    return np.require(values, requirements=["C", "W"])


def wrap_dataset(
    ds: xr.Dataset,
    max_cache_bytes: int,
    *,
    array_library: ModuleType | None = None,
    device: Any = None,
) -> xr.Dataset:
    """Replace all dask-backed data variables with ChunkCachedArray wrappers.

    Returns a shallow copy of the dataset. Each dask-backed data variable's
    internal ``variable._data`` is swapped for a ``ChunkCachedArray`` that
    caches chunks on vectorized indexing. Coordinate variables are loaded
    eagerly into memory to avoid dask task-graph overhead on every
    ``.isel()`` call.

    With an array library other than NumPy, data variables that are not dask-backed
    (held in memory, or loaded lazily by xarray) are wrapped as well, split into
    one chunk per level of their leading dimension, so that every variable
    returns data in that library.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset (not modified).
    max_cache_bytes : int
        Maximum cache size in bytes, per variable.
    array_library : module, optional
        Array library that chunks are converted to and gathers return, e.g. ``numpy``,
        ``torch``, ``cupy`` or ``array_api_strict``. Defaults to NumPy.
    device : optional
        Device within ``array_library``, passed to it untouched.

    Returns
    -------
    xr.Dataset
        Copy with dask arrays wrapped in ChunkCachedArray.
    """
    from dask.base import is_dask_collection

    converts_other_data = _placement_of_library(array_library, device) != NUMPY_PLACEMENT
    ds = ds.copy()
    # Load coordinates eagerly — they are small 1D arrays and keeping them
    # as dask arrays causes expensive task-graph construction on every .isel().
    for name in list(ds.coords):
        ds[name].load()
    for name in ds.data_vars:
        var = ds.variables[name]
        if is_duck_array(var._data) and is_dask_collection(var._data):
            var._data = ChunkCachedArray(var._data, max_cache_bytes, array_library, device=device)  # type: ignore[assignment, arg-type]
        elif converts_other_data and var.ndim > 0 and getattr(var._data, "array_placement", None) is None:
            level_chunks = dict(zip(var.dims, (1, *var.shape[1:]), strict=True))
            var._data = ChunkCachedArray(var.chunk(level_chunks)._data, max_cache_bytes, array_library, device=device)  # type: ignore[assignment, arg-type]
    return ds


class ChunkCachedArray(ExplicitlyIndexedNDArrayMixin):
    """Chunk-level LRU cache on top of a dask array for vectorized indexing.

    Implements xarray's ExplicitlyIndexed protocol so it can be used as
    a drop-in replacement for the dask array in ``da.data``. Xarray's
    ``.isel()`` with vectorized indexers will route through ``_vindex_get``,
    which uses the chunk cache. Other indexing modes delegate to the
    underlying dask array.

    On each vectorized index:
      1. Maps global indices -> (chunk_coord, local_index) per dimension.
      2. Fetches missing chunks via dask_array.blocks[...].compute(), converting
         each to the array library once, before it is cached.
      3. Assembles the result from cached chunks, in the array library.

    Parameters
    ----------
    dask_array : dask.array.Array
        Lazy array holding the data.
    max_cache_bytes : int
        Maximum cache size in bytes.
    array_library : module, optional
        Array library that chunks are converted to and gathers return, e.g. ``numpy``,
        ``torch``, ``cupy`` or ``array_api_strict``. Defaults to NumPy.
    device : optional
        Device within ``array_library``, passed to it untouched.
    """

    def __init__(
        self,
        dask_array: dask.array.Array,
        max_cache_bytes: int,
        array_library: ModuleType | None = None,
        *,
        device: Any = None,
    ) -> None:
        self.array = dask_array
        self.cache = ByteBoundedLRUCache(max_cache_bytes)
        self._array_placement = _placement_of_library(array_library, device)

        # Precompute chunk boundaries per dimension.
        # _boundaries[d] is a 1D array of cumulative chunk sizes, e.g., [0, 15, 30].
        self._boundaries: list[np.ndarray] = []
        for dim_chunks in dask_array.chunks:
            self._boundaries.append(np.concatenate(([0], np.cumsum(dim_chunks))))
        self._boundaries_in_library = [self._array_placement.asarray(b) for b in self._boundaries]
        if self._array_placement == NUMPY_PLACEMENT:
            self._dtype_in_library = dask_array.dtype
        else:
            self._dtype_in_library = self._array_placement.asarray(_host_array(np.empty(0, dask_array.dtype))).dtype

    @property
    def array_placement(self) -> ArrayPlacement:
        return self._array_placement

    def get_duck_array(self):
        raise NotImplementedError(
            "Full materialisation of the ChunkCachedArray is not supported, as it has serious (negative) "
            "performance implications. See discussion in https://github.com/Parcels-code/Parcels/issues/2910. "
            "Feel free to continue that discussion if you're running into this error message in your simulations."
        )

    def _chunk(self, key: tuple[int, ...]) -> Any:
        chunk_data = self.cache.get(key)
        if chunk_data is None:
            chunk_data = self.array.blocks[key].compute()
            if self._array_placement != NUMPY_PLACEMENT:
                chunk_data = self._array_placement.asarray(_host_array(chunk_data))
            self.cache.put(key, chunk_data)
        return chunk_data

    def _raw_vindex(self, *indices: Any) -> Any:
        """Vectorized indexing with chunk caching.

        Parameters
        ----------
        *indices : arrays of the array library
            One integer index array per dimension. The arrays are broadcast against each other.

        Returns
        -------
        array of the array library
            Array with the broadcast shape of ``indices`` holding the selected values.
        """
        ndim = len(self.array.chunks)
        assert len(indices) == ndim
        xp = self._array_placement.namespace

        # Step 0: The index arrays broadcast to a list of points. Their broadcast shape is
        # restored at the end.
        indices = tuple(self._array_placement.asarray(idx) for idx in indices)
        out_shape = np.broadcast_shapes(*(tuple(idx.shape) for idx in indices))
        n_points = int(np.prod(out_shape))
        if n_points == 0:
            return xp.empty(out_shape, dtype=self._dtype_in_library, device=self._array_placement.device)

        # Step 1: Map global indices to chunk coords and local indices.
        # Normalize negative indices (e.g. -1 → last element) to positive,
        # matching standard numpy fancy-indexing semantics.
        # Each dimension is mapped on its own index array, which can be much smaller
        # than the list of points, and only the result is broadcast to the points.
        out_of_bounds_per_dim = []
        chunk_ids = []
        local_indices = []
        for d, idx in enumerate(indices):
            size = self.array.shape[d]
            out_of_bounds_per_dim.append((idx < -size) | (idx >= size))
            idx = xp.astype(idx, xp.int64, copy=False)
            idx = xp.where(idx < 0, idx + size, idx)
            # Out-of-bounds points give chunk ids in [-1, numblocks], which still index the
            # boundaries, so the error can wait for the single host round trip below.
            cid = xp.searchsorted(self._boundaries_in_library[d], idx, side="right") - 1
            chunk_ids.append(cid)
            local_indices.append(idx - self._boundaries_in_library[d][cid])

        # One host round trip tells whether any index is out of bounds and which chunks are touched.
        summary = [xp.astype(xp.any(out_of_bounds), xp.int64) for out_of_bounds in out_of_bounds_per_dim]
        summary += [xp.min(cid) for cid in chunk_ids] + [xp.max(cid) for cid in chunk_ids]
        summary = to_numpy(xp.stack(summary)).tolist()
        for d, out_of_bounds in enumerate(out_of_bounds_per_dim):
            if summary[d]:
                size = self.array.shape[d]
                raise IndexError(
                    f"index {int(indices[d][out_of_bounds][0])} is out of bounds for axis {d} with size {size}"
                )
        lowest_chunk_ids, highest_chunk_ids = summary[ndim : 2 * ndim], summary[2 * ndim :]
        if lowest_chunk_ids == highest_chunk_ids:
            # Every point lies in the same chunk; its index arrays broadcast to the points themselves.
            return xp.reshape(self._chunk(tuple(lowest_chunk_ids))[tuple(local_indices)], out_shape)

        # Step 2: Group points by chunk. Encode each point's chunk coords as a single int
        # (a flat chunk key) and sort the points by it.
        numblocks = self.array.numblocks
        flat_keys = xp.reshape(xp.broadcast_to(ravel_multi_index(chunk_ids, numblocks), out_shape), (-1,))
        sort_order = xp.argsort(flat_keys, stable=False)
        sorted_flat_keys = flat_keys[sort_order]

        # Find group boundaries and the key of each group; the few numbers involved go to the host.
        group_starts = flatnonzero(sorted_flat_keys[1:] != sorted_flat_keys[:-1]) + 1
        group_keys = xp.concat([sorted_flat_keys[:1], sorted_flat_keys[group_starts]])
        n_groups = group_keys.shape[0]
        starts_and_keys = to_numpy(xp.concat([group_starts, group_keys])).tolist()
        boundaries = [0, *starts_and_keys[: n_groups - 1], n_points]

        # Gather each group from its chunk, in sorted order, then undo the sort with one gather.
        flat_local_indices = [xp.reshape(xp.broadcast_to(local, out_shape), (-1,)) for local in local_indices]
        values_by_group = []
        for g, flat_key in enumerate(starts_and_keys[n_groups - 1 :]):
            grp_indices = sort_order[boundaries[g] : boundaries[g + 1]]
            key = tuple(int(c) for c in unravel_index(flat_key, numblocks))
            local_idx = tuple(local[grp_indices] for local in flat_local_indices)
            values_by_group.append(self._chunk(key)[local_idx])

        sorted_values = values_by_group[0] if n_groups == 1 else xp.concat(values_by_group)
        return xp.reshape(sorted_values[xp.argsort(sort_order, stable=False)], out_shape)

    # --- ExplicitlyIndexed protocol ---

    def _vindex_get(self, indexer: VectorizedIndexer):
        # Dimensions not given an indexer in ``.isel()`` (e.g. a size-1 ``mockZ``)
        # arrive as slices. Under vectorized indexing semantics the output has the
        # broadcast dims of the array indexers first, then one dim per slice. Expand
        # each slice to an index array over its own trailing dim to match.
        key = indexer.tuple
        slices = [np.arange(n)[k] for k, n in zip(key, self.array.shape, strict=True) if isinstance(k, slice)]
        slice_grids = iter(np.meshgrid(*slices, indexing="ij", sparse=True))
        trailing = (np.newaxis,) * len(slices)
        return self._raw_vindex(
            *(next(slice_grids) if isinstance(k, slice) else np.asarray(k)[(..., *trailing)] for k in key)
        )

    def _oindex_get(self, indexer: OuterIndexer):
        # Delegate to dask for orthogonal indexing
        return self.array[indexer.tuple]

    def __getitem__(self, indexer):
        if isinstance(indexer, VectorizedIndexer):
            return self._vindex_get(indexer)
        if isinstance(indexer, OuterIndexer):
            return self._oindex_get(indexer)
        if isinstance(indexer, BasicIndexer):
            return self.array[indexer.tuple]
        return self.array[indexer]
