"""The Parcels backends (ChunkCachedArray, WindowedArray) holding field data in NumPy, array-api-strict and PyTorch.

Every gather is compared against NumPy fancy indexing of the full array, and must return an array of the
backend's library. The shared helpers the backends and the core use to place, convert and measure arrays
(``parcels._core.utils.array_backend``) are tested at the end.
"""

import functools
import io

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st

import parcels
import parcels._strategies.chunk_cached_array as ccast
from parcels._chunk_cached_array import ByteBoundedLRUCache, ChunkCachedArray
from parcels._core.mesh import FlatMesh
from parcels._core.utils.array_backend import (
    NUMPY_PLACEMENT,
    ArrayPlacement,
    array_namespace,
    asarray_like,
    cached_per_placement,
    device_of,
    host_array_for_conversion,
    indices_to_mask,
    namespace_of_library,
    nbytes_of,
    placement_of_library,
    to_numpy,
    values_like,
)
from parcels._core.warnings import FieldSetWarning
from parcels._datasets.structured.generated import simple_UV_dataset
from parcels._datasets.unstructured.generic import _ux_constant_flow_face_centered_2D
from parcels._prefetch import DataLoaderPrefetcher, NoPrefetcher
from parcels._windowed_array import WindowedArray

LIBRARIES = ["numpy", "array_api_strict", "torch"]


@pytest.fixture(params=LIBRARIES)
def library(request):
    return pytest.importorskip(request.param)


def as_library(library, values):
    return namespace_of_library(library).asarray(np.asarray(values))


def gather(backend, library, indices):
    """``backend._raw_vindex`` with the NumPy ``indices`` converted to ``library``."""
    return backend._raw_vindex(*(as_library(library, index) for index in indices))


def assert_gathered(gathered, library, expected):
    """``gathered`` is an array of ``library`` holding ``expected``."""
    assert array_namespace(gathered) is namespace_of_library(library)
    np.testing.assert_array_equal(to_numpy(gathered), expected)


@pytest.fixture
def host_array() -> np.ndarray:
    return np.random.default_rng(0).random((5, 4, 7, 9)).astype(np.float32)


def corner_indices(shape, npart, rng):
    """Index arrays laid out as the interpolators pass them: one 5-D broadcastable array per stored dimension."""
    per_axis = []
    for axis, n in enumerate(shape):
        lower = rng.integers(0, n - 1, npart)
        corner_shape = [1] * 5
        corner_shape[axis] = 2
        corner_shape[-1] = npart
        per_axis.append(np.stack([lower, lower + 1]).reshape(corner_shape))
    return per_axis


# ------------------------------------------------------------------------------------------------
# ChunkCachedArray
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def lazy_array(host_array):
    return da.from_array(host_array, chunks=(1, 2, 3, 4))


def test_chunk_cached_array_gathers_in_the_library(library, host_array, lazy_array):
    cca = ChunkCachedArray(lazy_array, 10**8, library)
    assert cca.array_placement.namespace is namespace_of_library(library)

    rng = np.random.default_rng(1)
    pointwise = [rng.integers(-n, n, 200) for n in host_array.shape]
    assert_gathered(gather(cca, library, pointwise), library, host_array[tuple(pointwise)])

    corners = corner_indices(host_array.shape, 50, rng)
    assert_gathered(gather(cca, library, corners), library, host_array[tuple(corners)])

    # a dimension no axis indexes gets a zero index that broadcasts against everything
    corners[1] = np.zeros((1,) * 5, dtype=np.int64)
    assert_gathered(gather(cca, library, corners), library, host_array[tuple(corners)])

    # every point in one chunk
    one_chunk = [np.array([0, 0]), np.array([1, 1]), np.array([0, 2]), np.array([3, 1])]
    assert_gathered(gather(cca, library, one_chunk), library, host_array[tuple(one_chunk)])


def test_chunk_cached_array_empty_and_out_of_bounds(library, host_array, lazy_array):
    cca = ChunkCachedArray(lazy_array, 10**8, library)
    empty = cca._raw_vindex(*(as_library(library, np.zeros((0, 3), dtype=np.int64)) for _ in host_array.shape))
    assert empty.shape == (0, 3)
    assert_gathered(empty, library, np.empty((0, 3), dtype=np.float32))

    out_of_bounds = [np.array([0, 1]), np.array([0, 1]), np.array([2, 7]), np.array([0, 1])]
    with pytest.raises(IndexError, match="index 7 is out of bounds for axis 2 with size 7"):
        cca._raw_vindex(*(as_library(library, i) for i in out_of_bounds))
    too_negative = [np.array([0]), np.array([-5]), np.array([0]), np.array([0])]
    with pytest.raises(IndexError, match="index -5 is out of bounds for axis 1 with size 4"):
        cca._raw_vindex(*(as_library(library, i) for i in too_negative))


def test_chunk_cached_array_with_a_cache_too_small_for_its_chunks(library, host_array, lazy_array):
    chunk_bytes = 2 * 3 * 4 * host_array.itemsize
    cca = ChunkCachedArray(lazy_array, chunk_bytes, library)
    rng = np.random.default_rng(2)
    for _ in range(3):
        corners = corner_indices(host_array.shape, 20, rng)
        assert_gathered(gather(cca, library, corners), library, host_array[tuple(corners)])
        assert cca.cache.current_bytes <= chunk_bytes


@given(data=st.data())
@settings(deadline=None, max_examples=30)
@pytest.mark.parametrize("library_name", LIBRARIES)
def test_chunk_cached_array_matches_numpy_for_any_chunking(library_name, data):
    library = pytest.importorskip(library_name)
    cca = data.draw(ccast.chunk_cached_array(array_library=library))
    key = data.draw(ccast.vectorized_indexer_key(cca.shape, allow_slices=False, min_index_side=1))
    expected = cca.array.compute()[key]
    assert_gathered(cca._raw_vindex(*(as_library(library, k) for k in key)), library, expected)


def test_lru_cache_measures_arrays_without_nbytes():
    xp = pytest.importorskip("array_api_strict")
    cache = ByteBoundedLRUCache(100)
    cache.put("a", xp.zeros((3, 4), dtype=xp.float64))  # 96 bytes
    assert cache.current_bytes == 96
    cache.put("b", xp.zeros(4, dtype=xp.int32))  # 16 bytes: evicts "a"
    assert cache.current_bytes == 16
    assert cache.get("a") is None


def test_to_chunk_cached_arrays_reports_the_placement(library):
    ds = simple_UV_dataset(mesh_str="flat")
    for chunked in (True, False):
        fieldset = parcels.FieldSet.from_sgrid_conventions(ds.chunk({"time": 1}) if chunked else ds, mesh=FlatMesh())
        assert fieldset.to_chunk_cached_arrays(library) is fieldset
        storage = fieldset.U.data.variable._data
        if library is np and not chunked:
            assert isinstance(storage, np.ndarray)  # eager NumPy fields are left alone
        else:
            assert isinstance(storage, ChunkCachedArray)
            assert fieldset.array_placement == storage.array_placement
            buf = io.StringIO()
            fieldset.describe(buf)
            expected_name = "ChunkCachedArray" if library is np else f"ChunkCachedArray ({library.__name__})"
            assert f"| {expected_name} " in buf.getvalue()
        assert fieldset.array_placement.namespace is namespace_of_library(library)


def test_to_chunk_cached_arrays_without_a_library_is_unchanged():
    ds = simple_UV_dataset(mesh_str="flat").chunk({"time": 1})
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds, mesh=FlatMesh()).to_chunk_cached_arrays()
    assert fieldset.array_placement == NUMPY_PLACEMENT
    buf = io.StringIO()
    fieldset.describe(buf)
    assert "| ChunkCachedArray " in buf.getvalue()


# ------------------------------------------------------------------------------------------------
# WindowedArray
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def time_chunked(host_array):
    return da.from_array(host_array, chunks=(1, -1, -1, -1))


def gather_levels(windowed, library, host_array, levels):
    """Gather at the given time levels and fixed spatial points, checking the values."""
    levels = np.asarray(levels)
    spatial = [np.resize(points, levels.size) for points in ([1, 2], [3, 4], [5, 6])]
    gathered = windowed._raw_vindex(*(as_library(library, i) for i in (levels, *spatial)))
    assert_gathered(gathered, library, host_array[(levels, *spatial)])


def test_windowed_array_gathers_in_the_library(library, host_array, time_chunked):
    rng = np.random.default_rng(3)
    for windowed in (True, False):
        array = WindowedArray(time_chunked, windowed=windowed, array_library=library)
        assert array.array_placement.namespace is namespace_of_library(library)
        corners = corner_indices(host_array.shape, 50, rng)
        assert_gathered(gather(array, library, corners), library, host_array[tuple(corners)])
        corners[1] = np.zeros((1,) * 5, dtype=np.int64)
        assert_gathered(gather(array, library, corners), library, host_array[tuple(corners)])
        negative = [np.array([-1, 0, -5]), np.array([0, -1, 2]), np.array([-7, 3, 6]), np.array([8, -9, 0])]
        assert_gathered(gather(array, library, negative), library, host_array[tuple(negative)])
        empty = array._raw_vindex(*(as_library(library, np.zeros(0, dtype=np.int64)) for _ in host_array.shape))
        assert empty.shape == (0,)
        array.close()


def test_windowed_array_rejects_out_of_range_levels(library, time_chunked):
    array = WindowedArray(time_chunked, windowed=True, array_library=library)
    with pytest.raises(IndexError, match="index 5 is out of bounds for axis 0 with size 5"):
        array._raw_vindex(*(as_library(library, np.array([i])) for i in (5, 0, 0, 0)))
    assert array.resident_levels == []


@pytest.mark.parametrize("prefetcher", [None, NoPrefetcher], ids=["thread-prefetcher", "no-prefetcher"])
def test_windowed_array_window_follows_the_requests(library, host_array, time_chunked, prefetcher):
    array = WindowedArray(time_chunked, windowed=True, array_library=library, prefetcher=prefetcher)
    assert array.resident_levels == []
    steps = [
        # (levels requested, levels resident afterwards, loads so far)
        ([0, 1], [0, 1], 2),
        ([1, 2], [1, 2], 3),  # advancing the clock evicts level 0 and loads only the new level
        ([2, 2], [2], 3),  # a single level evicts the other, and needs no load
        ([3, 2], [2, 3], 4),
        ([0, 4, 1], [0, 1, 2, 3, 4], 7),  # a wide request keeps what lies in its span and loads only what it asks
        ([4, 3], [3, 4], 7),
    ]
    for levels, resident, loads in steps:
        gather_levels(array, library, host_array, levels)
        assert array.resident_levels == resident
        assert array.loads == loads
    assert array.bytes_read == 7 * 4 * 7 * 9 * host_array.itemsize
    if prefetcher is NoPrefetcher:
        assert array.prefetch_hits == 0
    array.close()


def test_windowed_array_max_levels_bounds_the_window(library, host_array, time_chunked):
    array = WindowedArray(time_chunked, windowed=True, array_library=library, max_levels=2, prefetcher=NoPrefetcher)
    gather_levels(array, library, host_array, [1, 2])
    # within the span, levels that were not asked for are dropped, lowest first, to stay within max_levels
    gather_levels(array, library, host_array, [0, 3])
    assert array.resident_levels == [0, 3]
    # a request for more levels than max_levels gets them all, with a warning
    with pytest.warns(FieldSetWarning, match="exceeds max_level=2"):
        gather_levels(array, library, host_array, [0, 4, 1])
    assert array.resident_levels == [0, 1, 4]


def test_windowed_array_reads_ahead_without_changing_results(library, host_array, time_chunked):
    rng = np.random.default_rng(4)
    requests = [np.array([level, level + 1]) for level in range(4)] + [np.array([3, 4])]
    gathered = {}
    for name, prefetcher in [("none", NoPrefetcher), ("thread", None), ("dataloader", "dataloader")]:
        if prefetcher == "dataloader":
            pytest.importorskip("torch")
            prefetcher = functools.partial(DataLoaderPrefetcher, num_workers=0)
        array = WindowedArray(time_chunked, windowed=True, array_library=library, prefetcher=prefetcher)
        results = []
        for levels in requests:
            spatial = [rng.integers(0, n, levels.size) for n in host_array.shape[1:]]
            results.append(to_numpy(array._raw_vindex(*(as_library(library, i) for i in (levels, *spatial)))))
        gathered[name] = np.concatenate(results)
        assert array.loads == 5
        # the first request loads its two levels on demand; every later level was read ahead
        assert array.prefetch_hits == (0 if name == "none" else 3)
        array.close()
        rng = np.random.default_rng(4)
    np.testing.assert_array_equal(gathered["thread"], gathered["none"])
    np.testing.assert_array_equal(gathered["dataloader"], gathered["none"])


def test_windowed_array_backward_requests_read_ahead_backward(library, host_array, time_chunked):
    array = WindowedArray(time_chunked, windowed=True, array_library=library)
    for lowest in range(3, -1, -1):
        gather_levels(array, library, host_array, [lowest, lowest + 1])
    assert array.loads == 5
    assert array.prefetch_hits == 3
    array.close()


def test_windowed_array_orthogonal_indexing(library, host_array, time_chunked):
    array = WindowedArray(time_chunked, windowed=True, array_library=library, prefetcher=NoPrefetcher)
    # a range of levels is read from the source, lazily, without touching the window
    np.testing.assert_array_equal(np.asarray(array[(slice(1, 4), 0, slice(None), 3)]), host_array[1:4, 0, :, 3])
    assert array.resident_levels == []
    # a single level goes through the window
    assert_gathered(array[2, 1], library, host_array[2, 1])
    assert array.resident_levels == [2]
    eager = WindowedArray(host_array, windowed=False, array_library=library)
    assert_gathered(eager[(slice(1, 4), 0, slice(None), 3)], library, host_array[1:4, 0, :, 3])


def test_windowed_array_through_xarray_isel(library, host_array, time_chunked):
    dims = ("time", "depth", "YG", "XG")
    rng = np.random.default_rng(5)
    points = {dim: xr.Variable("points", rng.integers(0, n, 30)) for dim, n in zip(dims, host_array.shape, strict=True)}
    expected = xr.Variable(dims, host_array).isel(points)
    windowed = xr.Variable(dims, WindowedArray(time_chunked, windowed=True, array_library=library))
    np.testing.assert_array_equal(to_numpy(windowed.isel(points).data), expected.data)


# ------------------------------------------------------------------------------------------------
# FieldSet.to_windowed_arrays
# ------------------------------------------------------------------------------------------------
def test_to_windowed_arrays_reports_the_placement(library):
    ds = simple_UV_dataset(mesh_str="flat")
    ds_mixed = ds.chunk({"time": 1})
    ds_mixed["U"] = ds["U"].load()  # one eager field, one dask-backed field
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds_mixed, mesh=FlatMesh())
    assert fieldset.to_windowed_arrays(library) is fieldset

    lazy_storage = fieldset.V.data.variable._data
    assert isinstance(lazy_storage, WindowedArray) and lazy_storage._is_windowed
    eager_storage = fieldset.U.data.variable._data
    if library is np:
        assert isinstance(eager_storage, np.ndarray)  # eager NumPy fields are left alone
    else:
        assert isinstance(eager_storage, WindowedArray) and not eager_storage._is_windowed
    assert fieldset.array_placement == lazy_storage.array_placement
    assert fieldset.array_placement.namespace is namespace_of_library(library)

    # idempotent, with the same library
    fieldset.to_windowed_arrays(library)
    assert fieldset.V.data.variable._data is lazy_storage

    buf = io.StringIO()
    fieldset.describe(buf)
    expected_name = "WindowedArray" if library is np else f"WindowedArray ({library.__name__})"
    assert f"| {expected_name} " in buf.getvalue()


def test_to_windowed_arrays_refuses_a_second_library():
    array_api_strict = pytest.importorskip("array_api_strict")

    fieldset = parcels.FieldSet.from_sgrid_conventions(
        simple_UV_dataset(mesh_str="flat").chunk({"time": 1}), mesh=FlatMesh()
    )
    fieldset.to_windowed_arrays()
    with pytest.raises(ValueError, match="already held in a WindowedArray"):
        fieldset.to_windowed_arrays(array_api_strict)


@pytest.mark.parametrize("method", ["to_windowed_arrays", "to_chunk_cached_arrays"])
def test_other_libraries_need_a_structured_grid(method):
    torch = pytest.importorskip("torch")
    fieldset = parcels.FieldSet.from_ugrid_conventions(_ux_constant_flow_face_centered_2D().chunk({"time": 1}))
    with pytest.raises(NotImplementedError, match="structured"):
        getattr(fieldset, method)(torch)


# ------------------------------------------------------------------------------------------------
# Shared helpers (parcels._core.utils.array_backend)
# ------------------------------------------------------------------------------------------------
def test_placement_of_library(library):
    placement = placement_of_library(library)
    assert placement.namespace is namespace_of_library(library)
    assert placement == ArrayPlacement.of(as_library(library, [1.0]))
    assert placement_of_library(None) == NUMPY_PLACEMENT


def test_host_array_for_conversion():
    values = np.arange(12.0).reshape(3, 4)
    assert host_array_for_conversion(values) is values  # already C-contiguous, writable and in native byte order

    read_only = values.copy()
    read_only.flags.writeable = False
    for source in (read_only, values[:, ::2], values.astype(">f8"), [[1.0, 2.0]]):
        prepared = host_array_for_conversion(source)
        assert prepared.flags["C_CONTIGUOUS"] and prepared.flags["WRITEABLE"] and prepared.dtype.isnative
        np.testing.assert_array_equal(prepared, np.asarray(source))
    assert not np.shares_memory(host_array_for_conversion(read_only), read_only)


def test_asarray_like_takes_any_host_array(library):
    like = as_library(library, [0.0])
    values = np.arange(12.0).reshape(3, 4)
    read_only = values.copy()
    read_only.flags.writeable = False
    for source in (values, read_only, values[:, ::2], values.astype(">f8"), values.astype(np.float32)):
        converted = asarray_like(source, like=like)
        assert_gathered(converted, library, source)
        assert to_numpy(converted).dtype.newbyteorder("=") == source.dtype.newbyteorder("=")
    if library is np:
        assert asarray_like(read_only, like=like) is read_only  # NumPy computes on the host array itself
    else:
        assert not np.shares_memory(to_numpy(asarray_like(read_only, like=like)), read_only)


def test_values_like(library):
    xp = namespace_of_library(library)
    target = xp.zeros(3, dtype=xp.float64)

    # Python scalars are passed on as they are
    for scalar in (2.5, 3, True):
        assert values_like(scalar, target) is scalar

    # a list goes straight to the data type of the target, not through PyTorch's default float32
    precise = [0.1, 1.0 + 1e-12, 1e10 + 0.5]
    converted = values_like(precise, target)
    assert converted.dtype == target.dtype
    assert_gathered(converted, library, np.array(precise))
    target[...] = converted
    assert_gathered(target, library, np.array(precise))

    # arrays of other data types, NumPy scalars, and read-only host arrays
    converted = values_like(np.array([1, 2, 3], dtype=np.int32), target)
    assert converted.dtype == target.dtype
    assert_gathered(converted, library, np.array([1.0, 2.0, 3.0]))
    converted = values_like(np.float32(1.5), target)
    assert array_namespace(converted) is xp and converted.dtype == target.dtype and converted.shape == ()
    read_only = np.array([4.0, 5.0, 6.0])
    read_only.flags.writeable = False
    converted = values_like(read_only, target)
    assert_gathered(converted, library, read_only)
    if library is not np:
        assert not np.shares_memory(to_numpy(converted), read_only)
    assert device_of(converted) == device_of(target)


def test_cached_per_placement_converts_once_per_library_and_device(library):
    host_values = np.arange(4.0)
    copies = []
    conversions = []

    def convert(like):
        conversions.append(like)
        return asarray_like(host_values, like=like)

    first = cached_per_placement(copies, as_library(library, [0.0]), convert)
    assert cached_per_placement(copies, as_library(library, [1.0, 2.0]), convert) is first
    assert_gathered(first, library, host_values)
    assert len(conversions) == 1

    # NumPy arrays and values that are not arrays (which NumPy takes as they are) have entries of their own
    numpy_copy = cached_per_placement(copies, np.zeros(2), convert)
    not_an_array_copy = cached_per_placement(copies, 0.5, convert)
    assert isinstance(numpy_copy, np.ndarray) and isinstance(not_an_array_copy, np.ndarray)
    assert cached_per_placement(copies, 1.5, convert) is not_an_array_copy
    assert len(copies) == len(conversions) == (2 if library is np else 3)


def test_nbytes_of(library):
    xp = namespace_of_library(library)
    for dtype_name, itemsize in [
        ("bool", 1),
        ("int8", 1),
        ("int32", 4),
        ("float32", 4),
        ("float64", 8),
        ("complex128", 16),
    ]:
        assert nbytes_of(xp.zeros((3, 5), dtype=getattr(xp, dtype_name))) == 15 * itemsize, dtype_name
    assert nbytes_of(None) == 0
    assert nbytes_of([1.0, 2.0]) == 0


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize(
    "indices",
    [[0, 3], [-1, 2, -6], [4, 4, 1], [], [[0, 1], [5, -1]]],
    ids=["positive", "negative", "repeated", "empty", "two-dimensional"],
)
def test_indices_to_mask_matches_numpy_scatter(library, indices, dtype):
    indices = np.asarray(indices, dtype=dtype)
    expected = np.zeros(6, dtype=bool)
    expected[indices] = True

    mask = indices_to_mask(as_library(library, indices), 6)

    assert_gathered(mask, library, expected)


@pytest.mark.parametrize("out_of_bounds", [6, -7, 100])
def test_indices_to_mask_raises_the_numpy_index_error(library, out_of_bounds):
    indices = np.array([1, out_of_bounds, 2])
    with pytest.raises(IndexError) as numpy_error:
        np.zeros(6, dtype=bool)[indices] = True

    with pytest.raises(IndexError, match=f"^{numpy_error.value}$"):
        indices_to_mask(as_library(library, indices), 6)
