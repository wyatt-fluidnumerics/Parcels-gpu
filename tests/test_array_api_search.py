"""The grid search run on the array libraries Parcels computes on, compared against its NumPy result."""

from types import ModuleType

import numpy as np
import pytest

from parcels import FieldSet
from parcels._core.index_search import (
    GRID_SEARCH_ERROR,
    LEFT_OUT_OF_BOUNDS,
    RIGHT_OUT_OF_BOUNDS,
    _search_1d_array,
    _search_1d_columns,
    _search_time_index,
)
from parcels._core.spatialhash import _encode_morton3d
from parcels._core.utils.array_backend import array_namespace, namespace_of_library, to_numpy
from parcels._core.warnings import FieldSetWarning
from parcels._datasets.structured.generated import simple_UV_dataset, simple_UV_sigma_dataset
from parcels._datasets.structured.generic import datasets


@pytest.fixture(params=["numpy", "array_api_strict", "torch"])
def xp(request) -> ModuleType:
    """Array API namespace of a library to run the search on (PyTorch on the CPU)."""
    return namespace_of_library(pytest.importorskip(request.param))


@pytest.fixture(scope="module")
def rectilinear_grid():
    # Integer depth levels, so the search also meets integer coordinates
    return FieldSet.from_sgrid_conventions(datasets["ds_2d_left"]).data_g.grid


def _curvilinear_grid(mesh: str):
    ds = datasets["2d_left_rotated"].copy(deep=True)
    if mesh == "spherical":
        ds["lon"].attrs["units"] = "degrees"
        ds["lat"].attrs["units"] = "degrees"
    return FieldSet.from_sgrid_conventions(ds).data_g.grid


@pytest.fixture(scope="module", params=["flat", "spherical"])
def curvilinear_grid(request):
    return _curvilinear_grid(request.param)


def assert_matches_numpy(result, expected, xp: ModuleType, exact: bool = True) -> None:
    """``result`` is an array of ``xp`` with the dtype, shape and values of the NumPy result ``expected``."""
    assert array_namespace(result) is xp
    result_on_host = to_numpy(result)
    expected = np.asarray(expected)
    assert result_on_host.dtype == expected.dtype
    assert result_on_host.shape == expected.shape
    if exact:
        np.testing.assert_array_equal(result_on_host, expected)
    else:
        np.testing.assert_allclose(result_on_host, expected, rtol=1e-12, atol=1e-12)


def assert_positions_match_numpy(result: dict, expected: dict, xp: ModuleType) -> None:
    assert result.keys() == expected.keys()
    for axis in expected:
        assert_matches_numpy(result[axis]["index"], expected[axis]["index"], xp)
        assert_matches_numpy(result[axis]["bcoord"], expected[axis]["bcoord"], xp, exact=False)


def sample_positions(grid, n_points: int, dtype, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points spread over a margin around the grid's bounding box, so that some fall outside the grid."""
    rng = np.random.default_rng(seed)
    lon, lat, depth = grid.lon, grid.lat, grid.depth
    z = rng.uniform(depth.min() - 0.5, depth.max() + 0.5, n_points)
    y = rng.uniform(np.nanmin(lat) - 1.0, np.nanmax(lat) + 1.0, n_points)
    x = rng.uniform(np.nanmin(lon) - 1.0, np.nanmax(lon) + 1.0, n_points)
    return z.astype(dtype), y.astype(dtype), x.astype(dtype)


@pytest.mark.parametrize("coordinates", [np.linspace(-2.0, 5.0, 8), np.arange(6)], ids=["float64", "int64"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_search_1d_array(xp, coordinates, dtype):
    rng = np.random.default_rng(1)
    x = rng.uniform(coordinates[0] - 1.0, coordinates[-1] + 1.0, 50).astype(dtype)
    x[:2] = [coordinates[0] - 0.5, coordinates[-1] + 0.5]
    index_numpy, bcoord_numpy = _search_1d_array(coordinates, x)
    assert index_numpy[0] == LEFT_OUT_OF_BOUNDS
    assert index_numpy[1] == RIGHT_OUT_OF_BOUNDS

    index, bcoord = _search_1d_array(xp.asarray(coordinates), xp.asarray(x))

    assert_matches_numpy(index, index_numpy, xp)
    assert_matches_numpy(bcoord, bcoord_numpy, xp, exact=False)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_xgrid_search_rectilinear(xp, rectilinear_grid, dtype):
    z, y, x = sample_positions(rectilinear_grid, 200, dtype)
    expected = rectilinear_grid.search(z, y, x)
    assert np.any(expected["X"]["index"] == RIGHT_OUT_OF_BOUNDS)

    result = rectilinear_grid.search(xp.asarray(z), xp.asarray(y), xp.asarray(x))

    assert_positions_match_numpy(result, expected, xp)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_xgrid_search_curvilinear(xp, curvilinear_grid, dtype):
    z, y, x = sample_positions(curvilinear_grid, 300, dtype)
    expected = curvilinear_grid.search(z, y, x)
    assert np.any(expected["X"]["index"] == GRID_SEARCH_ERROR)

    result = curvilinear_grid.search(xp.asarray(z), xp.asarray(y), xp.asarray(x))

    assert_positions_match_numpy(result, expected, xp)


def test_xgrid_search_curvilinear_from_previous_cells(xp, curvilinear_grid):
    """With the cells of a previous search as guesses, points that left their cell are found in the spatial hash."""
    z, y, x = sample_positions(curvilinear_grid, 300, np.float64)
    first = curvilinear_grid.search(z, y, x)
    found = first["X"]["index"] >= 0
    z, y, x = z[found], y[found], x[found]
    ei = curvilinear_grid.ravel_index({axis: first[axis]["index"][found] for axis in curvilinear_grid.axes})
    y_moved, x_moved = y + 0.3, x - 0.2
    expected = curvilinear_grid.search(z, y_moved, x_moved, ei=ei)

    result = curvilinear_grid.search(xp.asarray(z), xp.asarray(y_moved), xp.asarray(x_moved), ei=xp.asarray(ei))

    assert_positions_match_numpy(result, expected, xp)


def test_spatialhash_query(xp, curvilinear_grid):
    _, y, x = sample_positions(curvilinear_grid, 300, np.float64, seed=2)
    y[:3] = [np.nan, np.inf, 1e4]
    x[:3] = [np.nan, np.inf, 1e4]
    spatialhash = curvilinear_grid.get_spatial_hash()
    # Infinite positions give NumPy invalid-value warnings on a spherical mesh
    with np.errstate(invalid="ignore"):
        j_numpy, i_numpy, coords_numpy = spatialhash.query(y, x)
        assert np.all(j_numpy[:3] == GRID_SEARCH_ERROR)

        j, i, coords = spatialhash.query(xp.asarray(y), xp.asarray(x))

    assert_matches_numpy(j, j_numpy, xp)
    assert_matches_numpy(i, i_numpy, xp)
    assert_matches_numpy(coords, coords_numpy, xp, exact=False)


def test_spatialhash_query_without_hits(xp, curvilinear_grid):
    y = x = np.array([1e4, -1e4])
    j_numpy, i_numpy, coords_numpy = curvilinear_grid.get_spatial_hash().query(y, x)

    j, i, coords = curvilinear_grid.get_spatial_hash().query(xp.asarray(y), xp.asarray(x))

    assert_matches_numpy(j, j_numpy, xp)
    assert_matches_numpy(i, i_numpy, xp)
    assert_matches_numpy(coords, coords_numpy, xp)


@pytest.mark.parametrize(
    "bounds",
    # A flat mesh has Python float z bounds of zero width, and z positions of zero
    [(0.0, 1.0, -2.0, 0.5, 0.0, 0.0), tuple(np.float64(bound) for bound in (0, 1, -2, 0.5, -1, 1))],
    ids=["python-floats", "numpy-scalars"],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_morton_codes(xp, bounds, dtype):
    rng = np.random.default_rng(3)
    x, y, z = (rng.uniform(-3.0, 2.0, 500).astype(dtype) for _ in range(3))
    if bounds[4] == bounds[5]:
        z = np.zeros_like(z)
    expected = _encode_morton3d(x, y, z, *bounds, bitwidth=1023)
    assert expected.dtype == np.uint32

    codes = _encode_morton3d(xp.asarray(x), xp.asarray(y), xp.asarray(z), *bounds, bitwidth=1023)

    assert_matches_numpy(codes, expected, xp)


def test_search_time_index(xp):
    field = FieldSet.from_sgrid_conventions(simple_UV_dataset(n_times=4, nz=2, ny=6, nx=5)).U
    time = np.linspace(0.0, float(field.time_interval.time_length_as_flt), 25)
    expected = _search_time_index(field, time)

    result = _search_time_index(field, xp.asarray(time))

    assert_positions_match_numpy(result, expected, xp)


def test_search_time_index_converts_each_time_axis_once(xp):
    """Fields of different models keep time axes of their own, each converted once per array library."""
    fields = [
        FieldSet.from_sgrid_conventions(simple_UV_dataset(n_times=n_times, nz=2, ny=6, nx=5)).U for n_times in (4, 7)
    ]
    times = [np.linspace(0.0, float(field.time_interval.time_length_as_flt), 25) for field in fields]
    expected = [_search_time_index(field, time) for field, time in zip(fields, times, strict=True)]

    for _ in range(2):
        for field, time, expected_positions in zip(fields, times, expected, strict=True):
            assert_positions_match_numpy(_search_time_index(field, xp.asarray(time)), expected_positions, xp)

    for field in fields:
        copies = [values for copy_xp, _, values in field.model._time_axis_copies if copy_xp is xp]
        assert len(copies) == 1
        assert array_namespace(copies[0]) is xp


def test_search_time_index_without_time_interval(xp):
    field = FieldSet.from_sgrid_conventions(simple_UV_dataset(n_times=1, nz=2, ny=6, nx=5)).U
    assert field.time_interval is None
    time = np.linspace(0.0, 100.0, 7)
    expected = _search_time_index(field, time)

    result = _search_time_index(field, xp.asarray(time))

    assert_positions_match_numpy(result, expected, xp)


@pytest.fixture(scope="module")
def sigma_field():
    """A field on a flat-bottomed sigma grid, whose layer depths change in time with the surface elevation."""
    n_times, nz, ny, nx = 3, 4, 6, 7
    rng = np.random.default_rng(5)
    bottom_depth_m = 10.0 + rng.uniform(0.0, 2.0, (ny, nx))
    surface_elevation_m = rng.uniform(-0.3, 0.3, (n_times, ny, nx))
    ds = simple_UV_sigma_dataset(n_times, nz, ny, nx, bottom_depth=bottom_depth_m, eta=surface_elevation_m)
    with pytest.warns(FieldSetWarning, match="Time-varying"):
        return FieldSet.from_sgrid_conventions(ds).U


def test_time_varying_depth_search_needs_numpy(xp, sigma_field):
    grid = sigma_field.grid
    rng = np.random.default_rng(6)
    n_points = 20
    x = rng.uniform(grid.lon[1], grid.lon[-2], n_points)
    y = rng.uniform(grid.lat[1], grid.lat[-2], n_points)
    z = rng.uniform(2.0, 8.0, n_points)
    time_positions = _search_time_index(sigma_field, np.full(n_points, 1800.0))["T"]
    ti, tau = time_positions["index"], time_positions["bcoord"]
    expected = grid.search(z, y, x, ti=ti, tau=tau)
    assert np.all(expected["Z"]["index"] >= 0)

    positions = [xp.asarray(values) for values in (z, y, x)]
    if xp is namespace_of_library(np):
        assert_positions_match_numpy(grid.search(*positions, ti=ti, tau=tau), expected, xp)
        return
    with pytest.raises(NotImplementedError, match="only supported with NumPy arrays"):
        grid.search(*positions, ti=xp.asarray(ti), tau=xp.asarray(tau))
    with pytest.raises(NotImplementedError, match="only supported with NumPy arrays"):
        _search_1d_columns(np.broadcast_to(np.linspace(0.0, 10.0, 4), (n_points, 4)), positions[0])


def test_ravel_unravel_index_round_trip(xp, rectilinear_grid):
    rng = np.random.default_rng(4)
    axis_indices = {
        axis: rng.integers(0, rectilinear_grid.get_axis_dim(axis), 100) for axis in rectilinear_grid.axes
    }
    expected_ei = rectilinear_grid.ravel_index(axis_indices)

    ei = rectilinear_grid.ravel_index({axis: xp.asarray(index) for axis, index in axis_indices.items()})
    assert_matches_numpy(ei, expected_ei, xp)

    # Particles store their cell as int32
    unraveled = rectilinear_grid.unravel_index(xp.astype(ei, xp.int32))
    assert unraveled.keys() == axis_indices.keys()
    for axis, index in axis_indices.items():
        assert_matches_numpy(unraveled[axis], index, xp)


def test_host_constants_are_converted_once(xp, curvilinear_grid):
    x = xp.asarray(np.array([1.0, 2.0]))
    spatialhash = curvilinear_grid.get_spatial_hash()

    table = spatialhash._hash_table_like(x)
    lon, lat = curvilinear_grid._node_coordinate_values(like=x)
    depth = curvilinear_grid._coordinate_values("depth", like=x)

    assert spatialhash._hash_table_like(x) is table
    assert curvilinear_grid._node_coordinate_values(like=x)[0] is lon
    assert curvilinear_grid._coordinate_values("depth", like=x) is depth
    for name, values in table.items():
        assert array_namespace(values) is xp, name
        np.testing.assert_array_equal(to_numpy(values), spatialhash._hash_table[name], err_msg=name)
    if xp is namespace_of_library(np):
        # NumPy computes on the host arrays themselves
        assert all(table[name] is spatialhash._hash_table[name] for name in table)
        assert lon is curvilinear_grid.lon
        assert lat is curvilinear_grid.lat
