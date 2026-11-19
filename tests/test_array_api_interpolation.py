"""The structured interpolators and the gather on array libraries other than NumPy.

Grid positions are computed on NumPy and converted to each library, and field data is held in that
library, so every interpolator runs on arrays of one library and must return arrays of it.
"""

import types
import warnings

import numpy as np
import pytest
import xarray as xr
from array_api_compat import is_numpy_namespace, is_torch_namespace
from dask import is_dask_collection

import parcels._core.utils.interpolation as i_u
import parcels._sgrid as sgrid
from parcels import FieldSet
from parcels._core.field import (
    _mask_outofbounds_values,
    _update_particle_states_interp_value,
    _update_particle_states_position,
)
from parcels._core.index_search import _search_time_index
from parcels._core.mesh import FlatMesh, SphericalMesh
from parcels._core.utils.array_backend import (
    ArrayPlacement,
    array_namespace,
    asarray_like,
    namespace_of_library,
    to_numpy,
)
from parcels._core.warnings import FieldEvalWarning
from parcels._core.xgrid import XGrid
from parcels._datasets.structured.generated import peninsula_dataset, simple_UV_dataset
from parcels.interpolators import (
    CGrid_Tracer,
    CGrid_Velocity,
    XFreeslip,
    XLinear,
    XLinear_Velocity,
    XLinearInvdistLandTracer,
    XNearest,
    XPartialslip,
)
from parcels.interpolators._xinterpolators import _gather_corners

LIBRARIES = ["numpy", "array_api_strict", "torch"]


@pytest.fixture(params=LIBRARIES)
def xp(request):
    """Array API namespace of the library under test."""
    return namespace_of_library(pytest.importorskip(request.param))


class _StandInBackend:
    """A Parcels backend array that holds its data in an array library and gathers there."""

    def __init__(self, array):
        self.array = array
        self.array_placement = ArrayPlacement.of(array)
        self.shape = tuple(array.shape)
        self.dtype = array.dtype
        self.ndim = array.ndim

    def _raw_vindex(self, *indices):
        return self.array[indices]


def _hold_field_data_in(fields, xp) -> None:
    """Replace the stored data of ``fields`` by arrays of ``xp``.

    xarray holds an ``array-api-strict`` array as it is, but would convert a PyTorch tensor to NumPy, so
    PyTorch data is held in a stand-in backend.
    """
    if is_numpy_namespace(xp):
        return
    for field in fields:
        variable = field.model.data.variables[field.name]
        values = xp.asarray(variable.values)
        variable._data = _StandInBackend(values) if is_torch_namespace(xp) else values


def _in_library(positions: dict, xp) -> dict:
    """Nested dict (with tuples) of NumPy arrays with every array converted to ``xp``."""
    converted = {}
    for key, value in positions.items():
        if isinstance(value, dict):
            converted[key] = _in_library(value, xp)
        elif isinstance(value, tuple):
            converted[key] = tuple(xp.asarray(array) for array in value)
        else:
            converted[key] = xp.asarray(value)
    return converted


def _positions(field, t, z, y, x) -> tuple[dict, dict]:
    grid_positions = field.grid.search(z, y, x)
    grid_positions.update(_search_time_index(field, t))
    return {"t": t, "z": z, "y": y, "x": x}, grid_positions


def _assert_matches_numpy(result, expected, xp):
    results = result if isinstance(result, tuple) else (result,)
    expecteds = expected if isinstance(expected, tuple) else (expected,)
    assert len(results) == len(expecteds)
    for actual, reference in zip(results, expecteds, strict=True):
        assert array_namespace(actual) is xp
        assert to_numpy(actual).dtype == reference.dtype
        np.testing.assert_allclose(to_numpy(actual), reference, rtol=1e-12, atol=1e-15)


@pytest.fixture
def node_coordinates_in_library(monkeypatch):
    """Have ``XGrid._node_coordinate_values`` return arrays of the library of ``like``.

    The method is being made library-neutral separately. Where it still returns NumPy, its result is
    converted here; where it already returns the library of ``like``, it is used as it is.
    """
    node_coordinate_values = XGrid._node_coordinate_values

    def in_library_of_like(grid, like):
        lon, lat = node_coordinate_values(grid, like=like)
        if array_namespace(lon) is array_namespace(like):
            return lon, lat
        return asarray_like(to_numpy(lon), like=like), asarray_like(to_numpy(lat), like=like)

    monkeypatch.setattr(XGrid, "_node_coordinate_values", in_library_of_like)


def _scalar_dataset() -> xr.Dataset:
    rng = np.random.default_rng(0)
    data = rng.uniform(1.0, 2.0, size=(3, 4, 5, 6))
    data[:, :, 1:3, 2:4] = 0.0  # land, for the inverse-distance land tracer
    return xr.Dataset(
        {"P": (["time", "depth", "lat", "lon"], data)},
        coords={
            "time": (["time"], [np.timedelta64(t, "s") for t in [0, 2, 4]], {"axis": "T"}),
            "depth": (["depth"], [0.0, 1.0, 2.0, 3.0], {"axis": "Z", "units": "m"}),
            "lat": (["lat"], np.arange(5.0), {"axis": "Y", "units": "m"}),
            "lon": (["lon"], np.arange(6.0), {"axis": "X", "units": "m"}),
            "x": (["x"], np.arange(6.0) + 0.5, {"axis": "X", "units": "m"}),
            "y": (["y"], np.arange(5.0) + 0.5, {"axis": "Y", "units": "m"}),
        },
    ).pipe(
        sgrid._attach_sgrid_metadata,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("x", "lon", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("y", "lat", sgrid.Padding.LOW),
            ),
            node_coordinates=("lon", "lat"),
            vertical_dimensions=(sgrid.FaceNodePadding("ZC", "depth", sgrid.Padding.HIGH),),
        ),
    )


def _scalar_positions(field, between_levels: bool):
    rng = np.random.default_rng(1)
    npart = 40
    if between_levels:
        t = rng.uniform(0.0, 4.0, npart)
        z = rng.uniform(0.0, 3.0, npart)
    else:
        t = np.zeros(npart)
        z = np.full(npart, 1.0)
    y = rng.uniform(0.0, 4.0, npart)
    x = rng.uniform(0.0, 5.0, npart)
    # some particles exactly on grid nodes, for the exact-hit branch of the land tracer
    y[:4] = np.round(y[:4])
    x[:4] = np.round(x[:4])
    return _positions(field, t, z, y, x)


@pytest.mark.parametrize("between_levels", [True, False])
@pytest.mark.parametrize(
    "interpolator", [XLinear(), XNearest(), CGrid_Tracer(), XLinearInvdistLandTracer()], ids=lambda i: type(i).__name__
)
def test_scalar_interpolators_match_numpy(xp, interpolator, between_levels):
    field = FieldSet.from_sgrid_conventions(_scalar_dataset()).P
    particle_positions, grid_positions = _scalar_positions(field, between_levels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the land tracer divides by zero on grid nodes
        expected = interpolator.interp(particle_positions, grid_positions, field)

        _hold_field_data_in([field], xp)
        result = interpolator.interp(_in_library(particle_positions, xp), _in_library(grid_positions, xp), field)

    _assert_matches_numpy(result, expected, xp)


def _velocity_fieldset(mesh_str: str) -> FieldSet:
    rng = np.random.default_rng(2)
    ds = simple_UV_dataset(n_times=3, nz=3, ny=6, nx=5, mesh_str=mesh_str)
    for name in ["U", "V", "W"]:
        values = rng.normal(size=ds["U"].shape)
        values[:, :, 2:4, 1:3] = 0.0  # land, for the slip interpolators
        ds[name] = (ds["U"].dims, values)
    mesh = SphericalMesh() if mesh_str == "spherical" else FlatMesh()
    return FieldSet.from_sgrid_conventions(ds, mesh=mesh)


def _velocity_positions(fieldset: FieldSet, between_levels: bool):
    rng = np.random.default_rng(3)
    npart = 40
    lon = fieldset.U.grid.lon
    lat = fieldset.U.grid.lat
    x = rng.uniform(lon[0], lon[-1], npart)
    y = rng.uniform(lat[1], lat[-2], npart)  # away from the poles of the spherical mesh
    if between_levels:
        time = fieldset.U.data.time.values
        t = rng.uniform(0.0, (time[-1] - time[0]) / np.timedelta64(1, "s"), npart)
        z = rng.uniform(0.0, 1.0, npart)
    else:
        t = np.zeros(npart)
        z = np.zeros(npart)
    return _positions(fieldset.U, t, z, y, x)


@pytest.mark.parametrize("mesh_str", ["flat", "spherical"])
@pytest.mark.parametrize("between_levels", [True, False])
@pytest.mark.parametrize("vector_name", ["UV", "UVW"])
@pytest.mark.parametrize(
    "interpolator",
    [XLinear_Velocity(), CGrid_Velocity(), XFreeslip(), XPartialslip()],
    ids=lambda i: type(i).__name__,
)
@pytest.mark.usefixtures("node_coordinates_in_library")
def test_vector_interpolators_match_numpy(xp, interpolator, vector_name, between_levels, mesh_str):
    fieldset = _velocity_fieldset(mesh_str)
    vectorfield = getattr(fieldset, vector_name)
    particle_positions, grid_positions = _velocity_positions(fieldset, between_levels)
    expected = interpolator.interp(particle_positions, grid_positions, vectorfield)

    _hold_field_data_in([fieldset.U, fieldset.V, fieldset.W], xp)
    result = interpolator.interp(_in_library(particle_positions, xp), _in_library(grid_positions, xp), vectorfield)

    _assert_matches_numpy(result, expected, xp)


@pytest.mark.parametrize(
    "interpolator, field_name",
    [(CGrid_Velocity(), "UV"), (CGrid_Tracer(), "P"), (XLinear(), "P"), (XNearest(), "U")],
    ids=lambda i: i if isinstance(i, str) else type(i).__name__,
)
@pytest.mark.usefixtures("node_coordinates_in_library")
def test_interpolators_on_fields_without_time_and_depth_match_numpy(xp, interpolator, field_name):
    """Fields whose time and depth are size-1 mock dimensions, with float32 coordinates and tracer data."""
    fieldset = FieldSet.from_sgrid_conventions(peninsula_dataset(xdim=12, ydim=8, grid_type="C"))
    rng = np.random.default_rng(7)
    npart = 30
    x = rng.uniform(0.0, 1e5, npart)
    y = rng.uniform(0.0, 5e4, npart)
    particle_positions, grid_positions = _positions(fieldset.U, np.zeros(npart), np.zeros(npart), y, x)
    field = getattr(fieldset, field_name)
    expected = interpolator.interp(particle_positions, grid_positions, field)

    _hold_field_data_in([fieldset.U, fieldset.V, fieldset.P], xp)
    result = interpolator.interp(_in_library(particle_positions, xp), _in_library(grid_positions, xp), field)

    _assert_matches_numpy(result, expected, xp)


def test_constant_field_matches_numpy(xp):
    fieldset = FieldSet.from_sgrid_conventions(_scalar_dataset())
    fieldset.add_constant_field("K", 2.5)
    x = np.linspace(0.5, 4.5, 7)
    expected = fieldset.K.interp_method.interp({"x": x}, {}, fieldset.K)

    result = fieldset.K.interp_method.interp({"x": xp.asarray(x)}, {}, fieldset.K)

    _assert_matches_numpy(result, expected, xp)


def _corner_levels(npart: int) -> dict:
    rng = np.random.default_rng(4)
    ti = rng.integers(0, 5, npart)
    zi = rng.integers(0, 4, npart)
    yi = rng.integers(0, 6, npart)
    xi = rng.integers(0, 7, npart)
    return {"T": (ti, ti + 1), "Z": (zi,), "Y": (yi, yi + 1), "X": (xi, xi + 1)}


@pytest.mark.parametrize("storage", ["eager", "backend", "lazy"])
@pytest.mark.parametrize("unindexed_depth", [False, True], ids=["all-indexed", "unindexed-depth"])
def test_gather_corners_on_each_storage(xp, storage, unindexed_depth):
    npart = 9
    values = np.random.default_rng(5).random((6, 1 if unindexed_depth else 5, 7, 8))
    axis_dim = {"X": "lon", "Y": "lat"} if unindexed_depth else {"X": "lon", "Y": "lat", "Z": "depth"}
    data = xr.DataArray(values, dims=("time", "depth", "lat", "lon"))
    levels = _corner_levels(npart)
    expected = _gather_corners(data, axis_dim, levels, npart)

    if storage == "lazy":
        # Lazily loaded data is gathered through xarray and computed to NumPy.
        data = data.chunk({"time": 2})
        result = _gather_corners(data, axis_dim, _in_library(levels, xp), npart)
        result = result.compute() if is_dask_collection(result) else result
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected)
        return

    library_values = xp.asarray(values)
    data.variable._data = library_values if storage == "eager" else _StandInBackend(library_values)
    result = _gather_corners(data, axis_dim, _in_library(levels, xp), npart)

    assert array_namespace(result) is xp
    assert result.shape == (2, 1, 2, 2, npart)
    np.testing.assert_array_equal(to_numpy(result), expected)
    result[...] = 0.0  # writable even where the unindexed depth repeats its corner
    if isinstance(result, np.ndarray):
        assert result.flags["C_CONTIGUOUS"]


def test_interpolation_utils_match_numpy(xp):
    rng = np.random.default_rng(6)
    npart = 11
    xsi, eta = rng.uniform(0, 1, npart), rng.uniform(0, 1, npart)
    px, py = rng.normal(size=(4, npart)), rng.normal(size=(4, npart))
    lib = _in_library({"xsi": xsi, "eta": eta, "px": px, "py": py}, xp)
    lib_px, lib_py = lib["px"], lib["py"]

    _assert_matches_numpy(i_u.phi2D_lin(lib["eta"], lib["xsi"]), i_u.phi2D_lin(eta, xsi), xp)
    _assert_matches_numpy(
        i_u._compute_jacobian_determinant(lib_py, lib_px, lib["eta"], lib["xsi"]),
        i_u._compute_jacobian_determinant(py, px, eta, xsi),
        xp,
    )
    for mesh in [FlatMesh(), SphericalMesh()]:
        _assert_matches_numpy(
            i_u._geodetic_distance(lib_py[0, :], lib_py[1, :], lib_px[0, :], lib_px[1, :], mesh, lib["eta"]),
            i_u._geodetic_distance(py[0], py[1], px[0], px[1], mesh, eta),
            xp,
        )


def _out_of_bounds_grid_positions() -> dict:
    return {
        "X": {"index": np.array([0, -1, 2, 1, -3])},
        "Y": {"index": np.array([1, 1, -2, 0, 0])},
        "Z": {"index": np.array([0, 0, 0, -1, 0])},
    }


def test_mask_outofbounds_values(xp):
    grid_positions = _out_of_bounds_grid_positions()
    expected = np.arange(1.0, 6.0)
    with pytest.warns(FieldEvalWarning):
        _mask_outofbounds_values(grid_positions, expected)

    value = xp.asarray(np.arange(1.0, 6.0))
    with pytest.warns(FieldEvalWarning):
        _mask_outofbounds_values(_in_library(grid_positions, xp), value)
    _assert_matches_numpy(value, expected, xp)


def test_update_particle_states(xp):
    grid_positions = _out_of_bounds_grid_positions()
    value = np.array([1.0, np.nan, 2.0, 3.0, 4.0])
    expected = types.SimpleNamespace(state=np.zeros(5, dtype=np.int32))
    _update_particle_states_position(expected, grid_positions)
    _update_particle_states_interp_value(expected, value)

    particles = types.SimpleNamespace(state=xp.zeros(5, dtype=xp.int32))
    _update_particle_states_position(particles, _in_library(grid_positions, xp))
    _update_particle_states_interp_value(particles, xp.asarray(value))

    assert array_namespace(particles.state) is xp
    np.testing.assert_array_equal(to_numpy(particles.state), expected.state)
    assert to_numpy(particles.state).dtype == np.int32
