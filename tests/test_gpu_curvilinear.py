"""GPU-versus-CPU tests of curvilinear grids and C-grid interpolation. They need CuPy and a GPU, and are skipped otherwise."""

import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

import parcels
import parcels._sgrid as sgrid
import parcels.tutorial
from parcels._core.index_search import GRID_SEARCH_ERROR
from parcels._core.mesh import FlatMesh, SphericalMesh
from parcels._core.spatialhash import _encode_morton3d
from parcels._core.utils.array_backend import is_gpu_array, to_numpy
from parcels._core.warnings import FieldEvalWarning
from parcels._datasets.structured.generated import peninsula_dataset
from parcels.interpolators import (
    CGrid_Tracer,
    CGrid_Velocity,
    XFreeslip,
    XLinearInvdistLandTracer,
    XNearest,
    XPartialslip,
    _gpu_xinterpolators,
)
from parcels.interpolators._gpu_xinterpolators import GpuCGrid_Tracer, GpuCGrid_Velocity, gpu_counterpart
from parcels.kernels import AdvectionRK2, AdvectionRK4_3D

cupy = pytest.importorskip("cupy")
try:
    cupy.cuda.runtime.getDeviceCount()
except Exception:
    pytest.skip("no GPU available", allow_module_level=True)


# ------------------------------------------------------------------------------------------------
# Synthetic curvilinear grids
# ------------------------------------------------------------------------------------------------
NX, NY, NZ, NT = 36, 28, 4, 5
TIME_LEVEL_SPACING_HOURS = 12
LAND_BLOCK_J = slice(10, 15)
LAND_BLOCK_I = slice(12, 19)
MESHES = {"spherical": SphericalMesh(), "flat": FlatMesh()}
OUTSIDE_POINTS_XY = {
    "spherical": (np.array([-80.0, 70.0, 0.0]), np.array([0.0, 60.0, -60.0])),
    "flat": (np.array([-1.0e6, 9.0e5, 0.0]), np.array([0.0, 8.0e5, -5.0e5])),
}


@dataclass(frozen=True)
class FacePaddings:
    """SGRID padding of the x, y and vertical face dimensions, which sets their face counts and C-grid offsets."""

    x: sgrid.Padding
    y: sgrid.Padding
    z: sgrid.Padding


NEMO_PADDINGS = FacePaddings(x=sgrid.Padding.LOW, y=sgrid.Padding.LOW, z=sgrid.Padding.HIGH)
# One face fewer than nodes along x and y, so that U, V, W and T all differ in shape, and C-grid offsets of
# (0, 0, 1) along (x, y, z) where NEMO's are (1, 1, 0).
MIXED_PADDINGS = FacePaddings(x=sgrid.Padding.NONE, y=sgrid.Padding.NONE, z=sgrid.Padding.LOW)


def curvilinear_node_coordinates(mesh: str) -> tuple[np.ndarray, np.ndarray]:
    """Rotated and smoothly distorted node lon/lat of shape (NY, NX): degrees on a sphere, metres on a flat mesh."""
    i_index, j_index = np.meshgrid(np.arange(NX, dtype=float), np.arange(NY, dtype=float))
    rotation_rad = np.deg2rad(20.0)
    distortion_cells = 0.15 * np.sin(2 * np.pi * j_index / NY) * np.cos(np.pi * i_index / NX)
    along_x_cells = i_index * np.cos(rotation_rad) - j_index * np.sin(rotation_rad) + distortion_cells
    along_y_cells = i_index * np.sin(rotation_rad) + j_index * np.cos(rotation_rad) - distortion_cells
    if mesh == "spherical":
        cell_size_deg = 2.0
        return -30.0 + cell_size_deg * along_x_cells, -25.0 + cell_size_deg * along_y_cells
    cell_size_m = 25_000.0
    return cell_size_m * along_x_cells, cell_size_m * along_y_cells


def rectilinear_node_coordinates(mesh: str) -> tuple[np.ndarray, np.ndarray]:
    """Unevenly spaced node lon of shape (NX,) and lat of shape (NY,): degrees on a sphere, metres on a flat mesh."""
    i_index, j_index = np.arange(NX, dtype=float), np.arange(NY, dtype=float)
    along_x_cells = i_index + 0.3 * np.sin(2 * np.pi * i_index / NX)
    along_y_cells = j_index + 0.3 * np.sin(np.pi * j_index / NY)
    if mesh == "spherical":
        cell_size_deg = 2.0
        return -30.0 + cell_size_deg * along_x_cells, -25.0 + cell_size_deg * along_y_cells
    cell_size_m = 25_000.0
    return cell_size_m * along_x_cells, cell_size_m * along_y_cells


def node_coordinates(mesh: str, rectilinear: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Node lon and lat of shape (NY, NX), of the curvilinear grid or of the rectilinear one."""
    if rectilinear:
        lon, lat = np.meshgrid(*rectilinear_node_coordinates(mesh))
        return lon, lat
    return curvilinear_node_coordinates(mesh)


def analytic_fields(with_land: bool) -> dict[str, np.ndarray]:
    """Smooth, time-varying U, V, W (m/s) and T (degC) of shape (NT, NZ, NY, NX), defined in grid-index space."""
    i_fraction = np.linspace(0.0, 1.0, NX)[None, None, None, :]
    j_fraction = np.linspace(0.0, 1.0, NY)[None, None, :, None]
    depth_decay = np.exp(-np.arange(NZ) / NZ)[None, :, None, None]
    time_modulation = (1.0 + 0.2 * np.arange(NT))[:, None, None, None]
    scale = depth_decay * time_modulation
    fields = {
        "U": 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * j_fraction) * np.cos(np.pi * i_fraction)) * scale,
        "V": 0.3 * np.sin(2 * np.pi * i_fraction) * np.cos(np.pi * j_fraction) * scale,
        "W": 1e-4 * np.sin(np.pi * i_fraction) * np.sin(np.pi * j_fraction) * scale,
        "T": 20.0 + 5.0 * i_fraction - 3.0 * j_fraction + 2.0 * scale,
    }
    for name, values in fields.items():
        values = np.broadcast_to(values, (NT, NZ, NY, NX)).astype(np.float32)
        if with_land:
            values[..., LAND_BLOCK_J, LAND_BLOCK_I] = 0.0
        fields[name] = values
    return fields


def curvilinear_dataset(
    mesh: str, grid_type: str, with_land: bool = False, paddings: FacePaddings = NEMO_PADDINGS
) -> xr.Dataset:
    """Curvilinear SGRID dataset in the NEMO layout written by ``parcels.convert.nemo_to_sgrid``.

    On an A-grid every field sits on the nodes. On a C-grid U sits on the x faces, V on the y
    faces, and W and T at the cell centres. The face dimensions are padded as ``paddings`` says.
    """
    lon, lat = curvilinear_node_coordinates(mesh)
    depth_m = np.linspace(0.0, 300.0, NZ)
    dim_sizes = {
        "time": NT,
        "depth": NZ,
        "depth_center": sgrid.get_n_faces(NZ, paddings.z),
        "y": NY,
        "y_center": sgrid.get_n_faces(NY, paddings.y),
        "x": NX,
        "x_center": sgrid.get_n_faces(NX, paddings.x),
    }
    if grid_type == "A":
        field_dims = dict.fromkeys(("U", "V", "W", "T"), ("time", "depth", "y", "x"))
    else:
        field_dims = {
            "U": ("time", "depth_center", "y_center", "x"),
            "V": ("time", "depth_center", "y", "x_center"),
            "W": ("time", "depth", "y_center", "x_center"),
            "T": ("time", "depth_center", "y_center", "x_center"),
        }
    horizontal_units = "degrees" if mesh == "spherical" else "m"
    times = np.datetime64("2000-01-01") + np.arange(NT) * np.timedelta64(TIME_LEVEL_SPACING_HOURS, "h")
    fields = analytic_fields(with_land)
    data_vars = {}
    for name, values in fields.items():
        dims = field_dims[name]
        data_vars[name] = (dims, values[tuple(slice(dim_sizes[dim]) for dim in dims)])
    depth_center_m = (depth_m + 0.5 * (depth_m[1] - depth_m[0]))[: dim_sizes["depth_center"]]
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": (["time"], times, {"axis": "T"}),
            "depth": (["depth"], depth_m, {"axis": "Z", "units": "m", "positive": "down"}),
            "depth_center": (["depth_center"], depth_center_m, {"axis": "Z"}),
            "x": (["x"], np.arange(NX), {"axis": "X"}),
            "x_center": (["x_center"], np.arange(dim_sizes["x_center"]), {"axis": "X"}),
            "y": (["y"], np.arange(NY), {"axis": "Y"}),
            "y_center": (["y_center"], np.arange(dim_sizes["y_center"]), {"axis": "Y"}),
            "lon": (["y", "x"], lon, {"units": horizontal_units}),
            "lat": (["y", "x"], lat, {"units": horizontal_units}),
        },
    )
    return sgrid._attach_sgrid_metadata(
        ds,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("x", "y"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("x_center", "x", paddings.x),
                sgrid.FaceNodePadding("y_center", "y", paddings.y),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("depth_center", "depth", paddings.z),),
        ),
    )


def rectilinear_cgrid_dataset(mesh: str, paddings: FacePaddings = NEMO_PADDINGS) -> xr.Dataset:
    """The C-grid dataset of ``curvilinear_dataset`` with its nodes on the 1D lon and lat of a rectilinear grid."""
    lon, lat = rectilinear_node_coordinates(mesh)
    ds = curvilinear_dataset(mesh, "C", paddings=paddings)
    return ds.assign_coords(lon=(["x"], lon, ds["lon"].attrs), lat=(["y"], lat, ds["lat"].attrs))


def curvilinear_fieldset(mesh: str, grid_type: str, on_gpu: bool, with_land: bool = False) -> parcels.FieldSet:
    ds = curvilinear_dataset(mesh, grid_type, with_land=with_land)
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds, mesh=MESHES[mesh])
    return fieldset.to_gpu_arrays() if on_gpu else fieldset


def points_inside_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    npoints: int,
    rng: np.random.Generator,
    i_range: tuple[float, float] = (0.05, NX - 1.05),
    j_range: tuple[float, float] = (0.05, NY - 1.05),
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear blends of the four nodes around random fractional node indices, returned as (x, y)."""
    i_fractional = rng.uniform(*i_range, npoints)
    j_fractional = rng.uniform(*j_range, npoints)
    i0 = np.floor(i_fractional).astype(int)
    j0 = np.floor(j_fractional).astype(int)
    xsi = i_fractional - i0
    eta = j_fractional - j0

    def blend(nodes: np.ndarray) -> np.ndarray:
        return (
            (1 - xsi) * (1 - eta) * nodes[j0, i0]
            + xsi * (1 - eta) * nodes[j0, i0 + 1]
            + xsi * eta * nodes[j0 + 1, i0 + 1]
            + (1 - xsi) * eta * nodes[j0 + 1, i0]
        )

    return blend(lon), blend(lat)


def sample_points(
    mesh: str, npoints: int = 400, seed: int = 0, rectilinear: bool = False, **inside_kwargs
) -> tuple[np.ndarray, ...]:
    """(t, z, y, x) samples inside the curvilinear (or rectilinear) grid, followed by a few points outside it."""
    rng = np.random.default_rng(seed)
    lon, lat = node_coordinates(mesh, rectilinear)
    x, y = points_inside_grid(lon, lat, npoints, rng, **inside_kwargs)
    x_outside, y_outside = OUTSIDE_POINTS_XY[mesh]
    x = np.concatenate([x, x_outside])
    y = np.concatenate([y, y_outside])
    z = rng.uniform(5.0, 290.0, x.size)
    t = rng.uniform(0.0, (NT - 1) * TIME_LEVEL_SPACING_HOURS * 3600.0, x.size)
    return t, z, y, x


def to_gpu(*arrays: np.ndarray) -> tuple:
    return tuple(cupy.asarray(array) for array in arrays)


def eval_including_outside_points(field, *points: np.ndarray):
    """Evaluate a Field or VectorField at points of which some lie outside the grid, where values are set to 0."""
    with pytest.warns(FieldEvalWarning, match="out-of-bounds"):
        return field.eval(*points)


def assert_search_matches(gpu_positions: dict, cpu_positions: dict) -> None:
    for axis, cpu_position in cpu_positions.items():
        gpu_index, gpu_bcoord = gpu_positions[axis]["index"], gpu_positions[axis]["bcoord"]
        assert is_gpu_array(gpu_index) and is_gpu_array(gpu_bcoord), axis
        assert gpu_index.dtype == cpu_position["index"].dtype, axis
        assert gpu_bcoord.dtype == cpu_position["bcoord"].dtype, axis
        np.testing.assert_array_equal(to_numpy(gpu_index), cpu_position["index"], err_msg=axis)
        np.testing.assert_allclose(to_numpy(gpu_bcoord), cpu_position["bcoord"], rtol=1e-6, atol=1e-6, err_msg=axis)


def assert_values_match(gpu_values, cpu_values, rtol: float = 1e-6, atol: float | None = None) -> None:
    """Compare interpolated values, by default to float32 round-off of the largest value.

    The CPU blends float32 corners with float32 weights in float32, while the GPU kernels blend in
    float64, so where corners nearly cancel the two differ by more than ``rtol`` of the result.
    """
    assert is_gpu_array(gpu_values)
    if atol is None:
        atol = 8 * np.finfo(np.float32).eps * float(np.max(np.abs(cpu_values), initial=0.0))
    np.testing.assert_allclose(to_numpy(gpu_values), cpu_values, rtol=rtol, atol=atol)


def assert_particles_match(pset_gpu, pset_cpu, variables=("x", "y", "z", "t"), rtol=1e-6, atol=1e-6) -> None:
    assert len(pset_gpu) == len(pset_cpu)
    np.testing.assert_array_equal(to_numpy(pset_gpu.particle_id), pset_cpu.particle_id)
    np.testing.assert_array_equal(to_numpy(pset_gpu.ei), pset_cpu.ei)
    for variable in variables:
        np.testing.assert_allclose(
            to_numpy(getattr(pset_gpu, variable)), getattr(pset_cpu, variable), rtol=rtol, atol=atol, err_msg=variable
        )


def DeleteErroredParticles(particles, fieldset):  # pragma: no cover
    particles[particles.state >= 50].state = parcels.StatusCode.Delete


# ------------------------------------------------------------------------------------------------
# Grid search and spatial hash
# ------------------------------------------------------------------------------------------------
def test_morton_codes_match_cpu_for_python_float_bounds():
    """A flat mesh gives the spatial hash Python float z bounds of zero width."""
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-1.0, 2.0, 500), rng.uniform(-3.0, 1.0, 500)
    z = np.zeros_like(x)
    for bounds in [(0.0, 1.0, -2.0, 0.5, 0.0, 0.0), tuple(np.float64(bound) for bound in (0, 1, -2, 0.5, -1, 1))]:
        codes_cpu = _encode_morton3d(x, y, z, *bounds, bitwidth=1023)
        codes_gpu = _encode_morton3d(*to_gpu(x, y, z), *bounds, bitwidth=1023)
        assert is_gpu_array(codes_gpu)
        np.testing.assert_array_equal(codes_gpu.get(), codes_cpu)


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
def test_spatial_hash_query_matches_cpu_and_reuses_gpu_table(mesh):
    grid = curvilinear_fieldset(mesh, "A", on_gpu=False).U.grid
    spatialhash = grid.get_spatial_hash()
    _, _, y, x = sample_points(mesh)
    y = np.concatenate([y, [np.nan, np.inf]])
    x = np.concatenate([x, [np.nan, np.inf]])

    with np.errstate(invalid="ignore"):
        j_cpu, i_cpu, coords_cpu = spatialhash.query(y, x)
    j_gpu, i_gpu, coords_gpu = spatialhash.query(*to_gpu(y, x))
    assert np.count_nonzero(j_cpu == GRID_SEARCH_ERROR) == 5
    np.testing.assert_array_equal(to_numpy(j_gpu), j_cpu)
    np.testing.assert_array_equal(to_numpy(i_gpu), i_cpu)
    assert_values_match(coords_gpu, coords_cpu, atol=1e-6)

    gpu_table = spatialhash._gpu_hash_table
    j_invalid, _, coords_invalid = spatialhash.query(*to_gpu(y[-2:], x[-2:]))
    assert spatialhash._gpu_hash_table is gpu_table
    np.testing.assert_array_equal(to_numpy(j_invalid), GRID_SEARCH_ERROR)
    np.testing.assert_array_equal(to_numpy(coords_invalid), -1.0)
    for name, host_values in spatialhash._hash_table.items():
        assert isinstance(host_values, np.ndarray), name
        np.testing.assert_array_equal(gpu_table[name].get(), host_values, err_msg=name)


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
def test_curvilinear_search_matches_cpu(mesh):
    grid_cpu = curvilinear_fieldset(mesh, "A", on_gpu=False).U.grid
    grid_gpu = curvilinear_fieldset(mesh, "A", on_gpu=True).U.grid
    _, z, y, x = sample_points(mesh)
    z_gpu, y_gpu, x_gpu = to_gpu(z, y, x)

    # Without a cell index every point goes through the spatial hash
    positions_cpu = grid_cpu.search(z, y, x)
    positions_gpu = grid_gpu.search(z_gpu, y_gpu, x_gpu)
    assert_search_matches(positions_gpu, positions_cpu)
    assert np.count_nonzero(positions_cpu["X"]["index"] == GRID_SEARCH_ERROR) == 3

    lon_gpu, _ = grid_gpu._node_coordinate_values(like=x_gpu)
    assert is_gpu_array(lon_gpu)
    assert grid_gpu._node_coordinate_values(like=x_gpu)[0] is lon_gpu

    # A correct cell index for every other point, and a stale one (another point's cell) for the rest
    ei = grid_cpu.ravel_index({axis: positions_cpu[axis]["index"] for axis in ("X", "Y", "Z")})
    stale_ei = ei.copy()
    stale_ei[::2] = np.roll(ei, 7)[::2]
    stale_positions_cpu = grid_cpu.search(z, y, x, ei=stale_ei)
    stale_positions_gpu = grid_gpu.search(z_gpu, y_gpu, x_gpu, ei=cupy.asarray(stale_ei))
    assert_search_matches(stale_positions_gpu, stale_positions_cpu)
    for axis in ("X", "Y"):
        np.testing.assert_array_equal(to_numpy(stale_positions_gpu[axis]["index"]), positions_cpu[axis]["index"])


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
def test_populate_indices_matches_cpu(mesh):
    _, z, y, x = sample_points(mesh, npoints=100)
    psets = {}
    for on_gpu in (False, True):
        pset = parcels.ParticleSet(curvilinear_fieldset(mesh, "A", on_gpu=on_gpu), x=x, y=y, z=z)
        pset.populate_indices()
        psets[on_gpu] = pset
    assert is_gpu_array(psets[True].ei)
    np.testing.assert_array_equal(psets[True].ei.get(), psets[False].ei)


# ------------------------------------------------------------------------------------------------
# Field evaluation
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mesh", ["spherical", "flat"])
@pytest.mark.parametrize("grid_type", ["A", "C"])
def test_curvilinear_eval_matches_cpu(mesh, grid_type):
    fieldsets = {on_gpu: curvilinear_fieldset(mesh, grid_type, on_gpu=on_gpu) for on_gpu in (False, True)}
    if grid_type == "C":
        assert type(fieldsets[True].UV.interp_method) is GpuCGrid_Velocity
        assert type(fieldsets[True].UVW.interp_method) is GpuCGrid_Velocity
        fieldsets[False].T.interp_method = CGrid_Tracer()
        fieldsets[True].T.interp_method = gpu_counterpart(CGrid_Tracer())
        assert type(fieldsets[True].T.interp_method) is GpuCGrid_Tracer
    t, z, y, x = sample_points(mesh)
    points_gpu = to_gpu(t, z, y, x)

    for vector_field_name in ("UV", "UVW"):
        velocity_cpu = eval_including_outside_points(getattr(fieldsets[False], vector_field_name), t, z, y, x)
        velocity_gpu = eval_including_outside_points(getattr(fieldsets[True], vector_field_name), *points_gpu)
        for component_gpu, component_cpu in zip(velocity_gpu, velocity_cpu, strict=True):
            assert_values_match(component_gpu, component_cpu)

    assert_values_match(
        eval_including_outside_points(fieldsets[True].T, *points_gpu),
        eval_including_outside_points(fieldsets[False].T, t, z, y, x),
    )


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
def test_rectilinear_cgrid_eval_matches_cpu(mesh):
    fieldsets = {}
    for on_gpu in (False, True):
        fieldset = parcels.FieldSet.from_sgrid_conventions(peninsula_dataset(grid_type="C", mesh_str=mesh))
        fieldset.P.interp_method = CGrid_Tracer()
        fieldsets[on_gpu] = fieldset.to_gpu_arrays() if on_gpu else fieldset
    assert type(fieldsets[True].UV.interp_method) is GpuCGrid_Velocity
    assert type(fieldsets[True].P.interp_method) is GpuCGrid_Tracer
    rng = np.random.default_rng(3)
    domain_scale = 1.0 / 1852.0 / 60.0 if mesh == "spherical" else 1.0
    x = rng.uniform(0.0, 1.0e5, 300) * domain_scale
    y = rng.uniform(0.0, 5.0e4, 300) * domain_scale
    t, z = np.zeros_like(x), np.zeros_like(x)
    points_gpu = to_gpu(t, z, y, x)

    velocity_cpu = fieldsets[False].UV.eval(t, z, y, x)
    velocity_gpu = fieldsets[True].UV.eval(*points_gpu)
    for component_gpu, component_cpu in zip(velocity_gpu, velocity_cpu, strict=True):
        assert_values_match(component_gpu, component_cpu)
    assert_values_match(fieldsets[True].P.eval(*points_gpu), fieldsets[False].P.eval(t, z, y, x))


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
@pytest.mark.parametrize("interpolator", [XFreeslip, XPartialslip])
def test_spatial_slip_interpolation_matches_cpu(mesh, interpolator):
    fieldsets = {on_gpu: curvilinear_fieldset(mesh, "A", on_gpu=on_gpu, with_land=True) for on_gpu in (False, True)}
    for fieldset in fieldsets.values():
        fieldset.UV.interp_method = interpolator()
        fieldset.UVW.interp_method = interpolator()
    t, z, y, x = sample_points(mesh, i_range=(9.0, 21.0), j_range=(7.0, 17.0))
    points_gpu = to_gpu(t, z, y, x)

    for vector_field_name in ("UV", "UVW"):
        velocity_cpu = eval_including_outside_points(getattr(fieldsets[False], vector_field_name), t, z, y, x)
        velocity_gpu = eval_including_outside_points(getattr(fieldsets[True], vector_field_name), *points_gpu)
        for component_gpu, component_cpu in zip(velocity_gpu, velocity_cpu, strict=True):
            assert_values_match(component_gpu, component_cpu)


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
@pytest.mark.parametrize("interpolator", [XNearest, XLinearInvdistLandTracer])
def test_scalar_interpolation_matches_cpu(mesh, interpolator):
    fieldsets = {on_gpu: curvilinear_fieldset(mesh, "A", on_gpu=on_gpu, with_land=True) for on_gpu in (False, True)}
    for fieldset in fieldsets.values():
        fieldset.T.interp_method = interpolator()
    t, z, y, x = sample_points(mesh, i_range=(9.0, 21.0), j_range=(7.0, 17.0))

    temperature_cpu = eval_including_outside_points(fieldsets[False].T, t, z, y, x)
    assert np.count_nonzero(temperature_cpu[:-3] == 0.0) > 0 and np.all(temperature_cpu[-3:] == 0.0)
    assert_values_match(eval_including_outside_points(fieldsets[True].T, *to_gpu(t, z, y, x)), temperature_cpu)


# ------------------------------------------------------------------------------------------------
# Fused C-grid kernels against the generic corner gather, both on the GPU
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def fused_cgrid_kernel_calls(monkeypatch) -> Counter:
    """Counts the calls the GPU C-grid interpolators make to their fused kernels."""
    calls = Counter()
    for name in ("sample_cgrid_horizontal_velocity", "sample_cgrid_vertical_velocity", "sample_cgrid_tracer"):
        original = getattr(_gpu_xinterpolators, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(_gpu_xinterpolators, name, counted)
    return calls


def gpu_cgrid_fieldset(
    mesh: str, rectilinear: bool, paddings: FacePaddings, chunked: bool, max_levels: int | None = None
) -> parcels.FieldSet:
    """C-grid fieldset on the GPU whose T is interpolated with ``CGrid_Tracer``; dask-backed (time-windowed) if chunked."""
    if rectilinear:
        ds = rectilinear_cgrid_dataset(mesh, paddings)
    else:
        ds = curvilinear_dataset(mesh, "C", paddings=paddings)
    if chunked:
        ds = ds.chunk({"time": 1})
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds, mesh=MESHES[mesh])
    fieldset.T.interp_method = CGrid_Tracer()
    return fieldset.to_gpu_arrays(max_levels=max_levels)


def eval_cgrid_fields(fieldset: parcels.FieldSet, points: tuple, fused: bool) -> dict[str, object]:
    """UV, UVW and T at ``points``, with the fused GPU interpolators or with the CPU interpolators they extend."""
    velocity_interpolator = GpuCGrid_Velocity if fused else CGrid_Velocity
    tracer_interpolator = GpuCGrid_Tracer if fused else CGrid_Tracer
    fieldset.UV.interp_method = velocity_interpolator()
    fieldset.UVW.interp_method = velocity_interpolator()
    fieldset.T.interp_method = tracer_interpolator()
    values = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FieldEvalWarning)
        for vector_field_name in ("UV", "UVW"):
            velocity = getattr(fieldset, vector_field_name).eval(*points)
            for component_name, component in zip("uvw", velocity, strict=False):
                values[f"{vector_field_name}.{component_name}"] = component
        values["T"] = fieldset.T.eval(*points)
    return values


def assert_fused_matches_generic(fused: dict, generic: dict, context: str) -> None:
    """Fused and generic values agree to float64 round-off, relative to the largest value of each."""
    for name, generic_values in generic.items():
        generic_values = to_numpy(generic_values)
        assert is_gpu_array(fused[name]), name
        np.testing.assert_allclose(
            to_numpy(fused[name]),
            generic_values,
            rtol=1e-12,
            atol=1e-12 * np.max(np.abs(generic_values)),
            err_msg=f"{name}, {context}",
        )


@pytest.mark.parametrize("chunked", [False, True], ids=["numpy", "dask"])
@pytest.mark.parametrize("paddings", [NEMO_PADDINGS, MIXED_PADDINGS], ids=["nemo_paddings", "mixed_paddings"])
@pytest.mark.parametrize("rectilinear", [False, True], ids=["curvilinear", "rectilinear"])
@pytest.mark.parametrize("mesh", ["spherical", "flat"])
def test_fused_cgrid_interpolation_matches_generic_gpu_path(
    mesh, rectilinear, paddings, chunked, fused_cgrid_kernel_calls
):
    """The fused C-grid interpolators against the interpolators they extend, run on the same GPU data.

    The particles include some outside the grid horizontally and vertically, some on the first time
    level (tau == 0), on their own and mixed with others, and some on an interior time level (tau == 1).
    With mixed paddings U, V, W and T all differ in shape. With dask-backed data each field keeps its own
    window of three time levels on the GPU, which every batch of particles fits in, and the levels sit in
    other slots for U than for the other fields.
    """
    fieldset = gpu_cgrid_fieldset(mesh, rectilinear, paddings, chunked, max_levels=3 if chunked else None)
    assert type(fieldset.UV.interp_method) is GpuCGrid_Velocity
    assert type(fieldset.UVW.interp_method) is GpuCGrid_Velocity
    assert type(fieldset.T.interp_method) is GpuCGrid_Tracer

    rng = np.random.default_rng(5)
    t, z, y, x = sample_points(mesh, npoints=300, seed=5, rectilinear=rectilinear)
    z[:10] = -10.0
    z[10:20] = 400.0
    time_level_spacing_s = TIME_LEVEL_SPACING_HOURS * 3600.0
    on_and_between_first_levels_t = rng.uniform(0.0, 1.0, t.size) * time_level_spacing_s
    on_and_between_first_levels_t[::3] = 0.0
    on_and_between_first_levels_t[1::3] = time_level_spacing_s
    time_batches = [
        ("between levels 3 and 4", rng.uniform(3.0, 4.0, t.size) * time_level_spacing_s),
        ("on the first level", np.zeros_like(t)),
        ("on and between levels 0 and 1", on_and_between_first_levels_t),
        ("between levels 1 and 3", rng.uniform(1.0, 3.0, t.size) * time_level_spacing_s),
        ("one time for all", np.array([1.5 * time_level_spacing_s])),
    ]
    if chunked:
        # With U's window holding levels 2 and 3, loading levels 3 and 4 puts them in other slots for U than for V.
        fieldset.U.eval(*to_gpu(np.array([2.5 * time_level_spacing_s]), z[100:101], y[100:101], x[100:101]))
    else:
        time_batches.append(("anywhere", t))

    for batch_name, batch_t in time_batches:
        points = to_gpu(batch_t, z, y, x)
        fused = eval_cgrid_fields(fieldset, points, fused=True)
        generic = eval_cgrid_fields(fieldset, points, fused=False)
        assert_fused_matches_generic(fused, generic, f"particles {batch_name}")
    n_batches = len(time_batches)
    assert fused_cgrid_kernel_calls == {
        "sample_cgrid_horizontal_velocity": 2 * n_batches,
        "sample_cgrid_vertical_velocity": n_batches,
        "sample_cgrid_tracer": n_batches,
    }


def test_fused_cgrid_interpolation_matches_generic_gpu_path_when_time_window_grows():
    """Particles spread over all five time levels make each field's two-level window grow to five."""
    fieldset = gpu_cgrid_fieldset("flat", rectilinear=True, paddings=NEMO_PADDINGS, chunked=True)
    t, z, y, x = sample_points("flat", npoints=300, seed=5, rectilinear=True)
    points = to_gpu(t, z, y, x)
    fused = eval_cgrid_fields(fieldset, points, fused=True)
    generic = eval_cgrid_fields(fieldset, points, fused=False)
    assert_fused_matches_generic(fused, generic, "particles anywhere")


# ------------------------------------------------------------------------------------------------
# Advection
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mesh", ["spherical", "flat"])
def test_cgrid_rk2_advection_matches_cpu(mesh):
    """Particles near the eastern edge are carried out of the grid and deleted."""
    lon, lat = curvilinear_node_coordinates(mesh)
    rng = np.random.default_rng(4)
    x_interior, y_interior = points_inside_grid(lon, lat, 200, rng, i_range=(2.0, NX - 6.0), j_range=(2.0, NY - 3.0))
    x_edge, y_edge = points_inside_grid(lon, lat, 100, rng, i_range=(NX - 1.3, NX - 1.05), j_range=(3.0, NY - 4.0))
    x, y = np.concatenate([x_interior, x_edge]), np.concatenate([y_interior, y_edge])
    z = rng.uniform(5.0, 290.0, x.size)

    psets = {}
    for on_gpu in (False, True):
        fieldset = curvilinear_fieldset(mesh, "C", on_gpu=on_gpu)
        if on_gpu:
            assert type(fieldset.UV.interp_method) is GpuCGrid_Velocity
        pset = parcels.ParticleSet(fieldset, x=x, y=y, z=z)
        pset.execute(
            [AdvectionRK2, DeleteErroredParticles],
            runtime=np.timedelta64(2, "D"),
            dt=np.timedelta64(1, "h"),
            verbose_progress=False,
        )
        psets[on_gpu] = pset
    assert is_gpu_array(psets[True].x)
    assert 200 <= len(psets[False]) < 300
    assert_particles_match(psets[True], psets[False])


def test_stale_cell_indices_fall_back_to_spatial_hash():
    _, z, y, x = sample_points("spherical", npoints=200)
    z, y, x = z[:200], y[:200], x[:200]
    psets = {}
    for on_gpu in (False, True):
        pset = parcels.ParticleSet(curvilinear_fieldset("spherical", "C", on_gpu=on_gpu), x=x, y=y, z=z)
        pset.populate_indices()
        correct_ei = to_numpy(pset.ei)
        stale_ei = correct_ei.copy()
        stale_ei[::2] = np.roll(correct_ei, 11, axis=0)[::2]
        assert np.any(stale_ei != correct_ei)
        pset.ei = cupy.asarray(stale_ei) if on_gpu else stale_ei
        pset.execute(AdvectionRK2, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(1, "h"), verbose_progress=False)
        psets[on_gpu] = pset
    assert_particles_match(psets[True], psets[False])

    reference = parcels.ParticleSet(curvilinear_fieldset("spherical", "C", on_gpu=False), x=x, y=y, z=z)
    reference.execute(AdvectionRK2, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(1, "h"), verbose_progress=False)
    assert_particles_match(psets[True], reference)


def test_nemo_3d_curvilinear_advection_matches_cpu():
    """NEMO C-grid on a curvilinear spherical grid, with W, from the tutorial data."""
    U = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/U")
    V = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/V")
    W = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/W")
    coords = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/mesh_mask")
    ds = parcels.convert.nemo_to_sgrid(fields=dict(U=U["uo"], V=V["vo"], W=W["wo"]), coords=coords)

    npart = 10
    x = np.linspace(1.9, 3.4, npart)
    y = np.linspace(52.5, 51.6, npart)
    z = np.ones(npart)
    psets = {}
    for on_gpu in (False, True):
        fieldset = parcels.FieldSet.from_sgrid_conventions(ds)
        if on_gpu:
            fieldset.to_gpu_arrays()
            assert type(fieldset.UVW.interp_method) is GpuCGrid_Velocity
        pset = parcels.ParticleSet(fieldset, x=x, y=y, z=z)
        pset.execute(
            AdvectionRK4_3D,
            runtime=np.timedelta64(3, "D") + np.timedelta64(18, "h"),
            dt=np.timedelta64(6, "h"),
            verbose_progress=False,
        )
        psets[on_gpu] = pset
    assert is_gpu_array(psets[True].z)
    assert_particles_match(psets[True], psets[False])


def test_nemo_global_curvilinear_advection_matches_cpu():
    """Global ORCA025 grid from the tutorial data, up to the tripolar fold.

    One fieldset serves both runs, so the large spatial hash is built once on the host and
    copied to the GPU for the second run.
    """
    U = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/U")
    V = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/V")
    coords = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/mesh_mask")
    ds = parcels.convert.nemo_to_sgrid(fields=dict(U=U, V=V), coords=coords)
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds)

    npart = 20
    x = 30 * np.ones(npart)
    y = np.linspace(-70, 88, npart)
    psets = {}
    for on_gpu in (False, True):
        if on_gpu:
            fieldset.to_gpu_arrays()
            assert type(fieldset.UV.interp_method) is GpuCGrid_Velocity
        pset = parcels.ParticleSet(fieldset, x=x, y=y)
        pset.execute(AdvectionRK2, runtime=np.timedelta64(160, "D"), dt=np.timedelta64(10, "D"), verbose_progress=False)
        psets[on_gpu] = pset
    assert is_gpu_array(psets[True].x)
    assert fieldset.U.grid.get_spatial_hash()._gpu_hash_table is not None
    assert_particles_match(psets[True], psets[False])
