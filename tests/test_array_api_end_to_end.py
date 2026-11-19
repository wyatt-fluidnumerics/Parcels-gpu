"""Whole simulations on array libraries other than NumPy, compared with the same simulation on NumPy.

Each simulation builds its FieldSet from NumPy data and then holds every field's data in the library
under test, so that the FieldSet reports that library and the ParticleSet allocates its particle data
there. At the end of the run every particle variable must still be an array of that library and must be
bitwise equal to the NumPy run, as must the particle file written during the run.

``array-api-strict`` computes with NumPy, so it is bitwise equal wherever it runs. PyTorch's float64
``sqrt``, ``sin`` and ``cos`` on the CPU are not always correctly rounded (about 1 % of ``sqrt`` and
0.15 % of ``sin``/``cos`` results are one unit in the last place away from NumPy's, measured with torch
2.14 on an AVX512 CPU). With the default float32 particle positions these differences are rounded away
in every simulation here; with float64 positions they can show, so that test compares PyTorch to within
a few units in the last place.
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow.parquet as pq
import pytest
import xarray as xr
from array_api_compat import is_numpy_namespace, is_torch_namespace
from xarray.core.indexing import ExplicitlyIndexedNDArrayMixin, NumpyIndexingAdapter

import parcels._sgrid as sgrid
from parcels import FieldOutOfBoundError, FieldSet, Particle, ParticleFile, ParticleSet, StatusCode, Variable
from parcels._core.mesh import FlatMesh, SphericalMesh
from parcels._core.particle import ParticleClass
from parcels._core.utils.array_backend import ArrayPlacement, array_namespace, namespace_of_library, to_numpy
from parcels._core.warnings import FieldSetWarning
from parcels._datasets.structured.generated import peninsula_dataset, simple_UV_dataset, simple_UV_sigma_dataset
from parcels.interpolators import XFreeslip, XPartialslip
from parcels.kernels import (
    AdvectionDiffusionEM,
    AdvectionDiffusionM1,
    AdvectionEE,
    AdvectionRK2,
    AdvectionRK2_3D,
    AdvectionRK4,
    AdvectionRK4_3D,
    AdvectionRK45,
    DiffusionUniformKh,
)

LIBRARIES = ["numpy", "array_api_strict", "torch"]
NUMPY_NAMESPACE = namespace_of_library(np)
MESHES = {"flat": FlatMesh(), "spherical": SphericalMesh()}


@pytest.fixture(params=LIBRARIES)
def library(request: pytest.FixtureRequest) -> str:
    """Name of the array library under test."""
    pytest.importorskip(request.param)
    return request.param


@pytest.fixture
def xp(library: str) -> ModuleType:
    """Array API namespace of the library under test."""
    return namespace_of_library(importlib.import_module(library))


# ------------------------------------------------------------------------------------------------
# Field data held in an array library
# ------------------------------------------------------------------------------------------------
class _TensorBackend(ExplicitlyIndexedNDArrayMixin):
    """A Parcels backend array that holds a field's data in a PyTorch tensor and gathers there.

    xarray converts a tensor stored in a variable to NumPy, so the tensor sits in this wrapper. It reports
    a NumPy data type, which is what xarray expects, and answers whatever xarray itself indexes from a
    NumPy copy.
    """

    def __init__(self, values: np.ndarray, xp: ModuleType):
        self.array = xp.asarray(values)
        self.array_placement = ArrayPlacement.of(self.array)
        self._numpy_dtype = values.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._numpy_dtype

    def _raw_vindex(self, *indices):
        return self.array[indices]

    def get_duck_array(self) -> np.ndarray:
        return to_numpy(self.array)

    def __getitem__(self, indexer):
        return NumpyIndexingAdapter(self.get_duck_array())[indexer]


def _hold_field_data_in(fieldset: FieldSet, xp: ModuleType) -> None:
    """Replace the stored data of every field of ``fieldset`` by an array of ``xp``.

    xarray holds an ``array-api-strict`` array as it is, but would convert a PyTorch tensor to NumPy, so
    PyTorch data is held in a stand-in backend. Constant fields are not part of ``fieldset.models`` and
    stay NumPy.
    """
    if is_numpy_namespace(xp):
        return
    for model in fieldset.models:
        for name in model.scalar_field_names:
            variable = model.data.variables[name]
            values = np.asarray(variable._data)
            variable._data = _TensorBackend(values, xp) if is_torch_namespace(xp) else xp.asarray(values)


# ------------------------------------------------------------------------------------------------
# Datasets
# ------------------------------------------------------------------------------------------------
HOUR = np.timedelta64(1, "h")
N_TIME_LEVELS = 4
TIME_LEVEL_SPACING = np.timedelta64(3, "h")
TIME_LEVELS = np.datetime64("2000-01-01") + np.arange(N_TIME_LEVELS) * TIME_LEVEL_SPACING

# Extent of the rectilinear A-grid, in metres on a flat mesh and in degrees on a sphere
RECTILINEAR_LON = {"flat": (0.0, 600e3), "spherical": (-20.0, 20.0)}
RECTILINEAR_LAT = {"flat": (0.0, 330e3), "spherical": (-15.0, 15.0)}


def _fraction_grids(shape: tuple[int, int, int, int]) -> tuple[np.ndarray, ...]:
    """Time-level index and depth, y and x positions as fractions of the grid, broadcastable to ``shape``."""
    nt, nz, ny, nx = shape
    t_index = np.arange(nt, dtype=float)[:, None, None, None]
    z_fraction = np.linspace(0.0, 1.0, nz)[None, :, None, None]
    j_fraction = np.linspace(0.0, 1.0, ny)[None, None, :, None]
    i_fraction = np.linspace(0.0, 1.0, nx)[None, None, None, :]
    return t_index, z_fraction, j_fraction, i_fraction


def _analytic_fields(shape: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    """Smooth, time- and depth-varying eastward-leaning U, V, W (m/s), temperature T (degC) and diffusivities (m2/s)."""
    t_index, z_fraction, j_fraction, i_fraction = _fraction_grids(shape)
    scale = (1.0 + 0.3 * t_index) * (1.0 - 0.5 * z_fraction)
    fields = {
        "U": 10.0 * (0.6 + 0.4 * np.sin(2 * np.pi * j_fraction) * np.cos(np.pi * i_fraction)) * scale,
        "V": 4.0 * np.sin(2 * np.pi * i_fraction) * np.cos(np.pi * j_fraction) * scale,
        "W": 2e-3 * np.sin(np.pi * i_fraction) * np.sin(np.pi * j_fraction) * scale,
        "T": 20.0 + 5.0 * i_fraction - 3.0 * j_fraction + t_index - 2.0 * z_fraction,
        "Kh_zonal": 40.0 + 30.0 * i_fraction * j_fraction + 0.0 * scale,
        "Kh_meridional": 25.0 + 20.0 * (1.0 - i_fraction) + 0.0 * scale,
    }
    return {name: np.broadcast_to(values, shape).copy() for name, values in fields.items()}


def rectilinear_agrid_dataset(mesh_str: str, nz: int = 2, field_names: tuple[str, ...] = ("U", "V")) -> xr.Dataset:
    """A-grid on 1D lon and lat, with the fields ``field_names`` of ``_analytic_fields`` on the nodes."""
    ds = simple_UV_dataset(n_times=N_TIME_LEVELS, nz=nz, ny=12, nx=10, maxdepth=200.0, mesh_str=mesh_str)
    ds = ds.assign_coords(
        time=(["time"], TIME_LEVELS, {"axis": "T"}),
        lon=(["XG"], np.linspace(*RECTILINEAR_LON[mesh_str], ds.sizes["XG"]), ds["lon"].attrs),
        lat=(["YG"], np.linspace(*RECTILINEAR_LAT[mesh_str], ds.sizes["YG"]), ds["lat"].attrs),
    )
    dims = ds["U"].dims
    fields = _analytic_fields(ds["U"].shape)
    ds = ds.drop_vars(["U", "V"])
    for name in field_names:
        ds[name] = (dims, fields[name])
    return ds


def rectilinear_positions(mesh_str: str, npart: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Random x and y in the middle of the rectilinear A-grid, which a few hours of flow either way do not leave."""
    rng = np.random.default_rng(seed)
    lon0, lon1 = RECTILINEAR_LON[mesh_str]
    lat0, lat1 = RECTILINEAR_LAT[mesh_str]
    x = lon0 + (lon1 - lon0) * rng.uniform(0.42, 0.58, npart)
    y = lat0 + (lat1 - lat0) * rng.uniform(0.35, 0.65, npart)
    return {"x": x, "y": y}


# Curvilinear grid in the NEMO layout: U on the x faces, V on the y faces, W and T at the cell centres
CURVILINEAR_NX, CURVILINEAR_NY, CURVILINEAR_NZ = 14, 12, 3
CURVILINEAR_DEPTH_M = np.linspace(0.0, 300.0, CURVILINEAR_NZ)


def curvilinear_node_coordinates(mesh_str: str) -> tuple[np.ndarray, np.ndarray]:
    """Rotated and smoothly distorted node lon/lat of shape (NY, NX): degrees on a sphere, metres on a flat mesh."""
    i_index, j_index = np.meshgrid(np.arange(CURVILINEAR_NX, dtype=float), np.arange(CURVILINEAR_NY, dtype=float))
    rotation_rad = np.deg2rad(20.0)
    distortion_cells = 0.15 * np.sin(2 * np.pi * j_index / CURVILINEAR_NY) * np.cos(np.pi * i_index / CURVILINEAR_NX)
    along_x_cells = i_index * np.cos(rotation_rad) - j_index * np.sin(rotation_rad) + distortion_cells
    along_y_cells = i_index * np.sin(rotation_rad) + j_index * np.cos(rotation_rad) - distortion_cells
    if mesh_str == "spherical":
        cell_size_deg = 0.5
        return -10.0 + cell_size_deg * along_x_cells, -5.0 + cell_size_deg * along_y_cells
    cell_size_m = 50e3
    return cell_size_m * along_x_cells, cell_size_m * along_y_cells


def curvilinear_dataset(mesh_str: str, grid_type: str) -> xr.Dataset:
    nt, nz, ny, nx = N_TIME_LEVELS, CURVILINEAR_NZ, CURVILINEAR_NY, CURVILINEAR_NX
    padding_xy, padding_z = sgrid.Padding.LOW, sgrid.Padding.HIGH
    dim_sizes = {
        "time": nt,
        "depth": nz,
        "depth_center": sgrid.get_n_faces(nz, padding_z),
        "y": ny,
        "y_center": sgrid.get_n_faces(ny, padding_xy),
        "x": nx,
        "x_center": sgrid.get_n_faces(nx, padding_xy),
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
    fields = _analytic_fields((nt, nz, ny, nx))
    data_vars = {}
    for name, dims in field_dims.items():
        data_vars[name] = (dims, fields[name][tuple(slice(dim_sizes[dim]) for dim in dims)])
    lon, lat = curvilinear_node_coordinates(mesh_str)
    horizontal_units = "degrees" if mesh_str == "spherical" else "m"
    depth_center_m = (CURVILINEAR_DEPTH_M + 0.5 * CURVILINEAR_DEPTH_M[1])[: dim_sizes["depth_center"]]
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": (["time"], TIME_LEVELS, {"axis": "T"}),
            "depth": (["depth"], CURVILINEAR_DEPTH_M, {"axis": "Z", "units": "m", "positive": "down"}),
            "depth_center": (["depth_center"], depth_center_m, {"axis": "Z"}),
            "x": (["x"], np.arange(nx), {"axis": "X"}),
            "x_center": (["x_center"], np.arange(dim_sizes["x_center"]), {"axis": "X"}),
            "y": (["y"], np.arange(ny), {"axis": "Y"}),
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
                sgrid.FaceNodePadding("x_center", "x", padding_xy),
                sgrid.FaceNodePadding("y_center", "y", padding_xy),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("depth_center", "depth", padding_z),),
        ),
    )


def curvilinear_positions(mesh_str: str, npart: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Bilinear blends of the four nodes around random fractional node indices in the middle of the curvilinear grid."""
    rng = np.random.default_rng(seed)
    lon, lat = curvilinear_node_coordinates(mesh_str)
    i_fractional = rng.uniform(4.0, 9.0, npart)
    j_fractional = rng.uniform(3.0, 8.0, npart)
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

    return {"x": blend(lon), "y": blend(lat), "z": rng.uniform(20.0, 250.0, npart)}


# ------------------------------------------------------------------------------------------------
# Kernels and particle classes
# ------------------------------------------------------------------------------------------------
def SampleTemperature(particles, fieldset):  # pragma: no cover
    particles.temperature = fieldset.T[particles]


def DeleteOutOfBounds(particles, fieldset):  # pragma: no cover
    particles.state[particles.state == StatusCode.ErrorOutOfBounds] = StatusCode.Delete


def Age(particles, fieldset):  # pragma: no cover
    particles.age += particles.dt


def InPlaceAdvectionEE(particles, fieldset):  # pragma: no cover
    """AdvectionEE as users often write it: float64 velocity steps added in place to float32 displacements."""
    (u, v) = fieldset.UV[particles]
    particles.dx += u * particles.dt
    particles.dy += v * particles.dt


SamplingParticle = Particle.add_variable(
    [Variable("temperature", dtype=np.float32, initial=np.nan), Variable("age", dtype=np.float64, initial=0.0)]
)
RK45Particle = Particle.add_variable(Variable("next_dt"))


def with_float64_positions(pclass: ParticleClass) -> ParticleClass:
    """``pclass`` with float64 positions and displacements instead of float32."""
    variables = []
    for variable in pclass.variables:
        if variable.name in ("x", "y", "z", "dx", "dy", "dz"):
            variable = Variable(variable.name, dtype=np.float64, to_write=variable.to_write, attrs=variable.attrs)
        variables.append(variable)
    return ParticleClass(variables)


# ------------------------------------------------------------------------------------------------
# Simulations
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Simulation:
    """A FieldSet, a release of particles and a run of kernels over them."""

    build_fieldset: Callable[[], FieldSet]
    release: Callable[[], dict[str, np.ndarray]]
    kernels: tuple[Callable, ...]
    dt: np.timedelta64
    runtime: np.timedelta64
    pclass: ParticleClass = Particle
    context: dict[str, float] = field(default_factory=dict)
    seed: int | None = None


def _rectilinear_agrid_fieldset(mesh_str: str, nz: int = 2, field_names=("U", "V"), constant_fields=None):
    def build() -> FieldSet:
        ds = rectilinear_agrid_dataset(mesh_str, nz=nz, field_names=field_names)
        fieldset = FieldSet.from_sgrid_conventions(ds, mesh=MESHES[mesh_str])
        for name, value in (constant_fields or {}).items():
            fieldset.add_constant_field(name, value)
        return fieldset

    return build


def _rectilinear_release(mesh_str: str, npart: int = 10, with_z: bool = False, t=None):
    def release() -> dict[str, np.ndarray]:
        positions = rectilinear_positions(mesh_str, npart)
        if with_z:
            positions["z"] = np.random.default_rng(1).uniform(60.0, 140.0, npart)
        if t is not None:
            positions["t"] = t
        return positions

    return release


def _agrid_advection(kernel, mesh_str: str, direction: int) -> Simulation:
    return Simulation(
        build_fieldset=_rectilinear_agrid_fieldset(mesh_str),
        release=_rectilinear_release(mesh_str),
        kernels=(kernel,),
        dt=direction * HOUR,
        runtime=np.timedelta64(6, "h"),
    )


def _agrid_advection_3d(kernel) -> Simulation:
    return Simulation(
        build_fieldset=_rectilinear_agrid_fieldset("flat", nz=4, field_names=("U", "V", "W")),
        release=_rectilinear_release("flat", with_z=True),
        kernels=(kernel,),
        dt=HOUR,
        runtime=np.timedelta64(6, "h"),
    )


def _peninsula_cgrid(mesh_str: str) -> Simulation:
    def build() -> FieldSet:
        return FieldSet.from_sgrid_conventions(
            peninsula_dataset(xdim=20, ydim=10, mesh_str=mesh_str, grid_type="C"), mesh=MESHES[mesh_str]
        )

    def release() -> dict[str, np.ndarray]:
        rng = np.random.default_rng(2)
        x = rng.uniform(5e3, 30e3, 8)
        y = rng.uniform(5e3, 45e3, 8)
        if mesh_str == "spherical":
            x, y = x / 1852.0 / 60.0, y / 1852.0 / 60.0
        return {"x": x, "y": y}

    return Simulation(
        build_fieldset=build,
        release=release,
        kernels=(AdvectionRK4,),
        dt=np.timedelta64(10, "m"),
        runtime=np.timedelta64(2, "h"),
    )


def _peninsula_agrid() -> Simulation:
    """The peninsula on an A-grid, whose velocities are float32."""

    def build() -> FieldSet:
        return FieldSet.from_sgrid_conventions(peninsula_dataset(xdim=20, ydim=10, grid_type="A"), mesh=MESHES["flat"])

    return Simulation(
        build_fieldset=build,
        release=lambda: {"x": np.linspace(5e3, 30e3, 8), "y": np.linspace(5e3, 45e3, 8)},
        kernels=(AdvectionRK4,),
        dt=np.timedelta64(10, "m"),
        runtime=np.timedelta64(2, "h"),
    )


def _curvilinear(grid_type: str, mesh_str: str, kernel) -> Simulation:
    def build() -> FieldSet:
        return FieldSet.from_sgrid_conventions(curvilinear_dataset(mesh_str, grid_type), mesh=MESHES[mesh_str])

    return Simulation(
        build_fieldset=build,
        release=lambda: curvilinear_positions(mesh_str, 10),
        kernels=(kernel,),
        dt=HOUR,
        runtime=np.timedelta64(6, "h"),
    )


def _slip(interpolator_class) -> Simulation:
    """The curvilinear A-grid with a block of land, which the slip interpolators treat at its edges."""

    def build() -> FieldSet:
        ds = curvilinear_dataset("flat", "A")
        for name in ("U", "V", "W"):
            ds[name].values[..., 5:7, 6:9] = 0.0
        fieldset = FieldSet.from_sgrid_conventions(ds, mesh=MESHES["flat"])
        fieldset.UV.interp_method = interpolator_class()
        fieldset.UVW.interp_method = interpolator_class()
        return fieldset

    return Simulation(
        build_fieldset=build,
        release=lambda: curvilinear_positions("flat", 10),
        kernels=(AdvectionRK2,),
        dt=HOUR,
        runtime=np.timedelta64(6, "h"),
    )


SIGMA_SHAPE = {"time": 4, "depth": 4, "y": 6, "x": 7}


def _sigma_depth() -> Simulation:
    """A flat-bottomed sigma grid whose layer depths change in time with the surface elevation."""
    n_times, nz, ny, nx = SIGMA_SHAPE.values()

    def build() -> FieldSet:
        rng = np.random.default_rng(5)
        bottom_depth_m = 10.0 + rng.uniform(0.0, 2.0, (ny, nx))
        surface_elevation_m = rng.uniform(-0.3, 0.3, (n_times, ny, nx))
        ds = simple_UV_sigma_dataset(n_times, nz, ny, nx, bottom_depth=bottom_depth_m, eta=surface_elevation_m)
        fields = _analytic_fields(ds["U"].shape)
        ds["U"] = (ds["U"].dims, 1e-5 * fields["U"])
        ds["V"] = (ds["V"].dims, 1e-5 * fields["V"])
        ds["W"] = (ds["W"].dims, 1e-3 * fields["W"])
        with pytest.warns(FieldSetWarning, match="Time-varying"):
            return FieldSet.from_sgrid_conventions(ds)

    def release() -> dict[str, np.ndarray]:
        rng = np.random.default_rng(6)
        lon = simple_UV_dataset(n_times=n_times, nz=nz, ny=ny, nx=nx, mesh_str="flat")["lon"].values
        lat = simple_UV_dataset(n_times=n_times, nz=nz, ny=ny, nx=nx, mesh_str="flat")["lat"].values
        return {
            "x": rng.uniform(lon[1], lon[-2], 6),
            "y": rng.uniform(lat[1], lat[-2], 6),
            "z": rng.uniform(2.0, 8.0, 6),
        }

    return Simulation(
        build_fieldset=build,
        release=release,
        kernels=(AdvectionRK4_3D,),
        dt=np.timedelta64(10, "m"),
        runtime=np.timedelta64(1, "h"),
    )


# Released at different times, so that the kernels first run on views of part of the set. The last three
# particles start near the eastern edge and flow out of the domain, where DeleteOutOfBounds deletes them.
STAGGERED_RELEASE_HOURS = np.array([0, 0, 1, 1, 2, 3, 5, 0, 1, 2])


def _staggered_release() -> dict[str, np.ndarray]:
    positions = rectilinear_positions("flat", STAGGERED_RELEASE_HOURS.size, seed=3)
    positions["x"][-3:] = [585e3, 590e3, 595e3]
    positions["t"] = STAGGERED_RELEASE_HOURS * HOUR
    return positions


STAGGERED = Simulation(
    build_fieldset=_rectilinear_agrid_fieldset("flat", field_names=("U", "V", "T")),
    release=_staggered_release,
    kernels=(AdvectionRK4, SampleTemperature, Age, DeleteOutOfBounds),
    dt=HOUR,
    runtime=np.timedelta64(7, "h"),
    pclass=SamplingParticle,
)


def _diffusion(kernel) -> Simulation:
    if kernel is DiffusionUniformKh:
        build = _rectilinear_agrid_fieldset("flat", constant_fields={"Kh_zonal": 50.0, "Kh_meridional": 20.0})
        kernels = (AdvectionRK2, kernel)
    else:
        build = _rectilinear_agrid_fieldset("flat", field_names=("U", "V", "Kh_zonal", "Kh_meridional"))
        kernels = (kernel,)
    return Simulation(
        build_fieldset=build,
        release=_rectilinear_release("flat"),
        kernels=kernels,
        dt=HOUR,
        runtime=np.timedelta64(6, "h"),
        context={"dres": 1000.0},
        seed=1234,
    )


SIMULATIONS: dict[str, Simulation] = {}
for _mesh_str in MESHES:
    for _kernel in (AdvectionRK2, AdvectionRK4, AdvectionEE):
        for _direction, _direction_name in ((1, "forward"), (-1, "backward")):
            SIMULATIONS[f"agrid-{_mesh_str}-{_kernel.__name__}-{_direction_name}"] = _agrid_advection(
                _kernel, _mesh_str, _direction
            )
for _kernel in (AdvectionRK2_3D, AdvectionRK4_3D):
    SIMULATIONS[f"agrid-flat-{_kernel.__name__}"] = _agrid_advection_3d(_kernel)
SIMULATIONS["peninsula-agrid-float32-data-AdvectionRK4"] = _peninsula_agrid()
for _mesh_str in MESHES:
    SIMULATIONS[f"peninsula-cgrid-{_mesh_str}-AdvectionRK4"] = _peninsula_cgrid(_mesh_str)
    for _grid_type in ("A", "C"):
        SIMULATIONS[f"curvilinear-{_grid_type}grid-{_mesh_str}-AdvectionRK2"] = _curvilinear(
            _grid_type, _mesh_str, AdvectionRK2
        )
SIMULATIONS["curvilinear-Cgrid-spherical-AdvectionRK4_3D"] = _curvilinear("C", "spherical", AdvectionRK4_3D)
SIMULATIONS["staggered-release-sampling-deletion"] = STAGGERED
SIMULATIONS["agrid-flat-AdvectionRK45"] = Simulation(
    build_fieldset=_rectilinear_agrid_fieldset("flat"),
    release=_rectilinear_release("flat"),
    kernels=(AdvectionRK45,),
    dt=HOUR,
    runtime=np.timedelta64(6, "h"),
    pclass=RK45Particle,
    context={"RK45_tol": 20.0, "RK45_min_dt": 60, "RK45_max_dt": 3 * 3600},
)
for _kernel in (DiffusionUniformKh, AdvectionDiffusionM1, AdvectionDiffusionEM):
    SIMULATIONS[f"diffusion-{_kernel.__name__}"] = _diffusion(_kernel)
for _interpolator_class in (XFreeslip, XPartialslip):
    SIMULATIONS[f"curvilinear-land-{_interpolator_class.__name__}-AdvectionRK2"] = _slip(_interpolator_class)
SIMULATIONS["agrid-flat-InPlaceAdvectionEE"] = Simulation(
    build_fieldset=_rectilinear_agrid_fieldset("flat"),
    release=_rectilinear_release("flat"),
    kernels=(InPlaceAdvectionEE,),
    dt=HOUR,
    runtime=np.timedelta64(6, "h"),
)
SIMULATIONS["sigma-time-varying-depth-AdvectionRK4_3D"] = _sigma_depth()

@dataclass(frozen=True)
class KnownFailure:
    reason: str
    error: type[Exception]


_MIXED_FLOAT_WRITE = KnownFailure(
    "array-api-strict refuses to write float64 values into a float32 array, in place or through __setitem__, "
    "where NumPy and PyTorch cast them (the standard leaves that cast to the library)",
    TypeError,
)
_TIME_VARYING_DEPTH_IS_NUMPY_ONLY = (
    "the vertical search in a time-varying (4D) depth, XGrid._search_time_varying_depth and "
    "index_search._search_1d_columns, runs on NumPy and xarray only"
)
EXPECTED_FAILURES: dict[tuple[str, str], KnownFailure] = {
    # The slip interpolators write float64 products into their slip factors, which are float32 because
    # the bcoords of a curvilinear search without cell guesses are
    ("array_api_strict", "curvilinear-land-XFreeslip-AdvectionRK2"): _MIXED_FLOAT_WRITE,
    ("array_api_strict", "curvilinear-land-XPartialslip-AdvectionRK2"): _MIXED_FLOAT_WRITE,
    ("array_api_strict", "agrid-flat-InPlaceAdvectionEE"): _MIXED_FLOAT_WRITE,
    ("array_api_strict", "sigma-time-varying-depth-AdvectionRK4_3D"): KnownFailure(
        _TIME_VARYING_DEPTH_IS_NUMPY_ONLY, AttributeError
    ),
    ("torch", "sigma-time-varying-depth-AdvectionRK4_3D"): KnownFailure(_TIME_VARYING_DEPTH_IS_NUMPY_ONLY, TypeError),
}


def expect_known_failure(request: pytest.FixtureRequest, library: str, name: str) -> None:
    known_failure = EXPECTED_FAILURES.get((library, name))
    if known_failure is not None:
        request.applymarker(pytest.mark.xfail(reason=known_failure.reason, raises=known_failure.error, strict=True))


def run(simulation: Simulation, xp: ModuleType, output_path: Path | None = None) -> ParticleSet:
    fieldset = simulation.build_fieldset()
    for name, value in simulation.context.items():
        fieldset.add_context(name, value)
    _hold_field_data_in(fieldset, xp)
    assert fieldset.array_placement.namespace is xp

    pset = ParticleSet(fieldset, pclass=simulation.pclass, **simulation.release())
    for name, values in pset._data.items():
        assert array_namespace(values) is xp, name

    output_file = None if output_path is None else ParticleFile(output_path, outputdt=HOUR)
    if simulation.seed is not None:
        np.random.seed(simulation.seed)
    pset.execute(
        list(simulation.kernels),
        dt=simulation.dt,
        runtime=simulation.runtime,
        output_file=output_file,
        verbose_progress=False,
    )
    return pset


_NUMPY_RESULTS: dict[tuple[str, bool], dict[str, np.ndarray]] = {}


def numpy_result(name: str, float64_positions: bool = False) -> dict[str, np.ndarray]:
    """Particle data at the end of the NumPy run of ``SIMULATIONS[name]``, computed once per session."""
    key = (name, float64_positions)
    if key not in _NUMPY_RESULTS:
        simulation = SIMULATIONS[name]
        if float64_positions:
            simulation = replace(simulation, pclass=with_float64_positions(simulation.pclass))
        pset = run(simulation, NUMPY_NAMESPACE)
        _NUMPY_RESULTS[key] = {variable: to_numpy(values).copy() for variable, values in pset._data.items()}
    return _NUMPY_RESULTS[key]


def assert_bitwise_equal(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    assert actual.dtype == expected.dtype, name
    np.testing.assert_array_equal(actual, expected, err_msg=name)
    assert actual.tobytes() == expected.tobytes(), name  # also tells -0.0 from 0.0


def assert_within_ulp(actual: np.ndarray, expected: np.ndarray, name: str, max_ulp: int) -> None:
    """Equal integers, NaN where NumPy has NaN, and floating values within ``max_ulp`` units in the last place."""
    assert actual.dtype == expected.dtype, name
    if not np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_array_equal(actual, expected, err_msg=name)
        return
    is_nan = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), is_nan, err_msg=name)
    np.testing.assert_array_max_ulp(actual[~is_nan], expected[~is_nan], maxulp=max_ulp)


def assert_matches_numpy(pset: ParticleSet, expected: dict[str, np.ndarray], xp: ModuleType, max_ulp: int = 0) -> None:
    assert len(pset) == expected["particle_id"].shape[0]
    assert pset._data.keys() == expected.keys()
    for name, values in pset._data.items():
        assert array_namespace(values) is xp, name
        if not is_numpy_namespace(xp):
            assert not isinstance(values, np.ndarray), name
        if max_ulp == 0:
            assert_bitwise_equal(to_numpy(values), expected[name], name)
        else:
            assert_within_ulp(to_numpy(values), expected[name], name, max_ulp)


# ------------------------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(SIMULATIONS))
def test_simulation_matches_numpy(request, library, xp, name):
    expect_known_failure(request, library, name)
    pset = run(SIMULATIONS[name], xp)
    assert_matches_numpy(pset, numpy_result(name), xp)


FLOAT64_SIMULATIONS = [
    "agrid-spherical-AdvectionRK4-forward",
    "agrid-flat-AdvectionRK4_3D",
    "peninsula-cgrid-spherical-AdvectionRK4",
    "curvilinear-Agrid-spherical-AdvectionRK2",
    "curvilinear-Cgrid-flat-AdvectionRK2",
    "curvilinear-Cgrid-spherical-AdvectionRK2",
    "curvilinear-Cgrid-spherical-AdvectionRK4_3D",
    "staggered-release-sampling-deletion",
    "agrid-flat-AdvectionRK45",
    "diffusion-AdvectionDiffusionM1",
]
# Measured: up to 4 units in the last place, on the curvilinear C-grid on a sphere, where the cell edge
# lengths are square roots of sums of two non-zero squares.
TORCH_FLOAT64_MAX_ULP = 16


@pytest.mark.parametrize("name", FLOAT64_SIMULATIONS)
def test_simulation_with_float64_positions_matches_numpy(library, xp, name):
    simulation = SIMULATIONS[name]
    pset = run(replace(simulation, pclass=with_float64_positions(simulation.pclass)), xp)
    max_ulp = TORCH_FLOAT64_MAX_ULP if library == "torch" else 0
    assert_matches_numpy(pset, numpy_result(name, float64_positions=True), xp, max_ulp=max_ulp)


def test_particles_leaving_the_domain_raise(xp):
    fieldset = _rectilinear_agrid_fieldset("flat")()
    _hold_field_data_in(fieldset, xp)
    pset = ParticleSet(fieldset, x=[300e3, 595e3], y=[150e3, 150e3])
    with pytest.raises(FieldOutOfBoundError, match="out-of-bound"):
        pset.execute(AdvectionRK4, dt=HOUR, runtime=np.timedelta64(4, "h"), verbose_progress=False)


def test_simulations_exercise_what_they_are_meant_to():
    staggered = numpy_result("staggered-release-sampling-deletion")
    assert staggered["particle_id"].shape[0] == STAGGERED_RELEASE_HOURS.size - 3
    assert np.all(np.isfinite(staggered["temperature"]))
    assert len(np.unique(staggered["age"])) > 1

    # RK45 doubled the time step of some particles and not of others
    rk45 = numpy_result("agrid-flat-AdvectionRK45")
    assert len(np.unique(rk45["next_dt"])) > 1


def test_particlefile_matches_numpy(xp, tmp_path):
    pset = run(STAGGERED, xp, tmp_path / "library.parquet")
    reference = run(STAGGERED, NUMPY_NAMESPACE, tmp_path / "numpy.parquet")
    assert_matches_numpy(pset, {name: to_numpy(values) for name, values in reference._data.items()}, xp)

    table = pq.read_table(tmp_path / "library.parquet")
    expected = pq.read_table(tmp_path / "numpy.parquet")
    assert expected.num_rows > STAGGERED_RELEASE_HOURS.size
    assert table.schema.equals(expected.schema, check_metadata=True)
    assert table.num_rows == expected.num_rows
    # Column by column rather than with `Table.equals`, which takes NaN (the temperature of particles
    # written before their first sample) as unequal to itself.
    for name in expected.column_names:
        assert_bitwise_equal(table.column(name).to_numpy(), expected.column(name).to_numpy(), name)
