"""Tests of the windowed backend holding field data in CuPy. They need CuPy and a GPU, and are skipped otherwise."""

import io

import numpy as np
import pyarrow.parquet as pq
import pytest
import xarray as xr
from array_api_compat import is_cupy_array, is_cupy_namespace

import parcels
import parcels._sgrid as sgrid
from parcels._core.mesh import FlatMesh
from parcels._core.utils.array_backend import NUMPY_PLACEMENT, to_numpy
from parcels._core.warnings import FieldSetWarning
from parcels._datasets.structured.generated import radial_rotation_dataset
from parcels._prefetch import NoPrefetcher
from parcels._windowed_array import WindowedArray
from parcels.kernels import AdvectionEE, AdvectionRK2, AdvectionRK4

cupy = pytest.importorskip("cupy")
try:
    cupy.cuda.runtime.getDeviceCount()
except Exception:
    pytest.skip("no GPU available", allow_module_level=True)


def rectilinear_dataset(nx: int = 60, ny: int = 40, nz: int = 6, nt: int = 4, chunked: bool = False) -> xr.Dataset:
    """Spherical A-grid with a smooth velocity field that varies in x, y, z and time."""
    lon_deg = np.linspace(-180.0, 180.0, nx)
    lat_deg = np.linspace(-80.0, 80.0, ny)
    depth_m = np.linspace(0.0, 1000.0, nz)
    lon2d, lat2d = np.meshgrid(lon_deg, lat_deg)
    u_surface_ms = 0.5 * np.cos(np.deg2rad(lat2d)) * np.sin(np.deg2rad(3 * lon2d))
    v_surface_ms = 0.3 * np.sin(np.deg2rad(2 * lat2d)) * np.cos(np.deg2rad(2 * lon2d))
    depth_decay = np.exp(-depth_m / 500.0)[:, None, None]
    time_modulation = (1.0 + 0.1 * np.arange(nt))[:, None, None, None]
    u_ms = (time_modulation * depth_decay[None] * u_surface_ms[None, None]).astype(np.float32)
    v_ms = (time_modulation * depth_decay[None] * v_surface_ms[None, None]).astype(np.float32)
    temperature_degC = (20.0 * depth_decay[None] * time_modulation + lat2d[None, None] / 10.0).astype(np.float32)

    dims = ["time", "depth", "YG", "XG"]
    ds = xr.Dataset(
        {"U": (dims, u_ms), "V": (dims, v_ms), "T": (dims, temperature_degC)},
        coords={
            "time": (["time"], xr.date_range("2000", periods=nt, freq="6h"), {"axis": "T"}),
            "depth": (["depth"], depth_m, {"axis": "Z", "units": "m"}),
            "YC": (["YC"], np.arange(ny) + 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(ny), {"axis": "Y"}),
            "XC": (["XC"], np.arange(nx) + 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(nx), {"axis": "X"}),
            "lat": (["YG"], lat_deg, {"axis": "Y", "units": "degrees_north"}),
            "lon": (["XG"], lon_deg, {"axis": "X", "units": "degrees_east"}),
        },
    )
    if chunked:
        ds = ds.chunk({"time": 1})
    return sgrid._attach_sgrid_metadata(
        ds,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("XG", "YG"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "XG", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("YC", "YG", sgrid.Padding.LOW),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("ZC", "depth", sgrid.Padding.BOTH),),
        ),
    )


def rectilinear_fieldset(on_gpu: bool, chunked: bool = False, prefetcher=None) -> parcels.FieldSet:
    fieldset = parcels.FieldSet.from_sgrid_conventions(
        rectilinear_dataset(chunked=chunked), mesh=parcels.SphericalMesh()
    )
    return fieldset.to_windowed_arrays(cupy, prefetcher=prefetcher) if on_gpu else fieldset


def seeded_particleset(fieldset: parcels.FieldSet, npart: int = 500, pclass=parcels.Particle, **kwargs):
    rng = np.random.default_rng(42)
    return parcels.ParticleSet(
        fieldset,
        pclass=pclass,
        x=rng.uniform(-170, 170, npart),
        y=rng.uniform(-70, 70, npart),
        z=rng.uniform(10, 900, npart),
        **kwargs,
    )


def assert_particles_match(pset_gpu, pset_cpu, variables=("x", "y", "z", "t"), rtol=1e-5, atol=1e-6):
    assert len(pset_gpu) == len(pset_cpu)
    np.testing.assert_array_equal(to_numpy(pset_gpu.particle_id), pset_cpu.particle_id)
    for variable in variables:
        np.testing.assert_allclose(
            to_numpy(getattr(pset_gpu, variable)), getattr(pset_cpu, variable), rtol=rtol, atol=atol, err_msg=variable
        )


# ------------------------------------------------------------------------------------------------
# WindowedArray in CuPy, indexing against NumPy
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def host_array() -> np.ndarray:
    return np.random.default_rng(0).random((5, 4, 7, 9)).astype(np.float32)


@pytest.mark.parametrize("index_module", [np, cupy])
def test_vectorized_indexing_matches_numpy(host_array, index_module):
    rng = np.random.default_rng(1)
    host_indices = [rng.integers(-n, n, 200) for n in host_array.shape]
    gpu_array = WindowedArray(host_array, windowed=False, array_library=cupy)
    gathered = gpu_array._raw_vindex(*(index_module.asarray(index) for index in host_indices))
    assert is_cupy_array(gathered)
    np.testing.assert_array_equal(gathered.get(), host_array[tuple(host_indices)])


def test_vectorized_indexing_broadcasts(host_array):
    gpu_array = WindowedArray(host_array, windowed=False, array_library=cupy)
    t = np.array([0, 3])[:, None, None]
    z = np.array([1, 2, 3])[None, :, None]
    y = np.arange(6)[None, None, :]
    x = np.arange(6)[None, None, :]
    np.testing.assert_array_equal(gpu_array._raw_vindex(t, z, y, x).get(), host_array[t, z, y, x])


def test_isel_through_xarray_matches_numpy(host_array):
    dims = ("time", "depth", "YG", "XG")
    rng = np.random.default_rng(2)
    points = {dim: rng.integers(0, n, 50) for dim, n in zip(dims, host_array.shape, strict=True)}
    expected = xr.Variable(dims, host_array).isel({dim: xr.Variable("points", index) for dim, index in points.items()})

    gpu_variable = xr.Variable(dims, WindowedArray(host_array, windowed=False, array_library=cupy))
    result = gpu_variable.isel({dim: xr.Variable("points", cupy.asarray(index)) for dim, index in points.items()})
    assert is_cupy_array(result.data)
    np.testing.assert_array_equal(result.data.get(), expected.data)


def test_isel_leaves_unindexed_dimension_as_trailing_axis(host_array):
    dims = ("time", "depth", "YG", "XG")
    single_depth = host_array[:, :1]
    rng = np.random.default_rng(3)
    points = {dim: rng.integers(0, n, 30) for dim, n in zip(dims, single_depth.shape, strict=True) if dim != "depth"}
    expected = xr.Variable(dims, single_depth).isel(
        {dim: xr.Variable("points", index) for dim, index in points.items()}
    )

    gpu_variable = xr.Variable(dims, WindowedArray(single_depth, windowed=False, array_library=cupy))
    result = gpu_variable.isel({dim: xr.Variable("points", cupy.asarray(index)) for dim, index in points.items()})
    assert result.dims == expected.dims
    np.testing.assert_array_equal(result.data.get(), expected.data)


def test_orthogonal_indexing_matches_numpy(host_array):
    gpu_array = WindowedArray(host_array, windowed=False, array_library=cupy)
    np.testing.assert_array_equal(gpu_array[1, 2].get(), host_array[1, 2])
    np.testing.assert_array_equal(gpu_array[(slice(1, 4), 0, slice(None), 3)].get(), host_array[1:4, 0, :, 3])
    np.testing.assert_array_equal(
        gpu_array._orthogonal_index((np.array([0, 2]), slice(None), np.array([1, 5, 6]), 0)).get(),
        host_array[[0, 2]][:, :, [1, 5, 6]][..., 0],
    )


def test_windowed_array_loads_levels_on_demand(host_array):
    import dask.array

    lazy_array = dask.array.from_array(host_array, chunks=(1, -1, -1, -1))
    gpu_array = WindowedArray(lazy_array, windowed=True, array_library=cupy)
    assert gpu_array.resident_levels == []
    spatial = (np.array([1, 2]), np.array([3, 4]), np.array([5, 6]))

    np.testing.assert_array_equal(
        gpu_array._raw_vindex(np.array([0, 1]), *spatial).get(), host_array[np.array([0, 1]), *spatial]
    )
    assert gpu_array.resident_levels == [0, 1]
    assert gpu_array.loads == 2

    # advancing the clock evicts level 0 and loads only the new level
    np.testing.assert_array_equal(
        gpu_array._raw_vindex(np.array([1, 2]), *spatial).get(), host_array[np.array([1, 2]), *spatial]
    )
    assert gpu_array.resident_levels == [1, 2]
    assert gpu_array.loads == 3

    # a single resident level needs no load
    np.testing.assert_array_equal(
        gpu_array._raw_vindex(np.array([2, 2]), *spatial).get(), host_array[np.array([2, 2]), *spatial]
    )
    assert gpu_array.loads == 3


def test_windowed_array_grows_for_wide_time_span(host_array):
    import dask.array

    lazy_array = dask.array.from_array(host_array, chunks=(1, -1, -1, -1))
    gpu_array = WindowedArray(lazy_array, windowed=True, array_library=cupy, max_levels=2)
    levels = np.array([0, 3, 1])
    spatial = (np.array([1, 2, 0]), np.array([3, 4, 0]), np.array([5, 6, 0]))
    with pytest.warns(FieldSetWarning, match="exceeds max_level=2"):
        gathered = gpu_array._raw_vindex(levels, *spatial)
    np.testing.assert_array_equal(gathered.get(), host_array[levels, *spatial])
    assert gpu_array.resident_levels == [0, 1, 3]


def test_windowed_array_rejects_out_of_range_level(host_array):
    import dask.array

    lazy_array = dask.array.from_array(host_array, chunks=(1, -1, -1, -1))
    gpu_array = WindowedArray(lazy_array, windowed=True, array_library=cupy)
    with pytest.raises(IndexError, match="out of bounds for axis 0"):
        gpu_array._raw_vindex(np.array([5]), np.array([0]), np.array([0]), np.array([0]))


# ------------------------------------------------------------------------------------------------
# FieldSet / ParticleSet on the GPU
# ------------------------------------------------------------------------------------------------
def test_to_windowed_arrays_in_cupy_wraps_fields_and_is_idempotent():
    fieldset = rectilinear_fieldset(on_gpu=True)
    assert is_cupy_namespace(fieldset.array_placement.namespace)
    wrapper = fieldset.U.data.variable._data
    assert isinstance(wrapper, WindowedArray)
    fieldset.to_windowed_arrays(cupy)
    assert fieldset.U.data.variable._data is wrapper

    buf = io.StringIO()
    fieldset.describe(buf)
    assert "WindowedArray (cupy)" in buf.getvalue()


def test_particle_data_lives_on_gpu():
    pset = seeded_particleset(rectilinear_fieldset(on_gpu=True))
    assert all(is_cupy_array(values) for values in pset._data.values())
    cpu_fieldset = parcels.FieldSet.from_sgrid_conventions(rectilinear_dataset(), mesh=parcels.SphericalMesh())
    assert cpu_fieldset.array_placement == NUMPY_PLACEMENT


def test_field_eval_matches_cpu():
    rng = np.random.default_rng(5)
    x, y, z = rng.uniform(-170, 170, 300), rng.uniform(-70, 70, 300), rng.uniform(10, 900, 300)
    t = rng.uniform(0, 3 * 6 * 3600, 300)
    cpu_fieldset = rectilinear_fieldset(on_gpu=False)
    gpu_fieldset = rectilinear_fieldset(on_gpu=True)

    u_cpu, v_cpu = cpu_fieldset.UV.eval(t, z, y, x)
    u_gpu, v_gpu = gpu_fieldset.UV.eval(*(cupy.asarray(values) for values in (t, z, y, x)))
    np.testing.assert_allclose(u_gpu.get(), u_cpu, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(v_gpu.get(), v_cpu, rtol=1e-6, atol=1e-12)

    temperature_cpu = cpu_fieldset.T.eval(t, z, y, x)
    temperature_gpu = gpu_fieldset.T.eval(*(cupy.asarray(values) for values in (t, z, y, x)))
    np.testing.assert_allclose(temperature_gpu.get(), temperature_cpu, rtol=1e-6)


@pytest.mark.parametrize("kernel", [AdvectionEE, AdvectionRK2, AdvectionRK4])
@pytest.mark.parametrize("chunked", [False, True])
def test_advection_matches_cpu(kernel, chunked):
    psets = {}
    for on_gpu in (False, True):
        fieldset = rectilinear_fieldset(on_gpu=on_gpu, chunked=chunked and on_gpu)
        pset = seeded_particleset(fieldset)
        pset.execute(kernel, runtime=np.timedelta64(12, "h"), dt=np.timedelta64(30, "m"), verbose_progress=False)
        psets[on_gpu] = pset
    assert is_cupy_array(psets[True].x)
    assert_particles_match(psets[True], psets[False])


def test_backward_advection_matches_cpu():
    psets = {}
    for on_gpu in (False, True):
        fieldset = rectilinear_fieldset(on_gpu=on_gpu, chunked=on_gpu)
        pset = seeded_particleset(fieldset, t=np.full(500, np.datetime64("2000-01-01T18:00")))
        pset.execute(AdvectionRK2, runtime=np.timedelta64(12, "h"), dt=-np.timedelta64(30, "m"), verbose_progress=False)
        psets[on_gpu] = pset
    assert_particles_match(psets[True], psets[False])


def test_flat_mesh_2d_advection_matches_cpu():
    psets = {}
    for on_gpu in (False, True):
        fieldset = parcels.FieldSet.from_sgrid_conventions(radial_rotation_dataset(), mesh=FlatMesh())
        if on_gpu:
            fieldset.to_windowed_arrays(cupy)
        pset = parcels.ParticleSet(fieldset, x=np.linspace(32, 50, 40), y=np.full(40, 30.0))
        pset.execute(AdvectionRK4, runtime=np.timedelta64(6, "h"), dt=np.timedelta64(10, "m"), verbose_progress=False)
        psets[on_gpu] = pset
    assert_particles_match(psets[True], psets[False], variables=("x", "y", "t"))


def test_staggered_release_times_match_cpu():
    """Particles with different clocks finish at different iterations, so kernels get masked particle views."""
    release_times = np.datetime64("2000-01-01T00:00") + np.arange(500) % 7 * np.timedelta64(20, "m")
    psets = {}
    for on_gpu in (False, True):
        fieldset = rectilinear_fieldset(on_gpu=on_gpu)
        pset = seeded_particleset(fieldset, t=release_times)
        pset.execute(
            AdvectionRK2, endtime=np.datetime64("2000-01-01T09:00"), dt=np.timedelta64(30, "m"), verbose_progress=False
        )
        psets[on_gpu] = pset
    assert_particles_match(psets[True], psets[False])


def SampleTemperature(particles, fieldset):  # pragma: no cover
    particles.temperature = fieldset.T[particles]


def DeleteParticlesNorthOf40(particles, fieldset):  # pragma: no cover
    particles[particles.y > 40].state = parcels.StatusCode.Delete


def WarmParticlesDriftEast(particles, fieldset):  # pragma: no cover
    warm = particles.temperature > 15
    particles[warm].dx += 0.001
    particles.age += np.where(warm, 2.0, 1.0) * particles.dt


def test_custom_kernels_match_cpu():
    pclass = parcels.Particle.add_variable(
        [parcels.Variable("temperature", dtype=np.float32), parcels.Variable("age", dtype=np.float32)]
    )
    kernels = [AdvectionRK2, SampleTemperature, WarmParticlesDriftEast, DeleteParticlesNorthOf40]
    psets = {}
    for on_gpu in (False, True):
        fieldset = rectilinear_fieldset(on_gpu=on_gpu)
        pset = seeded_particleset(fieldset, pclass=pclass)
        pset.execute(kernels, runtime=np.timedelta64(12, "h"), dt=np.timedelta64(30, "m"), verbose_progress=False)
        psets[on_gpu] = pset
    assert 0 < len(psets[False]) < 500
    assert_particles_match(psets[True], psets[False], variables=("x", "y", "z", "t", "temperature", "age"))


def DeleteErroredParticles(particles, fieldset):  # pragma: no cover
    particles[particles.state >= 50].state = parcels.StatusCode.Delete


def test_out_of_bounds_particles_are_deleted_like_cpu():
    psets = {}
    for on_gpu in (False, True):
        fieldset = rectilinear_fieldset(on_gpu=on_gpu)
        rng = np.random.default_rng(7)
        pset = parcels.ParticleSet(
            fieldset, x=rng.uniform(-179.9, 179.9, 400), y=rng.uniform(-79.9, 79.9, 400), z=rng.uniform(1, 999, 400)
        )
        pset.execute(
            [AdvectionRK2, DeleteErroredParticles],
            runtime=np.timedelta64(18, "h"),
            dt=np.timedelta64(60, "m"),
            verbose_progress=False,
        )
        psets[on_gpu] = pset
    assert_particles_match(psets[True], psets[False])


def test_particlefile_output_matches_cpu(tmp_path):
    tables = {}
    for on_gpu in (False, True):
        fieldset = rectilinear_fieldset(on_gpu=on_gpu)
        pset = seeded_particleset(fieldset, npart=100)
        path = tmp_path / f"trajectories_gpu_{on_gpu}.parquet"
        output_file = parcels.ParticleFile(path, outputdt=np.timedelta64(2, "h"))
        pset.execute(
            AdvectionRK2,
            runtime=np.timedelta64(6, "h"),
            dt=np.timedelta64(30, "m"),
            output_file=output_file,
            verbose_progress=False,
        )
        tables[on_gpu] = pq.read_table(path).to_pandas()

    assert len(tables[False]) == 100 * 4
    assert list(tables[True].columns) == list(tables[False].columns)
    for column in tables[False].columns:
        np.testing.assert_allclose(tables[True][column], tables[False][column], rtol=1e-5, atol=1e-6, err_msg=column)


def test_particles_added_on_gpu_stay_on_gpu():
    fieldset = rectilinear_fieldset(on_gpu=True)
    pset = seeded_particleset(fieldset, npart=10)
    pset += seeded_particleset(fieldset, npart=5)
    assert len(pset) == 15
    assert all(is_cupy_array(values) for values in pset._data.values())
    pset.execute(AdvectionRK2, runtime=np.timedelta64(1, "h"), dt=np.timedelta64(30, "m"), verbose_progress=False)
    assert len(np.unique(to_numpy(pset.particle_id))) == 15


# ------------------------------------------------------------------------------------------------
# Reading time levels ahead
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt_sign", [1, -1])
def test_prefetched_levels_are_used_and_change_nothing(dt_sign):
    """A run crossing every time level, with and without reading ahead, forward and backward."""
    psets = {}
    for prefetch in (False, True):
        fieldset = rectilinear_fieldset(on_gpu=True, chunked=True, prefetcher=None if prefetch else NoPrefetcher)
        start = np.datetime64("2000-01-01T00:00") if dt_sign == 1 else np.datetime64("2000-01-01T18:00")
        pset = seeded_particleset(fieldset, npart=200, t=np.full(200, start))
        pset.execute(
            AdvectionRK2, runtime=np.timedelta64(18, "h"), dt=dt_sign * np.timedelta64(30, "m"), verbose_progress=False
        )
        psets[prefetch] = pset
        gpu_array = fieldset.models[0].data.variables["U"]._data
        assert gpu_array.loads == 4
        # The forward run starts exactly on the first level and needs only that one at first, so
        # every later level arrives prefetched. The backward run starts between two levels.
        expected_hits = 3 if dt_sign == 1 else 2
        assert gpu_array.prefetch_hits == (expected_hits if prefetch else 0)
    for variable in ("x", "y", "z", "t"):
        np.testing.assert_array_equal(getattr(psets[True], variable).get(), getattr(psets[False], variable).get())
