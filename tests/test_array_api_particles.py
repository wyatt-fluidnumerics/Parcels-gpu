"""Particle data, particle views, the kernel loop and particle output on Array API libraries.

Each test runs on NumPy, array-api-strict and PyTorch (CPU) and compares with a NumPy reference.
No test samples a field, so search and interpolation are not involved.
"""

import importlib

import numpy as np
import pyarrow.parquet as pq
import pytest

from parcels import FieldOutOfBoundError, Particle, ParticleFile, ParticleSet, StatusCode, Variable
from parcels._core.kernel import _clip_dt_to_endtime
from parcels._core.particle import get_default_particle
from parcels._core.utils.array_backend import array_namespace, to_numpy

LIBRARIES = {
    "numpy": "array_api_compat.numpy",
    "array_api_strict": "array_api_strict",
    "torch": "array_api_compat.torch",
}


@pytest.fixture(params=list(LIBRARIES))
def xp(request):
    if request.param == "torch":
        pytest.importorskip("torch")
    if request.param == "array_api_strict":
        pytest.importorskip("array_api_strict")
    return importlib.import_module(LIBRARIES[request.param])


@pytest.fixture
def numpy_xp():
    return importlib.import_module(LIBRARIES["numpy"])


def place_particle_data(pset: ParticleSet, xp) -> ParticleSet:
    """Move the particle data of `pset` to the library `xp`, keeping every data type."""
    pset._data = {name: xp.asarray(values) for name, values in pset._data.items()}
    return pset


def assert_in_library(array, xp) -> None:
    assert array_namespace(array) is xp
    if xp.__name__ != "array_api_compat.numpy":
        assert not isinstance(array, np.ndarray)


def assert_particle_data_equal(data: dict, expected: dict, xp) -> None:
    assert data.keys() == expected.keys()
    for name, values in data.items():
        assert_in_library(values, xp)
        values_np = to_numpy(values)
        assert values_np.dtype == expected[name].dtype, name
        np.testing.assert_array_equal(values_np, expected[name], err_msg=name)


NPARTICLES = 8


def make_pset(fieldset, xp, pclass=Particle, **kwargs) -> ParticleSet:
    pset = ParticleSet(
        fieldset,
        pclass=pclass,
        x=np.linspace(0.1, 0.9, NPARTICLES),
        y=np.linspace(0.0, 1.0, NPARTICLES),
        **kwargs,
    )
    return place_particle_data(pset, xp)


# ---------------------------------------------------------------------------------------------------
# Write patterns on a ParticleSet and on views
# ---------------------------------------------------------------------------------------------------
# Each pattern is applied to `p` (the ParticleSet or a view of it); `expect` applies the same change
# with NumPy to `a` (a dict of the ParticleSet's arrays) at `rows` (the rows that `p` selects, in order).


def _assign(p):
    p.x = p.y * 2


def _assign_expect(a, rows):
    a["x"][rows] = a["y"][rows] * 2


def _iadd(p):
    p.x += 1.5


def _iadd_expect(a, rows):
    a["x"][rows] += 1.5


def _isub(p):
    p.x -= p.y


def _isub_expect(a, rows):
    a["x"][rows] -= a["y"][rows]


def _imul(p):
    p.x *= 3


def _imul_expect(a, rows):
    a["x"][rows] *= 3


def _masked_assign(p):
    p.x[p.y > 0.5] = 7


def _masked_assign_expect(a, rows):
    a["x"][rows[a["y"][rows] > 0.5]] = 7


def _masked_iadd(p):
    p.x[p.y > 0.5] += 2


def _masked_iadd_expect(a, rows):
    a["x"][rows[a["y"][rows] > 0.5]] += 2


def _ei_column(p):
    p.ei[:, 0] = p.state + 3


def _ei_column_expect(a, rows):
    a["ei"][rows, 0] = a["state"][rows] + 3


def _subview_assign(p):
    p[p.y > 0.5].x = 9


def _subview_assign_expect(a, rows):
    a["x"][rows[a["y"][rows] > 0.5]] = 9


def _subview_iadd(p):
    p[p.y > 0.5].x += 1


def _subview_iadd_expect(a, rows):
    a["x"][rows[a["y"][rows] > 0.5]] += 1


def _subview_state(p):
    p[p.y > 0.5].state = StatusCode.Delete


def _subview_state_expect(a, rows):
    a["state"][rows[a["y"][rows] > 0.5]] = StatusCode.Delete


def _subview_of_subview(p):
    sub = p[p.y > 0.2]
    sub[sub.x < 0.6].x = -1.0


def _subview_of_subview_expect(a, rows):
    rows = rows[a["y"][rows] > 0.2]
    a["x"][rows[a["x"][rows] < 0.6]] = -1.0


def _subview_integers(p):
    p[[0, 2]].x = 5.0


def _subview_integers_expect(a, rows):
    a["x"][rows[[0, 2]]] = 5.0


def _subview_slice(p):
    p[1:3].x += 1


def _subview_slice_expect(a, rows):
    a["x"][rows[1:3]] += 1


def _subview_single_integer(p):
    p[1].x = 4.0


def _subview_single_integer_expect(a, rows):
    a["x"][rows[1]] = 4.0


def _temporary_subview_in_place(p):
    p[p.y > 0.5].x[0] = 8.0


def _temporary_subview_in_place_expect(a, rows):
    a["x"][rows[a["y"][rows] > 0.5][0]] = 8.0


def _handed_out_array_in_place(p):
    x = p.x
    x[1] = 6.0
    p.y[p.x > 5] = 0.25


def _handed_out_array_in_place_expect(a, rows):
    a["x"][rows[1]] = 6.0
    a["y"][rows[a["x"][rows] > 5]] = 0.25


def _mixed_dtype_assign(p):
    p.dx = p.dx + 0.1 * p.dt


def _mixed_dtype_assign_expect(a, rows):
    a["dx"][rows] = a["dx"][rows] + 0.1 * a["dt"][rows]


WRITE_PATTERNS = {
    "assign": (_assign, _assign_expect),
    "iadd": (_iadd, _iadd_expect),
    "isub": (_isub, _isub_expect),
    "imul": (_imul, _imul_expect),
    "masked_assign": (_masked_assign, _masked_assign_expect),
    "masked_iadd": (_masked_iadd, _masked_iadd_expect),
    "ei_column": (_ei_column, _ei_column_expect),
    "subview_assign": (_subview_assign, _subview_assign_expect),
    "subview_iadd": (_subview_iadd, _subview_iadd_expect),
    "subview_state": (_subview_state, _subview_state_expect),
    "subview_of_subview": (_subview_of_subview, _subview_of_subview_expect),
    "subview_integers": (_subview_integers, _subview_integers_expect),
    "subview_slice": (_subview_slice, _subview_slice_expect),
    "subview_single_integer": (_subview_single_integer, _subview_single_integer_expect),
    "temporary_subview_in_place": (_temporary_subview_in_place, _temporary_subview_in_place_expect),
    "handed_out_array_in_place": (_handed_out_array_in_place, _handed_out_array_in_place_expect),
    "mixed_dtype_assign": (_mixed_dtype_assign, _mixed_dtype_assign_expect),
}

TOP_LEVEL_MASK = np.array([True, False, True, True, False, True, True, True])
TOP_LEVEL_INTEGERS = np.array([6, 0, 3, 7, 2, 5])  # unsorted, so views keep this order
TARGETS = {
    "particleset": None,
    "mask_view": TOP_LEVEL_MASK,
    "integer_view": TOP_LEVEL_INTEGERS,
    "slice_view": slice(1, 7),
}


def _rows_of(target_index) -> np.ndarray:
    rows = np.arange(NPARTICLES)
    return rows if target_index is None else rows[target_index]


@pytest.mark.parametrize("finish", ["write_back", "drop"])
@pytest.mark.parametrize("target", list(TARGETS))
@pytest.mark.parametrize("pattern", list(WRITE_PATTERNS))
def test_write_patterns(fieldset, xp, pattern, target, finish):
    apply, expect = WRITE_PATTERNS[pattern]
    pset = make_pset(fieldset, xp)
    expected = {name: to_numpy(values).copy() for name, values in pset._data.items()}

    target_index = TARGETS[target]
    if target_index is None:
        particles = pset
    elif isinstance(target_index, slice):
        particles = pset[target_index]
    else:
        particles = pset[xp.asarray(target_index)]

    apply(particles)
    if finish == "write_back" and particles is not pset:
        particles._write_back()
    del particles

    expect(expected, _rows_of(target_index))
    assert_particle_data_equal(pset._data, expected, xp)


def test_view_read_returns_cached_library_array(fieldset, xp):
    pset = make_pset(fieldset, xp)
    view = pset[pset.y > 0.3]
    x = view.x
    assert_in_library(x, xp)
    assert view.x is x
    np.testing.assert_array_equal(to_numpy(x), to_numpy(pset.x)[to_numpy(pset.y) > 0.3])


def test_view_assignment_writes_through_and_keeps_cache(fieldset, xp):
    pset = make_pset(fieldset, xp)
    view = pset[pset.y > 0.3]
    x = view.x
    view.x = 3.0
    assert view.x is x
    np.testing.assert_array_equal(to_numpy(x), 3.0)
    np.testing.assert_array_equal(to_numpy(pset.x)[to_numpy(pset.y) > 0.3], 3.0)


def test_single_particle_view_returns_scalars(fieldset, xp):
    pset = make_pset(fieldset, xp)
    expected_x = to_numpy(pset.x)[2]
    x = pset[2].x
    assert x.shape == ()
    assert float(x) == expected_x
    pset[2].x = 0.5
    assert float(to_numpy(pset.x)[2]) == np.float32(0.5)


def test_nested_view_keeps_parent_arrays_current(fieldset, xp):
    pset = make_pset(fieldset, xp)
    view = pset[pset.y > 0.1]
    x = view.x
    view[view.x > 0.5].x = 5.0
    # The parent's handed-out array shows what the nested view wrote
    np.testing.assert_array_equal(to_numpy(x) == 5.0, to_numpy(pset.x)[to_numpy(pset.y) > 0.1] == 5.0)
    assert bool(xp.any(x == 5.0))

    # A change in place to the parent's array is seen by a nested view and survives the parent's
    # write-back together with what the nested view wrote
    x[0] = 42.0
    nested = view[view.x > 40]
    np.testing.assert_array_equal(to_numpy(nested.x), [42.0])
    nested.x[0] = 43.0
    del nested
    view._write_back()
    x_np = to_numpy(pset.x)
    assert x_np[to_numpy(pset.y) > 0.1][0] == 43.0
    assert np.count_nonzero(x_np == 5.0) == 4


def test_repr_matches_numpy(fieldset, xp, numpy_xp):
    pset = make_pset(fieldset, xp)
    reference = make_pset(fieldset, numpy_xp)
    assert repr(pset[3]) == repr(reference[3])
    assert repr(pset) == repr(reference)


# ---------------------------------------------------------------------------------------------------
# ParticleSet bookkeeping
# ---------------------------------------------------------------------------------------------------


def _bookkeeping_data(fieldset, xp) -> dict:
    pset = make_pset(fieldset, xp)
    pset.state[1] = StatusCode.Delete
    pset.state[4] = StatusCode.ErrorOutOfBounds
    results = {
        "data_indices": pset.data_indices("state", [StatusCode.Delete, StatusCode.ErrorOutOfBounds]),
        "data_indices_inverted": pset.data_indices("state", StatusCode.Success, invert=True),
        "error_particles": pset._error_particles,
    }
    for array in results.values():
        assert_in_library(array, xp)
    results = {name: to_numpy(array) for name, array in results.items()}
    results["num_error_particles"] = pset._num_error_particles
    results["len"] = len(pset)

    pset.remove_indices(xp.asarray([0, 3]))
    pset.remove_indices(-1)
    other = make_pset(fieldset, xp)
    pset.add(other)
    results["after_remove_and_add"] = {name: to_numpy(values) for name, values in pset._data.items()}
    for values in pset._data.values():
        assert_in_library(values, xp)
    return results


def test_bookkeeping_matches_numpy(fieldset, xp, numpy_xp):
    results = _bookkeeping_data(fieldset, xp)
    reference = _bookkeeping_data(fieldset, numpy_xp)
    assert results.keys() == reference.keys()
    for name in ("data_indices", "data_indices_inverted", "error_particles"):
        np.testing.assert_array_equal(results[name], reference[name])
        assert results[name].dtype == reference[name].dtype
    assert results["num_error_particles"] == reference["num_error_particles"] == 2
    assert results["len"] == reference["len"] == NPARTICLES
    for name, values in reference["after_remove_and_add"].items():
        np.testing.assert_array_equal(results["after_remove_and_add"][name], values, err_msg=name)
    assert reference["after_remove_and_add"]["particle_id"].shape == (NPARTICLES - 3 + NPARTICLES,)


def test_remove_indices_out_of_bounds_raises(fieldset, xp):
    pset = make_pset(fieldset, xp)
    with pytest.raises(IndexError, match="out of bounds"):
        pset.remove_indices(NPARTICLES)


def test_clip_dt_to_endtime(xp):
    dt = xp.asarray([1.0, 1.0, 1.0, 1.0])
    time_to_endtime = xp.asarray([2.0, 0.5, 0.0, -1.0])
    clipped, dt_to_restore = _clip_dt_to_endtime(dt, time_to_endtime, 1)
    assert_in_library(clipped, xp)
    np.testing.assert_array_equal(to_numpy(clipped), [1.0, 0.5, 0.0, 0.0])
    ((newly_clipped, values),) = dt_to_restore
    np.testing.assert_array_equal(to_numpy(newly_clipped), [False, True, True, True])
    np.testing.assert_array_equal(to_numpy(values), [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------------------------------
# Kernel loop
# ---------------------------------------------------------------------------------------------------

Float64Particle = get_default_particle(np.float64).add_variable(
    [Variable("counter", dtype=np.float32), Variable("npasses", dtype=np.int32, initial=0)]
)


def Drift(particles, fieldset):  # pragma: no cover
    particles.dx += 0.1 * particles.dt


def CountRightHandParticles(particles, fieldset):  # pragma: no cover
    particles[particles.x > 0.3].counter += 1
    particles.counter[particles.y > 0.6] += 10


def RepeatEvenParticlesOnce(particles, fieldset):  # pragma: no cover
    first_pass = particles.npasses == 0
    particles.npasses += 1
    particles.state[particles.state == StatusCode.Repeat] = StatusCode.Success
    particles.state[first_pass & (particles.particle_id % 2 == 0)] = StatusCode.Repeat


def StopOneParticle(particles, fieldset):  # pragma: no cover
    particles.state[particles.particle_id == 3] = StatusCode.StopExecution


def DeleteFarParticles(particles, fieldset):  # pragma: no cover
    particles[particles.x > 1.05].state = StatusCode.Delete


def ResetNpasses(particles, fieldset):  # pragma: no cover
    particles.npasses = 0


STAGGERED_RELEASE = [np.timedelta64(int(s), "s") for s in (0, 0, 1, 2, 2, 3, 5, 6)]


def _execute(fieldset, xp, output_path=None) -> ParticleSet:
    pset = make_pset(fieldset, xp, pclass=Float64Particle, t=STAGGERED_RELEASE)
    output_file = None if output_path is None else ParticleFile(output_path, outputdt=np.timedelta64(1, "s"))
    pset.execute(
        [Drift, CountRightHandParticles, RepeatEvenParticlesOnce, ResetNpasses, StopOneParticle, DeleteFarParticles],
        runtime=np.timedelta64(8, "s"),
        dt=np.timedelta64(1, "s"),
        output_file=output_file,
        verbose_progress=False,
    )
    return pset


def test_execute_matches_numpy(fieldset, xp, numpy_xp):
    pset = _execute(fieldset, xp)
    reference = _execute(fieldset, numpy_xp)
    expected = {name: to_numpy(values) for name, values in reference._data.items()}
    assert len(reference) < NPARTICLES  # some particles were deleted
    assert_particle_data_equal(pset._data, expected, xp)


def test_execute_output_matches_numpy(fieldset, xp, numpy_xp, tmp_path):
    _execute(fieldset, xp, tmp_path / "library.parquet")
    _execute(fieldset, numpy_xp, tmp_path / "numpy.parquet")
    table = pq.read_table(tmp_path / "library.parquet")
    reference = pq.read_table(tmp_path / "numpy.parquet")
    assert reference.num_rows > NPARTICLES
    assert table.schema.equals(reference.schema, check_metadata=True)
    assert table.equals(reference)


def test_particlefile_writes_view(fieldset, xp, numpy_xp, tmp_path):
    tables = {}
    for name, library in (("library", xp), ("numpy", numpy_xp)):
        pset = make_pset(fieldset, library, t=STAGGERED_RELEASE)
        pfile = ParticleFile(tmp_path / f"{name}.parquet", outputdt=np.timedelta64(1, "s"))
        pfile.write(pset[pset.y > 0.4], t=0.0)
        pfile.write(pset, t=0.0)
        pfile.close()
        tables[name] = pq.read_table(tmp_path / f"{name}.parquet")
    # All rows of the view, then the two particles of the ParticleSet released at t=0
    assert tables["numpy"].num_rows == 5 + 2
    assert tables["library"].equals(tables["numpy"])


def test_execute_default_particle_out_of_place_drift(fieldset, xp, numpy_xp):
    def OutOfPlaceDrift(particles, fieldset):  # pragma: no cover
        particles.dx = particles.dx + 0.1 * particles.dt

    psets = {}
    for name, library in (("library", xp), ("numpy", numpy_xp)):
        pset = make_pset(fieldset, library, t=STAGGERED_RELEASE)
        pset.execute(OutOfPlaceDrift, runtime=np.timedelta64(4, "s"), dt=np.timedelta64(1, "s"), verbose_progress=False)
        psets[name] = pset
    expected = {name: to_numpy(values) for name, values in psets["numpy"]._data.items()}
    assert_particle_data_equal(psets["library"]._data, expected, xp)


def test_execute_raises_particle_error(fieldset, xp):
    def FlagOutOfBounds(particles, fieldset):  # pragma: no cover
        particles[particles.particle_id == 5].state = StatusCode.ErrorOutOfBounds

    pset = make_pset(fieldset, xp, t=STAGGERED_RELEASE)
    with pytest.raises(FieldOutOfBoundError):
        pset.execute(FlagOutOfBounds, runtime=np.timedelta64(4, "s"), dt=np.timedelta64(1, "s"), verbose_progress=False)
