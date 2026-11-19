"""Particle data, particle views, the kernel loop and particle output on Array API libraries.

Each test runs on NumPy, array-api-strict and PyTorch (CPU) and compares with a NumPy reference.
No test samples a field, so search and interpolation are not involved.
"""

import gc
import importlib

import numpy as np
import pyarrow.parquet as pq
import pytest

from parcels import (
    FieldOutOfBoundError,
    OutsideTimeInterval,
    Particle,
    ParticleFile,
    ParticleSet,
    StatusCode,
    Variable,
)
from parcels._core.kernel import _clip_dt_to_endtime, _kernel_view
from parcels._core.particle import get_default_particle
from parcels._core.utils.array_backend import array_namespace, is_array, to_numpy

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
NESTED_MASK = np.array([True, True, False, True, False, True])  # over the six particles of TOP_LEVEL_MASK
TARGETS = {
    "particleset": None,
    "mask_view": TOP_LEVEL_MASK,
    "integer_view": TOP_LEVEL_INTEGERS,
    "slice_view": slice(1, 7),
    "kernel_view": TOP_LEVEL_MASK,
    "kernel_nested_view": TOP_LEVEL_MASK,  # NESTED_MASK of the kernel loop's view
}
# Patterns that change in place an array that `p` hands out. Of all targets, only the ParticleSet and the
# kernel loop's view hand out arrays whose changes reach the ParticleSet.
IN_PLACE_PATTERNS = ("masked_assign", "masked_iadd", "ei_column", "handed_out_array_in_place")


def _rows_of(target) -> np.ndarray:
    rows = np.arange(NPARTICLES)
    if TARGETS[target] is not None:
        rows = rows[TARGETS[target]]
    if target == "kernel_nested_view":
        rows = rows[NESTED_MASK]
    return rows


def _reaches_particleset(pattern, target) -> bool:
    if pattern == "temporary_subview_in_place":
        # changes in place what a view of `p` hands out, which no view writes back
        return False
    return pattern not in IN_PLACE_PATTERNS or target in ("particleset", "kernel_view")


@pytest.mark.parametrize("target", list(TARGETS))
@pytest.mark.parametrize("pattern", list(WRITE_PATTERNS))
def test_write_patterns(fieldset, xp, pattern, target):
    if target == "slice_view" and pattern in IN_PLACE_PATTERNS:
        pytest.skip("A slice view hands out a view of the ParticleSet's array in these libraries; nothing relies on it")
    apply, expect = WRITE_PATTERNS[pattern]
    pset = make_pset(fieldset, xp)
    expected = {name: to_numpy(values).copy() for name, values in pset._data.items()}

    target_index = TARGETS[target]
    kernel_view = None
    if target_index is None:
        particles = pset
    elif target in ("kernel_view", "kernel_nested_view"):
        kernel_view = _kernel_view(pset, xp.asarray(target_index))
        particles = kernel_view if target == "kernel_view" else kernel_view[xp.asarray(NESTED_MASK)]
    elif isinstance(target_index, slice):
        particles = pset[target_index]
    else:
        particles = pset[xp.asarray(target_index)]

    apply(particles)
    if kernel_view is not None:
        kernel_view._write_back()

    if _reaches_particleset(pattern, target):
        expect(expected, _rows_of(target))
    assert_particle_data_equal(pset._data, expected, xp)


def test_view_read_returns_cached_library_array(fieldset, xp):
    pset = make_pset(fieldset, xp)
    expected_x = to_numpy(pset.x)[to_numpy(pset.y) > 0.3]

    kernel_view = _kernel_view(pset, pset.y > 0.3)
    x = kernel_view.x
    assert_in_library(x, xp)
    assert kernel_view.x is x
    np.testing.assert_array_equal(to_numpy(x), expected_x)

    # Any other view gathers again on every read
    view = pset[pset.y > 0.3]
    x = view.x
    assert_in_library(x, xp)
    assert view.x is not x
    np.testing.assert_array_equal(to_numpy(view.x), expected_x)


def test_view_assignment_writes_through_and_keeps_cache(fieldset, xp):
    pset = make_pset(fieldset, xp)
    selected = to_numpy(pset.y) > 0.3
    view = pset[pset.y > 0.3]
    view.x = 3.0
    np.testing.assert_array_equal(to_numpy(view.x), 3.0)
    np.testing.assert_array_equal(to_numpy(pset.x)[selected], 3.0)

    # The kernel loop's view keeps what is assigned in its cached array until it writes back
    kernel_view = _kernel_view(pset, pset.y > 0.3)
    x = kernel_view.x
    kernel_view.x = 5.0
    assert kernel_view.x is x
    np.testing.assert_array_equal(to_numpy(x), 5.0)
    np.testing.assert_array_equal(to_numpy(pset.x)[selected], 3.0)
    kernel_view._write_back()
    np.testing.assert_array_equal(to_numpy(pset.x)[selected], 5.0)
    assert kernel_view.x is not x


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
    selected = to_numpy(pset.y) > 0.1
    expected_x = to_numpy(pset.x)[selected].copy()
    view = _kernel_view(pset, pset.y > 0.1)
    x = view.x
    view[view.x > 0.5].x = 5.0
    expected_x[expected_x > 0.5] = 5.0
    # The parent's handed-out array shows what the nested view wrote
    np.testing.assert_array_equal(to_numpy(x), expected_x)
    assert bool(xp.any(x == 5.0))

    # A change in place to the parent's array is seen by a nested view. A change in place to what the
    # nested view hands out is not written anywhere; assigning the variable is, and survives the
    # parent's write-back together with what the nested view wrote before
    x[0] = 42.0
    nested = view[view.x > 40]
    np.testing.assert_array_equal(to_numpy(nested.x), [42.0])
    nested.x[0] = 43.0
    np.testing.assert_array_equal(to_numpy(nested.x), [42.0])
    nested.x = 44.0
    view._write_back()
    x_np = to_numpy(pset.x)
    assert x_np[selected][0] == 44.0
    assert np.count_nonzero(x_np == 5.0) == 4


def test_repr_matches_numpy(fieldset, xp, numpy_xp):
    pset = make_pset(fieldset, xp)
    reference = make_pset(fieldset, numpy_xp)
    assert repr(pset[3]) == repr(reference[3])
    assert repr(pset) == repr(reference)


def test_reading_a_view_does_not_undo_later_writes(fieldset, xp):
    pset = make_pset(fieldset, xp)
    view = pset[xp.asarray(TOP_LEVEL_MASK)]
    x = view.x
    nested_x = view[xp.asarray(NESTED_MASK)].x
    pset.x[...] = 100.0
    np.testing.assert_array_equal(to_numpy(view.x), 100.0)
    del view, x, nested_x
    gc.collect()
    np.testing.assert_array_equal(to_numpy(pset.x), 100.0)


def test_read_and_write_orders_between_particleset_and_views(fieldset, xp):
    pset = make_pset(fieldset, xp)
    expected_x = to_numpy(pset.x).copy()
    rows = _rows_of("mask_view")
    nested_rows = rows[NESTED_MASK]
    view = pset[xp.asarray(TOP_LEVEL_MASK)]
    nested = view[xp.asarray(NESTED_MASK)]

    # a nested view assigns: its parent view and the ParticleSet read what it wrote
    nested.x = 9.0
    expected_x[nested_rows] = 9.0
    np.testing.assert_array_equal(to_numpy(view.x), expected_x[rows])
    np.testing.assert_array_equal(to_numpy(pset.x), expected_x)

    # the parent view assigns after the nested view read: the nested view reads what the parent wrote
    held = nested.x
    view.x = view.x + 1
    expected_x[rows] += np.float32(1)
    del held
    gc.collect()
    np.testing.assert_array_equal(to_numpy(nested.x), expected_x[nested_rows])
    np.testing.assert_array_equal(to_numpy(pset.x), expected_x)

    # the ParticleSet changes after both views read: both read the change
    pset.x[...] = 7.0
    np.testing.assert_array_equal(to_numpy(view.x), 7.0)
    np.testing.assert_array_equal(to_numpy(nested.x), 7.0)


def test_read_and_write_orders_in_the_kernel_loop_view(fieldset, xp):
    pset = make_pset(fieldset, xp)
    expected_x = to_numpy(pset.x).copy()
    rows = _rows_of("kernel_view")
    nested_rows = rows[NESTED_MASK]
    kernel_view = _kernel_view(pset, xp.asarray(TOP_LEVEL_MASK))

    # the array of a nested view is held while the parent assigns every particle
    held = kernel_view[kernel_view.x > 0.4].x
    kernel_view.x = kernel_view.x + 1
    expected_x[rows] += np.float32(1)
    del held
    gc.collect()

    # a nested view assigns: the parent reads it, then assigns some particles again
    nested = kernel_view[xp.asarray(NESTED_MASK)]
    nested.x = 5.0
    expected_x[nested_rows] = 5.0
    np.testing.assert_array_equal(to_numpy(kernel_view.x), expected_x[rows])
    kernel_view.x[0] = -2.0
    expected_x[rows[0]] = -2.0
    np.testing.assert_array_equal(to_numpy(nested.x), expected_x[nested_rows])

    # nothing reaches the ParticleSet before the write-back, which writes everything once
    np.testing.assert_array_equal(to_numpy(pset.x)[rows], to_numpy(make_pset(fieldset, xp).x)[rows])
    kernel_view._write_back()
    np.testing.assert_array_equal(to_numpy(pset.x), expected_x)

    # after the write-back the ParticleSet's arrays are what the kernel loop's view reads
    pset.x[...] = 3.0
    gc.collect()
    np.testing.assert_array_equal(to_numpy(kernel_view.x), 3.0)
    np.testing.assert_array_equal(to_numpy(pset.x), 3.0)


def test_negative_index_out_of_range_raises(fieldset, xp):
    pset = make_pset(fieldset, xp)
    too_negative = xp.asarray([-NPARTICLES - 1])
    with pytest.raises(IndexError, match=f"index {-NPARTICLES - 1} is out of bounds for axis 0 with size {NPARTICLES}"):
        pset[too_negative].x  # noqa: B018
    with pytest.raises(IndexError, match="out of bounds"):
        pset[too_negative].x = 1.0
    np.testing.assert_array_equal(to_numpy(pset[xp.asarray([-1, -NPARTICLES])].particle_id), [NPARTICLES - 1, 0])

    view = pset[xp.asarray(TOP_LEVEL_MASK)]
    nselected = int(np.count_nonzero(TOP_LEVEL_MASK))
    with pytest.raises(IndexError, match=f"index {-nselected - 1} is out of bounds for axis 0 with size {nselected}"):
        view[xp.asarray([-nselected - 1])]
    np.testing.assert_array_equal(to_numpy(view[xp.asarray([-1])].particle_id), [NPARTICLES - 1])


def test_view_indexed_with_mask_over_whole_particleset(fieldset, xp):
    pset = make_pset(fieldset, xp)
    first = np.array([True, True, True, False, False, True, False, False])
    whole = np.array([False, False, True, True, True, False, False, False])
    view = pset[xp.asarray(first)]
    np.testing.assert_array_equal(to_numpy(view[xp.asarray(whole)].particle_id), [2, 3, 4])
    assert len(view[xp.asarray(whole)]) == NPARTICLES
    assert len(view[xp.asarray([True, False, True, False])]) == NPARTICLES
    view[xp.asarray(whole)].x = 9.0
    expected_x = to_numpy(make_pset(fieldset, xp).x)
    expected_x[whole] = 9.0
    np.testing.assert_array_equal(to_numpy(pset.x), expected_x)

    # Inside the kernel loop, such a view reads and writes the newest values of every particle, whether
    # the kernel loop's view selects it or not
    kernel_view = _kernel_view(pset, xp.asarray(first))
    x = kernel_view.x
    kernel_view.x = -1.0
    expected_x[first] = -1.0
    within = np.array([True, False, True, False, False, False, False, False])
    np.testing.assert_array_equal(to_numpy(kernel_view[xp.asarray(within)].x), [-1.0, -1.0])
    kernel_view[xp.asarray(within)].x = 3.0
    expected_x[within] = 3.0
    np.testing.assert_array_equal(to_numpy(kernel_view[xp.asarray(whole)].x), expected_x[whole])
    kernel_view[xp.asarray(whole)].x = 4.0
    expected_x[whole] = 4.0
    np.testing.assert_array_equal(to_numpy(x), expected_x[first])
    assert len(kernel_view[xp.asarray(within)]) == len(kernel_view[xp.asarray(whole)]) == NPARTICLES
    kernel_view._write_back()
    np.testing.assert_array_equal(to_numpy(pset.x), expected_x)


def test_list_assignment_keeps_precision(fieldset, xp):
    pset = make_pset(fieldset, xp, pclass=Float64Particle)
    values = [0.1 * (i + 1) for i in range(NPARTICLES)]  # none is a float32 value
    pset.dx = values
    np.testing.assert_array_equal(to_numpy(pset.dx), np.asarray(values))
    pset[xp.asarray(TOP_LEVEL_MASK)].dy = values[:6]
    np.testing.assert_array_equal(to_numpy(pset.dy)[TOP_LEVEL_MASK], np.asarray(values[:6]))
    kernel_view = _kernel_view(pset, xp.asarray(TOP_LEVEL_MASK))
    kernel_view.dz = values[:6]
    kernel_view._write_back()
    np.testing.assert_array_equal(to_numpy(pset.dz)[TOP_LEVEL_MASK], np.asarray(values[:6]))

    if xp.__name__ == "array_api_strict":
        return  # array-api-strict's isin takes integers only
    # Values are compared as NumPy builds them: float64 for Python floats
    np.testing.assert_array_equal(to_numpy(pset.data_indices("dx", [values[2], values[5]])), [2, 5])
    float32_pset = make_pset(fieldset, xp)
    float32_pset.x = values
    assert to_numpy(float32_pset.data_indices("x", [values[2]])).shape == (0,)


def test_views_and_particlesets_are_not_arrays(fieldset, xp):
    pset = make_pset(fieldset, xp)
    kernel_view = _kernel_view(pset, xp.asarray(TOP_LEVEL_MASK))
    for particles in (pset, pset[xp.asarray(TOP_LEVEL_MASK)], pset[2], kernel_view, kernel_view[1:3]):
        assert not is_array(particles)
        assert array_namespace(particles, pset.x) is xp
        with pytest.raises(AttributeError):
            particles.__array_namespace__  # noqa: B018
    assert kernel_view._cache == {}


def test_particlefile_write_leaves_views_and_particleset_untouched(fieldset, xp, tmp_path):
    pset = make_pset(fieldset, xp, t=STAGGERED_RELEASE)
    kernel_view = _kernel_view(pset, pset.y > 0.4)
    kernel_view.x = 0.25  # cached, not yet written back
    expected = {name: to_numpy(values).copy() for name, values in pset._data.items()}
    pfile = ParticleFile(tmp_path / "views.parquet", outputdt=np.timedelta64(1, "s"))
    pfile.write(kernel_view, t=0.0)
    pfile.write(pset[pset.y > 0.4], t=0.0)
    pfile.close()

    assert list(kernel_view._cache) == ["x"]
    assert_particle_data_equal(pset._data, expected, xp)
    table = pq.read_table(tmp_path / "views.parquet")
    selected = to_numpy(pset.y) > 0.4
    np.testing.assert_array_equal(table.column("x").to_numpy()[:5], 0.25)
    np.testing.assert_array_equal(table.column("x").to_numpy()[5:], expected["x"][selected])
    np.testing.assert_array_equal(table.column("particle_id").to_numpy()[:5], expected["particle_id"][selected])


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


def CountWithNestedRead(particles, fieldset):  # pragma: no cover
    previous = particles[particles.counter >= 0].counter  # noqa: F841
    particles.counter = particles.counter + 1


@pytest.mark.parametrize("release", ["together", "staggered"])
def test_count_with_nested_read(fieldset, xp, release):
    if release == "together":
        release_s = np.zeros(NPARTICLES, dtype=np.int64)
    else:
        release_s = np.array(STAGGERED_RELEASE) // np.timedelta64(1, "s")
    pset = make_pset(fieldset, xp, pclass=Float64Particle, t=release_s * np.timedelta64(1, "s"))
    pset.execute(CountWithNestedRead, runtime=np.timedelta64(8, "s"), dt=np.timedelta64(1, "s"), verbose_progress=False)
    np.testing.assert_array_equal(to_numpy(pset.counter), (8 - release_s).astype(np.float32))


def test_error_traceback_does_not_undo_later_writes(fieldset, xp):
    def FlagOutOfBounds(particles, fieldset):  # pragma: no cover
        particles[particles.particle_id == 5].state = StatusCode.ErrorOutOfBounds

    pset = make_pset(fieldset, xp, t=STAGGERED_RELEASE)
    try:
        pset.execute(FlagOutOfBounds, runtime=np.timedelta64(4, "s"), dt=np.timedelta64(1, "s"), verbose_progress=False)
    except FieldOutOfBoundError:
        pset.x[...] = 7.0
        pset.z[...] = -1.0
    gc.collect()
    np.testing.assert_array_equal(to_numpy(pset.x), 7.0)
    np.testing.assert_array_equal(to_numpy(pset.z), -1.0)


@pytest.mark.parametrize(
    "error_code, error",
    [(StatusCode.ErrorOutOfBounds, FieldOutOfBoundError), (StatusCode.ErrorOutsideTimeInterval, OutsideTimeInterval)],
)
def test_particle_error_message_matches_numpy(fieldset, xp, numpy_xp, error_code, error):
    def Flag(particles, fieldset):  # pragma: no cover
        particles[particles.particle_id >= 5].state = error_code

    def message_and_flagged(library):
        pset = make_pset(fieldset, library, t=STAGGERED_RELEASE)
        with pytest.raises(error) as excinfo:
            pset.execute(Flag, runtime=np.timedelta64(4, "s"), dt=np.timedelta64(1, "s"), verbose_progress=False)
        flagged = to_numpy(pset.state) == error_code
        return str(excinfo.value), {name: to_numpy(pset._data[name])[flagged] for name in ("t", "z", "y", "x")}

    message, flagged = message_and_flagged(numpy_xp)
    if error_code == StatusCode.ErrorOutsideTimeInterval:
        assert message == f"Field sampled outside time domain at time {flagged['t']!r}."
    else:
        assert message == (
            f"Field sampled out-of-bound, at (z={flagged['z']!r}, y={flagged['y']!r}, x={flagged['x']!r})"
        )
        assert "dtype=float32" in message
    assert message_and_flagged(xp)[0] == message
