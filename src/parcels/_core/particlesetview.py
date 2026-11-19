import weakref

import numpy as np

from parcels._core.utils.array_backend import array_namespace, asarray_like, device_of, flatnonzero, is_array
from parcels._repr_utils import particlesetview_repr

_VIEW_ATTRIBUTES = ("_data", "_index", "_pclass", "_parent", "_arrays", "_sources", "_write_backs")


class ParticleSetView:
    """Class to be used in a kernel that links a View of the ParticleSet (on the kernel level) to a ParticleSet.

    Reading a variable gathers the selected particles' values into an array of the compute library,
    which the view keeps and hands out on every further read. Assigning a variable writes through to
    the ParticleSet at once. Changes made in place to a handed-out array reach the ParticleSet when
    the view writes back: on assignment of that variable, when `_write_back` is called (the kernel
    loop does so after every kernel), and otherwise as soon as nothing holds the array any more.

    A view of a view (``particles[mask]`` inside a kernel) gathers from, and writes back to, the
    arrays of the view it was taken from, so neither ever holds values older than what the other wrote.
    """

    def __init__(self, data, index, pclass, parent=None):
        self._data = data
        self._index = index if parent is not None else _normalize_index(index, data["particle_id"])
        self._pclass = pclass
        self._parent = parent
        self._arrays = {}
        self._sources = {}
        self._write_backs = {}

    def __getattr__(self, name):
        if name in _VIEW_ATTRIBUTES:
            raise AttributeError(name)
        if name in self._data:
            # If this ParticleSetView represents a single particle (integer
            # index), return the underlying scalar directly to preserve
            # user-facing semantics (e.g., `pset[0].time` should be a number).
            if isinstance(self._index, int):
                return _gather_rows(self._data[name], self._index)
            return self._get(name)
        return self._data[name][self._index]

    def __setattr__(self, name, value):
        if name in _VIEW_ATTRIBUTES:
            object.__setattr__(self, name, value)
        elif isinstance(self._index, int):
            target = self._data[name]
            _put_rows(target, self._index, _values_like(value, target))
        elif name in self._arrays:
            array = self._arrays[name]
            if value is not array:
                array[...] = _values_like(value, array)
            self._write_back_variable(name)
        else:
            source = self._source(name)
            _write_rows_to(self._owner(), name, source, self._index, _values_like(value, source))

    def __repr__(self):
        return particlesetview_repr(self)

    def __getitem__(self, index):
        # normalize single-element tuple indexing (e.g., (inds,))
        if isinstance(index, tuple) and len(index) == 1:
            index = index[0]
        if isinstance(self._index, int):
            raise TypeError("A ParticleSetView of a single particle cannot be indexed.")

        reference = self._data["particle_id"]
        xp = array_namespace(reference)
        nselected = self._size()

        # Boolean mask (could be local-length or global-length)
        if _is_array_or_list(index):
            index = asarray_like(index, like=reference)
            if index.dtype == xp.bool:
                if index.shape[0] == reference.shape[0]:
                    # global mask
                    local_mask = self._select(index)
                elif index.shape[0] == nselected:
                    local_mask = index
                else:
                    raise ValueError(
                        f"Boolean index has incompatible length {index.shape[0]} for selection of size {nselected}"
                    )
                return ParticleSetView(self._data, local_mask, self._pclass, parent=self)

        # Integer array/list, slice or single integer relative to the local view
        # (boolean masks were handled above). A view of a view selects with a
        # boolean mask over the particles of the view it is taken from.
        if is_array(index):
            _raise_if_out_of_bounds(index, nselected)
            positions = xp.astype(xp.where(index < 0, index + nselected, index), xp.int64)
            local_mask = xp.isin(xp.arange(nselected, dtype=xp.int64, device=device_of(reference)), positions)
            return ParticleSetView(self._data, local_mask, self._pclass, parent=self)
        local_mask = xp.zeros(nselected, dtype=xp.bool, device=device_of(reference))
        if isinstance(index, (slice, int)):
            local_mask[index] = True
            return ParticleSetView(self._data, local_mask, self._pclass, parent=self)

        # Fallback: try to assign directly (preserves previous behaviour for other index types)
        try:
            local_mask[...] = index
            return ParticleSetView(self._data, local_mask, self._pclass, parent=self)
        except Exception as e:
            raise TypeError(f"Unsupported index type for ParticleSetView.__getitem__: {type(index)!r}") from e

    def __len__(self):
        if self._parent is not None:
            return len(self._parent)
        if is_array(self._index):
            return self._index.shape[0]
        return len(self._index)

    def _owner(self):
        """Where this view's selection points into: the enclosing view, or the ParticleSet's data."""
        return self._data if self._parent is None else self._parent

    def _source(self, name):
        """The array of `name` that this view's selection indexes."""
        if self._parent is None:
            return self._data[name]
        return self._parent._get(name)

    def _get(self, name):
        """The array of `name` for the selected particles, gathered on first use."""
        array = self._arrays.get(name)
        if array is None:
            source = self._source(name)
            gathered = _gather_rows(source, self._index)
            # The view hands out an alias of the gathered array, so that once the view and every user
            # have let go of it, the finalizer can still write the gathered values back.
            array = gathered[...]
            write_back = weakref.finalize(array, _write_rows_to, self._owner(), name, source, self._index, gathered)
            write_back.atexit = False
            self._arrays[name] = array
            self._sources[name] = source
            self._write_backs[name] = write_back
        return array

    def _write_back_variable(self, name):
        """Write the array gathered for `name` into the view's parent, and from there up to the ParticleSet."""
        _write_rows_to(self._owner(), name, self._sources[name], self._index, self._arrays[name])

    def _write_back(self):
        """Write every gathered array back to the ParticleSet and let go of them, so the next read gathers again."""
        for name in list(self._arrays):
            self._write_back_variable(name)
            self._write_backs[name].detach()
        self._arrays.clear()
        self._sources.clear()
        self._write_backs.clear()

    def _size(self):
        """Number of particles that this view selects."""
        if isinstance(self._index, slice):
            return len(range(*self._index.indices(self._data["particle_id"].shape[0])))
        if self._index.dtype == array_namespace(self._index).bool:
            return int(array_namespace(self._index).count_nonzero(self._index))
        return self._index.shape[0]

    def _select(self, values):
        """The entries of `values`, which has one entry per particle in the ParticleSet, that this view selects."""
        if self._parent is not None:
            values = self._parent._select(values)
        return _gather_rows(values, self._index)


def _is_array_or_list(index) -> bool:
    return is_array(index) or isinstance(index, list)


def _is_integer_array(index) -> bool:
    return is_array(index) and array_namespace(index).isdtype(index.dtype, "integral")


def _normalize_index(index, reference):
    """A ParticleSet index as a Python int (a single particle), a slice, or an array of the library of `reference`.

    Negative entries of integer arrays are wrapped, since not every library gathers with them.
    """
    if isinstance(index, tuple) and len(index) == 1:
        index = index[0]
    if isinstance(index, np.integer) or (is_array(index) and index.ndim == 0):
        return int(index)
    if _is_array_or_list(index):
        index = asarray_like(index, like=reference)
        if _is_integer_array(index):
            index = array_namespace(index).where(index < 0, index + reference.shape[0], index)
    return index


def _raise_if_out_of_bounds(indices, size: int):
    """Raise the IndexError NumPy raises for integer `indices` into an axis of length `size`."""
    xp = array_namespace(indices)
    out_of_bounds = (indices < -size) | (indices >= size)
    if bool(xp.any(out_of_bounds)):
        first = int(xp.reshape(indices, (-1,))[flatnonzero(xp.reshape(out_of_bounds, (-1,)))[0]])
        raise IndexError(f"index {first} is out of bounds for axis 0 with size {size}")


def _values_like(value, target):
    """`value` ready to be assigned into `target`: an array of the library, device and data type of `target`.

    Python scalars are passed on as they are.
    """
    if not is_array(value) and isinstance(value, (bool, int, float, complex)):
        return value
    value = asarray_like(value, like=target)
    if value.dtype != target.dtype:
        value = array_namespace(target).astype(value, target.dtype)
    return value


def _gather_rows(source, index):
    """The rows of `source` (one row per particle) that `index` selects."""
    if _is_integer_array(index):
        return array_namespace(source).take(source, index, axis=0)
    if source.ndim > 1 and not is_array(index):
        return source[index, ...]
    return source[index]


def _put_rows(target, index, values):
    """Assign `values` to the rows of `target` that `index` selects.

    Rows selected by an integer array are assigned through the equivalent boolean mask, as not every
    library assigns through integer arrays. A row selected more than once gets its last value, as in NumPy.
    """
    if not _is_integer_array(index):
        if target.ndim > 1 and not is_array(index):
            target[index, ...] = values
        else:
            target[index] = values
        return

    xp = array_namespace(target)
    nselected = index.shape[0]
    if nselected == 0:
        return
    values = xp.asarray(values, dtype=target.dtype, device=device_of(target))
    values = xp.broadcast_to(values, (nselected, *target.shape[1:]))
    order = xp.argsort(index, stable=True)
    sorted_rows = xp.take(index, order)
    all_rows = xp.arange(target.shape[0], dtype=index.dtype, device=device_of(target))
    last_selection = xp.searchsorted(sorted_rows, all_rows, side="right") - 1
    is_selected = last_selection >= 0
    last_selection = xp.where(is_selected, last_selection, 0)
    is_selected = is_selected & (xp.take(sorted_rows, last_selection) == all_rows)
    target[is_selected] = xp.take(values, xp.take(order, last_selection[is_selected]), axis=0)


def _write_rows_to(owner, name, source, index, values):
    """Write `values` into the rows `index` of `source`, which is `owner`'s array of `name`.

    A view as `owner` passes the write on to its own parent. Nothing is written once `owner` no
    longer holds `source` (a view that wrote back and let go of its arrays, or a ParticleSet whose
    arrays were replaced when particles were added or removed): the selection no longer applies.
    """
    if isinstance(owner, ParticleSetView):
        if owner._arrays.get(name) is not source:
            return
        _put_rows(source, index, values)
        owner._write_back_variable(name)
    elif owner.get(name) is source:
        _put_rows(source, index, values)
