"""Views of some of the particles of a ParticleSet.

Reading a variable of a view gathers the values of the selected particles into an array of the
compute library. Assigning a variable writes values into the selected particles.

The view that the kernel loop hands a kernel caches what it reads (``cache_arrays=True``). The first
read of a variable gathers it from the ParticleSet's arrays and every later read returns that same
array. Assigning the variable, or changing the array in place (``particles.state[mask] = code``),
changes only that array. After each kernel call the kernel loop calls ``_write_back``, which writes
each cached array to the ParticleSet's arrays once and drops the cache.

A view taken from a view (``particles[mask]``) gathers from the values of the view it was taken from
on every read, and assigning a variable writes into those values and assigns them back to that view.
Taken from the kernel loop's view, it therefore reads and writes the cached arrays. Every other view
caches nothing: each read gathers from the ParticleSet's arrays and each assignment writes straight
into them. A view taken inside a kernel with a mask over the whole ParticleSet that selects particles
the kernel loop's view does not, writes the kernel loop's cached array of a variable to the
ParticleSet before using it, and reloads it after assigning to it.
"""

import numpy as np

from parcels._core.utils.array_backend import array_namespace, asarray_like, device_of, flatnonzero, is_array
from parcels._repr_utils import particlesetview_repr

_VIEW_ATTRIBUTES = ("_data", "_index", "_pclass", "_parent", "_cache", "_caching_view")


class ParticleSetView:
    """Class to be used in a kernel that links a View of the ParticleSet (on the kernel level) to a ParticleSet.

    Only the view that the kernel loop hands a kernel returns the same array on every read. Any other
    view gathers the values again on every read, so changing the array it returned in place, as in
    ``pset[mask].x[0] = 1`` or ``particles[mask].x[0] = 1`` inside a kernel, is not written to the
    ParticleSet. Assign the variable instead: ``view.x = values``.
    """

    def __init__(self, data, index, pclass, parent=None, *, cache_arrays=False):
        self._data = data
        # `index` selects rows of the parent's values, or of the ParticleSet's arrays without a parent
        self._index = index if parent is not None else _normalize_index(index, data["particle_id"])
        self._pclass = pclass
        self._parent = parent
        self._cache = {} if cache_arrays else None
        self._caching_view = None

    def __getattr__(self, name):
        if name in _VIEW_ATTRIBUTES or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        if name not in self._data:
            return self._data[name][self._index]
        if self._cache is None:
            return self._read(name)
        if name not in self._cache:
            self._cache[name] = self._read(name)
        return self._cache[name]

    def __setattr__(self, name, value):
        if name in _VIEW_ATTRIBUTES:
            object.__setattr__(self, name, value)
        elif self._cache is not None:
            array = getattr(self, name)
            if value is not array:
                array[...] = _values_like(value, array)
        elif self._parent is not None:
            parent_values = getattr(self._parent, name)
            _put_rows(parent_values, self._index, _values_like(value, parent_values))
            setattr(self._parent, name, parent_values)
        else:
            if self._caching_view is not None:
                self._caching_view._flush(name)
            target = self._data[name]
            _put_rows(target, self._index, _values_like(value, target))
            if self._caching_view is not None:
                self._caching_view._refresh(name)

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
                    # global mask: the particles it selects, whether or not this view selects them
                    return self._view_of_particleset_mask(index)
                if index.shape[0] != nselected:
                    raise ValueError(
                        f"Boolean index has incompatible length {index.shape[0]} for selection of size {nselected}"
                    )
                return ParticleSetView(self._data, index, self._pclass, parent=self)
            if not _is_integer_array(index):
                raise IndexError("arrays used as indices must be of integer (or boolean) type")

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

        # Fallback: try to assign directly
        try:
            local_mask[...] = index
            return ParticleSetView(self._data, local_mask, self._pclass, parent=self)
        except Exception as e:
            raise TypeError(f"Unsupported index type for ParticleSetView.__getitem__: {type(index)!r}") from e

    def __len__(self):
        # The parent's length: for a boolean mask view that is the ParticleSet's length, not the number selected
        if self._parent is not None:
            return len(self._parent)
        if is_array(self._index):
            return self._index.shape[0]
        return len(self._index)

    def _read(self, name):
        """The selected particles' values of `name`: the cached array if there is one, otherwise gathered."""
        if self._cache is not None and name in self._cache:
            return self._cache[name]
        if self._parent is not None:
            return _gather_rows(getattr(self._parent, name), self._index)
        if self._caching_view is not None:
            self._caching_view._flush(name)
        return _gather_rows(self._data[name], self._index)

    def _write_back(self):
        """Write each cached array to the ParticleSet once and drop the cache, so that the next read gathers again."""
        for name in self._cache:
            self._flush(name)
        self._cache.clear()

    def _flush(self, name):
        """Write the cached array of `name`, if there is one, to the ParticleSet, and keep it cached."""
        if name in self._cache:
            _put_rows(self._data[name], self._index, self._cache[name])

    def _refresh(self, name):
        """Update the cached array of `name`, if there is one, in place from the ParticleSet."""
        if name in self._cache:
            self._cache[name][...] = _gather_rows(self._data[name], self._index)

    def _view_of_particleset_mask(self, mask):
        """A view of the particles that `mask`, with one entry per particle of the ParticleSet, selects."""
        top = self
        while top._parent is not None:
            top = top._parent
        caching_view = top if top._cache is not None else top._caching_view
        if caching_view is None:
            return ParticleSetView(self._data, mask, self._pclass)

        # Inside the kernel loop, the particles must be read and written where the newest values are: in
        # the kernel loop's cached arrays for its own particles (it selects them with a boolean mask)
        xp = array_namespace(mask)
        local_mask = _gather_rows(mask, caching_view._index)
        if int(xp.count_nonzero(local_mask)) == int(xp.count_nonzero(mask)):
            return ParticleSetView(self._data, local_mask, self._pclass, parent=caching_view)
        view = ParticleSetView(self._data, mask, self._pclass)
        view._caching_view = caching_view
        return view

    def _size(self):
        """Number of particles that this view selects."""
        if isinstance(self._index, slice):
            return len(range(*self._index.indices(self._data["particle_id"].shape[0])))
        if self._index.dtype == array_namespace(self._index).bool:
            return int(array_namespace(self._index).count_nonzero(self._index))
        return self._index.shape[0]


def _is_array_or_list(index) -> bool:
    return is_array(index) or isinstance(index, list)


def _is_integer_array(index) -> bool:
    return is_array(index) and array_namespace(index).isdtype(index.dtype, "integral")


def _normalize_index(index, reference):
    """A ParticleSet index as a Python int (a single particle), a slice, or an array of the library of `reference`.

    Negative entries of integer arrays are wrapped, since not every library gathers with them. Entries
    out of range are kept, so that reading or writing through the index raises IndexError.
    """
    if isinstance(index, tuple) and len(index) == 1:
        index = index[0]
    if isinstance(index, np.integer) or (is_array(index) and index.ndim == 0):
        return int(index)
    if isinstance(index, list) and len(index) == 0:
        # NumPy indexes with an empty list as with an empty integer array
        return asarray_like(index, like=reference, dtype="int64")
    if _is_array_or_list(index):
        index = asarray_like(index, like=reference)
        if _is_integer_array(index):
            nparticles = reference.shape[0]
            in_range_negative = (index < 0) & (index >= -nparticles)
            index = array_namespace(index).where(in_range_negative, index + nparticles, index)
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

    Python scalars are passed on as they are. Other non-arrays (lists) are built in the data type of
    `target` directly, as PyTorch would otherwise build Python floats in float32.
    """
    if not is_array(value) and isinstance(value, (bool, int, float, complex)):
        return value
    xp = array_namespace(target)
    if not is_array(value):
        return xp.asarray(value, dtype=target.dtype, device=device_of(target))
    value = asarray_like(value, like=target)
    if value.dtype != target.dtype:
        value = xp.astype(value, target.dtype)
    return value


def _gather_rows(source, index):
    """The rows of `source` (one row per particle) that `index` selects."""
    if _is_integer_array(index):
        _raise_if_out_of_bounds(index, source.shape[0])
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

    _raise_if_out_of_bounds(index, target.shape[0])
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
