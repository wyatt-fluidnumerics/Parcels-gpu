import numpy as np

from parcels._core.utils.array_backend import array_namespace, is_array
from parcels._repr_utils import particlesetview_repr


class ParticleSetView:
    """Class to be used in a kernel that links a View of the ParticleSet (on the kernel level) to a ParticleSet."""

    def __init__(self, data, index, pclass):
        self._data = data
        self._index = index
        self._pclass = pclass

    def __getattr__(self, name):
        # Return a proxy that behaves like the underlying numpy array but
        # writes back into the parent arrays when sliced/modified. This
        # enables constructs like `particles.dlon[mask] += vals` to update
        # the parent arrays rather than temporary copies.
        if name in self._data:
            # If this ParticleSetView represents a single particle (integer
            # index), return the underlying scalar directly to preserve
            # user-facing semantics (e.g., `pset[0].time` should be a number).
            if isinstance(self._index, (int, np.integer)):
                return self._data[name][self._index]
            if is_array(self._index) and self._index.ndim == 0:
                return self._data[name][int(self._index)]
            return ParticleSetViewArray(self._data, self._index, name)
        return self._data[name][self._index]

    def __setattr__(self, name, value):
        if name in ["_data", "_index", "_pclass"]:
            object.__setattr__(self, name, value)
        else:
            self._data[name][self._index] = value

    def __repr__(self):
        return particlesetview_repr(self)

    def __getitem__(self, index):
        # normalize single-element tuple indexing (e.g., (inds,))
        if isinstance(index, tuple) and len(index) == 1:
            index = index[0]

        base = self._index
        xp = array_namespace(base)
        new_index = np.zeros_like(base, dtype=bool)

        # Boolean mask (could be local-length or global-length)
        if _is_array_or_list(index) and xp.asarray(index).dtype == bool:
            arr = xp.asarray(index)
            if arr.size == base.size:
                # global mask
                new_index = arr
            elif arr.size == int(np.sum(base)):
                new_index[base] = arr
            else:
                raise ValueError(
                    f"Boolean index has incompatible length {arr.size} for selection of size {int(np.sum(base))}"
                )
            return ParticleSetView(self._data, new_index, self._pclass)

        # Integer array/list, slice or single integer relative to the local view
        # (boolean masks were handled above). Normalize and map to global
        # particle indices for both boolean-base and integer-base `self._index`.
        if _is_array_or_list(index) or isinstance(index, (slice, int)):
            # convert list/ndarray to ndarray, keep slice/int as-is
            idx = xp.asarray(index) if _is_array_or_list(index) else index
            if base.dtype == bool:
                particle_idxs = np.flatnonzero(base)
                sel = particle_idxs[idx]
            else:
                base_arr = xp.asarray(base)
                sel = base_arr[idx]
            new_index[sel] = True
            return ParticleSetView(self._data, new_index, self._pclass)

        # Fallback: try to assign directly (preserves previous behaviour for other index types)
        try:
            new_index[base] = index
            return ParticleSetView(self._data, new_index, self._pclass)
        except Exception as e:
            raise TypeError(f"Unsupported index type for ParticleSetView.__getitem__: {type(index)!r}") from e

    def __len__(self):
        return len(self._index)


def _is_array_or_list(index) -> bool:
    return is_array(index) or isinstance(index, list)


def _unwrap(other):
    """Return ndarray for ParticleSetViewArray or the value unchanged."""
    return other.__array__() if isinstance(other, ParticleSetViewArray) else other


def _unwrap_nested(argument):
    """Unwrap ParticleSetViewArrays that are passed directly or inside a list or tuple."""
    if isinstance(argument, (list, tuple)):
        return type(argument)(_unwrap_nested(item) for item in argument)
    return _unwrap(argument)


def _asarray(other):
    """Return the selected array for ParticleSetViewArray, otherwise return argument."""
    return other.__array__() if isinstance(other, ParticleSetViewArray) else other


class ParticleSetViewArray:
    """Array-like proxy for a ParticleSetView that writes through to the parent arrays when mutated."""

    def __init__(self, data, index, name):
        self._data = data
        self._index = index
        self._name = name

    def __array__(self, dtype=None):
        arr = self._data[self._name][self._index]
        return arr.astype(dtype) if dtype is not None else arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if any(isinstance(out, ParticleSetViewArray) for out in kwargs.get("out", ())):
            return NotImplemented
        return getattr(ufunc, method)(*(_unwrap(value) for value in inputs), **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        return func(*_unwrap_nested(args), **{name: _unwrap_nested(value) for name, value in kwargs.items()})

    def __repr__(self):
        return repr(self.__array__())

    def __len__(self):
        return len(self.__array__())

    def _to_global_index(self, subindex=None):
        """Return a global index (boolean mask or integer indices) that
        addresses the parent arrays. If `subindex` is provided it selects
        within the current local view and maps back to the global index.
        """
        base = self._index
        if subindex is None:
            return base
        xp = array_namespace(base)

        # If subindex is a boolean array, support both local-length masks
        # (length == base.sum()) and global-length masks (length == base.size).
        if _is_array_or_list(subindex) and xp.asarray(subindex).dtype == bool:
            arr = xp.asarray(subindex)
            if arr.size == base.size:
                # already a global mask
                return arr
            if arr.size == int(np.sum(base)):
                global_mask = np.zeros_like(base, dtype=bool)
                global_mask[base] = arr
                return global_mask
            raise ValueError(
                f"Boolean index has incompatible length {arr.size} for selection of size {int(np.sum(base))}"
            )

        # Handle tuple indexing where the first axis indexes particles
        # and later axes index into the per-particle array shape (e.g. ei[:, igrid])
        if isinstance(subindex, tuple):
            first, *rest = subindex
            # map the first index (local selection) to global particle indices
            if base.dtype == bool:
                particle_idxs = np.flatnonzero(base)
                first_arr = xp.asarray(first) if _is_array_or_list(first) else first
                sel = particle_idxs[first_arr]
            else:
                base_arr = xp.asarray(base)
                sel = base_arr[first]

            # if rest contains a single int (e.g., column), return tuple index
            if len(rest) == 1:
                return (sel, rest[0])
            # return full tuple (sel, ...) for higher-dim cases
            return tuple([sel] + rest)

        # If base is a boolean mask over the parent array and subindex is
        # an integer or slice relative to the local view, map it to integer
        # indices in the parent array.
        if base.dtype == bool:
            if isinstance(subindex, (slice, int)):
                rel = np.flatnonzero(base)[subindex]
                return rel
            # If subindex is an integer/array selection (relative to the
            # local view) map those to global integer indices.
            arr = xp.asarray(subindex)
            if arr.dtype != bool:
                particle_idxs = np.flatnonzero(base)
                sel = particle_idxs[arr]
                return sel
            # Otherwise treat subindex as a boolean mask relative to the
            # local view and expand to a global boolean mask.
            global_mask = np.zeros_like(base, dtype=bool)
            global_mask[base] = arr
            return global_mask

        # If base is an array of integer indices
        base_arr = xp.asarray(base)
        try:
            return base_arr[subindex]
        except Exception:
            return base_arr[xp.asarray(subindex, dtype=bool)]

    def __getitem__(self, subindex):
        # Handle tuple indexing (e.g. [:, igrid]) by applying the tuple
        # to the local selection first. This covers the common case
        # `particles.ei[:, igrid]` where `ei` is a 2D parent array and the
        # second index selects the grid index.
        if isinstance(subindex, tuple):
            local = self._data[self._name][self._index]
            return local[subindex]

        new_index = self._to_global_index(subindex)
        return ParticleSetViewArray(self._data, new_index, self._name)

    def __setitem__(self, subindex, value):
        tgt = self._to_global_index(subindex)
        self._data[self._name][tgt] = value

    # in-place ops must write back into the parent array
    def __iadd__(self, other):
        vals = self._data[self._name][self._index] + _unwrap(other)
        self._data[self._name][self._index] = vals
        return self

    def __isub__(self, other):
        vals = self._data[self._name][self._index] - _unwrap(other)
        self._data[self._name][self._index] = vals
        return self

    def __imul__(self, other):
        vals = self._data[self._name][self._index] * _unwrap(other)
        self._data[self._name][self._index] = vals
        return self

    # Provide simple numpy-like evaluation for binary ops by delegating to ndarray
    def __add__(self, other):
        return self.__array__() + _unwrap(other)

    def __sub__(self, other):
        return self.__array__() - _unwrap(other)

    def __mul__(self, other):
        return self.__array__() * _unwrap(other)

    def __truediv__(self, other):
        return self.__array__() / _unwrap(other)

    def __floordiv__(self, other):
        return self.__array__() // _unwrap(other)

    def __pow__(self, other):
        return self.__array__() ** _unwrap(other)

    def __neg__(self):
        return -self.__array__()

    def __pos__(self):
        return +self.__array__()

    def __abs__(self):
        return abs(self.__array__())

    # Right-hand operations to handle cases like `scalar - ParticleSetViewArray`
    def __radd__(self, other):
        return _unwrap(other) + self.__array__()

    def __rsub__(self, other):
        return _unwrap(other) - self.__array__()

    def __rmul__(self, other):
        return _unwrap(other) * self.__array__()

    def __rtruediv__(self, other):
        return _unwrap(other) / self.__array__()

    def __rfloordiv__(self, other):
        return _unwrap(other) // self.__array__()

    def __rpow__(self, other):
        return _unwrap(other) ** self.__array__()

    # Comparison operators should return plain numpy boolean arrays so that
    # expressions like `mask = particles.gridID == gid` produce an ndarray
    # usable for indexing (rather than another ParticleSetViewArray).
    def __eq__(self, other):
        left = self.__array__()
        right = _asarray(other)
        return left == right

    def __ne__(self, other):
        left = self.__array__()
        right = _asarray(other)
        return left != right

    def __lt__(self, other):
        left = self.__array__()
        right = _asarray(other)
        return left < right

    def __le__(self, other):
        left = self.__array__()
        right = _asarray(other)
        return left <= right

    def __gt__(self, other):
        left = self.__array__()
        right = _asarray(other)
        return left > right

    def __ge__(self, other):
        left = self.__array__()
        right = _asarray(other)
        return left >= right

    # Allow attribute access like .dtype etc. by forwarding to the ndarray
    def __getattr__(self, item):
        arr = self.__array__()
        return getattr(arr, item)
