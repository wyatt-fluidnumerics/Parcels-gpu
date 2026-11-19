"""Helpers for writing Parcels code against the Array API standard.

Parcels computes on whatever array library its particle and field arrays belong to (NumPy, CuPy,
PyTorch, ...). Code that does so gets the library's namespace from the arrays it is handed,
``xp = array_namespace(x)``, and calls ``xp.<function>(x)``. Parcels supports libraries whose arrays
are eager and mutable.

Parcels never asks where an array lives. Where a host-side constant (a grid coordinate, a lookup
table) has to meet particle arrays, it is converted with ``asarray_like(values, like=x)``, which
places it wherever ``x`` is, and kept with ``cached_per_placement`` so that it is converted once per
library and device. Values assigned into an array are converted with ``values_like``. The only
conversion back is ``to_numpy``, for file output. Every NumPy array handed to another library goes
through ``host_array_for_conversion`` first.

The Parcels backends name the library they hold data in as a module (``placement_of_library``) and
measure what they cache with ``nbytes_of``.

The functions from ``atleast_1d`` on cover NumPy functions and indexing that Parcels uses and that
the standard does not have under any name.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import array_api_compat
import array_api_compat.numpy as compat_numpy
import numpy as np
from array_api_compat import is_array_api_obj, is_numpy_array, to_device
from array_api_compat import device as device_of

__all__ = [
    "NUMPY_PLACEMENT",
    "ArrayPlacement",
    "array_namespace",
    "asarray_like",
    "atleast_1d",
    "cached_per_placement",
    "copy_of",
    "deg2rad",
    "device_of",
    "flatnonzero",
    "host_array_for_conversion",
    "indices_to_mask",
    "is_array",
    "isclose",
    "namespace_of_library",
    "nbytes_of",
    "placement_of_library",
    "ravel_multi_index",
    "to_numpy",
    "unravel_index",
    "values_like",
]


def is_array(value: Any) -> bool:
    """Whether ``value`` is an array of a library Parcels can compute on."""
    return is_array_api_obj(value)


def array_namespace(*values: Any) -> ModuleType:
    """Array API namespace of the arrays among ``values``.

    Python scalars, ``None`` and other non-arrays are ignored, and NumPy is returned when no
    value is an array. Arrays of different libraries raise ``TypeError``.
    """
    arrays = [value for value in values if is_array_api_obj(value)]
    if not arrays:
        return compat_numpy
    return array_api_compat.array_namespace(*arrays)


def namespace_of_library(library: ModuleType | None) -> ModuleType:
    """Array API namespace for an array library given as a module, e.g. ``numpy``, ``cupy`` or ``torch``.

    ``None`` means NumPy. Libraries that implement the standard in their own namespace are returned
    as they are.
    """
    if library is None:
        return compat_numpy
    name = library.__name__
    if name in ("numpy", "cupy", "torch"):
        return importlib.import_module(f"array_api_compat.{name}")
    if name.startswith("array_api_compat.") or hasattr(library, "__array_api_version__"):
        return library
    raise TypeError(f"{name!r} is not an array library Parcels can compute on.")


@dataclass(frozen=True)
class ArrayPlacement:
    """The array library, and the device within it, that a FieldSet's field data is returned in.

    ``device`` is whatever the library uses to name a device and is passed back to it untouched;
    ``None`` leaves the choice to the library.
    """

    namespace: ModuleType
    device: Any = None

    @classmethod
    def of(cls, array: Any) -> ArrayPlacement:
        if is_numpy_array(array):
            return NUMPY_PLACEMENT
        return cls(array_namespace(array), device_of(array))

    def asarray(self, values: Any) -> Any:
        if self.device is None:
            return self.namespace.asarray(values)
        return self.namespace.asarray(values, device=self.device)


NUMPY_PLACEMENT = ArrayPlacement(compat_numpy)


def placement_of_library(array_library: ModuleType | None, device: Any = None) -> ArrayPlacement:
    """Where arrays created in ``array_library`` on ``device`` live, with the device named as the library names it.

    ``None`` means NumPy. Naming the device the library's way lets placements reported by different
    backends, and by arrays themselves, compare equal.
    """
    probe = ArrayPlacement(namespace_of_library(array_library), device).asarray(np.empty(0, dtype=np.float64))
    return ArrayPlacement.of(probe)


def host_array_for_conversion(values: Any) -> np.ndarray:
    """``values`` as a C-contiguous, writable NumPy array in native byte order, ready to hand to any array library.

    A copy is made only when ``values`` is not one already. Libraries that share memory with the NumPy array
    they are given must not be handed memory they could write through but should not (xarray hands out
    dimension coordinates read-only), and not every library reads other byte orders or strided memory.
    """
    values = np.asarray(values)
    if not values.dtype.isnative:
        values = values.astype(values.dtype.newbyteorder("="))
    return np.require(values, requirements=["C", "W"])


def _ready_for_library(xp: ModuleType, values: Any) -> Any:
    """``values``, with a NumPy array prepared by ``host_array_for_conversion`` when it goes to a library other than NumPy."""
    if xp is not compat_numpy and is_numpy_array(values):
        return host_array_for_conversion(values)
    return values


def asarray_like(values: Any, like: Any, *, dtype: str | None = None) -> Any:
    """``values`` as an array of the same library, and in the same place, as the array ``like``.

    ``dtype`` is the name of a standard data type (``"float64"``, ``"int32"``, ...); without it the
    data type of ``values`` is kept.
    """
    xp = array_namespace(like)
    values = _ready_for_library(xp, values)
    if not is_array_api_obj(like):
        return xp.asarray(values, dtype=None if dtype is None else getattr(xp, dtype))
    return xp.asarray(values, dtype=None if dtype is None else getattr(xp, dtype), device=device_of(like))


def values_like(value: Any, target: Any) -> Any:
    """``value`` as an array of the library, device and data type of the array ``target``, e.g. to assign into it.

    Python scalars are returned as they are. Sequences are converted straight to the data type of
    ``target``, without passing through the library's default floating-point type.
    """
    if not is_array_api_obj(value) and isinstance(value, (bool, int, float, complex)):
        return value
    xp = array_namespace(target)
    value = _ready_for_library(xp, value)
    return xp.asarray(value, dtype=target.dtype, device=device_of(target))


def cached_per_placement(copies: list[tuple[ModuleType, Any, Any]], like: Any, convert: Callable[[Any], Any]) -> Any:
    """The entry of ``copies`` for the array library and device of ``like``, made with ``convert(like)`` on first use.

    For host-side constants that meet particle arrays, so that each is converted once per library and device.
    ``copies`` is a list kept by the owner of the constant. Entries are found by comparing devices because the
    device objects of some libraries (CuPy) are not hashable. A ``like`` that is not an array (a Python scalar or
    sequence, which NumPy takes as it is) has no device.
    """
    xp = array_namespace(like)
    device = device_of(like) if is_array_api_obj(like) else None
    for copy_xp, copy_device, values in copies:
        if copy_xp is xp and copy_device == device:
            return values
    values = convert(like)
    copies.append((xp, device, values))
    return values


def to_numpy(array: Any) -> np.ndarray:
    """``array`` as a NumPy array, for handing data to libraries that only read host memory."""
    if is_numpy_array(array) or not is_array_api_obj(array):
        return np.asarray(array)
    try:
        return np.from_dlpack(array)
    except (TypeError, BufferError, RuntimeError, ValueError, AttributeError):
        return np.asarray(to_device(array, "cpu"))


def copy_of(array: Any) -> Any:
    """A copy of ``array`` that shares no memory with it."""
    return array_namespace(array).asarray(array, copy=True)


def nbytes_of(value: Any) -> int:
    """Size of the array ``value`` in bytes, also for libraries whose arrays have no ``nbytes`` (array-api-strict).

    Values that are not arrays count as zero bytes.
    """
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    if not is_array_api_obj(value):
        return 0
    xp = array_namespace(value)
    dtype = value.dtype
    if xp.isdtype(dtype, "bool"):
        itemsize = 1
    elif xp.isdtype(dtype, "integral"):
        itemsize = xp.iinfo(dtype).bits // 8
    elif xp.isdtype(dtype, "real floating"):
        itemsize = xp.finfo(dtype).bits // 8
    else:
        # finfo describes one component of a complex number
        itemsize = 2 * xp.finfo(dtype).bits // 8
    return math.prod(value.shape) * itemsize


def atleast_1d(value: Any) -> Any:
    """``value`` as an array with at least one dimension. Non-arrays become NumPy arrays."""
    if not is_array_api_obj(value):
        return np.atleast_1d(value)
    if value.ndim == 0:
        return array_namespace(value).reshape(value, (1,))
    return value


def flatnonzero(mask: Any) -> Any:
    """Indices of the non-zero entries of a one-dimensional array."""
    return array_namespace(mask).nonzero(mask)[0]


def deg2rad(angle_deg: Any) -> Any:
    return angle_deg * (math.pi / 180.0)


def isclose(a: Any, b: Any, *, rtol: float = 1e-5, atol: float = 1e-8) -> Any:
    """Elementwise ``|a - b| <= atol + rtol * |b|``, as ``np.isclose`` without NaN handling."""
    xp = array_namespace(a, b)
    magnitude_of_b = xp.abs(b) if is_array_api_obj(b) else abs(b)
    return xp.abs(a - b) <= atol + rtol * magnitude_of_b


def ravel_multi_index(indices: Sequence[Any], dims: Sequence[int]) -> Any:
    """Flat index of the C-ordered position ``indices`` in an array of shape ``dims``."""
    flat_index = indices[0]
    for index, size in zip(indices[1:], dims[1:], strict=True):
        flat_index = flat_index * int(size) + index
    return flat_index


def unravel_index(flat_index: Any, dims: Sequence[int]) -> tuple[Any, ...]:
    """Position in an array of shape ``dims`` of the C-ordered flat index ``flat_index``."""
    indices = []
    remainder = flat_index
    for size in reversed(dims[1:]):
        indices.append(remainder % int(size))
        remainder = remainder // int(size)
    indices.append(remainder)
    return tuple(reversed(indices))


def indices_to_mask(indices: Any, size: int) -> Any:
    """Boolean mask over ``size`` items that is True at the integer ``indices``, which NumPy would index with.

    Negative indices count from the end. An index outside ``-size <= index < size`` raises the ``IndexError``
    NumPy raises. Scattering through the mask stands in for assigning through ``indices``, which not every
    library supports; an item that ``indices`` select more than once is selected once.
    """
    xp = array_namespace(indices)
    flat_indices = xp.reshape(indices, (-1,))
    out_of_bounds = (flat_indices < -size) | (flat_indices >= size)
    if bool(xp.any(out_of_bounds)):
        first = int(flat_indices[flatnonzero(out_of_bounds)[0]])
        raise IndexError(f"index {first} is out of bounds for axis 0 with size {size}")
    flat_indices = xp.astype(flat_indices, xp.int64, copy=False)
    flat_indices = xp.where(flat_indices < 0, flat_indices + size, flat_indices)
    return xp.isin(xp.arange(size, dtype=xp.int64, device=device_of(indices)), flat_indices)
