"""Helpers for writing Parcels code against the Array API standard.

Parcels computes on whatever array library its particle and field arrays belong to (NumPy, CuPy,
PyTorch, ...). Code that does so gets the library's namespace from the arrays it is handed,
``xp = array_namespace(x)``, and calls ``xp.<function>(x)``. Parcels supports libraries whose arrays
are eager and mutable.

Parcels never asks where an array lives. Where a host-side constant (a grid coordinate, a lookup
table) has to meet particle arrays, it is converted with ``asarray_like(values, like=x)``, which
places it wherever ``x`` is. The only conversion back is ``to_numpy``, for file output.

The functions below ``to_numpy`` cover NumPy functions that Parcels uses and that the standard does
not have under any name.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import array_api_compat
import array_api_compat.numpy as compat_numpy
import numpy as np
from array_api_compat import is_array_api_obj, is_numpy_array, to_device
from array_api_compat import device as device_of

__all__ = [
    "ArrayPlacement",
    "array_namespace",
    "asarray_like",
    "atleast_1d",
    "atleast_2d",
    "copy_of",
    "deg2rad",
    "device_of",
    "flatnonzero",
    "is_array",
    "isclose",
    "nanmax",
    "namespace_of_library",
    "nanmin",
    "rad2deg",
    "ravel_multi_index",
    "to_numpy",
    "unravel_index",
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


def asarray_like(values: Any, like: Any, *, dtype: str | None = None) -> Any:
    """``values`` as an array of the same library, and in the same place, as the array ``like``.

    ``dtype`` is the name of a standard data type (``"float64"``, ``"int32"``, ...); without it the
    data type of ``values`` is kept.
    """
    xp = array_namespace(like)
    if not is_array_api_obj(like):
        return xp.asarray(values, dtype=None if dtype is None else getattr(xp, dtype))
    return xp.asarray(values, dtype=None if dtype is None else getattr(xp, dtype), device=device_of(like))


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


def atleast_1d(value: Any) -> Any:
    """``value`` as an array with at least one dimension. Non-arrays become NumPy arrays."""
    if not is_array_api_obj(value):
        return np.atleast_1d(value)
    if value.ndim == 0:
        return array_namespace(value).reshape(value, (1,))
    return value


def atleast_2d(value: Any) -> Any:
    """``value`` as an array with at least two dimensions, new axes leading as in ``np.atleast_2d``."""
    if not is_array_api_obj(value):
        return np.atleast_2d(value)
    if value.ndim < 2:
        return array_namespace(value).reshape(value, (1,) * (2 - value.ndim) + tuple(value.shape))
    return value


def flatnonzero(mask: Any) -> Any:
    """Indices of the non-zero entries of a one-dimensional array."""
    return array_namespace(mask).nonzero(mask)[0]


def deg2rad(angle_deg: Any) -> Any:
    return angle_deg * (math.pi / 180.0)


def rad2deg(angle_rad: Any) -> Any:
    return angle_rad * (180.0 / math.pi)


def isclose(a: Any, b: Any, *, rtol: float = 1e-5, atol: float = 1e-8) -> Any:
    """Elementwise ``|a - b| <= atol + rtol * |b|``, as ``np.isclose`` without NaN handling."""
    xp = array_namespace(a, b)
    magnitude_of_b = xp.abs(b) if is_array_api_obj(b) else abs(b)
    return xp.abs(a - b) <= atol + rtol * magnitude_of_b


def nanmin(array: Any) -> Any:
    """Smallest value of ``array`` ignoring NaN, as a zero-dimensional array."""
    xp = array_namespace(array)
    return xp.min(xp.where(xp.isnan(array), xp.inf, array))


def nanmax(array: Any) -> Any:
    """Largest value of ``array`` ignoring NaN, as a zero-dimensional array."""
    xp = array_namespace(array)
    return xp.max(xp.where(xp.isnan(array), -xp.inf, array))


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

