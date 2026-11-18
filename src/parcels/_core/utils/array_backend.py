"""Helpers for code that runs on either NumPy (CPU) or CuPy (GPU) arrays.

Most ``np.<function>(array)`` calls dispatch to CuPy on their own when ``array`` is a CuPy
array (NEP 18). Array *creation* (``np.zeros``, ``np.full``, ``np.arange``, ``np.asarray``)
does not, so code that creates arrays asks ``array_namespace`` which module to create them in.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np


def array_namespace(*arrays: Any) -> ModuleType:
    """Return ``cupy`` if any of ``arrays`` is a CuPy array, otherwise ``numpy``."""
    # An array can only be a CuPy array if something has already imported cupy, so CPU-only
    # runs never import it here.
    cupy = sys.modules.get("cupy")
    if cupy is None:
        return np
    return cupy.get_array_module(*arrays)


def is_gpu_array(array: Any) -> bool:
    cupy = sys.modules.get("cupy")
    return cupy is not None and isinstance(array, cupy.ndarray)


def is_array(array: Any) -> bool:
    return isinstance(array, np.ndarray) or is_gpu_array(array)


def to_numpy(array: Any) -> np.ndarray:
    """Return ``array`` as a NumPy array, copying from the GPU when needed."""
    if is_gpu_array(array):
        return array.get()
    return np.asarray(array)


def import_cupy() -> ModuleType:
    try:
        import cupy
    except ImportError as e:
        raise ImportError("Running Parcels on a GPU requires CuPy, which is not installed in this environment.") from e
    return cupy
