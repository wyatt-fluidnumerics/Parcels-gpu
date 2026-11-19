from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Hashable
from typing import Any

from parcels._core.utils.array_backend import array_namespace


def _nbytes(array: Any) -> int:
    """Size of ``array`` in bytes, also for array types without an ``nbytes`` attribute."""
    nbytes = getattr(array, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    xp = array_namespace(array)
    dtype = array.dtype
    if xp.isdtype(dtype, "bool"):
        itemsize = 1
    elif xp.isdtype(dtype, "integral"):
        itemsize = xp.iinfo(dtype).bits // 8
    elif xp.isdtype(dtype, "real floating"):
        itemsize = xp.finfo(dtype).bits // 8
    else:
        # finfo describes one component of a complex number
        itemsize = 2 * xp.finfo(dtype).bits // 8
    return math.prod(array.shape) * itemsize


class ByteBoundedLRUCache:
    """LRU cache bounded by total stored bytes.

    Keys are hashable (typically chunk coordinate tuples).
    Values are arrays whose size in bytes drives eviction.
    """

    def __init__(self, max_bytes: int) -> None:
        assert max_bytes > 0
        self._max_bytes = max_bytes
        self._cache: OrderedDict[Hashable, Any] = OrderedDict()
        self._current_bytes = 0

    @property
    def current_bytes(self) -> int:
        return self._current_bytes

    def get(self, key: Hashable) -> Any | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: Hashable, value: Any) -> None:
        nbytes = _nbytes(value)
        if nbytes > self._max_bytes:
            return
        # Remove existing entry if present (will be re-inserted at end)
        if key in self._cache:
            self._current_bytes -= _nbytes(self._cache.pop(key))
        # Evict LRU entries until there's room
        while self._current_bytes + nbytes > self._max_bytes:
            _, evicted = self._cache.popitem(last=False)
            self._current_bytes -= _nbytes(evicted)
        self._cache[key] = value
        self._current_bytes += nbytes

    def clear(self) -> None:
        self._cache.clear()
        self._current_bytes = 0
