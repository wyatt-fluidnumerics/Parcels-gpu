"""GPU counterparts of the pre-built interpolators for structured grids.

``FieldSet.to_gpu_arrays()`` installs them in place of the interpolator they extend. Each one
evaluates a field in GPU kernels that read the bracketing corners straight from the field's
``GpuArray`` buffer, where the interpolator it extends gathers the corners into arrays and blends
those. Whenever the field data or the particle positions are not on the GPU, they defer to the
interpolator they extend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import parcels._typing as ptyping
from parcels._core.utils.array_backend import is_gpu_array
from parcels._gpu_array import GpuArray, lowest_and_highest
from parcels._gpu_array.kernels import (
    sample_cgrid_horizontal_velocity,
    sample_cgrid_tracer,
    sample_cgrid_vertical_velocity,
    sample_linear,
)
from parcels.interpolators._base import ScalarInterpolator, VectorInterpolator
from parcels.interpolators._xinterpolators import (
    CGrid_Tracer,
    CGrid_Velocity,
    XLinear,
    XLinear_Velocity,
    _get_offsets_dictionary,
)

if TYPE_CHECKING:
    from parcels._core.field import Field, VectorField

__all__ = ["GpuCGrid_Tracer", "GpuCGrid_Velocity", "GpuXLinear", "GpuXLinear_Velocity", "gpu_counterpart"]


def _gpu_array_of(field: Field) -> GpuArray | None:
    variable = getattr(field.data, "variable", None)
    backend = getattr(variable, "_data", None)
    return backend if isinstance(backend, GpuArray) else None


def _component_gpu_arrays(vectorfield: VectorField) -> list[GpuArray] | None:
    """The GpuArrays of U, V and, if the vector field has it, W; None unless every one of them has a GpuArray."""
    components = [vectorfield.U, vectorfield.V]
    if vectorfield.W:
        components.append(vectorfield.W)
    gpu_arrays = [_gpu_array_of(component) for component in components]
    if any(gpu_array is None for gpu_array in gpu_arrays):
        return None
    return gpu_arrays


def _time_level_extremes(gpu_array: GpuArray, grid_positions: dict) -> tuple[int, int] | None:
    """Range of the particles' time indices, found once so every component of a vector field can reuse it."""
    time_index = grid_positions["T"]["index"]
    if gpu_array.shape[0] == 1 or time_index.size == 0:
        return None
    return lowest_and_highest(time_index)


def _positions_are_on_gpu(grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]]) -> bool:
    return is_gpu_array(grid_positions["X"]["index"]) and is_gpu_array(grid_positions["X"]["bcoord"])


class GpuXLinear(XLinear):
    """Trilinear interpolation on a regular grid, evaluated in a single GPU kernel."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        """Trilinear interpolation on a regular grid."""
        gpu_array = _gpu_array_of(field)
        if gpu_array is None or not _positions_are_on_gpu(grid_positions):
            return super().interp(particle_positions, grid_positions, field)
        return sample_linear(gpu_array, grid_positions)


class GpuXLinear_Velocity(XLinear_Velocity):  # noqa:  N801
    """Trilinear interpolation on a regular grid for VectorFields of velocity, one GPU kernel per component."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]],
        vectorfield: VectorField,
    ):
        """Trilinear interpolation on a regular grid for VectorFields of velocity."""
        gpu_arrays = _component_gpu_arrays(vectorfield)
        if gpu_arrays is None or not _positions_are_on_gpu(grid_positions):
            return super().interp(particle_positions, grid_positions, vectorfield)

        time_level_extremes = _time_level_extremes(gpu_arrays[0], grid_positions)
        u = sample_linear(gpu_arrays[0], grid_positions, time_level_extremes)
        v = sample_linear(gpu_arrays[1], grid_positions, time_level_extremes)
        if vectorfield.grid.mesh.is_spherical():
            u /= vectorfield.grid.deg2m * np.cos(np.deg2rad(particle_positions["y"]))
            v /= vectorfield.grid.deg2m

        if vectorfield.W:
            w = sample_linear(gpu_arrays[2], grid_positions, time_level_extremes)
        else:
            w = np.zeros_like(u)
        return u, v, w


class GpuCGrid_Velocity(CGrid_Velocity):  # noqa:  N801
    """C-grid interpolation of velocity, evaluated in one GPU kernel for U and V and one for W."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]],
        vectorfield: VectorField,
    ):
        """C-grid interpolation of velocity, following Delandmeter and Van Sebille (2019)."""
        gpu_arrays = _component_gpu_arrays(vectorfield)
        if gpu_arrays is None or not _positions_are_on_gpu(grid_positions):
            return super().interp(particle_positions, grid_positions, vectorfield)

        grid = vectorfield.grid
        offsets = _get_offsets_dictionary(grid)
        node_lon, node_lat = grid._node_coordinate_values(like=grid_positions["X"]["index"])
        time_level_extremes = _time_level_extremes(gpu_arrays[0], grid_positions)
        horizontal = sample_cgrid_horizontal_velocity(
            gpu_arrays[0],
            gpu_arrays[1],
            grid_positions,
            node_lon,
            node_lat,
            particle_positions["y"],
            offsets,
            spherical=grid.mesh.is_spherical(),
            deg2m=grid.deg2m,
            time_level_extremes=time_level_extremes,
        )
        if vectorfield.W:
            w = sample_cgrid_vertical_velocity(
                gpu_arrays[2],
                grid_positions,
                offsets,
                clipping_shape=gpu_arrays[0].shape,
                time_level_extremes=time_level_extremes,
            )
        else:
            w = np.zeros_like(horizontal.u)
        return horizontal.u, horizontal.v, w


class GpuCGrid_Tracer(CGrid_Tracer):  # noqa:  N801
    """C-grid interpolation of a tracer, constant over each cell, evaluated in a single GPU kernel."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        """C-grid interpolation of a tracer, following Delandmeter and Van Sebille (2019)."""
        gpu_array = _gpu_array_of(field)
        if gpu_array is None or not _positions_are_on_gpu(grid_positions):
            return super().interp(particle_positions, grid_positions, field)
        return sample_cgrid_tracer(gpu_array, grid_positions, _get_offsets_dictionary(field.grid))


_GPU_COUNTERPARTS: dict[type, type] = {
    XLinear: GpuXLinear,
    XLinear_Velocity: GpuXLinear_Velocity,
    CGrid_Velocity: GpuCGrid_Velocity,
    CGrid_Tracer: GpuCGrid_Tracer,
}


def gpu_counterpart(
    interpolator: ScalarInterpolator | VectorInterpolator,
) -> ScalarInterpolator | VectorInterpolator:
    """Return the GPU counterpart of a pre-built interpolator, or the interpolator itself if it has none.

    Only exact matches are replaced, so subclasses and user-written interpolators are left alone.
    """
    counterpart = _GPU_COUNTERPARTS.get(type(interpolator))
    return interpolator if counterpart is None else counterpart()
