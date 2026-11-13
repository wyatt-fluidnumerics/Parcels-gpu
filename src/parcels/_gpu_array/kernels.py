"""GPU kernels that do, per particle and in one launch, what the NumPy code does array by array.

Each kernel here mirrors one piece of host code, named in its docstring, and must keep
giving the same results as that code, to round-off.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np

from parcels._core.utils.array_backend import import_cupy

_INDEX_HELPERS = """
__device__ inline long long parcels_clamp_index(long long index, long long size) {
    return index < 0 ? 0 : (index >= size ? size - 1 : index);
}

// A negative index counts from the end of its axis, as in NumPy.
__device__ inline long long parcels_wrap_index(long long index, long long size) {
    return parcels_clamp_index(index < 0 ? index + size : index, size);
}
"""


@functools.cache
def _search_sorted_kernel():
    cupy = import_cupy()
    return cupy.ElementwiseKernel(
        "raw C coordinate, int64 n_coordinates, X x, int64 left_out_of_bounds, int64 right_out_of_bounds",
        "int64 index, B bcoord",
        """
        // number of coordinates strictly below x, i.e. np.searchsorted(coordinate, x, side="left")
        long long n_below;
        if (x != x) {
            n_below = n_coordinates;  // NaN sorts after every coordinate
        } else {
            long long low = 0;
            long long high = n_coordinates;
            while (low < high) {
                long long middle = (low + high) / 2;
                if (coordinate[middle] < x) low = middle + 1; else high = middle;
            }
            n_below = low;
        }
        long long cell = n_below - 1;
        cell = cell < 0 ? 0 : (cell > n_coordinates - 2 ? n_coordinates - 2 : cell);

        // The cell width is formed in the coordinate's own precision, as the host code does.
        C lower = coordinate[cell];
        C cell_width = coordinate[cell + 1] - lower;
        bcoord = ((B)x - (B)lower) / (B)cell_width;

        if (x < coordinate[0]) cell = left_out_of_bounds;
        if (x > coordinate[n_coordinates - 1]) cell = right_out_of_bounds;
        index = cell;
        """,
        "parcels_search_sorted",
    )


def search_sorted(coordinate: Any, x: Any, left_out_of_bounds: int, right_out_of_bounds: int) -> tuple[Any, Any]:
    """Locate ``x`` in the strictly increasing 1D ``coordinate``: cell index and barycentric coordinate.

    Mirrors the body of ``parcels._core.index_search._search_1d_array`` for a coordinate
    with at least two values.
    """
    cupy = import_cupy()
    index = cupy.empty(x.shape, dtype=np.int64)
    bcoord = cupy.empty(x.shape, dtype=np.result_type(x.dtype, coordinate.dtype))
    _search_sorted_kernel()(coordinate, coordinate.size, x, left_out_of_bounds, right_out_of_bounds, index, bcoord)
    return index, bcoord


@functools.cache
def _sample_linear_kernel():
    cupy = import_cupy()
    return cupy.ElementwiseKernel(
        # Indices and weights arrive in whatever integer and float types the search produced,
        # and elementwise kernels do not cast their inputs, so each gets a type of its own.
        "raw T buffer, int64 lower_slot, int64 upper_slot, A tau_in, I zi, B zeta_in, "
        "J yi, C eta_in, K xi, D xsi_in, int64 n_slots, int64 nz, int64 ny, int64 nx",
        "V value",
        """
        double tau = tau_in;
        double zeta = zeta_in;
        double eta = eta_in;
        double xsi = xsi_in;
        long long slots[2] = {parcels_clamp_index(lower_slot, n_slots), parcels_clamp_index(upper_slot, n_slots)};
        long long z_levels[2] = {parcels_wrap_index(zi, nz), parcels_clamp_index((long long)zi + 1, nz)};
        long long y_levels[2] = {parcels_wrap_index(yi, ny), parcels_clamp_index((long long)yi + 1, ny)};
        long long x_levels[2] = {parcels_wrap_index(xi, nx), parcels_clamp_index((long long)xi + 1, nx)};
        long long level_size = nz * ny * nx;

        // A weight of exactly zero leaves its upper level unread, so that level cannot contribute a NaN.
        bool blend_in_time = tau != 0;
        bool blend_in_depth = zeta != 0;
        int n_time_levels_read = blend_in_time ? 2 : 1;
        int n_depth_levels_read = blend_in_depth ? 2 : 1;

        // The two x neighbours of every corner pair are read back to back: they nearly always share
        // a cache line, which another read in between would give thousands of other threads the
        // chance to evict.
        double at_corner[2][2][2][2];  // [time][depth][y][x]
        for (int it = 0; it < n_time_levels_read; ++it) {
            for (int iz = 0; iz < n_depth_levels_read; ++iz) {
                for (int iy = 0; iy < 2; ++iy) {
                    long long row = slots[it] * level_size + (z_levels[iz] * ny + y_levels[iy]) * nx;
                    at_corner[it][iz][iy][0] = buffer[row + x_levels[0]];
                    at_corner[it][iz][iy][1] = buffer[row + x_levels[1]];
                }
            }
        }

        double corners[2][2];
        for (int iy = 0; iy < 2; ++iy) {
            for (int ix = 0; ix < 2; ++ix) {
                double at_depth_level[2] = {0.0, 0.0};
                for (int iz = 0; iz < n_depth_levels_read; ++iz) {
                    at_depth_level[iz] = blend_in_time
                        ? at_corner[0][iz][iy][ix] * (1 - tau) + at_corner[1][iz][iy][ix] * tau
                        : at_corner[0][iz][iy][ix];
                }
                corners[iy][ix] = blend_in_depth
                    ? at_depth_level[0] * (1 - zeta) + at_depth_level[1] * zeta
                    : at_depth_level[0];
            }
        }
        value = (1 - xsi) * (1 - eta) * corners[0][0]
              + xsi * (1 - eta) * corners[0][1]
              + (1 - xsi) * eta * corners[1][0]
              + xsi * eta * corners[1][1];
        """,
        "parcels_sample_linear",
        preamble=_INDEX_HELPERS,
    )


def sample_linear(gpu_array: Any, grid_positions: dict, time_level_extremes: tuple[int, int] | None = None) -> Any:
    """Linear interpolation in time, depth, y and x of a ``(time, Z, Y, X)`` GpuArray at node-based positions.

    Mirrors ``parcels.interpolators.XLinear.interp``: the same corners (index and index + 1
    on every axis, clipped to the array), blended in the same order: time, then depth, then
    the horizontal bilinear weights. The blend is always done in float64, whereas the host
    code blends in float32 when the data and every weight are float32.
    """
    cupy = import_cupy()
    xi, xsi = grid_positions["X"]["index"], grid_positions["X"]["bcoord"]
    yi, eta = grid_positions["Y"]["index"], grid_positions["Y"]["bcoord"]
    zi, zeta = grid_positions["Z"]["index"], grid_positions["Z"]["bcoord"]
    ti, tau = grid_positions["T"]["index"], grid_positions["T"]["bcoord"]

    levels = gpu_array.bracketed_buffer(ti, level_extremes=time_level_extremes)
    n_slots, nz, ny, nx = levels.buffer.shape

    value = cupy.empty(
        np.broadcast_shapes(tau.shape, zeta.shape, eta.shape, xsi.shape),
        dtype=np.result_type(levels.buffer.dtype, tau.dtype, zeta.dtype, eta.dtype, xsi.dtype),
    )
    _sample_linear_kernel()(
        levels.buffer, levels.lower_slot, levels.upper_slot, tau, zi, zeta, yi, eta, xi, xsi, n_slots, nz, ny, nx, value
    )
    return value


_CGRID_HELPERS = """
// Where the two time levels around a particle start in a (slot, Z, Y, X) field buffer, and the shape of one level.
struct ParcelsTimeBracket {
    long long lower_level_start;
    long long upper_level_start;
    long long nz;
    long long ny;
    long long nx;
};

__device__ inline ParcelsTimeBracket parcels_time_bracket(
    long long lower_slot, long long upper_slot, long long n_slots, long long nz, long long ny, long long nx
) {
    long long level_size = nz * ny * nx;
    ParcelsTimeBracket bracket;
    bracket.lower_level_start = parcels_clamp_index(lower_slot, n_slots) * level_size;
    bracket.upper_level_start = parcels_clamp_index(upper_slot, n_slots) * level_size;
    bracket.nz = nz;
    bracket.ny = ny;
    bracket.nx = nx;
    return bracket;
}

// The field value at (zi, yi, xi), linear in time. Indices outside the field are wrapped as NumPy does
// and then clamped. A tau of exactly zero leaves the upper level unread, so it cannot contribute a NaN.
template <typename Buffer>
__device__ inline double parcels_sample_in_time(
    const Buffer& buffer, const ParcelsTimeBracket& bracket, long long zi, long long yi, long long xi, double tau
) {
    long long cell = (parcels_wrap_index(zi, bracket.nz) * bracket.ny + parcels_wrap_index(yi, bracket.ny)) * bracket.nx
                     + parcels_wrap_index(xi, bracket.nx);
    double at_lower_time = buffer[bracket.lower_level_start + cell];
    if (tau == 0) {
        return at_lower_time;
    }
    double at_upper_time = buffer[bracket.upper_level_start + cell];
    return at_lower_time * (1 - tau) + at_upper_time * tau;
}

// np.remainder for floating point numbers: the result takes the sign of the divisor.
template <typename Float>
__device__ inline Float parcels_floored_remainder(Float dividend, Float divisor) {
    Float remainder = fmod(dividend, divisor);
    if (remainder == 0) {
        return copysign(Float(0), divisor);
    }
    if ((remainder < 0) != (divisor < 0)) {
        remainder += divisor;
    }
    return remainder;
}

// Wraps the corner longitudes into [-180, 180), then moves each by 360 degrees where needed to lie within
// 180 degrees of the first corner. This runs in the longitudes' own precision, as on the host: adding
// 180 degrees rounds a float32 longitude by up to 8e-6 degrees, which is not small next to a cell width.
template <typename Lon>
__device__ inline void parcels_unwrap_corner_longitudes(Lon corner_lon[4]) {
    for (int corner = 0; corner < 4; ++corner) {
        corner_lon[corner] = parcels_floored_remainder<Lon>(corner_lon[corner] + Lon(180), Lon(360)) - Lon(180);
    }
    for (int corner = 1; corner < 4; ++corner) {
        if (corner_lon[corner] - corner_lon[0] > Lon(180)) {
            corner_lon[corner] = corner_lon[corner] - Lon(360);
        }
        if (-corner_lon[corner] + corner_lon[0] > Lon(180)) {
            corner_lon[corner] = corner_lon[corner] + Lon(360);
        }
    }
}

// i_u.phi2D_lin(eta, xsi) applied to four corner values ordered counter-clockwise from (eta, xsi) = (0, 0).
template <typename Eta, typename Xsi>
__device__ inline double parcels_bilinear(const double corner_values[4], Eta eta, Xsi xsi) {
    return (1 - xsi) * (1 - eta) * corner_values[0] + xsi * (1 - eta) * corner_values[1]
           + xsi * eta * corner_values[2] + (1 - xsi) * eta * corner_values[3];
}

// i_u._geodetic_distance
__device__ inline double parcels_geodetic_distance(
    double lat1, double lat2, double lon1, double lon2, bool spherical, double lat, double deg2m
) {
    if (spherical) {
        double rad = M_PI / 180.0;
        double zonal = (lon2 - lon1) * deg2m * cos(rad * lat);
        double meridional = (lat2 - lat1) * deg2m;
        return sqrt(zonal * zonal + meridional * meridional);
    }
    return sqrt((lon2 - lon1) * (lon2 - lon1) + (lat2 - lat1) * (lat2 - lat1));
}

// i_u._compute_jacobian_determinant
template <typename Eta, typename Xsi>
__device__ inline double parcels_jacobian_determinant(const double py[4], const double px[4], Eta eta, Xsi xsi) {
    const double dphidxsi[4] = {eta - 1, 1 - eta, eta, -eta};
    const double dphideta[4] = {xsi - 1, -xsi, xsi, 1 - xsi};
    double dxdxsi = 0;
    double dxdeta = 0;
    double dydxsi = 0;
    double dydeta = 0;
    for (int corner = 0; corner < 4; ++corner) {
        dxdxsi += dphidxsi[corner] * px[corner];
        dxdeta += dphideta[corner] * px[corner];
        dydxsi += dphidxsi[corner] * py[corner];
        dydeta += dphideta[corner] * py[corner];
    }
    return dxdxsi * dydeta - dxdeta * dydxsi;
}
"""


@functools.cache
def _cgrid_horizontal_velocity_kernel():
    cupy = import_cupy()
    return cupy.ElementwiseKernel(
        "raw P u_data, int64 u_lower_slot, int64 u_upper_slot, int64 u_n_slots, int64 u_nz, int64 u_ny, int64 u_nx, "
        "raw Q v_data, int64 v_lower_slot, int64 v_upper_slot, int64 v_n_slots, int64 v_nz, int64 v_ny, int64 v_nx, "
        "raw L lon, raw M lat, bool coordinates_are_2d, int64 n_node_y, int64 n_node_x, "
        "bool spherical, float64 deg2m, int64 offset_y, int64 offset_x, int64 ydim, int64 xdim, "
        "A tau_in, I zi, J yi, K xi, C eta_in, D xsi_in, Y particle_y_in",
        "R u, R v",
        """
        double tau = tau_in;
        // eta and xsi keep their own precision, so that 1 - eta and 1 - xsi round as they do on the host:
        // in float32 where the curvilinear search gives float32 barycentric coordinates.
        C eta = eta_in;
        D xsi = xsi_in;

        // px and py: the cell's corner nodes counter-clockwise from (yi, xi)
        long long node_x[2] = {parcels_wrap_index(xi, n_node_x), parcels_wrap_index((long long)xi + 1, n_node_x)};
        long long node_y[2] = {parcels_wrap_index(yi, n_node_y), parcels_wrap_index((long long)yi + 1, n_node_y)};
        const int corner_node_x[4] = {0, 1, 1, 0};
        const int corner_node_y[4] = {0, 0, 1, 1};
        L corner_lon[4];
        double py[4];
        for (int corner = 0; corner < 4; ++corner) {
            long long x_node = node_x[corner_node_x[corner]];
            long long y_node = node_y[corner_node_y[corner]];
            corner_lon[corner] = coordinates_are_2d ? lon[y_node * n_node_x + x_node] : lon[x_node];
            py[corner] = coordinates_are_2d ? lat[y_node * n_node_x + x_node] : lat[y_node];
        }
        if (spherical) {
            parcels_unwrap_corner_longitudes(corner_lon);
        }
        double px[4] = {corner_lon[0], corner_lon[1], corner_lon[2], corner_lon[3]};

        double c1 = parcels_geodetic_distance(py[0], py[1], px[0], px[1], spherical, parcels_bilinear(py, 0.0, xsi), deg2m);
        double c2 = parcels_geodetic_distance(py[1], py[2], px[1], px[2], spherical, parcels_bilinear(py, eta, 1.0), deg2m);
        double c3 = parcels_geodetic_distance(py[2], py[3], px[2], px[3], spherical, parcels_bilinear(py, 1.0, xsi), deg2m);
        double c4 = parcels_geodetic_distance(py[3], py[0], px[3], px[0], spherical, parcels_bilinear(py, eta, 0.0), deg2m);

        // U: the two corners are the X faces
        ParcelsTimeBracket u_levels = parcels_time_bracket(u_lower_slot, u_upper_slot, u_n_slots, u_nz, u_ny, u_nx);
        long long yi_o = parcels_clamp_index((long long)yi + offset_y, ydim);
        long long xi_1 = parcels_clamp_index((long long)xi + 1, xdim);
        double U0 = parcels_sample_in_time(u_data, u_levels, zi, yi_o, xi, tau) * c4;
        double U1 = parcels_sample_in_time(u_data, u_levels, zi, yi_o, xi_1, tau) * c2;
        double Uvel = (1 - xsi) * U0 + xsi * U1;

        // V: the two corners are the Y faces
        ParcelsTimeBracket v_levels = parcels_time_bracket(v_lower_slot, v_upper_slot, v_n_slots, v_nz, v_ny, v_nx);
        long long yi_1 = parcels_clamp_index((long long)yi + 1, ydim);
        long long xi_o = parcels_clamp_index((long long)xi + offset_x, xdim);
        double V0 = parcels_sample_in_time(v_data, v_levels, zi, yi, xi_o, tau) * c1;
        double V1 = parcels_sample_in_time(v_data, v_levels, zi, yi_1, xi_o, tau) * c3;
        double Vvel = (1 - eta) * V0 + eta * V1;

        double jac = parcels_jacobian_determinant(py, px, eta, xsi);
        if (spherical) {
            jac *= deg2m;
        }
        // In the precision of the particle's y, as on the host, where a float32 latitude gives a float32 cosine.
        Y conversion = spherical ? Y(deg2m) * cos(particle_y_in * Y(M_PI / 180.0)) : Y(1);
        u = ((-(1 - eta) * Uvel - (1 - xsi) * Vvel) * px[0]
             + ((1 - eta) * Uvel - xsi * Vvel) * px[1]
             + (eta * Uvel + xsi * Vvel) * px[2]
             + (-eta * Uvel + (1 - xsi) * Vvel) * px[3]) / jac / conversion;
        v = ((-(1 - eta) * Uvel - (1 - xsi) * Vvel) * py[0]
             + ((1 - eta) * Uvel - xsi * Vvel) * py[1]
             + (eta * Uvel + xsi * Vvel) * py[2]
             + (-eta * Uvel + (1 - xsi) * Vvel) * py[3]) / jac / conversion;
        """,
        "parcels_cgrid_horizontal_velocity",
        preamble=_INDEX_HELPERS + _CGRID_HELPERS,
    )


@functools.cache
def _cgrid_vertical_velocity_kernel():
    cupy = import_cupy()
    return cupy.ElementwiseKernel(
        "raw P w_data, int64 lower_slot, int64 upper_slot, int64 n_slots, int64 nz, int64 ny, int64 nx, "
        "int64 offset_z, int64 offset_y, int64 offset_x, int64 zdim, int64 ydim, int64 xdim, "
        "A tau_in, I zi, J yi, K xi, B zeta_in",
        "R w",
        """
        double tau = tau_in;
        // zeta keeps its own precision, so that 1 - zeta rounds as it does on the host
        B zeta = zeta_in;
        ParcelsTimeBracket w_levels = parcels_time_bracket(lower_slot, upper_slot, n_slots, nz, ny, nx);

        // W: the two corners are the Z faces
        long long yi_o = parcels_clamp_index((long long)yi + offset_y, ydim);
        long long xi_o = parcels_clamp_index((long long)xi + offset_x, xdim);
        long long zi_0 = parcels_clamp_index((long long)zi + offset_z, zdim);
        long long zi_1 = parcels_clamp_index((long long)zi + offset_z + 1, zdim);
        double W0 = parcels_sample_in_time(w_data, w_levels, zi_0, yi_o, xi_o, tau);
        double W1 = parcels_sample_in_time(w_data, w_levels, zi_1, yi_o, xi_o, tau);
        w = W0 * (1 - zeta) + W1 * zeta;
        """,
        "parcels_cgrid_vertical_velocity",
        preamble=_INDEX_HELPERS + _CGRID_HELPERS,
    )


@functools.cache
def _cgrid_tracer_kernel():
    cupy = import_cupy()
    return cupy.ElementwiseKernel(
        "raw P buffer, int64 lower_slot, int64 upper_slot, int64 n_slots, int64 nz, int64 ny, int64 nx, "
        "int64 offset_z, int64 offset_y, int64 offset_x, A tau_in, I zi, J yi, K xi",
        "R value",
        """
        ParcelsTimeBracket levels = parcels_time_bracket(lower_slot, upper_slot, n_slots, nz, ny, nx);
        long long zi_o = parcels_clamp_index((long long)zi + offset_z, nz);
        long long yi_o = parcels_clamp_index((long long)yi + offset_y, ny);
        long long xi_o = parcels_clamp_index((long long)xi + offset_x, nx);
        value = parcels_sample_in_time(buffer, levels, zi_o, yi_o, xi_o, tau_in);
        """,
        "parcels_cgrid_tracer",
        preamble=_INDEX_HELPERS + _CGRID_HELPERS,
    )


@dataclass(frozen=True)
class HorizontalVelocity:
    """The ``u`` and ``v`` that ``CGrid_Velocity.interp`` returns, as CuPy arrays."""

    u: Any
    v: Any


def sample_cgrid_horizontal_velocity(
    u_data: Any,
    v_data: Any,
    grid_positions: dict,
    node_lon: Any,
    node_lat: Any,
    particle_y: Any,
    offsets: dict[str, int],
    spherical: bool,
    deg2m: float,
    time_level_extremes: tuple[int, int] | None = None,
) -> HorizontalVelocity:
    """``u`` and ``v`` of a C-grid velocity field whose face-normal components are ``(time, Z, Y, X)`` GpuArrays.

    Mirrors the ``u`` and ``v`` part of ``parcels.interpolators.CGrid_Velocity.interp``, step by step:
    the corner coordinates (with longitudes unwrapped on a spherical mesh), the cell edge lengths, the
    face velocities scaled by those lengths, and their transformation by the cell's Jacobian. As on the
    host, the indices into V are clipped to the shape of U. ``node_lon`` and ``node_lat`` are the grid's
    node coordinates, 1D on a rectilinear grid and 2D on a curvilinear one.

    The arithmetic is done in float64, except for three steps that the host does in the precision of
    their inputs, which is float32 for particle positions and for the barycentric coordinates of the
    curvilinear search: the longitude unwrapping (precision of the coordinates), the weights ``1 - xsi``
    and ``1 - eta`` (of the barycentric coordinates) and the conversion to degrees on a spherical mesh
    (of ``particle_y``). In float64 these would differ from the host by more than round-off of the
    result, by far more near the poles.
    """
    cupy = import_cupy()
    xi, xsi = grid_positions["X"]["index"], grid_positions["X"]["bcoord"]
    yi, eta = grid_positions["Y"]["index"], grid_positions["Y"]["bcoord"]
    zi = grid_positions["Z"]["index"]
    ti, tau = grid_positions["T"]["index"], grid_positions["T"]["bcoord"]

    n_time_levels, _, ydim, xdim = u_data.shape
    u_levels = u_data.bracketed_buffer(ti, n_time_levels, time_level_extremes)
    v_levels = v_data.bracketed_buffer(ti, n_time_levels, time_level_extremes)

    coordinates_are_2d = node_lon.ndim == 2
    if coordinates_are_2d:
        n_node_y, n_node_x = node_lon.shape
    else:
        n_node_y, n_node_x = node_lat.size, node_lon.size

    shape = np.broadcast_shapes(tau.shape, zi.shape, yi.shape, eta.shape, xi.shape, xsi.shape, particle_y.shape)
    dtype = np.result_type(
        u_levels.buffer.dtype, v_levels.buffer.dtype, tau.dtype, eta.dtype, xsi.dtype, node_lon.dtype, node_lat.dtype
    )
    u = cupy.empty(shape, dtype=dtype)
    v = cupy.empty(shape, dtype=dtype)
    _cgrid_horizontal_velocity_kernel()(
        u_levels.buffer, u_levels.lower_slot, u_levels.upper_slot, *u_levels.buffer.shape,
        v_levels.buffer, v_levels.lower_slot, v_levels.upper_slot, *v_levels.buffer.shape,
        node_lon, node_lat, coordinates_are_2d, n_node_y, n_node_x,
        spherical, deg2m, offsets["Y"], offsets["X"], ydim, xdim,
        tau, zi, yi, xi, eta, xsi, particle_y,
        u, v,
    )  # fmt: skip
    return HorizontalVelocity(u=u, v=v)


def sample_cgrid_vertical_velocity(
    w_data: Any,
    grid_positions: dict,
    offsets: dict[str, int],
    clipping_shape: tuple[int, ...],
    time_level_extremes: tuple[int, int] | None = None,
) -> Any:
    """``w`` of a C-grid velocity field whose vertical component is a ``(time, Z, Y, X)`` GpuArray.

    Mirrors the ``w`` part of ``parcels.interpolators.CGrid_Velocity.interp``: linear in depth between
    the two Z faces of the cell, both always read. As on the host, the indices are clipped to
    ``clipping_shape``, the ``(time, Z, Y, X)`` shape of U, and the weight ``1 - zeta`` is formed in the
    precision of zeta. The rest is done in float64.
    """
    cupy = import_cupy()
    xi = grid_positions["X"]["index"]
    yi = grid_positions["Y"]["index"]
    zi, zeta = grid_positions["Z"]["index"], grid_positions["Z"]["bcoord"]
    ti, tau = grid_positions["T"]["index"], grid_positions["T"]["bcoord"]

    n_time_levels, zdim, ydim, xdim = clipping_shape
    w_levels = w_data.bracketed_buffer(ti, n_time_levels, time_level_extremes)

    w = cupy.empty(
        np.broadcast_shapes(tau.shape, zi.shape, zeta.shape, yi.shape, xi.shape),
        dtype=np.result_type(w_levels.buffer.dtype, tau.dtype, zeta.dtype),
    )
    _cgrid_vertical_velocity_kernel()(
        w_levels.buffer, w_levels.lower_slot, w_levels.upper_slot, *w_levels.buffer.shape,
        offsets["Z"], offsets["Y"], offsets["X"], zdim, ydim, xdim,
        tau, zi, yi, xi, zeta,
        w,
    )  # fmt: skip
    return w


def sample_cgrid_tracer(gpu_array: Any, grid_positions: dict, offsets: dict[str, int]) -> Any:
    """Mirrors ``parcels.interpolators.CGrid_Tracer.interp``: the value of the cell holding each particle, linear in time."""
    cupy = import_cupy()
    xi = grid_positions["X"]["index"]
    yi = grid_positions["Y"]["index"]
    zi = grid_positions["Z"]["index"]
    ti, tau = grid_positions["T"]["index"], grid_positions["T"]["bcoord"]

    levels = gpu_array.bracketed_buffer(ti)

    value = cupy.empty(
        np.broadcast_shapes(tau.shape, zi.shape, yi.shape, xi.shape),
        dtype=np.result_type(levels.buffer.dtype, tau.dtype),
    )
    _cgrid_tracer_kernel()(
        levels.buffer, levels.lower_slot, levels.upper_slot, *levels.buffer.shape,
        offsets["Z"], offsets["Y"], offsets["X"],
        tau, zi, yi, xi,
        value,
    )  # fmt: skip
    return value
