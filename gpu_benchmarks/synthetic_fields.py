"""Lazily generated, global-scale synthetic SGRID datasets for Parcels performance benchmarks.

Both grids are spherical A-grids (U and V on the nodes), with an analytic, smooth, time-varying
velocity field. Field data is dask-backed with one chunk per time level, so a time level is only
materialized when Parcels' time-window cache asks for it.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import dask.array as da
import numpy as np
import xarray as xr
from particle_seeds import ParticleSeeds

import parcels._sgrid as sgrid


# ------------------------------------------------------------------------------------------------
# Grid specification
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SyntheticGridSpec:
    resolution_deg: float = 1.0 / 12.0
    nz: int = 50
    n_time_levels: int = 2
    time_level_spacing_hours: float = 24.0
    max_depth_m: float = 5000.0
    curvilinear_distortion_deg: float = 1.0

    @property
    def nx(self) -> int:
        return int(round(360.0 / self.resolution_deg))

    @property
    def ny(self) -> int:
        return int(round(180.0 / self.resolution_deg))

    @property
    def time_span_seconds(self) -> float:
        return (self.n_time_levels - 1) * self.time_level_spacing_hours * 3600.0

    def field_bytes_per_time_level(self) -> int:
        return self.nz * self.ny * self.nx * np.dtype(np.float32).itemsize


def rectilinear_node_coordinates_deg(spec: SyntheticGridSpec) -> tuple[np.ndarray, np.ndarray]:
    lon_deg = (-180.0 + spec.resolution_deg * np.arange(spec.nx)).astype(np.float32)
    lat_deg = np.linspace(
        -90.0 + 0.5 * spec.resolution_deg, 90.0 - 0.5 * spec.resolution_deg, spec.ny, dtype=np.float64
    ).astype(np.float32)
    return lon_deg, lat_deg


def curvilinear_node_coordinates_deg(spec: SyntheticGridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Smoothly distorted global lon/lat nodes, shape (ny, nx).

    The displacement is at most ``curvilinear_distortion_deg`` and its gradient is at most
    ``curvilinear_distortion_deg * pi / 180`` per degree, so cells stay convex quadrilaterals for any
    resolution as long as that gradient is well below 1. The latitude displacement carries a cos^2(lat)
    factor so nodes never cross the poles.
    """
    lon_1d_deg, lat_1d_deg = rectilinear_node_coordinates_deg(spec)
    lon_regular_deg, lat_regular_deg = np.meshgrid(lon_1d_deg.astype(np.float64), lat_1d_deg.astype(np.float64))
    lon_rad = np.deg2rad(lon_regular_deg)
    lat_rad = np.deg2rad(lat_regular_deg)
    amplitude_deg = spec.curvilinear_distortion_deg
    lon_deg = lon_regular_deg + amplitude_deg * np.sin(2.0 * lat_rad) * np.cos(lon_rad)
    lat_deg = lat_regular_deg + 0.5 * amplitude_deg * np.sin(2.0 * lon_rad) * np.cos(lat_rad) ** 2
    return lon_deg.astype(np.float32), lat_deg.astype(np.float32)


def depth_levels_m(spec: SyntheticGridSpec) -> np.ndarray:
    """Quadratically stretched depth levels (fine near the surface, like most ocean models)."""
    if spec.nz == 1:
        return np.zeros(1, dtype=np.float32)
    stretched_fraction = (np.arange(spec.nz) / (spec.nz - 1)) ** 2
    return (spec.max_depth_m * stretched_fraction).astype(np.float32)


# ------------------------------------------------------------------------------------------------
# Analytic velocity, generated one time level (dask chunk) at a time
# ------------------------------------------------------------------------------------------------
_EDDY_PERIOD_SECONDS = 10.0 * 86400.0
_DEPTH_DECAY_SCALE_M = 1000.0


def _surface_velocity_ms(lon_deg: np.ndarray, lat_deg: np.ndarray, time_s: float, component: str) -> np.ndarray:
    """Smooth analytic surface flow in m/s (not divergence-free; only smoothness matters here).

    The cos(lat) factor on U keeps the converted zonal velocity in deg/s bounded near the poles.
    """
    lon_rad = np.deg2rad(lon_deg, dtype=np.float32)
    lat_rad = np.deg2rad(lat_deg, dtype=np.float32)
    phase_rad = np.float32(2.0 * np.pi * time_s / _EDDY_PERIOD_SECONDS)
    if component == "U":
        return (0.5 * np.cos(lat_rad) ** 2 * np.sin(3.0 * lon_rad + phase_rad)).astype(np.float32)
    return (0.3 * np.sin(2.0 * lat_rad) * np.cos(2.0 * lon_rad - phase_rad)).astype(np.float32)


def _velocity_time_level_block(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    depth_m: np.ndarray,
    time_s: np.ndarray,
    component: str,
    block_info: dict | None = None,
) -> np.ndarray:
    time_index = block_info[None]["chunk-location"][0]
    surface_ms = _surface_velocity_ms(lon_deg, lat_deg, float(time_s[time_index]), component)
    depth_decay = np.exp(-depth_m.astype(np.float32) / np.float32(_DEPTH_DECAY_SCALE_M))
    block = np.empty((1, depth_m.size, *surface_ms.shape), dtype=np.float32)
    np.multiply(depth_decay[:, None, None], surface_ms[None, :, :], out=block[0])
    return block


def _lazy_velocity_component(
    lon_deg_2d: np.ndarray, lat_deg_2d: np.ndarray, depth_m: np.ndarray, time_s: np.ndarray, component: str
) -> da.Array:
    ny, nx = lon_deg_2d.shape
    generate_block = functools.partial(_velocity_time_level_block, lon_deg_2d, lat_deg_2d, depth_m, time_s, component)
    return da.map_blocks(
        generate_block,
        chunks=((1,) * time_s.size, (depth_m.size,), (ny,), (nx,)),
        dtype=np.float32,
        meta=np.empty((0, 0, 0, 0), dtype=np.float32),
        name=f"synthetic-{component}-{ny}x{nx}x{depth_m.size}",
    )


# ------------------------------------------------------------------------------------------------
# SGRID dataset assembly
# ------------------------------------------------------------------------------------------------
def _sgrid_metadata() -> sgrid.SGrid2DMetadata:
    return sgrid.SGrid2DMetadata(
        cf_role="grid_topology",
        topology_dimension=2,
        node_dimensions=("XG", "YG"),
        node_coordinates=("lon", "lat"),
        face_dimensions=(
            sgrid.FaceNodePadding("XC", "XG", sgrid.Padding.LOW),
            sgrid.FaceNodePadding("YC", "YG", sgrid.Padding.LOW),
        ),
        vertical_dimensions=(sgrid.FaceNodePadding("ZC", "depth", sgrid.Padding.BOTH),),
    )


def _assemble_dataset(
    spec: SyntheticGridSpec, lon_deg: np.ndarray, lat_deg: np.ndarray, is_curvilinear: bool
) -> xr.Dataset:
    depth_m = depth_levels_m(spec)
    time_s = np.arange(spec.n_time_levels) * spec.time_level_spacing_hours * 3600.0
    if is_curvilinear:
        lon_deg_2d, lat_deg_2d = lon_deg, lat_deg
        lon_coord = (["YG", "XG"], lon_deg, {"axis": "X", "units": "degrees_east"})
        lat_coord = (["YG", "XG"], lat_deg, {"axis": "Y", "units": "degrees_north"})
    else:
        lon_deg_2d, lat_deg_2d = np.meshgrid(lon_deg, lat_deg)
        lon_coord = (["XG"], lon_deg, {"axis": "X", "units": "degrees_east"})
        lat_coord = (["YG"], lat_deg, {"axis": "Y", "units": "degrees_north"})

    velocity_dims = ["time", "depth", "YG", "XG"]
    u_ms = _lazy_velocity_component(lon_deg_2d, lat_deg_2d, depth_m, time_s, "U")
    v_ms = _lazy_velocity_component(lon_deg_2d, lat_deg_2d, depth_m, time_s, "V")
    times = np.datetime64("2000-01-01T00:00:00", "ns") + (time_s * 1e9).astype("timedelta64[ns]")

    ds = xr.Dataset(
        {"U": (velocity_dims, u_ms), "V": (velocity_dims, v_ms)},
        coords={
            "time": (["time"], times, {"axis": "T"}),
            "depth": (["depth"], depth_m, {"axis": "Z", "units": "m", "positive": "down"}),
            "YC": (["YC"], np.arange(spec.ny) + 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(spec.ny), {"axis": "Y"}),
            "XC": (["XC"], np.arange(spec.nx) + 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(spec.nx), {"axis": "X"}),
            "lon": lon_coord,
            "lat": lat_coord,
        },
    )
    return sgrid._attach_sgrid_metadata(ds, _sgrid_metadata())


def rectilinear_global_dataset(spec: SyntheticGridSpec) -> xr.Dataset:
    lon_deg, lat_deg = rectilinear_node_coordinates_deg(spec)
    return _assemble_dataset(spec, lon_deg, lat_deg, is_curvilinear=False)


def curvilinear_global_dataset(spec: SyntheticGridSpec) -> xr.Dataset:
    lon_deg, lat_deg = curvilinear_node_coordinates_deg(spec)
    return _assemble_dataset(spec, lon_deg, lat_deg, is_curvilinear=True)


# ------------------------------------------------------------------------------------------------
# Particle seeding
# ------------------------------------------------------------------------------------------------
def seed_particles_uniformly(
    spec: SyntheticGridSpec,
    npart: int,
    rng: np.random.Generator,
    lon_bounds_deg: tuple[float, float] = (-170.0, 170.0),
    lat_bounds_deg: tuple[float, float] = (-70.0, 70.0),
) -> ParticleSeeds:
    """Uniform seeding away from the non-periodic dateline seam and the poles.

    Depth is drawn between the second and second-to-last level so particles start bracketed vertically.
    """
    depth_m = depth_levels_m(spec)
    if depth_m.size > 2:
        depth_bounds_m = (float(depth_m[1]), float(depth_m[-2]))
    else:
        depth_bounds_m = (float(depth_m[0]), float(depth_m[-1]))
    return ParticleSeeds(
        lon_deg=rng.uniform(*lon_bounds_deg, npart),
        lat_deg=rng.uniform(*lat_bounds_deg, npart),
        depth_m=rng.uniform(*depth_bounds_m, npart),
    )
