"""Loader and ocean-point particle seeding for the MOI (Mercator PSY4V3R1, global 1/12 degree NEMO) benchmark data.

Mirrors the conversion in parcels-benchmarks/benchmarks/moi_curvilinear.py, but opens the NetCDF files
with xarray directly instead of through intake.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from particle_seeds import ParticleSeeds

import parcels

MOI_DATA_DIRECTORY = Path(
    "/mnt/beegfs/wyatt/parcels-benchmarks/data/surf-data/parcels-benchmarks/data/"
    "Parcelsv4_Benchmarking_data/Parcels_Benchmarks_MOi_data"
)
MOI_MESH_FILE = MOI_DATA_DIRECTORY / "PSY4V3R1_mesh_hgr.nc"
_MOI_VARIABLE_FOR_COMPONENT = {"U": "vozocrtx", "V": "vomecrty", "W": "vovecrtz"}
_SEEDING_EXCLUDED_ABS_LAT_DEG = 80.0


# ------------------------------------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------------------------------------
def _daily_files(component: str) -> list[Path]:
    files = sorted(MOI_DATA_DIRECTORY.glob(f"psy4v3r1-daily_{component}_*.nc"))
    if not files:
        raise FileNotFoundError(f"No MOI {component} files found in {MOI_DATA_DIRECTORY}")
    return files


_VERTICAL_DIMENSION_FOR_COMPONENT = {"U": "deptht", "V": "deptht", "W": "depthw"}
_DEPTH_LEVELS_PER_CHUNK = 10


def _open_component_float32(component: str, n_days: int | None, n_depth_levels: int | None) -> xr.Dataset:
    """Lazily open one velocity component as float32, with land (the CF fill value) set to NaN.

    CF decoding would promote these float32 values to float64 (their scale_factor is a float64 1.0), doubling
    the transient host memory of every time-level read, so masking and scaling are applied here in float32.
    The data is chunked along depth so a time-level read never materializes one 50-level float64 block.
    """
    files = _daily_files(component)[:n_days]
    vertical_dimension = _VERTICAL_DIMENSION_FOR_COMPONENT[component]
    ds = xr.open_mfdataset(
        files,
        concat_dim="time_counter",
        combine="nested",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        chunks={vertical_dimension: _DEPTH_LEVELS_PER_CHUNK},
        engine="netcdf4",
        mask_and_scale=False,
    )
    variable_name = _MOI_VARIABLE_FOR_COMPONENT[component]
    ds = ds[[variable_name]].rename_vars({variable_name: component})
    if n_depth_levels is not None:
        ds = ds.isel({vertical_dimension: slice(0, n_depth_levels)})

    raw = ds[component]
    fill_value = raw.attrs.get("_FillValue", raw.attrs.get("missing_value"))
    scale_factor = np.float32(raw.attrs.get("scale_factor", 1.0))
    add_offset = np.float32(raw.attrs.get("add_offset", 0.0))
    velocity_ms = raw.astype(np.float32)
    if fill_value is not None:
        velocity_ms = velocity_ms.where(velocity_ms != np.float32(fill_value))
    if scale_factor != 1 or add_offset != 0:
        velocity_ms = velocity_ms * scale_factor + add_offset
    ds[component] = velocity_ms.assign_attrs(
        {k: v for k, v in raw.attrs.items() if k not in ("_FillValue", "missing_value", "scale_factor", "add_offset")}
    )
    return ds


def moi_sgrid_dataset(
    include_w: bool = False, n_days: int | None = None, n_depth_levels: int | None = None
) -> xr.Dataset:
    """Dask-backed SGRID dataset of the MOI velocities, keeping only the top ``n_depth_levels`` levels if given."""
    components = ["U", "V", "W"] if include_w else ["U", "V"]
    fields = {component: _open_component_float32(component, n_days, n_depth_levels) for component in components}
    w_depth_m = _open_component_float32("W", n_days, n_depth_levels)["depthw"].load()

    mesh = xr.open_dataset(MOI_MESH_FILE)[["glamf", "gphif"]].isel(t=0).load()
    mesh["depthw"] = w_depth_m
    return parcels.convert.nemo_to_sgrid(fields=fields, coords=mesh)


# ------------------------------------------------------------------------------------------------
# Particle seeding over open ocean
# ------------------------------------------------------------------------------------------------
def _eroded_surface_ocean_mask() -> np.ndarray:
    """Cells whose surface U and V, and those of all 8 neighbours, are finite and non-zero."""
    first_u_file, first_v_file = _daily_files("U")[0], _daily_files("V")[0]
    u_surface_ms = xr.open_dataset(first_u_file)["vozocrtx"].isel(deptht=0).values
    v_surface_ms = xr.open_dataset(first_v_file)["vomecrty"].isel(deptht=0).values
    is_wet = np.isfinite(u_surface_ms) & np.isfinite(v_surface_ms) & (u_surface_ms != 0) & (v_surface_ms != 0)

    is_wet_with_wet_neighbours = np.zeros_like(is_wet)
    interior = is_wet[1:-1, 1:-1].copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            interior &= is_wet[1 + dy : is_wet.shape[0] - 1 + dy, 1 + dx : is_wet.shape[1] - 1 + dx]
    is_wet_with_wet_neighbours[1:-1, 1:-1] = interior
    return is_wet_with_wet_neighbours


def seed_particles_in_ocean(
    npart: int,
    rng: np.random.Generator,
    depth_bounds_m: tuple[float, float] = (1.0, 100.0),
    n_depth_levels: int | None = None,
) -> ParticleSeeds:
    """Random positions inside randomly chosen open-ocean cells of the F-point (glamf, gphif) mesh.

    A particle is placed at random bilinear weights inside the quad of nodes (j, i)-(j+1, i+1) of an ocean
    cell, with the corner longitudes unwrapped relative to node (j, i) so cells on the dateline stay compact.
    With ``n_depth_levels``, the depth range is clipped to the retained W levels so particles start in the grid.
    """
    if n_depth_levels is not None:
        w_depth_m = xr.open_dataset(_daily_files("W")[0])["depthw"].values[:n_depth_levels]
        depth_bounds_m = (max(depth_bounds_m[0], float(w_depth_m[0])), min(depth_bounds_m[1], float(w_depth_m[-1])))
    mesh = xr.open_dataset(MOI_MESH_FILE)[["glamf", "gphif"]].isel(t=0)
    node_lon_deg = mesh["glamf"].values.astype(np.float64)
    node_lat_deg = mesh["gphif"].values.astype(np.float64)

    candidate_mask = _eroded_surface_ocean_mask()
    candidate_mask[-2:, :] = False
    candidate_mask[:, -2:] = False
    candidate_mask &= np.abs(node_lat_deg) < _SEEDING_EXCLUDED_ABS_LAT_DEG
    candidate_j, candidate_i = np.nonzero(candidate_mask)
    chosen = rng.integers(0, candidate_j.size, npart)
    j, i = candidate_j[chosen], candidate_i[chosen]

    corner_lon_deg = np.stack(
        [node_lon_deg[j, i], node_lon_deg[j, i + 1], node_lon_deg[j + 1, i + 1], node_lon_deg[j + 1, i]]
    )
    corner_lat_deg = np.stack(
        [node_lat_deg[j, i], node_lat_deg[j, i + 1], node_lat_deg[j + 1, i + 1], node_lat_deg[j + 1, i]]
    )
    corner_lon_deg = corner_lon_deg[0] + ((corner_lon_deg - corner_lon_deg[0] + 180.0) % 360.0 - 180.0)

    xsi = rng.uniform(0.05, 0.95, npart)
    eta = rng.uniform(0.05, 0.95, npart)
    bilinear_weights = np.stack([(1 - xsi) * (1 - eta), xsi * (1 - eta), xsi * eta, (1 - xsi) * eta])
    lon_deg = (bilinear_weights * corner_lon_deg).sum(axis=0)
    lon_deg = (lon_deg + 180.0) % 360.0 - 180.0
    lat_deg = (bilinear_weights * corner_lat_deg).sum(axis=0)
    return ParticleSeeds(lon_deg=lon_deg, lat_deg=lat_deg, depth_m=rng.uniform(*depth_bounds_m, npart))
