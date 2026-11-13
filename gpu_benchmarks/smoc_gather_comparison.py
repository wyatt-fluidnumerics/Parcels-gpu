"""Real-data comparison of `_gather_corners` implementations on a surface-drift simulation.

Data: CMEMS SMOC global 1/12 degree hourly surface currents (regular lat-lon A-grid, one depth level),
read lazily from NetCDF as the run advances. Simulation: particles seeded on ocean points, AdvectionRK2,
dt = 15 min, hourly ParticleFile output, particles that leave the ocean are deleted.

Only `parcels.interpolators._xinterpolators._gather_corners` differs between variants:
  upstream  : DataArray-level `isel` with flattened (*counts x npart) point indexers
  flattened : variable-level `isel` with flattened point indexers
  broadcast : variable-level `isel` with one broadcastable indexer per axis (what the source tree has)

Wall time is split into field reading (NetCDF -> window), output writing, and the remainder.

Usage: python gpu_benchmarks/smoc_gather_comparison.py --device cpu --backend windowed --npart 100000
"""

import argparse
import json
import socket
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from run_benchmark import DeleteErroredParticles

import parcels
import parcels.interpolators._xinterpolators as xinterpolators
from parcels._core._windowed_array import WindowedArray
from parcels._core.particlefile import ParticleFile
from parcels._gpu_array import GpuArray
from parcels.interpolators._xinterpolators import _CORNER_AXES

SMOC_GLOB = (
    "/mnt/beegfs/wyatt/parcels-benchmarks/data/surf-data/parcels-benchmarks/data/"
    "Parcelsv4_Benchmarking_data/Parcels_Benchmarks_SMOC_data/SMOC_*.nc"
)


# ------------------------------------------------------------------------------------------------
# The two earlier `_gather_corners` implementations
# ------------------------------------------------------------------------------------------------
def _flattened_point_indexers(data, axis_dim, levels, npart, make_indexer):
    dims = {axis: ("time" if axis == "T" else axis_dim.get(axis)) for axis in _CORNER_AXES}
    counts = tuple(len(levels[axis]) for axis in _CORNER_AXES)
    shape = (*counts, npart)
    selection_dict = {}
    for i, axis in enumerate(_CORNER_AXES):
        if dims[axis] is not None and dims[axis] in data.dims:
            stacked = np.stack(np.broadcast_arrays(*levels[axis]))
            other_slots = tuple(j for j in range(len(_CORNER_AXES)) if j != i)
            in_slot_i = np.expand_dims(stacked, other_slots)
            selection_dict[dims[axis]] = make_indexer(np.broadcast_to(in_slot_i, shape).reshape(-1))
    return selection_dict, shape


def gather_corners_upstream(data, axis_dim, levels, npart):
    selection_dict, shape = _flattened_point_indexers(
        data, axis_dim, levels, npart, lambda index: xr.DataArray(index, dims="points")
    )
    return data.isel(selection_dict).data.reshape(shape)


def gather_corners_flattened(data, axis_dim, levels, npart):
    selection_dict, shape = _flattened_point_indexers(
        data, axis_dim, levels, npart, lambda index: xr.Variable("points", index)
    )
    indexable = data if isinstance(data, WindowedArray) else data.variable
    return indexable.isel(selection_dict).data.reshape(shape)


GATHER_VARIANTS = {
    "upstream": gather_corners_upstream,
    "flattened": gather_corners_flattened,
    "broadcast": xinterpolators._gather_corners,
}


# ------------------------------------------------------------------------------------------------
# Wall-time attribution
# ------------------------------------------------------------------------------------------------
class Stopwatch:
    def __init__(self) -> None:
        self.total_s = 0.0

    def wrap(self, function):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.total_s += time.perf_counter() - start

        return timed


FIELD_READ_STOPWATCH = Stopwatch()
FIELD_WAIT_STOPWATCH = Stopwatch()
OUTPUT_WRITE_STOPWATCH = Stopwatch()
WindowedArray._read_level = FIELD_READ_STOPWATCH.wrap(WindowedArray._read_level)
GpuArray._read_level = FIELD_READ_STOPWATCH.wrap(GpuArray._read_level)
GpuArray._prefetched_or_read_level = FIELD_WAIT_STOPWATCH.wrap(GpuArray._prefetched_or_read_level)
ParticleFile.write = OUTPUT_WRITE_STOPWATCH.wrap(ParticleFile.write)


def synchronize(device: str) -> None:
    if device == "gpu":
        import cupy

        cupy.cuda.runtime.deviceSynchronize()


# ------------------------------------------------------------------------------------------------
# Fieldset and particles
# ------------------------------------------------------------------------------------------------
def open_smoc() -> xr.Dataset:
    ds = xr.open_mfdataset(SMOC_GLOB, chunks={"time": 1}, combine="by_coords")
    # The SMOC files carry no CF axis attributes, which the Copernicus converter uses to find the coordinates
    for coordinate_name, axis in (("longitude", "X"), ("latitude", "Y"), ("depth", "Z"), ("time", "T")):
        ds[coordinate_name].attrs["axis"] = axis
    return ds


def build_fieldset(device: str, backend: str) -> parcels.FieldSet:
    ds = open_smoc()
    ds_sgrid = parcels.convert.copernicusmarine_to_sgrid(fields={"U": ds["utotal"], "V": ds["vtotal"]})
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds_sgrid)
    if device == "gpu":
        return fieldset.to_gpu_arrays()
    if backend == "windowed":
        return fieldset.to_windowed_arrays()
    return fieldset.to_chunk_cached_arrays()


@dataclass
class OceanSeeds:
    lon_deg: np.ndarray
    lat_deg: np.ndarray


def seed_ocean_particles(npart: int, seed: int) -> OceanSeeds:
    """Random positions inside cells whose four corners are ocean, between 60S and 60N."""
    first_level = open_smoc()["utotal"].isel(time=0, depth=0)
    is_ocean = np.isfinite(first_level.values)
    all_corners_ocean = is_ocean[:-1, :-1] & is_ocean[:-1, 1:] & is_ocean[1:, :-1] & is_ocean[1:, 1:]
    lat_deg = first_level["latitude"].values.astype(np.float64)
    lon_deg = first_level["longitude"].values.astype(np.float64)
    all_corners_ocean &= (np.abs(lat_deg[:-1]) < 60.0)[:, None]

    rng = np.random.default_rng(seed)
    cell_j, cell_i = np.nonzero(all_corners_ocean)
    chosen = rng.integers(0, cell_j.size, npart)
    cell_j, cell_i = cell_j[chosen], cell_i[chosen]
    return OceanSeeds(
        lon_deg=lon_deg[cell_i] + rng.uniform(0.05, 0.95, npart) * (lon_deg[cell_i + 1] - lon_deg[cell_i]),
        lat_deg=lat_deg[cell_j] + rng.uniform(0.05, 0.95, npart) * (lat_deg[cell_j + 1] - lat_deg[cell_j]),
    )


# ------------------------------------------------------------------------------------------------
# One simulation
# ------------------------------------------------------------------------------------------------
@dataclass
class RunResult:
    variant: str
    device: str
    backend: str
    npart: int
    npart_at_end: int
    runtime_hours: float
    nsteps: int
    wall_s: float
    field_read_s: float
    output_write_s: float
    remainder_s: float
    remainder_s_per_step: float
    hostname: str


def run_simulation(
    variant: str, device: str, backend: str, seeds: OceanSeeds, runtime_hours: float, output_parent_directory: str
) -> RunResult:
    xinterpolators._gather_corners = GATHER_VARIANTS[variant]
    dt = np.timedelta64(15, "m")
    nsteps = int(runtime_hours * 4)
    FIELD_READ_STOPWATCH.total_s = 0.0
    FIELD_WAIT_STOPWATCH.total_s = 0.0
    OUTPUT_WRITE_STOPWATCH.total_s = 0.0

    with tempfile.TemporaryDirectory(dir=output_parent_directory) as output_directory:
        start = time.perf_counter()
        fieldset = build_fieldset(device, backend)
        pset = parcels.ParticleSet(fieldset, x=seeds.lon_deg, y=seeds.lat_deg)
        output_file = ParticleFile(Path(output_directory) / "trajectories.parquet", outputdt=np.timedelta64(1, "h"))
        pset.execute(
            [parcels.kernels.AdvectionRK2, DeleteErroredParticles],
            runtime=nsteps * dt,
            dt=dt,
            output_file=output_file,
            verbose_progress=False,
        )
        synchronize(device)
        wall_s = time.perf_counter() - start

    remainder_s = wall_s - FIELD_READ_STOPWATCH.total_s - OUTPUT_WRITE_STOPWATCH.total_s
    return RunResult(
        variant=variant,
        device=device,
        backend="gpu_array" if device == "gpu" else backend,
        npart=seeds.lon_deg.size,
        npart_at_end=len(pset),
        runtime_hours=runtime_hours,
        nsteps=nsteps,
        wall_s=wall_s,
        field_read_s=FIELD_READ_STOPWATCH.total_s,
        output_write_s=OUTPUT_WRITE_STOPWATCH.total_s,
        remainder_s=remainder_s,
        remainder_s_per_step=remainder_s / nsteps,
        hostname=socket.gethostname(),
    )


def warm_page_cache(runtime_hours: float) -> None:
    """Read the time levels the runs will use once, so every variant sees the same file-system state."""
    ds = open_smoc()
    n_levels = int(runtime_hours) + 2
    for name in ("utotal", "vtotal"):
        for level in range(n_levels):
            ds[name].isel(time=level).values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--backend", choices=["windowed", "chunk_cached"], default="windowed", help="cpu only")
    parser.add_argument("--npart", type=int, nargs="+", default=[100_000])
    parser.add_argument("--runtime-hours", type=float, default=48.0)
    parser.add_argument("--variants", nargs="+", choices=list(GATHER_VARIANTS), default=None)
    parser.add_argument("--results-file", default="gpu_benchmarks/results/smoc_gather_comparison.jsonl")
    parser.add_argument("--output-parent-directory", default="gpu_benchmarks/logs")
    parser.add_argument("--no-prefetch", action="store_true", help="gpu only: do not read the next time level ahead")
    args = parser.parse_args()

    GpuArray.prefetch_next_level = not args.no_prefetch
    variants = args.variants or (["flattened", "broadcast"] if args.device == "gpu" else list(GATHER_VARIANTS))
    warm_page_cache(args.runtime_hours)
    # one short untimed run compiles GPU kernels and imports everything
    run_simulation(
        "broadcast",
        args.device,
        args.backend,
        seed_ocean_particles(1000, seed=1),
        runtime_hours=1.0,
        output_parent_directory=args.output_parent_directory,
    )

    for npart in args.npart:
        seeds = seed_ocean_particles(npart, seed=0)
        for variant in variants:
            result = run_simulation(
                variant, args.device, args.backend, seeds, args.runtime_hours, args.output_parent_directory
            )
            print(
                f"[{result.device}/{result.backend}] npart={result.npart:>9d} {variant:>9s}: wall={result.wall_s:8.2f} s "
                f"(field read {result.field_read_s:6.2f}, output {result.output_write_s:6.2f}, "
                f"rest {result.remainder_s:8.2f} = {1e3 * result.remainder_s_per_step:8.2f} ms/step) "
                f"survivors={result.npart_at_end} "
                f"prefetch={GpuArray.prefetch_next_level} kernel loop waited on fields {FIELD_WAIT_STOPWATCH.total_s:6.2f} s",
                flush=True,
            )
            with open(args.results_file, "a") as results_file:
                results_file.write(json.dumps(asdict(result)) + "\n")
