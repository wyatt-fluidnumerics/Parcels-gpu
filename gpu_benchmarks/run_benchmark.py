"""Benchmark Parcels advection throughput (particle-steps per second) on large domains, CPU vs GPU.

Usage (from the repository root):
    python gpu_benchmarks/run_benchmark.py --case rectilinear --device cpu --npart 10000 100000 --nsteps 10

Per invocation the fieldset is built once and warmed up once with a tiny particle set (field time-window
loading, spatial-hash construction, kernel compilation). Then, for every requested particle count, a fresh
ParticleSet runs one untimed warm-up step followed by ``--nsteps`` timed steps. One JSON line per particle
count is appended to ``--results-file``.
"""

from __future__ import annotations

import argparse
import cProfile
import dataclasses
import datetime
import hashlib
import json
import resource
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import parcels
from parcels._core.utils.time import timedelta_to_float

BENCHMARK_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_DIRECTORY.parent
sys.path.insert(0, str(BENCHMARK_DIRECTORY))

import moi_fields
import synthetic_fields
from particle_seeds import ParticleSeeds

KERNEL_CHOICES = ["AdvectionRK2", "AdvectionRK4", "AdvectionEE", "AdvectionRK2_3D", "AdvectionRK4_3D"]
FIELD_WARMUP_NPART = 100


# ------------------------------------------------------------------------------------------------
# Result record
# ------------------------------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    case: str
    device: str
    kernel: str
    npart_requested: int
    npart_at_timed_start: int
    npart_at_timed_end: int
    nsteps: int
    dt_s: float
    field_shape_tzyx: list[int]
    resolution_deg: float | None
    fieldset_setup_s: float
    field_warmup_s: float
    warmup_step_s: float
    timed_s: float
    s_per_step: float
    particle_steps_per_s: float
    output_enabled: bool
    outputdt_s: float | None
    output_bytes: int | None
    field_time_levels_loaded_in_timed_run: int | None
    field_bytes_loaded_in_timed_run: int | None
    device_memory_pool_bytes: int | None
    device_memory_used_bytes: int | None
    host_peak_rss_bytes: int
    particle_array_module: str
    hostname: str
    git_branch: str
    git_commit: str
    source_tree_sha1: str
    timestamp_utc: str


# ------------------------------------------------------------------------------------------------
# Provenance helpers (no git binary on the compute nodes, so .git is read directly)
# ------------------------------------------------------------------------------------------------
def _git_branch_and_commit(repository_root: Path) -> tuple[str, str]:
    git_directory = repository_root / ".git"
    head = (git_directory / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return "detached", head
    ref = head.removeprefix("ref: ")
    branch = ref.removeprefix("refs/heads/")
    ref_file = git_directory / ref
    if ref_file.exists():
        return branch, ref_file.read_text().strip()
    packed_refs = git_directory / "packed-refs"
    if packed_refs.exists():
        for line in packed_refs.read_text().splitlines():
            if line.endswith(f" {ref}"):
                return branch, line.split()[0]
    return branch, "unknown"


def _source_tree_sha1(source_directory: Path) -> str:
    """Fingerprint of the parcels sources, so runs on an uncommitted working tree stay distinguishable."""
    digest = hashlib.sha1()
    for path in sorted(source_directory.rglob("*.py")):
        digest.update(str(path.relative_to(source_directory)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


# ------------------------------------------------------------------------------------------------
# Device helpers
# ------------------------------------------------------------------------------------------------
def _synchronize(device: str) -> None:
    if device == "gpu":
        import cupy

        cupy.cuda.runtime.deviceSynchronize()


def _device_memory_bytes(device: str) -> tuple[int | None, int | None]:
    """(CuPy memory-pool size, device memory in use). The pool keeps freed blocks, so its size tracks the peak."""
    if device != "gpu":
        return None, None
    import cupy

    free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    return int(cupy.get_default_memory_pool().total_bytes()), int(total_bytes - free_bytes)


def _place_fieldset_on_device(fieldset: parcels.FieldSet, device: str) -> parcels.FieldSet:
    if device == "cpu":
        return fieldset.to_windowed_arrays()
    try:
        import cupy
    except ImportError as e:
        raise RuntimeError("--device gpu needs CuPy; run in the gpu-cuda12 pixi environment.") from e
    if not hasattr(fieldset, "to_gpu_arrays"):
        raise NotImplementedError("FieldSet.to_gpu_arrays() is not implemented in this parcels checkout yet.")
    return fieldset.to_gpu_arrays()


def _windowed_array_load_counters(fieldset: parcels.FieldSet) -> tuple[int | None, int | None]:
    """Total (time levels loaded, bytes read) over all time-windowed field caches, if any."""
    field_arrays: list[object] = []
    for model in fieldset.models:
        field_arrays += list(model.__dict__.get("_windowed", {}).values())
        for field_name in model.scalar_field_names:
            field_arrays.append(model.data.variables[field_name]._data)

    time_levels_loaded = 0
    bytes_loaded = 0
    found_any_window = False
    for field_array in field_arrays:
        if hasattr(field_array, "loads"):
            found_any_window = True
            time_levels_loaded += int(field_array.loads)
            bytes_loaded += int(field_array.bytes_read)
    if not found_any_window:
        return None, None
    return time_levels_loaded, bytes_loaded


def _counter_difference(after: int | None, before: int | None) -> int | None:
    if after is None or before is None:
        return None
    return after - before


def _array_module_name(array: object) -> str:
    return type(array).__module__.split(".")[0]


# ------------------------------------------------------------------------------------------------
# Kernels
# ------------------------------------------------------------------------------------------------
def DeleteErroredParticles(particles, fieldset):
    """Delete particles that errored (ran aground, left the grid) so large runs never abort."""
    errored = particles.state >= parcels.StatusCode.Error
    particles[errored].state = parcels.StatusCode.Delete


# ------------------------------------------------------------------------------------------------
# Cases
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    fieldset: parcels.FieldSet
    resolution_deg: float | None
    has_vertical_velocity: bool
    seed_particles: Callable[[int, np.random.Generator], ParticleSeeds]


def _build_case(args: argparse.Namespace) -> BenchmarkCase:
    if args.case in ("rectilinear", "curvilinear"):
        spec = synthetic_fields.SyntheticGridSpec(
            resolution_deg=args.resolution_deg, nz=args.nz, n_time_levels=args.n_time_levels
        )
        make_dataset = (
            synthetic_fields.rectilinear_global_dataset
            if args.case == "rectilinear"
            else synthetic_fields.curvilinear_global_dataset
        )
        fieldset = parcels.FieldSet.from_sgrid_conventions(
            make_dataset(spec), mesh=parcels.SphericalMesh(), skip_field_data_validation=True
        )

        def seed(npart: int, rng: np.random.Generator) -> ParticleSeeds:
            return synthetic_fields.seed_particles_uniformly(spec, npart, rng)

        return BenchmarkCase(args.case, fieldset, args.resolution_deg, False, seed)

    fieldset = parcels.FieldSet.from_sgrid_conventions(
        moi_fields.moi_sgrid_dataset(include_w=args.moi_include_w, n_depth_levels=args.moi_nz)
    )
    ocean_seeds = moi_fields.seed_particles_in_ocean(
        max(args.npart + [FIELD_WARMUP_NPART]), np.random.default_rng(args.seed), n_depth_levels=args.moi_nz
    )

    def seed_from_ocean_pool(npart: int, rng: np.random.Generator) -> ParticleSeeds:
        chosen = rng.permutation(ocean_seeds.lon_deg.size)[:npart]
        return ParticleSeeds(ocean_seeds.lon_deg[chosen], ocean_seeds.lat_deg[chosen], ocean_seeds.depth_m[chosen])

    return BenchmarkCase("moi", fieldset, None, args.moi_include_w, seed_from_ocean_pool)


def _make_particleset(case: BenchmarkCase, npart: int, rng: np.random.Generator) -> parcels.ParticleSet:
    seeds = case.seed_particles(npart, rng)
    return parcels.ParticleSet(case.fieldset, x=seeds.lon_deg, y=seeds.lat_deg, z=seeds.depth_m)


# ------------------------------------------------------------------------------------------------
# Timed runs
# ------------------------------------------------------------------------------------------------
def _time_execute(
    pset: parcels.ParticleSet,
    kernels: list,
    dt: np.timedelta64,
    nsteps: int,
    device: str,
    output_file: parcels.ParticleFile | None = None,
    profiler: cProfile.Profile | None = None,
) -> float:
    _synchronize(device)
    start_s = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    pset.execute(kernels, runtime=dt * nsteps, dt=dt, output_file=output_file, verbose_progress=False)
    _synchronize(device)
    if profiler is not None:
        profiler.disable()
    return time.perf_counter() - start_s


def _assert_runtime_fits_in_field_time_span(fieldset: parcels.FieldSet, dt: np.timedelta64, total_steps: int) -> None:
    time_interval = fieldset.time_interval
    if time_interval is None:
        return
    available_s = timedelta_to_float(time_interval.right - time_interval.left)
    requested_s = timedelta_to_float(dt) * total_steps
    if requested_s > available_s:
        raise ValueError(
            f"{total_steps} steps of {timedelta_to_float(dt)} s ({requested_s} s, including the warm-up step) exceed "
            f"the field time span of {available_s} s. Reduce --nsteps/--dt-minutes or raise --n-time-levels."
        )


def run(args: argparse.Namespace) -> list[BenchmarkResult]:
    kernel = getattr(parcels.kernels, args.kernel)
    dt = np.timedelta64(int(round(args.dt_minutes * 60)), "s")
    git_branch, git_commit = _git_branch_and_commit(REPOSITORY_ROOT)
    source_tree_sha1 = _source_tree_sha1(REPOSITORY_ROOT / "src" / "parcels")

    setup_start_s = time.perf_counter()
    case = _build_case(args)
    if kernel.__name__.endswith("_3D") and not case.has_vertical_velocity:
        raise ValueError(f"{args.kernel} needs a W field; only --case moi with --moi-include-w provides one.")
    case = dataclasses.replace(case, fieldset=_place_fieldset_on_device(case.fieldset, args.device))
    fieldset_setup_s = time.perf_counter() - setup_start_s
    _assert_runtime_fits_in_field_time_span(case.fieldset, dt, args.nsteps + 1)

    kernels = [kernel, DeleteErroredParticles]
    rng = np.random.default_rng(args.seed)
    field_warmup_s = _time_execute(_make_particleset(case, FIELD_WARMUP_NPART, rng), kernels, dt, 1, args.device)
    field_shape_tzyx = [int(n) for n in case.fieldset.U.data.shape]
    print(
        f"[setup] case={case.name} device={args.device} field_shape_tzyx={field_shape_tzyx} "
        f"fieldset_setup_s={fieldset_setup_s:.2f} field_warmup_s={field_warmup_s:.2f}",
        flush=True,
    )

    results = []
    for npart in args.npart:
        pset = _make_particleset(case, npart, rng)
        warmup_step_s = _time_execute(pset, kernels, dt, 1, args.device)
        npart_at_timed_start = len(pset)

        output_file = None
        output_path = None
        outputdt_s = None
        if args.output:
            outputdt_s = 60.0 * (args.outputdt_minutes if args.outputdt_minutes is not None else args.dt_minutes)
            output_path = args.output_dir / (
                f"{case.name}_{args.device}_{npart}_{datetime.datetime.now(datetime.UTC):%Y%m%dT%H%M%S%f}.parquet"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = parcels.ParticleFile(output_path, outputdt=np.timedelta64(int(outputdt_s), "s"))

        profiler = cProfile.Profile() if args.cprofile_dir is not None else None
        levels_loaded_before_timed_run, bytes_loaded_before_timed_run = _windowed_array_load_counters(case.fieldset)
        timed_s = _time_execute(pset, kernels, dt, args.nsteps, args.device, output_file, profiler)
        levels_loaded_after_timed_run, bytes_loaded_after_timed_run = _windowed_array_load_counters(case.fieldset)

        output_bytes = None
        if output_path is not None and output_path.exists():
            output_bytes = output_path.stat().st_size
            if not args.keep_output:
                output_path.unlink()
        if profiler is not None:
            args.cprofile_dir.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(args.cprofile_dir / f"{case.name}_{args.device}_{args.kernel}_{npart}.prof")

        device_memory_pool_bytes, device_memory_used_bytes = _device_memory_bytes(args.device)
        result = BenchmarkResult(
            case=case.name,
            device=args.device,
            kernel=args.kernel,
            npart_requested=npart,
            npart_at_timed_start=npart_at_timed_start,
            npart_at_timed_end=len(pset),
            nsteps=args.nsteps,
            dt_s=timedelta_to_float(dt),
            field_shape_tzyx=field_shape_tzyx,
            resolution_deg=case.resolution_deg,
            fieldset_setup_s=fieldset_setup_s,
            field_warmup_s=field_warmup_s,
            warmup_step_s=warmup_step_s,
            timed_s=timed_s,
            s_per_step=timed_s / args.nsteps,
            particle_steps_per_s=npart_at_timed_start * args.nsteps / timed_s,
            output_enabled=args.output,
            outputdt_s=outputdt_s,
            output_bytes=output_bytes,
            field_time_levels_loaded_in_timed_run=_counter_difference(
                levels_loaded_after_timed_run, levels_loaded_before_timed_run
            ),
            field_bytes_loaded_in_timed_run=_counter_difference(
                bytes_loaded_after_timed_run, bytes_loaded_before_timed_run
            ),
            device_memory_pool_bytes=device_memory_pool_bytes,
            device_memory_used_bytes=device_memory_used_bytes,
            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            particle_array_module=_array_module_name(pset._data["x"]),
            hostname=socket.gethostname(),
            git_branch=git_branch,
            git_commit=git_commit,
            source_tree_sha1=source_tree_sha1,
            timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        )
        print(
            f"[result] npart={npart:>10d} survivors={result.npart_at_timed_end:>10d} "
            f"s/step={result.s_per_step:9.4f} particle-steps/s={result.particle_steps_per_s:12.4g} "
            f"warmup_step_s={warmup_step_s:7.3f} arrays={result.particle_array_module}",
            flush=True,
        )
        args.results_file.parent.mkdir(parents=True, exist_ok=True)
        with args.results_file.open("a") as results_stream:
            results_stream.write(json.dumps(dataclasses.asdict(result)) + "\n")
        results.append(result)
        del pset
    return results


# ------------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------------
def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=["rectilinear", "curvilinear", "moi"], default="rectilinear")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--npart", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--nsteps", type=int, default=10)
    parser.add_argument("--dt-minutes", type=float, default=15.0)
    parser.add_argument("--kernel", choices=KERNEL_CHOICES, default="AdvectionRK2")
    parser.add_argument("--resolution-deg", type=float, default=1.0 / 12.0, help="synthetic cases only")
    parser.add_argument("--nz", type=int, default=50, help="synthetic cases only")
    parser.add_argument("--n-time-levels", type=int, default=2, help="synthetic cases only; levels are 24 h apart")
    parser.add_argument("--moi-include-w", action="store_true", help="also load W (needed for *_3D kernels)")
    parser.add_argument("--moi-nz", type=int, default=None, help="keep only the top N MOI depth levels")
    parser.add_argument("--output", action="store_true", help="write a ParticleFile during the timed run")
    parser.add_argument("--outputdt-minutes", type=float, default=None, help="defaults to --dt-minutes")
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_DIRECTORY / "logs" / "particlefiles")
    parser.add_argument("--keep-output", action="store_true", help="keep the ParticleFile instead of deleting it")
    parser.add_argument("--results-file", type=Path, default=BENCHMARK_DIRECTORY / "results" / "results.jsonl")
    parser.add_argument("--cprofile-dir", type=Path, default=None, help="dump cProfile stats of each timed run here")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
