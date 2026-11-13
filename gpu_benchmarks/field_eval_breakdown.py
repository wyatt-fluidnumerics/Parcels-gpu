"""How a GPU RK2 step splits between the fused kernels and the code around them.

Synthetic global rectilinear A-grid (same as run_benchmark.py --case rectilinear). Every stage is
followed by a device synchronize, so the times are real GPU times and not launch times.

Stages, per UV field evaluation:
  search        : time search plus the three 1D grid searches (fused kernels)
  sample        : sampling U and V (fused kernels) and the spherical unit conversion
  full eval     : fieldset.UV[particles], i.e. search + sample + the bookkeeping around them
                  (NaN checks, time-interval check, ei update, state codes, out-of-bounds masking)
and per step:
  RK2 step      : one pset.execute step (two evaluations, the integrator arithmetic, the kernel loop)

Usage: python gpu_benchmarks/field_eval_breakdown.py --npart 1000000 10000000
"""

import argparse
import time
from collections.abc import Callable

import cupy
import numpy as np
import synthetic_fields

import parcels
from parcels._core.index_search import _search_time_index


def seconds_per_call(function: Callable[[], object], repeats: int) -> float:
    function()
    cupy.cuda.runtime.deviceSynchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    cupy.cuda.runtime.deviceSynchronize()
    return (time.perf_counter() - start) / repeats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npart", type=int, nargs="+", default=[1_000_000, 10_000_000])
    parser.add_argument("--resolution-deg", type=float, default=1 / 12)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    spec = synthetic_fields.SyntheticGridSpec(resolution_deg=args.resolution_deg, nz=args.nz, n_time_levels=2)
    fieldset = parcels.FieldSet.from_sgrid_conventions(
        synthetic_fields.rectilinear_global_dataset(spec), mesh=parcels.SphericalMesh(), skip_field_data_validation=True
    ).to_gpu_arrays()
    dt = np.timedelta64(15, "m")

    for npart in args.npart:
        seeds = synthetic_fields.seed_particles_uniformly(spec, npart, np.random.default_rng(0))
        pset = parcels.ParticleSet(fieldset, x=seeds.lon_deg, y=seeds.lat_deg, z=seeds.depth_m)
        pset.execute(parcels.kernels.AdvectionRK2, runtime=dt, dt=dt, verbose_progress=False)

        t, z, y, x = pset.t, pset.z, pset.y, pset.x
        grid = fieldset.U.grid
        grid_positions = dict(_search_time_index(fieldset.U, t))
        grid_positions.update(grid.search(z, y, x))
        particle_positions = {"t": t, "z": z, "y": y, "x": x}

        def search():
            _search_time_index(fieldset.U, t)
            grid.search(z, y, x)

        def sample():
            fieldset.UV.interp_method.interp(particle_positions, grid_positions, fieldset.UV)

        def full_eval():
            fieldset.UV[pset]

        def rk2_step():
            pset.execute(parcels.kernels.AdvectionRK2, runtime=dt, dt=dt, verbose_progress=False)

        search_s = seconds_per_call(search, args.repeats)
        sample_s = seconds_per_call(sample, args.repeats)
        full_eval_s = seconds_per_call(full_eval, args.repeats)
        rk2_step_s = seconds_per_call(rk2_step, args.repeats)
        bookkeeping_s = full_eval_s - search_s - sample_s
        outside_evals_s = rk2_step_s - 2 * full_eval_s
        print(
            f"npart={npart:>9d}  per evaluation [ms]: search {1e3 * search_s:7.2f}  sample {1e3 * sample_s:7.2f}  "
            f"bookkeeping {1e3 * bookkeeping_s:7.2f}  (full eval {1e3 * full_eval_s:7.2f})   "
            f"per RK2 step [ms]: total {1e3 * rk2_step_s:7.2f}  outside the two evaluations {1e3 * outside_evals_s:7.2f}",
            flush=True,
        )
