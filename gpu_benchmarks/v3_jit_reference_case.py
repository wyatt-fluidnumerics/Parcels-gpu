"""The case behind the published v3-JIT timing, run with this fork on the GPU.

Reference: Parcels-code/parcels-benchmarks PR 4 ("Benchmark vectorized kernels"), where v3 JIT is
reported to take 8 s for 1 million particles. Its script (benchmark_vectorized_kernels.py) runs
AdvectionEE on `radial_rotation_dataset()` (200 x 200, flat mesh, 2D) for 3 hours at dt = 30 s,
i.e. 360 steps, with particles on lon = linspace(32, 50, npart), lat = 30.

Usage: python gpu_benchmarks/v3_jit_reference_case.py --npart 1000000
"""

import argparse
import time

import cupy
import numpy as np

import parcels
from parcels._core.mesh import FlatMesh
from parcels._datasets.structured.generated import radial_rotation_dataset

RUNTIME = np.timedelta64(3, "h")
DT = np.timedelta64(30, "s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npart", type=int, nargs="+", default=[1_000_000])
    args = parser.parse_args()

    fieldset = parcels.FieldSet.from_sgrid_conventions(radial_rotation_dataset(), mesh=FlatMesh()).to_gpu_arrays()
    warmup = parcels.ParticleSet(fieldset, x=np.linspace(32, 50, 100), y=np.full(100, 30.0))
    warmup.execute(parcels.kernels.AdvectionEE, runtime=DT, dt=DT, verbose_progress=False)

    for npart in args.npart:
        pset = parcels.ParticleSet(fieldset, x=np.linspace(32, 50, npart), y=np.full(npart, 30.0))
        cupy.cuda.runtime.deviceSynchronize()
        start = time.perf_counter()
        pset.execute(parcels.kernels.AdvectionEE, runtime=RUNTIME, dt=DT, verbose_progress=False)
        cupy.cuda.runtime.deviceSynchronize()
        wall_s = time.perf_counter() - start
        nsteps = int(RUNTIME / DT)
        print(
            f"[gpu] npart={npart:>9d} AdvectionEE {nsteps} steps: wall={wall_s:7.2f} s "
            f"({1e3 * wall_s / nsteps:6.2f} ms/step, {npart * nsteps / wall_s:.3e} particle-steps/s)",
            flush=True,
        )
