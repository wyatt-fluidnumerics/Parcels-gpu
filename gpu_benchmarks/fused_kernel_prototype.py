"""Upper-bound experiment: one fused CUDA kernel doing a whole RK2 advection step per particle.

Not wired into Parcels. It measures how fast a fully fused step can go on a global 1/12 degree
rectilinear spherical A-grid (binary searches in lon/lat/depth, 16-corner space-time linear
sampling of U and V, RK2 midpoint update), to compare with the array-at-a-time GpuArray path
measured by run_benchmark.py on the same grid size.

Usage: python gpu_benchmarks/fused_kernel_prototype.py --npart 1000000 10000000 --nsteps 20
"""

import argparse
import time

import cupy as cp
import numpy as np

EARTH_DEG2M = 1852.0 * 60.0

_RK2_STEP_SOURCE = r"""
__device__ void search_sorted(const double* coordinate, int n, double value, int* index, double* bcoord) {
    int low = 0;
    int high = n - 1;
    while (high - low > 1) {
        int middle = (low + high) / 2;
        if (coordinate[middle] <= value) low = middle; else high = middle;
    }
    *index = low;
    *bcoord = (value - coordinate[low]) / (coordinate[low + 1] - coordinate[low]);
}

__device__ void sample_uv(const float* u, const float* v,
                          const double* lon, const double* lat, const double* depth,
                          int nx, int ny, int nz, double level_interval_s,
                          double t, double z, double y, double x,
                          double* u_degs, double* v_degs) {
    int xi, yi, zi;
    double xsi, eta, zeta;
    search_sorted(lon, nx, x, &xi, &xsi);
    search_sorted(lat, ny, y, &yi, &eta);
    search_sorted(depth, nz, z, &zi, &zeta);
    double tau = t / level_interval_s;

    double u_ms = 0.0;
    double v_ms = 0.0;
    long long level_size = (long long)nz * ny * nx;
    for (int it = 0; it < 2; ++it) {
        double weight_t = it == 0 ? 1.0 - tau : tau;
        for (int iz = 0; iz < 2; ++iz) {
            double weight_tz = weight_t * (iz == 0 ? 1.0 - zeta : zeta);
            for (int iy = 0; iy < 2; ++iy) {
                double weight_tzy = weight_tz * (iy == 0 ? 1.0 - eta : eta);
                long long row = it * level_size + ((long long)(zi + iz) * ny + (yi + iy)) * nx + xi;
                u_ms += weight_tzy * ((1.0 - xsi) * u[row] + xsi * u[row + 1]);
                v_ms += weight_tzy * ((1.0 - xsi) * v[row] + xsi * v[row + 1]);
            }
        }
    }
    const double deg2m = 1852.0 * 60.0;
    const double deg2rad = 3.14159265358979323846 / 180.0;
    *u_degs = u_ms / (deg2m * cos(deg2rad * y));
    *v_degs = v_ms / deg2m;
}

extern "C" __global__
void rk2_step(const float* u, const float* v,
              const double* lon, const double* lat, const double* depth,
              int nx, int ny, int nz, double level_interval_s, double dt_s,
              float* x, float* y, const float* z, double* t, int npart) {
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= npart) return;
    double xp = x[p], yp = y[p], zp = z[p], tp = t[p];

    double u1, v1, u2, v2;
    sample_uv(u, v, lon, lat, depth, nx, ny, nz, level_interval_s, tp, zp, yp, xp, &u1, &v1);
    double x_mid = xp + u1 * 0.5 * dt_s;
    double y_mid = yp + v1 * 0.5 * dt_s;
    sample_uv(u, v, lon, lat, depth, nx, ny, nz, level_interval_s, tp + 0.5 * dt_s, zp, y_mid, x_mid, &u2, &v2);

    x[p] = (float)(xp + u2 * dt_s);
    y[p] = (float)(yp + v2 * dt_s);
    t[p] = tp + dt_s;
}
"""


# ------------------------------------------------------------------------------------------------
# Grid and velocity fields, created directly on the GPU
# ------------------------------------------------------------------------------------------------
def make_fields(nx: int, ny: int, nz: int) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    lon_deg = cp.linspace(-180.0, 180.0, nx)
    lat_deg = cp.linspace(-80.0, 80.0, ny)
    depth_m = cp.linspace(0.0, 5000.0, nz)
    u_ms = cp.random.random((2, nz, ny, nx), dtype=cp.float32) - cp.float32(0.5)
    v_ms = cp.random.random((2, nz, ny, nx), dtype=cp.float32) - cp.float32(0.5)
    return lon_deg, lat_deg, depth_m, u_ms, v_ms


# ------------------------------------------------------------------------------------------------
# Timing
# ------------------------------------------------------------------------------------------------
def time_fused_rk2(npart: int, nsteps: int, fields: tuple, dt_s: float, level_interval_s: float) -> float:
    lon_deg, lat_deg, depth_m, u_ms, v_ms = fields
    nz, ny, nx = u_ms.shape[1:]
    rng = cp.random.default_rng(0)
    x_deg = rng.uniform(-170.0, 170.0, npart).astype(cp.float32)
    y_deg = rng.uniform(-70.0, 70.0, npart).astype(cp.float32)
    z_m = rng.uniform(10.0, 4900.0, npart).astype(cp.float32)
    t_s = cp.zeros(npart, dtype=cp.float64)

    rk2_step = cp.RawKernel(_RK2_STEP_SOURCE, "rk2_step")
    threads_per_block = 256
    n_blocks = (npart + threads_per_block - 1) // threads_per_block
    arguments = (
        u_ms, v_ms, lon_deg, lat_deg, depth_m,
        np.int32(nx), np.int32(ny), np.int32(nz), np.float64(level_interval_s), np.float64(dt_s),
        x_deg, y_deg, z_m, t_s, np.int32(npart),
    )  # fmt: skip

    rk2_step((n_blocks,), (threads_per_block,), arguments)
    cp.cuda.runtime.deviceSynchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        rk2_step((n_blocks,), (threads_per_block,), arguments)
    cp.cuda.runtime.deviceSynchronize()
    return (time.perf_counter() - start) / nsteps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npart", type=int, nargs="+", default=[10**6, 10**7])
    parser.add_argument("--nsteps", type=int, default=20)
    parser.add_argument("--nx", type=int, default=4320)
    parser.add_argument("--ny", type=int, default=2160)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--dt-minutes", type=float, default=15.0)
    args = parser.parse_args()

    fields = make_fields(args.nx, args.ny, args.nz)
    print(f"grid (t, z, y, x) = (2, {args.nz}, {args.ny}, {args.nx}); device = {cp.cuda.Device().id}")
    for npart in args.npart:
        seconds_per_step = time_fused_rk2(
            npart, args.nsteps, fields, dt_s=args.dt_minutes * 60.0, level_interval_s=86400.0
        )
        print(
            f"[fused RK2] npart={npart:>10d} s/step={seconds_per_step:9.5f} "
            f"particle-steps/s={npart / seconds_per_step:10.3e}"
        )
