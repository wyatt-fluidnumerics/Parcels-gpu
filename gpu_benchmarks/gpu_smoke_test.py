"""Smoke test of the CuPy stack on a GPU node, plus a random-gather bandwidth ceiling estimate.

The gather test mimics the trilinear corner gather that dominates Parcels interpolation: random
(t, z, y, x) cell lookups into a global-1/12-degree-sized float32 array.
"""

import time

import cupy as cp
import numpy as np


# ------------------------------------------------------------------------------------------------
# Device and toolchain report
# ------------------------------------------------------------------------------------------------
def report_device() -> None:
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    print(f"device name            : {props['name'].decode()}")
    print(f"compute capability     : {device.compute_capability}")
    print(f"device memory total GB : {total_bytes / 1e9:.2f}")
    print(f"device memory free GB  : {free_bytes / 1e9:.2f}")
    print(f"multiprocessors        : {props['multiProcessorCount']}")
    print(f"driver CUDA version    : {cp.cuda.runtime.driverGetVersion()}")
    print(f"runtime CUDA version   : {cp.cuda.runtime.runtimeGetVersion()}")
    print(f"cupy version           : {cp.__version__}")


def check_raw_kernel_compiles() -> None:
    add_one = cp.RawKernel(
        r"""
        extern "C" __global__ void add_one(float* values, int n) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < n) values[i] += 1.0f;
        }
        """,
        "add_one",
    )
    values = cp.zeros(1000, dtype=cp.float32)
    add_one((4,), (256,), (values, cp.int32(values.size)))
    assert float(values.sum()) == 1000.0
    print("RawKernel (NVRTC) compile + launch : OK")


def check_numpy_dispatch() -> None:
    values = cp.linspace(0, 1, 10)
    dispatched = np.where(values > 0.5, np.sqrt(values), 0.0)
    assert isinstance(dispatched, cp.ndarray)
    print("np.where/np.sqrt on cupy arrays dispatch to cupy : OK")


# ------------------------------------------------------------------------------------------------
# Random-gather ceiling at global 1/12 degree size
# ------------------------------------------------------------------------------------------------
def time_random_corner_gather(n_particles: int, nx: int = 4320, ny: int = 2160, nz: int = 50) -> None:
    field_bytes = 2 * nz * ny * nx * 4
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    if field_bytes * 1.5 > free_bytes:
        nz = max(1, int(free_bytes / 1.5 / (2 * ny * nx * 4)))
        print(f"reducing nz to {nz} to fit device memory")
    field_tzyx = cp.random.random((2, nz, ny, nx), dtype=cp.float32)

    rng = cp.random.default_rng(0)
    xi = rng.integers(0, nx - 1, n_particles, dtype=cp.int64)
    yi = rng.integers(0, ny - 1, n_particles, dtype=cp.int64)
    zi = rng.integers(0, nz - 1, n_particles, dtype=cp.int64)
    flat_field = field_tzyx.ravel()
    strides = (nz * ny * nx, ny * nx, nx, 1)

    def gather_16_corners() -> cp.ndarray:
        base_index = zi * strides[1] + yi * strides[2] + xi
        total = cp.zeros(n_particles, dtype=cp.float32)
        for ti in (0, 1):
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        offset = ti * strides[0] + dz * strides[1] + dy * strides[2] + dx
                        total += flat_field[base_index + offset]
        return total

    gather_16_corners()
    cp.cuda.Stream.null.synchronize()
    n_repeats = 5
    start = time.perf_counter()
    for _ in range(n_repeats):
        gather_16_corners()
    cp.cuda.Stream.null.synchronize()
    seconds_per_gather = (time.perf_counter() - start) / n_repeats
    print(
        f"16-corner gather (unfused cupy), npart={n_particles:>10d}: "
        f"{1e3 * seconds_per_gather:8.3f} ms  -> {n_particles / seconds_per_gather / 1e6:9.1f} M particle-evals/s"
    )


if __name__ == "__main__":
    report_device()
    check_raw_kernel_compiles()
    check_numpy_dispatch()
    for n_particles in (10**5, 10**6, 10**7):
        time_random_corner_gather(n_particles)
        cp.get_default_memory_pool().free_all_blocks()
