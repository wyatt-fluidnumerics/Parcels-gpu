#!/bin/bash
# Single-core CPU baseline. Usage: sbatch gpu_benchmarks/bench_franklin_cpu.sbatch --case rectilinear --npart 10000 100000
# All arguments are passed through to run_benchmark.py (--device cpu is preset).
# Memory: the CPU time window keeps 2 time levels of U and V resident on the host, which at 1/12 degree
# and nz=50 is 4 x 1.9 GB (synthetic) or 4 x 2.6 GB (MOI), plus a transient copy while a level is
# concatenated into the window, hence 32G rather than 16G.
#SBATCH --job-name=parcels-cpu-bench
#SBATCH --partition=cpu
#SBATCH --nodelist=franklin
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/beegfs/wyatt/Parcels-gpu/gpu_benchmarks/logs/%x-%j.out

set -euo pipefail
cd /mnt/beegfs/wyatt/Parcels-gpu
export PATH="$HOME/.pixi/bin:$PATH"
export PIXI_CACHE_DIR=/tmp/pixi-cache-$USER
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PARCELS=0.0.0+gpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "host: $(hostname)  date: $(date -Is)  args: $*"
pixi run --as-is -e test python gpu_benchmarks/run_benchmark.py --device cpu "$@"
