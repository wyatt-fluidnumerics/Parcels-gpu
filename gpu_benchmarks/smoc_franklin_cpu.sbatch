#!/bin/bash
# Real-data (SMOC surface drift) comparison of _gather_corners implementations on one CPU core.
# Arguments, if any, are passed to smoc_gather_comparison.py for a single run; without arguments the default set runs.
#SBATCH --job-name=parcels-smoc-cpu
#SBATCH --partition=cpu
#SBATCH --nodelist=franklin
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/beegfs/wyatt/Parcels-gpu/gpu_benchmarks/logs/%x-%j.out

set -euo pipefail
cd /mnt/beegfs/wyatt/Parcels-gpu
export PATH="$HOME/.pixi/bin:$PATH"
export PIXI_CACHE_DIR=/tmp/pixi-cache-$USER
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PARCELS=0.0.0+gpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "host: $(hostname)  date: $(date -Is)"
run() { pixi run --as-is -e test python gpu_benchmarks/smoc_gather_comparison.py --device cpu "$@" 2>&1 | grep "^\[cpu"; }
if [ "$#" -gt 0 ]; then
    run "$@"
else
    run --backend windowed --npart 100000 --runtime-hours 48
    run --backend chunk_cached --npart 100000 --runtime-hours 48
    run --backend windowed --npart 1000000 --runtime-hours 48
fi
