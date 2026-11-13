#!/bin/bash
# Usage: sbatch gpu_benchmarks/bench_oram_gpu.sbatch --case rectilinear --npart 100000 1000000 10000000 --nsteps 10
# All arguments are passed through to run_benchmark.py (--device gpu is preset; a later --device overrides it).
#SBATCH --job-name=parcels-gpu-bench
#SBATCH --partition=gpu
#SBATCH --nodelist=oram
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/beegfs/wyatt/Parcels-gpu/gpu_benchmarks/logs/%x-%j.out

set -euo pipefail
cd /mnt/beegfs/wyatt/Parcels-gpu
export PATH="$HOME/.pixi/bin:$PATH"
export PIXI_CACHE_DIR=/tmp/pixi-cache-$USER
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PARCELS=0.0.0+gpu
export OMP_NUM_THREADS=1

echo "host: $(hostname)  date: $(date -Is)  args: $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
pixi run --as-is -e gpu-cuda12 python gpu_benchmarks/run_benchmark.py --device gpu "$@"
