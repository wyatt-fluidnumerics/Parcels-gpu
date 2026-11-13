#!/bin/bash
#SBATCH --job-name=parcels-gpu-smoke
#SBATCH --partition=gpu
#SBATCH --nodelist=oram
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/beegfs/wyatt/Parcels-gpu/gpu_benchmarks/logs/%x-%j.out

set -euo pipefail
cd /mnt/beegfs/wyatt/Parcels-gpu
export PATH="$HOME/.pixi/bin:$PATH"
export PIXI_CACHE_DIR=/tmp/pixi-cache-$USER
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PARCELS=0.0.0+gpu

echo "host: $(hostname)  date: $(date -Is)"
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.memory --format=csv
pixi run --as-is -e gpu-cuda12 python gpu_benchmarks/gpu_smoke_test.py
