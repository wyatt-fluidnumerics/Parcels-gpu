#!/bin/bash
#SBATCH --job-name=parcels-gpu-pytest
#SBATCH --partition=gpu
#SBATCH --nodelist=oram
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/beegfs/wyatt/Parcels-gpu/gpu_benchmarks/logs/%x-%j.out

# Runs pytest on a V100. Arguments are passed through to pytest, e.g.
#   sbatch gpu_benchmarks/pytest_oram_gpu.sbatch tests/test_gpu_array.py -x

set -euo pipefail
cd /mnt/beegfs/wyatt/Parcels-gpu
export PATH="$HOME/.pixi/bin:$PATH"
export PIXI_CACHE_DIR=/tmp/pixi-cache-$USER
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PARCELS=0.0.0+gpu

echo "host: $(hostname)  date: $(date -Is)"
pixi run --as-is -e gpu-cuda12 pytest -p no:cacheprovider "$@" 2>&1 | grep -v "^ WARN\|│\|╭\|╰\|⚠\|·\|help: e.g.\|= \"12\" }"
