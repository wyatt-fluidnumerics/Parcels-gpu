#!/bin/bash
# Run a command on one V100 on oram and wait for it, e.g.
#   gpu_benchmarks/srun_oram_gpu.sh pytest tests/test_gpu_array.py -x -q
# The command runs inside the gpu-cuda12 pixi environment from the repository root.
set -euo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.pixi/bin:$PATH"
export PIXI_CACHE_DIR=/tmp/pixi-cache-$USER
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PARCELS=0.0.0+gpu
cd "$repository_root"
srun --partition=gpu --nodelist=oram --gres=gpu:v100:1 --cpus-per-task="${GPU_JOB_CPUS:-2}" \
    --mem="${GPU_JOB_MEM:-6G}" --time="${GPU_JOB_TIME:-00:20:00}" --job-name=parcels-gpu-dev \
    pixi run --as-is -e gpu-cuda12 "$@" 2>&1 | grep -v "^ WARN\|│\|╭\|╰\|⚠\|·\|help: e.g.\|= \"12\" }"
