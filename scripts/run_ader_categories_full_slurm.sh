#!/usr/bin/env bash
# Submit from the existing SoC project root after syncing this runner.
#SBATCH --job-name=amazon-ader
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=runs/amazon-ader-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the SoC project root}"
mkdir -p runs
export TMPDIR="$HOME/.cache/ader-tmp"
mkdir -p "$TMPDIR"
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=1
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export TF_DETERMINISTIC_OPS=0
export ADER_SEED="${ADER_SEED:-0}"
gpu_log="runs/amazon-ader-gpu-${SLURM_JOB_ID}.csv"
export ADER_GPU_MONITOR_FILE="$gpu_log"
(
  while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$gpu_log"
    sleep 2
  done
) &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT
ader_python=local/ader-verify-env/bin/python
ader_cuda_libs="$($ader_python -c 'import glob, sysconfig; print(":".join(sorted(glob.glob(sysconfig.get_paths()["purelib"] + "/nvidia/*/lib"))))')"
export LD_LIBRARY_PATH="${ader_cuda_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
hostname
nvidia-smi
"$HOME/.local/bin/uv" run --no-project --python "$ader_python" --with duckdb==2.0.0 \
  python scripts/run_ader_categories_full.py \
  --periods runs/category-periods-v3-1012758 \
  --events runs/category-sequences-1010635/train_events.parquet \
  --source local/sources/ADER-348fc21 \
  --seed "$ADER_SEED" \
  --output "runs/amazon-ader-categories-seed${ADER_SEED}-${SLURM_JOB_ID}"
