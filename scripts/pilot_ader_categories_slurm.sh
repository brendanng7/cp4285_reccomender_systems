#!/usr/bin/env bash
# Submit from the SoC project root after copying this script and pilot_ader_categories.py.
#SBATCH --job-name=ader-category-pilot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=runs/ader-category-pilot-%j.log

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
gpu_log="runs/ader-category-gpu-${SLURM_JOB_ID}.csv"
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
  python scripts/pilot_ader_categories.py \
  --periods runs/category-periods-v3-1012758 \
  --events runs/category-sequences-1010635/train_events.parquet \
  --source local/sources/ADER-348fc21 \
  --output "runs/ader-category-pilot-${SLURM_JOB_ID}"
