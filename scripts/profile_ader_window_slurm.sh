#!/usr/bin/env bash
# Submit from the existing cluster working directory; create runs/ first.
#SBATCH --job-name=ader-window0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=runs/ader-window0-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the verification root}"
export TMPDIR="$HOME/.cache/ader-tmp"
mkdir -p "$TMPDIR"
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=1
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
ader_python=local/ader-verify-env/bin/python
ader_cuda_libs="$($ader_python -c 'import glob, sysconfig; print(":".join(sorted(glob.glob(sysconfig.get_paths()["purelib"] + "/nvidia/*/lib"))))')"
export LD_LIBRARY_PATH="${ader_cuda_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
hostname
nvidia-smi
printf 'Slurm CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
"$ader_python" scripts/profile_ader_window.py \
  --source local/sources/ADER-348fc21 \
  --data local/ader-window-inputs \
  --output "runs/ader-window0-${SLURM_JOB_ID}"
