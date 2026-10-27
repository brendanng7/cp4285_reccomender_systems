#!/usr/bin/env bash
# Submit only after all three training-only seed jobs complete successfully.
#SBATCH --job-name=amazon-ader-validation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/amazon-ader-validation-%j.log

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
ader_python=local/ader-verify-env/bin/python
ader_cuda_libs="$($ader_python -c 'import glob, sysconfig; print(":".join(sorted(glob.glob(sysconfig.get_paths()["purelib"] + "/nvidia/*/lib"))))')"
export LD_LIBRARY_PATH="${ader_cuda_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
hostname
nvidia-smi
"$HOME/.local/bin/uv" run --no-project --python "$ader_python" --with duckdb==2.0.0 \
  python scripts/evaluate_ader_categories_validation.py \
  --sequences runs/category-sequences-1010635 \
  --periods runs/category-periods-v3-1012758 \
  --source local/sources/ADER-348fc21 \
  --run-dir runs/amazon-ader-categories-seed0-1013143 \
  --run-dir runs/amazon-ader-categories-seed1-1013599 \
  --run-dir runs/amazon-ader-categories-seed2-1013600 \
  --output "runs/amazon-ader-validation-${SLURM_JOB_ID}"
