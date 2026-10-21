#!/usr/bin/env bash
# Submit from the verification bundle/repository root after creating runs/.
#SBATCH --job-name=ader-verify
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:nv:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=runs/ader-gpu-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the verification root}"
ader_python=local/ader-verify-env/bin/python
hostname
nvidia-smi
printf 'Slurm CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"

# Make the CUDA libraries installed by TensorFlow's extra visible to its loader.
ader_cuda_libs="$($ader_python -c 'import glob, sysconfig; print(":".join(sorted(glob.glob(sysconfig.get_paths()["purelib"] + "/nvidia/*/lib"))))')"
export LD_LIBRARY_PATH="${ader_cuda_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1
"$ader_python" scripts/verify_ader.py \
  --source local/sources/ADER-348fc21 \
  --output "runs/ader-gpu-${SLURM_JOB_ID}" --device gpu
