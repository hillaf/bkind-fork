#!/usr/bin/env bash
#SBATCH --account=project_2015338
#SBATCH --job-name=bkind-infer
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:00:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/roihu_scripts/environment.sh"
: "${IMAGES:?Set IMAGES to the image directory}"
: "${CHECKPOINT:?Set CHECKPOINT to the trained checkpoint}"
srun python "$SCRIPT_DIR/run.py" infer --device cuda --images "$IMAGES" --checkpoint "$CHECKPOINT" \
  --output "${OUTPUT:-$ROOT/outputs/infer-$SLURM_JOB_ID}" "$@"
