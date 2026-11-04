#!/usr/bin/env bash
#SBATCH --account=project_2015338
#SBATCH --job-name=bkind
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=12:00:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/roihu_scripts/environment.sh"
: "${DATA:?Set DATA to an absolute root with train/video and val/video frame directories}"
OUTPUT="${OUTPUT:-$ROOT/outputs/train-$SLURM_JOB_ID}"
mkdir -p "$OUTPUT"
module list 2> "$OUTPUT/modules.txt"
python -m pip freeze > "$OUTPUT/packages.txt"
srun python "$SCRIPT_DIR/run.py" check --device cuda
srun python "$SCRIPT_DIR/run.py" train --device cuda --config "${CONFIG:-$SCRIPT_DIR/custom.yaml}" \
  --data "$DATA" --output "$OUTPUT" --epochs "${EPOCHS:-30}" --batch-size "${BATCH_SIZE:-4}" \
  --workers "${WORKERS:-4}" --threads "$OMP_NUM_THREADS" "$@"
