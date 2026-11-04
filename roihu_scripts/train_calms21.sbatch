#!/usr/bin/env bash
#SBATCH --account=project_2015338
#SBATCH --job-name=bkind-calms21
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=36:00:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/roihu_scripts/environment.sh"
DATA="${DATA:-/scratch/project_2015338/DATA/calms21-bkind-paper}"
OUTPUT="${OUTPUT:-$ROOT/outputs/calms21-paper-$SLURM_JOB_ID}"
mkdir -p "$OUTPUT"
module list 2> "$OUTPUT/modules.txt"
python -m pip freeze > "$OUTPUT/packages.txt"
srun python "$SCRIPT_DIR/run.py" check --device cuda
srun python "$SCRIPT_DIR/run.py" train --device cuda --config "$SCRIPT_DIR/calms21_paper.yaml" \
  --data "$DATA" --output "$OUTPUT" --threads 4 "$@"
