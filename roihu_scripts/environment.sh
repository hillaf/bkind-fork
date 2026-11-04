#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${BKIND_MODULE:=python-pytorch/2.13}"
: "${BKIND_VENV:=/projappl/project_2015338/bkind-venv-$(uname -m)}"
if [[ "$BKIND_VENV" == *"/PROJECT"* ]]; then
  echo "BKIND_VENV still contains PROJECT. Run: export BKIND_VENV=/projappl/project_2015338/bkind-venv-aarch64" >&2
  exit 2
fi
if [[ ! -f "$BKIND_VENV/bkind-module.txt" ]]; then
  echo "Cannot find setup metadata: $BKIND_VENV/bkind-module.txt" >&2
  if [[ -f "$BKIND_VENV/bin/activate" ]]; then
    echo "The venv exists here, but setup metadata is missing. Check whether setup completed." >&2
  else
    echo "No venv activation script at: $BKIND_VENV/bin/activate" >&2
  fi
  echo "Set BKIND_VENV to the environment used for training and export it before sbatch." >&2
  echo "Earlier setup versions defaulted to: $ROOT/.venv-roihu-$(uname -m)" >&2
  exit 2
fi
[[ "$(cat "$BKIND_VENV/bkind-module.txt")" == "$BKIND_MODULE" ]] || { echo 'Venv module mismatch' >&2; exit 2; }
[[ "$(cat "$BKIND_VENV/bkind-arch.txt")" == "$(uname -m)" ]] || { echo 'Venv architecture mismatch' >&2; exit 2; }
module purge
module load "$BKIND_MODULE"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 CW_FORCE_CONDA_ACTIVATE=1
source "$BKIND_VENV/bin/activate"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.torch-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
