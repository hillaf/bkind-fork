#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${BKIND_MODULE:=python-pytorch/2.13}"
: "${BKIND_VENV:=/projappl/project_2015338/bkind-venv-$(uname -m)}"
if [[ "$BKIND_VENV" == *"/PROJECT"* ]]; then
  echo "BKIND_VENV still contains PROJECT. Run: export BKIND_VENV=/projappl/project_2015338/bkind-venv-aarch64" >&2
  exit 2
fi
if [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
  echo 'Deactivate the current venv/Conda environment before setup.' >&2; exit 2
fi
module purge
module load "$BKIND_MODULE"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 CW_FORCE_CONDA_ACTIVATE=1
export TORCH_HOME="${TORCH_HOME:-$ROOT/.torch-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.pip-cache}"
if [[ ! -e "$BKIND_VENV" ]]; then python3 -m venv --system-site-packages "$BKIND_VENV"; fi
source "$BKIND_VENV/bin/activate"
python -m pip install --upgrade-strategy only-if-needed -r "$SCRIPT_DIR/requirements.txt"
python -m pip check
python "$SCRIPT_DIR/run.py" check --device cpu
python "$SCRIPT_DIR/run.py" download-weights --device cpu
printf '%s\n' "$BKIND_MODULE" > "$BKIND_VENV/bkind-module.txt"
uname -m > "$BKIND_VENV/bkind-arch.txt"
python -m pip freeze > "$BKIND_VENV/bkind-packages.txt"
echo "Ready: $BKIND_VENV"
