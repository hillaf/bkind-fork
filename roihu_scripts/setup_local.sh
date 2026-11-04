#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${BKIND_LOCAL_VENV:=$ROOT/.venv-local}"
"${PYTHON:-python3.12}" -m venv "$BKIND_LOCAL_VENV"
"$BKIND_LOCAL_VENV/bin/python" -m pip install torch==2.10.0 torchvision==0.25.0 -r "$SCRIPT_DIR/requirements.txt"
"$BKIND_LOCAL_VENV/bin/python" -m pip check
export TORCH_HOME="${TORCH_HOME:-$ROOT/.torch-cache}"
"$BKIND_LOCAL_VENV/bin/python" "$SCRIPT_DIR/run.py" download-weights --device cpu
