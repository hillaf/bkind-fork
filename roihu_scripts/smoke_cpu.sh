#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv-local/bin/python}"
OUTPUT="${OUTPUT:-$ROOT/outputs/cpu-smoke-$(date +%Y%m%dT%H%M%S)}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.torch-cache}"
"$PYTHON" "$SCRIPT_DIR/make_smoke_data.py" "$OUTPUT/data"
# Four adjacent pairs from five frames; two pairs per batch. Exercise all losses.
"$PYTHON" "$SCRIPT_DIR/run.py" train --device cpu --data "$OUTPUT/data" --output "$OUTPUT/train" \
  --image-size 64 --nkpts 4 --frame-gap 1 --batch-size 2 --workers 0 --epochs 1 --curriculum 0 --max-steps 2
"$PYTHON" "$SCRIPT_DIR/run.py" train --device cpu --data "$OUTPUT/data" --output "$OUTPUT/finetune" \
  --image-size 64 --nkpts 4 --frame-gap 1 --batch-size 2 --workers 0 --epochs 1 --curriculum 0 --max-steps 1 \
  --finetune "$OUTPUT/train/last.pt"
"$PYTHON" "$SCRIPT_DIR/run.py" train --device cpu --data "$OUTPUT/data" --output "$OUTPUT/resume" \
  --image-size 64 --nkpts 4 --frame-gap 1 --batch-size 2 --workers 0 --epochs 2 --curriculum 0 --max-steps 3 \
  --resume "$OUTPUT/train/last.pt"
"$PYTHON" "$SCRIPT_DIR/run.py" infer --device cpu --checkpoint "$OUTPUT/train/last.pt" \
  --images "$OUTPUT/data/val/clip_val" --output "$OUTPUT/inference"
echo "CPU smoke workflow passed: $OUTPUT"
"$PYTHON" "$SCRIPT_DIR/verify_smoke.py" "$OUTPUT"
