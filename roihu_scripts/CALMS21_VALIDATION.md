# CalMS21 preparation validation — 2026-11-03

Completed preparation of all ten cable-free training videos selected by the
unmodified upstream CalMS21 mouse loader. Exact result: **25,977 temporal pairs**,
**51,954 unique RGB PNGs**, **6,597,311,587 image bytes (6.144 GiB)**. The dataset
occupies about 6.3 GiB with allocation/metadata overhead; 60.3 GiB remained free.
Original MP4s were not altered and no test video was used.

Passed checks:

- All ten source videos report 30/1 constant FPS; source metadata and original
  frame counts are recorded in inventory.json and summary.json.
- Every pair is exactly (7*i, 7*i+6), matching the upstream range expression.
  No frame is reused across pairs. All pair source paths are in train/.
- All 51,954 PNG headers/modes/dimensions/CRC checks passed (RGB, 256×256).
- Every PNG's SHA256 matched its extraction-time hash.
- Independently decoding original frames 0, 6 and 7 and applying the prescribed
  Pillow resize yielded bit-identical stored pixels.
- Pair-manifest loader returned the expected nine-item tuple and normalized
  (3,256,256) tensors; representative beginning/middle/end pairs were checked.
- An additional small fixture verified rotation-transform parity and rejection
  of wrong frame gaps and attempts to access validation/test as discovery data.
- All 13 selected optimizer/data/model hyperparameters match the upstream
  config/CalMS21.yaml exactly.
- Explicit disk-reserve and output-budget tests stopped before any frame decode.
- Exclusive-writer lock rejected a concurrent preparer. Resuming a completed
  run skipped all ten videos and left the exact pair manifest unchanged.
- Python compilation, shell syntax for Slurm/setup scripts and git diff checks
  passed.

Preparation was sequential with one decoder thread, one filter thread, bounded
frame memory and periodic short pauses. Verification was also sequential and
throttled. No local model training or GPU workload ran during this task.

The local prepared root contains validation.json, source/frame/pair manifests,
code_manifest.json and pair_preview.jpg. See CALMS21_REPRODUCTION.md for transfer,
Roihu launch commands, wrapper differences and evaluation limits.

Not verified in this task: a full neural-network training pass using the new
train-only manifest path, actual Slurm/GH200 training or performance, paper MAP,
MARS semi-supervised training/annotations or MARS-Pose PCK. Use the supplied
separate GPU-smoke command before the 15-epoch run. Earlier dummy-data CPU tests
covered the underlying architecture/loss, fine-tuning/resume and inference;
they are not a claim of a completed CalMS21 reproduction.
