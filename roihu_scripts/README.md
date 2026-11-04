# B-KinD training and fine-tuning on CSC Roihu

For CalMS21 paper-style discovery, use [CALMS21_REPRODUCTION.md](CALMS21_REPRODUCTION.md)
and `train_calms21.sbatch`; the generic cow defaults below are different.

CSC project and Slurm account: **`project_2015338`**.

Repository: `BKind/BKinD`; upstream base commit
`9e737219aff1bcf2d471833835edd3115a967d62`. These scripts add a single-device
runner for the upstream B-KinD model and SSIM/perceptual, rotation and separation
losses. They prepare jobs; they do not connect to CSC or submit automatically.

## Data: temporal pairs are required

B-KinD learns from two frames of the same video. Independent images/crops from
the AutoLink experiments are not a valid substitute for a temporal sequence.
Use fixed-camera videos, modest motion between paired frames, and little
occlusion. This upstream implementation has no multi-animal identity tracking
or support for per-frame bounding boxes. For cow crops, use consistently framed
tracked clips; fluctuating crop transforms can create artificial motion.

Keep whole videos, and preferably cameras/sessions, within one split:

```text
/scratch/project_2015338/bkind_frames/
  train/
    video_001/frame_000001.png ...
    video_002/frame_000001.png ...
  val/
    video_003/frame_000001.png ...
  test/                         # optional, never read by the training runner
    video_004/frame_000001.png ...
```

Files must sort in temporal order: use zero-padded frame numbers. Do not mix
clips in one directory. Each clip needs more than `frame_gap` frames. Only image
files are paired; `.DS_Store` and other non-image entries are ignored.
`frame_gap` is a frame-index offset, not seconds. Default gap 6 corresponds to
0.2 seconds at 30 extracted fps; at 10 fps, set gap 2 for 0.2 seconds. Choose a
gap with visible but moderate animal motion. The loader samples start positions
at stride `frame_gap // 10 + 1`, matching upstream.

Example extraction if `ffmpeg` is available in your preprocessing environment:

```bash
mkdir -p /scratch/project_2015338/bkind_frames/train/video_001
ffmpeg -i /path/to/video_001.mp4 -vf fps=30 \
  /scratch/project_2015338/bkind_frames/train/video_001/frame_%08d.png
```

Repeat per clip and split; no FFmpeg module is required for already extracted
frames. No GT landmark labels enter keypoint-discovery training. Fine-tuning
below means continuing self-supervised discovery on new videos, not supervised
training on anatomical GT points. A downstream regression is a separate task.

## Environment and required module

Use **Roihu-GPU** (`roihu-gpu.csc.fi`) to create the GPU environment: its Grace
CPUs are ARM/aarch64. Roihu-CPU uses x86_64, so use a separate venv there.
Never transfer a Mac venv or a Roihu-CPU venv to Roihu-GPU.

The default is `module purge; module load python-pytorch/2.13`. CSC documents
2.10 as another available version. CUDA and cuDNN are already included; no
separate CUDA/cuDNN, python-data, Conda or compiler module is needed. The venv
uses `--system-site-packages` to inherit Torch and torchvision, then installs
PIQ 0.8.0, PyYAML, NumPy and Pillow without replacing the module's Torch.

Sources checked 2026-10-22:
[CSC PyTorch](https://docs.csc.fi/apps/pytorch/),
[CSC Python venv guide](https://docs.csc.fi/support/tutorials/python-usage-guide/),
[Roihu architecture](https://docs.csc.fi/support/faq/roihu/).

Transfer the repository excluding environments, caches, large data and outputs:

```bash
rsync -av --exclude='.git' --exclude='.venv-*' --exclude='.torch-cache' \
  --exclude='.pip-cache' --exclude='outputs' --exclude='__pycache__' \
  /path/to/BKind/BKinD/ USER@roihu-gpu.csc.fi:/scratch/project_2015338/BKind/BKinD/
```

On the GPU login node, outside any active venv or Conda environment:

```bash
cd /scratch/project_2015338/BKind/BKinD
export BKIND_VENV=/projappl/project_2015338/bkind-venv-aarch64
bash roihu_scripts/setup_venv.sh
```

Setup records module, architecture and package versions. It downloads the
upstream ResNet50 and VGG19 ImageNet weights (~98 MB and ~548 MB) into the shared
`.torch-cache`. Compute jobs reuse them without internet access. Set `TORCH_HOME`
to another shared cache if needed and use the same value for setup and jobs.
`BKIND_MODULE` may override the module but must match between setup and jobs.
Do not run heavy training on login nodes.

## Training from ImageNet initialization

Submit from the repository root; scripts use `SLURM_SUBMIT_DIR` to find code:

```bash
export BKIND_VENV=/projappl/project_2015338/bkind-venv-aarch64
export DATA=/scratch/project_2015338/bkind_frames
export OUTPUT=/scratch/project_2015338/BKind/BKinD/outputs/discovery-01
sbatch --account=project_2015338 roihu_scripts/train.sbatch
```

Defaults in `roihu_scripts/custom.yaml`: 8 learned points, 256×256 RGB images,
frame gap 6, batch size 4, SGD lr 0.001, momentum 0.9, weight decay 0.0001,
30 epochs, lr multiplied by 0.1 every 30 epochs, and rotation/separation losses
starting at epoch index 4. This uses upstream loss weights and ImageNet input
normalization. The jobs request one GH200 and 18 CPU cores; reduce batch size
if memory is insufficient. Single-GPU float32 only; no mixed precision/DDP.

Override YAML options with CLI arguments after the sbatch script:

```bash
sbatch --account=project_2015338 roihu_scripts/train.sbatch --frame-gap 2 --nkpts 10
```

`BATCH_SIZE`, `EPOCHS`, `WORKERS`, `CONFIG` and `OUTPUT` are also environment
overrides. CLI arguments override YAML values; lists replace rather than append
(the upstream config reader appends perceptual weights). `image_size` must be
at least 64 and a multiple of 64. Prefer the default 256 for real experiments.

A first GPU smoke check, in a fresh output directory:

```bash
OUTPUT=/scratch/project_2015338/BKind/BKinD/outputs/gpu-smoke \
  sbatch --account=project_2015338 --partition=gputest --time=00:15:00 \
  roihu_scripts/train.sbatch --epochs 1 --max-steps 2 --limit-batches 1 \
  --curriculum 0 --batch-size 2 --image-size 128
```

This deliberately limits the data pass. It verifies the full loss branch, not
convergence or an unbiased validation score.

## Fine-tuning and resume

Weights-only fine-tuning loads a complete matching B-KinD checkpoint, starts a
fresh optimizer and epoch count, and lets you choose a new learning rate/data:

```bash
export OUTPUT=/scratch/project_2015338/BKind/BKinD/outputs/finetune-01
sbatch --account=project_2015338 roihu_scripts/train.sbatch \
  --finetune /scratch/project_2015338/pretrained_bkind/checkpoint.pth.tar --lr 0.0001
```

Use the same `--nkpts` as the checkpoint; shape mismatches fail rather than
silently skipping parameters. Legacy `module.` prefixes are handled. This does
not download or assume the existence of an official pretrained B-KinD file.
Checkpoints must contain plain tensor state dictionaries; arbitrary pickle
objects are not loaded. For a legacy inference checkpoint, supply the matching
`--nkpts` and `--image-size`; runner checkpoints remember these settings.

Resume restores weights, optimizer, total step, epoch and best validation loss:

```bash
export OUTPUT=/scratch/project_2015338/BKind/BKinD/outputs/discovery-01
sbatch --account=project_2015338 roihu_scripts/train.sbatch \
  --resume "$OUTPUT/last.pt" --epochs 60
```

Core data/model/optimizer/loss settings must match. Epoch count may increase.
The runner saves at epoch boundaries. When a smoke limit interrupts an epoch,
resume restarts that epoch's shuffled data pass; it does not restore exact
sampler position or random-generator state. Use `--finetune` for legacy
checkpoints lacking runner metadata. Do not use an existing checkpoint output
folder for a new independent run.

## Validation, outputs and inference

Training reads only train/video frame directories; validation reads only val.
Keep clips from the same session out of both splits. Resolved source paths are
recorded and overlap is rejected (this cannot detect copied duplicate images).
Test images are never used for model selection.

Outputs: atomic `last.pt`, validation-selected `best.pt`, `history.jsonl`, and
`run.json` containing config and exact temporal pair paths. Batch losses and
finite-gradient checks run during training. VGG19 loss features are frozen,
while gradients flow from them to the decoder.

Validation is MSE against the SSIM-derived motion reconstruction target, not
anatomical GT error. Unlike upstream validation (which compares motion output
to raw normalized RGB), this matches the model's reconstruction target. Full
rotation/perceptual losses are used during training according to curriculum;
validation reconstructs without the additional rotation branches.

```bash
export CHECKPOINT="$OUTPUT/best.pt"
export IMAGES=/scratch/project_2015338/bkind_frames/test/video_004
export OUTPUT=/scratch/project_2015338/BKind/BKinD/outputs/test-keypoints
sbatch --account=project_2015338 roihu_scripts/infer.sbatch
```

Inference writes bright keypoint overlays and JSON coordinates. Normalized
coordinates are `(x,y)` in [-1,1], unlike AutoLink's `(row,column)` convention.
Use the saved JSON for downstream linear regression; discovered indices have
no predefined anatomical identities. No label regression or test scoring is
part of these training scripts.

Monitor using `squeue --me`, `tail -f slurm-JOBID.out`, and
`sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,AllocTRES`.

## Local execution checks

```bash
PYTHON=python3.12 bash roihu_scripts/setup_local.sh
bash roihu_scripts/smoke_cpu.sh
```

The smoke workflow uses separate synthetic moving-shape train/val clips, each
five frames. It exercises the real pretrained backbone, VGG19/SSIM loss,
rotation and separation losses, validation, checkpoint save/reload, weights-only
fine-tuning, optimizer resume and five-frame inference at 64×64. This is a
software sanity check, not evidence of anatomical quality.

See `VALIDATION.md` for observed results. Roihu modules, Slurm and GH200 remain
unverified until the GPU smoke job is executed on CSC.
