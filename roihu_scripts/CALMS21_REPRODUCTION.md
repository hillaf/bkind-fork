# CalMS21 B-KinD discovery setup and reproduction audit

Prepared 2026-11-03 for CSC project **project_2015338**. Repository:
`/Users/hillafred/Sandbox/BKind/BKinD`; source videos:
`/Users/hillafred/Sandbox/BKind/task1_videos_mp4`; prepared dataset:
`/Users/hillafred/Sandbox/BKind/calms21-bkind-paper`.

## Protocol and what this reproduces

The paper trains discovery on cable-free CalMS21 training videos. For CalMS21
behavior scores, it then trains a separate classifier using the official labeled
training split and scores the official test split. MARS-Pose is a different
experiment: semi-supervised training with 10/50/100/500 labeled images and about
1,500 test images, scored with 0.5-cm PCK over nose/ears/tail. Merely predicting
MARS images with a frozen CalMS21 discovery model does not reproduce that result.
A frozen linear probe would be another experiment and must be labeled as such.

Primary reference: [paper, final v2 including supplement](https://arxiv.org/html/2112.05121v2),
Sections 4.1–4.3, Appendix A.1 and Appendix B.
The local upstream release does not contain the MARS-Pose semi-supervised
training/evaluation implementation. Its keypoint regression script is for
Human3.6M, not MARS. MARS images, anatomical annotations, official split,
physical image calibration and experiment-specific implementation are still
needed before claiming a paper-equivalent MARS-Pose score.

This deliverable prepares **self-supervised CalMS21 discovery**. It does not run
Roihu training, download MARS data, or compute a MARS score.

## Exactly selected inputs

The available task-1 MP4 tree has 70 training videos and 19 test videos (about
27.6 GiB compressed combined). The local upstream loader
`dataloader/mouse_dataset.py::generate_pair_images` hard-codes these ten
cable-free training IDs:

```text
mouse001 mouse002 mouse051 mouse052 mouse053
mouse054 mouse055 mouse056 mouse057 mouse058
```

All ten are present, with constant 30-FPS 1024×570 MPEG-4 video according to
ffprobe. The frame-count inventory gives **25,977 pairs / 51,954 stored frames**,
matching the roughly 26k-pair setting described in the paper's data-size study.
The selection rule is copied from the actual upstream loader, not inferred by
looking at labels. No behavior labels, GT points, crops or masks are used.

Pair i within each video uses original zero-based frame indices:

```text
(0,6), (7,13), (14,20), ... while second index < original frame count
```

The pair gap is 6 original frames / 0.2 seconds; pair-start stride is 7. All
selected pairs are retained, and no source frame is reused between pairs. Test
videos are not decoded or sampled by this preparation. Do not create a val set
from test videos: the upstream CalMS21 entry point does not use held-out test
images for discovery validation. For its 15-epoch default, upstream
`val_schedule=20` means no scheduled validation occurs before training ends.

## Storage and machine-load controls

Only the two required frames per pair are stored. Full source resolution and
unused timestamps are not extracted to disk. MP4 files and earlier datasets are
untouched; no hard links or symlinks are created.

Each selected frame is decoded as RGB, resized from the **entire fixed camera
view** to 256×256 with Pillow bilinear antialiasing, and stored as lossless PNG
(compression level 1). This matches the upstream full-view square resize; the
constant aspect transformation is intentional. There are no moving crops,
per-frame boxes, segmentation masks or temporal resampling. Saving at model
resolution avoids redundant resizing during every training epoch. MP4 inputs
may already differ from the original SEQ release; exact bitwise reproduction of
those original pixels is not asserted.

Preparation uses a single sequential video process, one FFmpeg decoder thread,
one filter thread and one PNG writer. It pauses 0.1 seconds every 32 stored
frames and retains only a few decoded frames in memory. The source decode is
sequential, avoiding thousands of costly random MP4 seeks. The pipeline does
not run local model training or use the GPU while decoding.

It checks free space every 32 frames, stops below **30 GiB free**, and refuses
to exceed **10 GiB output**. The measured dataset is expected to need roughly
7 GiB, rather than extracting every full-resolution frame of every video.
No source data is deleted to make space. If interrupted, completed videos are
skipped on restart; the current incomplete video is decoded again. The final
pair manifest is published only after all ten videos finish.

```bash
cd /Users/hillafred/Sandbox/BKind/BKinD
.venv-local/bin/python roihu_scripts/prepare_calms21.py \
  --videos ../task1_videos_mp4 --output ../calms21-bkind-paper \
  --mode prepare --reserve-gib 30 --max-output-gib 10
```

The same command resumes. `--mode inventory` only reads metadata and reports
counts. Local preparation log: `/private/tmp/bkind-calms21-prepare.log`.

Files under the prepared root:

- `train/<video>/frame_<original-index>.png`: sparse lossless frames.
- `train/<video>/complete.json`: per-frame hashes and completed source inventory.
- `train_pairs.jsonl`: exact source indices, times and relative frame paths.
- `inventory.json`, `summary.json`: selected IDs, codec/FPS/size and actual counts.
- `validation.json`: integrity and pair/loader checks, produced after extraction.

**Use the pair-manifest loader.** Ordinary directory-based pairing on these
sparse frames would create the wrong temporal gaps.

## Training parameters and launch

Settings are encoded in `roihu_scripts/calms21_paper.yaml` and follow the local
upstream `config/CalMS21.yaml`, optimizer in `train_video.py`, and the CalMS21
hyperparameter row in the paper:

| Setting | Value |
|---|---|
| Dataset format | `calms21_pairs` (explicit frame indices) |
| Discovery data | all 25,977 pairs from ten cable-free train videos |
| Validation | none; never use CalMS21 test to select discovery checkpoints |
| Learned points | **10 total for the full scene**, not 10 per mouse |
| Model input | 256×256 full RGB view |
| Pair gap / start stride | 6 / 7 source frames at 30 FPS |
| Batch size | **5** |
| Epochs | **15** |
| Optimizer | SGD, momentum 0.9, weight decay 0.0001 |
| Initial learning rate | **0.001** |
| LR schedule | ×0.1 at epoch index 15 (so constant during a 15-epoch run) |
| Rotation/separation | enabled at epoch index **4**, matching released code |
| Rotations | +90°, 180°, −90° |
| Perceptual initial scales | `[100.0, 1.6, 2.3, 1.8, 2.8, 100.0]` |
| Initialization | upstream ImageNet ResNet50 and VGG19 loss weights |
| Input normalization | ImageNet RGB mean/std |
| Precision/devices | float32, one GH200; no AMP or DDP |
| Seed | 0 (recorded choice, not a claim of the authors' exact seed) |

There are 5,196 optimizer steps per full epoch, 77,940 across 15 epochs with
the final short batch retained, as in upstream. Loss magnitudes are not directly
comparable with the prior cow experiment's validation losses.

Transfer the prepared dataset and changed repository files to CSC, excluding
local virtual environments, caches and previous outputs. The 29-GB MP4 tree is
not required on compute nodes for this prepared-frame training.

```bash
rsync -av /Users/hillafred/Sandbox/BKind/calms21-bkind-paper/ \
  USER@roihu-gpu.csc.fi:/scratch/project_2015338/DATA/calms21-bkind-paper/

# Transfer/update BKinD code separately, including dataloader/pair_manifest_dataset.py
# and all changed model/loss files. A native venv cannot be copied from the Mac.
```

On **Roihu-GPU**, from the repository root:

```bash
export BKIND_MODULE=python-pytorch/2.13
export BKIND_VENV=/projappl/project_2015338/bkind-venv-aarch64
export DATA=/scratch/project_2015338/DATA/calms21-bkind-paper
export OUTPUT=/scratch/project_2015338/BKind/BKinD/outputs/calms21-paper-01
sbatch roihu_scripts/train_calms21.sbatch
```

Use the dedicated `train_calms21.sbatch`, not the generic `train.sbatch`: the
generic script supplies cow defaults of 30 epochs and batch size 4 as CLI flags,
which override a selected YAML.

The CalMS21 script pins account `project_2015338`, requests one GH200, 18 CPU cores and
36 hours. Run a short GPU check first in a **different** output directory:

```bash
OUTPUT=/scratch/project_2015338/BKind/BKinD/outputs/calms21-gpu-smoke \
  sbatch --partition=gputest --time=00:15:00 roihu_scripts/train_calms21.sbatch \
  --max-steps 2 --curriculum 0
```

Do not use smoke limits for the real experiment. If runtime exceeds an
allocation, resume from the last completed epoch with the same config and
`--resume /absolute/path/last.pt`; the runner does not checkpoint every batch.
The final epoch's `last.pt` is the discovery checkpoint for this protocol.
No validation-selected `best.pt` is produced in train-only mode. Leave any
old cow checkpoint out of this run: start from ImageNet initialization.

CSC's [PyTorch module documentation](https://docs.csc.fi/apps/pytorch/) lists
`python-pytorch/2.13`; it includes CUDA/cuDNN. No extra CUDA, cuDNN or python-data
module is required. The existing venv setup uses `--system-site-packages`,
PIQ 0.8.0, PyYAML, NumPy and Pillow. FFmpeg is needed only for preparation or
later direct video decoding, not for training on prepared PNGs.

## Wrapper changes versus the original release

| Area | Our behavior | Reproduction implication |
|---|---|---|
| Network | upstream ResNet50, GlobalNet and reconstruction decoder retained | architecture/checkpoint tensor layout preserved |
| Device grids | use tensor device/dtype instead of unconditional `.cuda()` | CPU checks and Roihu single GPU supported |
| Loss | upstream SSIM target, VGG19 features, rotation and separation retained | no cow foreground objective or extra regularization |
| PIQ kernels | explicit floating dtype | fixes modern Torch integer-kernel error |
| Loss division | out-of-place rather than `/=` | modern autograd compatibility; same formula |
| VGG | frozen loss weights, eval mode | original optimizer excluded these weights anyway; avoids unused gradient storage |
| Reduction string | equality rather than Python object identity | fixes brittle string comparison |
| Config parser | explicit YAML defaults; CLI overrides; replace lists | fixes upstream perceptual-weight list append behavior |
| Initialization | `pretrained` is respected; strict model loading | new runs use ImageNet, fine-tuning loads supplied weights |
| CalMS21 pairs | explicit sparse manifest; exact upstream ten IDs and gap/stride | avoids wrong re-pairing after storage thinning |
| Validation | generic cow mode uses SSIM-target MSE; CalMS21 mode disables validation | avoids upstream RGB/MSE mismatch and test leakage |
| Checkpoints | plain single-device state, atomic epoch saves; metadata and optimizer | no DataParallel prefix; legacy prefixes stripped on load |
| Selection | CalMS21 uses final checkpoint after 15 epochs | no test-selected or cow-validation-selected model |
| Inference | our still-image runner explicitly calls `eval()` | upstream `extract_features.py` comments out both eval calls; audit downstream feature extraction rather than silently copying that behavior |
| Numeric environment | release environment pins Python 3.9.13, Torch 1.11.0, torchvision 0.12.0 and CUDA 11.3; Roihu uses the current module/GH200 | meaningful modern replication, not bit-for-bit identical |

Main implementation changes are in `roihu_scripts/run.py`,
`dataloader/pair_manifest_dataset.py`, `model/unsupervised_model.py`,
`model/vggforLoss.py`, `loss/compute_loss.py` and `loss/ssim_loss.py`.
The original `train_video.py` entry point remains available but should not be
used with the sparse prepared tree.

## MARS and later evaluation

The existing `run.py infer` accepts individual RGB images. Use original full
views with both mice, matching discovery input framing. It returns unsupervised
indices, not named MARS anatomical landmarks. A separate mapping can quantify
transfer but is not the paper's MARS semi-supervised experiment.

For CalMS21 behavior reproduction, extract model features at **every** frame of
all official train and test videos using the frozen discovery checkpoint; do
not train the behavior classifier only on the sparse discovery samples. The
paper evaluates keypoints with confidence/covariance features and multiple
classifier seeds. The released `extract_features.py` and downstream classifier
code also require modernization/verification before claiming the reported MAP.
Our generic still-image inference currently exports coordinates/overlays only;
it is suitable for visual checks, not a substitute for that full evaluation.

No discovery result or downstream paper score is claimed until training and the
matching evaluation procedure have actually run.

Preparation completed and all integrity checks passed. Actual image storage:
6.144 GiB (about 6.3 GiB on disk), leaving 60.3 GiB free. See
[CALMS21_VALIDATION.md](CALMS21_VALIDATION.md) for checks and unverified scope.
