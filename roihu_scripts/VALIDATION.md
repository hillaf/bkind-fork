# B-KinD setup validation — 2026-10-22

Executed locally on macOS ARM using Python 3.12.12, Torch 2.10.0,
torchvision 0.25.0, PIQ 0.8.0, PyYAML 6.0.3, NumPy 2.5.4 and Pillow 13.0.0.
`pip check` passed. Official upstream ResNet50 and VGG19 weights were downloaded
with hash verification and used in the CPU test; neither was replaced by a
random network or surrogate loss.

Command from `BKind/BKinD`:

```bash
OUTPUT=/Users/hillafred/Sandbox/BKind/BKinD/outputs/cpu-smoke-20261022 \
  bash roihu_scripts/smoke_cpu.sh
```

Dummy dataset: separate moving-shape train and validation clips, each with five
64×64 frames. Gap 1 produces four temporal pairs per split. Batch size 2, four
learned keypoints, workers 0, four CPU threads. `curriculum=0` exercises the
SSIM/VGG reconstruction, all three rotation branches and separation losses.
This is an execution check; it does not test a real animal dataset or convergence.

| Workflow | Observed result |
|---|---|
| Train | Two optimizer steps; batch losses 4.160819 and 3.929081; train mean 4.044950 |
| Validation | SSIM-target MSE 0.118734; best and last checkpoints saved |
| Weights-only fine-tune | Fresh optimizer/epoch; one step; loss 3.917496; validation MSE 0.118715 |
| Resume | Model and optimizer restored from step 2/epoch 1; advanced to step 3; validation MSE 0.118502 |
| Inference | Separate process loaded the checkpoint and produced keypoints/overlays for all five validation frames |

`verify_smoke.py` independently checked saved step/epoch counts, nonempty
optimizer state, parameter changes after fine-tuning and resume, finite recorded
metrics, four temporal pairs per split, disjoint train/val paths, best checkpoint
existence, and five finite (4,2) keypoint arrays bounded by [-1,1] with overlays.
The full workflow passed after correcting a PIQ Gaussian-filter dtype failure.
The smoke script now runs this artifact verification at the end.

Python compilation, `bash -n` for every setup/environment/Slurm script, and
`git diff --check` passed. Later input-validation guards for nonpositive thread
counts, negative smoke limits and completed resume targets were syntax-checked;
they do not alter the tested loss/model path.

Compatibility changes to upstream:

- Coordinate grids follow input device/dtype instead of calling `.cuda()`.
- VGG loss module is device-aware and frozen; gradients still flow to predictions.
- Loss division uses out-of-place operations for current autograd compatibility.
- PIQ Gaussian kernels receive explicit floating dtype.
- Custom frame pairing skips non-directories and non-image entries and rejects
  nonpositive frame gaps.
- String reduction comparison uses equality rather than object identity.

The new runner has a distinct CLI so existing upstream entry points are retained.
Validation measures the SSIM reconstruction target, correcting upstream's
comparison of a motion reconstruction to raw normalized RGB. This is explicitly
documented and means validation-loss values are not directly comparable with
old upstream logs.

Unverified here: actual Roihu module execution, Linux ARM venv setup, Slurm
submission, CUDA/GH200 execution, Torch 2.13 behavior, full-size training memory,
real video datasets, or an official externally provided B-KinD checkpoint.
Use the documented short GPU smoke job before launching a full experiment.
No CSC jobs were submitted and no real training dataset was prepared.
