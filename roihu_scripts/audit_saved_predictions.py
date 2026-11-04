"""Audit positional collapse using existing B-KinD predictions; no retraining."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
r=Path(__file__).resolve().parents[1]
pred=r/'outputs/infer-2491633'
data=r.parents[1]/'DeepLabCut/datasets/20260907T141211Z_balanced_nonpartial_strong_aug_train_val_test_8kp_tailbase'
out=r/'outputs/infer-2491633_diagnosis';out.mkdir(exist_ok=True)
report={}
for split in ['train','val','test']:
 records=json.loads((data/split/'annotations.json').read_text())['samples']
 original=[s for s in records if not s.get('is_augmented')]
 points=[];inside=[];lookup={}
 for s in records:
  p=pred/split/'images'/(s['sample_id']+'.png.json')
  if p.exists():lookup[s['sample_id']]=np.array(json.loads(p.read_text())['keypoints_xy_pixels'])
 for s in original:
  xy=lookup[s['sample_id']];points.append(xy)
  mask=np.array(Image.open(data/split/'masks'/s['crop_file_name']).convert('L'))>0
  x=np.rint(xy[:,0]).astype(int);y=np.rint(xy[:,1]).astype(int)
  valid=(x>=0)&(y>=0)&(x<mask.shape[1])&(y<mask.shape[0]);v=np.zeros(len(x),dtype=bool);v[valid]=mask[y[valid],x[valid]];inside.append(v)
 points=np.array(points);entry=dict(original_images=len(points),mean_xy_pixels=points.mean(0).tolist(),
  std_xy_pixels=points.std(0).tolist(),inside_cow_mask_fraction=np.mean(inside,axis=0).tolist(),
  range_xy_pixels=np.ptp(points,axis=0).tolist())
 errors=[];actual=[];expected=[]
 for s in records:
  aug=s.get('augmentation')
  if not aug or s['sample_id'] not in lookup:continue
  parent=lookup[aug['parent_sample_id']];child=lookup[s['sample_id']]
  transformed=np.c_[parent,np.ones(len(parent))]@np.array(aug['affine_matrix_2x3']).T
  errors.append(np.linalg.norm(child-transformed,axis=1))
  actual.append(np.linalg.norm(child-parent,axis=1));expected.append(np.linalg.norm(transformed-parent,axis=1))
 if errors:entry['augmentation_response']=dict(pairs=len(errors),median_affine_response_error_px=np.median(errors,axis=0).tolist(),
  median_actual_displacement_px=np.median(actual,axis=0).tolist(),median_affine_expected_displacement_px=np.median(expected,axis=0).tolist(),
  note='Arbitrary affine transforms include flips/contrast and are not the exact 90-degree training loss; descriptive diagnostic only.')
 report[split]=entry
(out/'prediction_audit.json').write_text(json.dumps(report,indent=2))
text='# B-KinD prediction/training audit\n\nConfirmed prediction behavior (original images only; pixel units refer to 640×640 crops):\n\n| Index | Test std x | Test std y | Test x range | Test y range | Inside cow mask |\n|---|---:|---:|---:|---:|---:|\n'
t=report['test']
for i in range(len(t['std_xy_pixels'])):
 sx,sy=t['std_xy_pixels'][i];rx,ry=t['range_xy_pixels'][i];text+=f"| {i} | {sx:.4f} | {sy:.4f} | {rx:.4f} | {ry:.4f} | {100*t['inside_cow_mask_fraction'][i]:.1f}% |\n"
text+='\nPoint 3 is nearly input-independent and lies in background on the original test crops. Other points respond somewhat to image content; the model is not literally constant overall. This is positional-template behavior, not all points collapsing to the same coordinate.\n\nThe model checkpoint, training run.json, history.jsonl and training log are not available locally, so the selected epoch, actual settings, full-loss stage and checkpoint identity cannot yet be verified.\n\nCode/data risks to investigate:\n\n1. best.pt is selected using SSIM-target validation MSE only. A shortcut reconstruction can improve this metric while landmarks remain uninformative. Selection can also retain an epoch before curriculum=4 enables rotation/separation. Compare checkpoint epochs and best.pt versus last.pt before changing training.\n2. Slow motion, independently moving crop boxes and segmentation/crop changes can make the temporal reconstruction target weak or dominated by preprocessing. View frame-pair/SSIM targets from the actual run before increasing frame_gap.\n3. Training frames are tight 224×224 behavior crops, whereas these test inputs are independently extracted masked 640×640 crops, both resized by the runner. Framing/background changes may cause a distribution shift. Audit predictions on held-out frames from the training-style sequences as well.\n4. The current history records total training loss and validation MSE only; it cannot show whether perceptual, rotation and separation terms were balanced or whether the keypoint branch had useful gradients. Log each term, keypoint variation and reconstruction sensitivity to shuffled keypoints.\n\nNo verified sign was found of constant coordinates being hard-coded, stale image caching, or permissive checkpoint loading in the local runner: it reads each image separately, performs model inference, and loads weights strictly. Those code checks do not establish which remote script/checkpoint was used.\n\nThe paper uses a stationary-background motion signal, and specifically uses the same crop bounding box for a paired human example; it also describes the center-collapse risk and separation loss: https://arxiv.org/pdf/2112.05121 .\n'
(out/'REPORT.md').write_text(text)
print(text.split('Code/data risks')[0])
