"""Audit input sensitivity and heatmap saturation of a complete B-KinD checkpoint."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import transform,load_weights
from model.unsupervised_model import Model
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);r=Path(__file__).resolve().parents[1];sandbox=r.parents[1]
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True);cfg=saved['config']
model=Model(cfg['nkpts'],pretrained=False,output_shape=(cfg['image_size']//4,)*2);load_weights(model,saved);model.eval();t=transform(cfg['image_size'])
a.output.mkdir(parents=True,exist_ok=True)
@torch.inference_mode()
def predict(image):
 (x,y),logits=model(t(image).unsqueeze(0));prob=logits.flatten(2).softmax(-1);entropy=-(prob*prob.clamp_min(1e-30).log()).sum(-1)/math.log(prob.shape[-1])
 return dict(keypoints_xy_normalized=torch.stack((x[0],y[0]),-1).numpy().tolist(),heatmap_peak=prob.max(-1).values[0].numpy().tolist(),heatmap_normalized_entropy=entropy[0].numpy().tolist())
data=sandbox/'DeepLabCut/datasets/20260907T141211Z_balanced_nonpartial_direct_train_val_test_8kp_tailbase/test/images'
records=[];diff=[]
for path in sorted(data.glob('*.png')):
 result=predict(Image.open(path).convert('RGB'));result['source']=str(path);records.append(result)
 old=json.loads((r/'outputs/infer-2491633/test/images'/(path.name+'.json')).read_text())
 diff.append(float(np.abs(np.array(result['keypoints_xy_normalized'])-np.array(old['keypoints_xy_normalized'])).max()))
# Same image under controlled geometric/content changes; preserve keypoint indices.
image=Image.open(sorted(data.glob('*.png'))[0]).convert('RGB');controls={}
for name,img in [('original',image),('rot90',image.transpose(Image.Transpose.ROTATE_90)),('rot180',image.transpose(Image.Transpose.ROTATE_180)),('blank_black',Image.new('RGB',image.size,'black')),('blank_gray',Image.new('RGB',image.size,(128,128,128)))]:controls[name]=predict(img)
# Also test frames in the actual training-style input distribution.
manifest=[json.loads(v) for v in (r.parent/'bkind-small-662/manifest.jsonl').read_text().splitlines()];sampled={}
for v in manifest:
 if v['split']=='val':sampled.setdefault(v['behavior'],v)
validation=[]
for v in sampled.values():
 path=r.parent/'bkind-small-662'/v['destination']/'00.jpg';value=predict(Image.open(path).convert('RGB'));value['behavior']=v['behavior'];validation.append(value)
points=np.array([v['keypoints_xy_normalized'] for v in records]);style=np.array([v['keypoints_xy_normalized'] for v in validation]);orig=np.array(controls['original']['keypoints_xy_normalized']);rot=np.array(controls['rot90']['keypoints_xy_normalized']);expected=np.stack((orig[:,1],-orig[:,0]),axis=1)
report=dict(checkpoint=str(a.checkpoint.resolve()),epoch=saved['epoch'],step=saved['step'],config=cfg,
 max_difference_from_saved_inference_normalized=max(diff),median_difference_from_saved_inference_normalized=float(np.median(diff)),
 original_test_std_xy_at_640px=(points.std(0)*319.5).tolist(),
 test_mean_peak=np.mean([v['heatmap_peak'] for v in records],axis=0).tolist(),
 test_mean_normalized_entropy=np.mean([v['heatmap_normalized_entropy'] for v in records],axis=0).tolist(),
 training_style_val_frames=len(validation),training_style_std_xy_at_224px=(style.std(0)*111.5).tolist(),
 rot90_equivariance_error_at_640px=(np.linalg.norm(rot-expected,axis=1)*319.5).tolist(),controls=controls,test_records=records,training_style_records=validation)
(a.output/'checkpoint_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('controls','test_records','training_style_records','config')},indent=2))
