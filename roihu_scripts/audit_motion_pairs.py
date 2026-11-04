"""Inspect sampled real training pairs and the actual SSIM reconstruction target."""
from pathlib import Path
import json,sys
from collections import defaultdict
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import transform,compute_ssim
r=Path(__file__).resolve().parents[1];dataset=r.parent/'bkind-small-662';out=r/'outputs/infer-2491633_diagnosis';out.mkdir(exist_ok=True)
torch.set_num_threads(4);t=transform(256);by_class=defaultdict(list)
for row in map(json.loads,(dataset/'manifest.jsonl').read_text().splitlines()):
 if row['split']=='train':by_class[row['behavior']].append(row)
results=[];tiles=[]
with torch.inference_mode():
 for behavior,clips in sorted(by_class.items()):
  for clip in clips[:2]:
   folder=dataset/clip['destination'];images=sorted(folder.glob('*.jpg'))
   starts=[0,6,11];x=torch.stack([t(Image.open(images[i]).convert('RGB')) for i in starts]);y=torch.stack([t(Image.open(images[i+4]).convert('RGB')) for i in starts]);target=compute_ssim(x,y)
   for j,i in enumerate(starts):
    value=target[j].numpy();results.append(dict(behavior=behavior,clip=clip['destination'],frame0=images[i].name,frame1=images[i+4].name,
      zero_predictor_mse=float(np.mean(value**2)),mean_motion_target=float(value.mean()),fraction_target_gt_0_1=float((value>.1).mean())))
   if len(tiles)<len(by_class) and (not tiles or tiles[-1][0]!=behavior):
    target_img=Image.fromarray((target[0,0].clamp(0,1).numpy()*255).astype('uint8')).convert('RGB');canvas=Image.new('RGB',(768,290),'#152030');d=ImageDraw.Draw(canvas)
    for k,img in enumerate([Image.open(images[0]).convert('RGB'),Image.open(images[4]).convert('RGB'),target_img]):canvas.paste(ImageOps.fit(img,(256,256)),(k*256,24))
    for k,label in enumerate([behavior+' t','t + 0.4s','SSIM target (clipped display)']):d.text((k*256+4,6),label,fill='white')
    tiles.append((behavior,canvas))
values=np.array([r['zero_predictor_mse'] for r in results]);summary=dict(clips=len(results)//3,pairs=len(results),frame_gap=4,
 zero_prediction_MSE=dict(min=float(values.min()),median=float(np.median(values)),max=float(values.max())),
 note='Sample of the locally prepared subset, not verified as the remote training run dataset; this target energy alone cannot establish collapse cause.',pairs_detail=results)
(out/'motion_pair_audit.json').write_text(json.dumps(summary,indent=2))
canvas=Image.new('RGB',(1536,((len(tiles)+1)//2)*290),'#152030')
for i,(_,tile) in enumerate(tiles):canvas.paste(tile,(i%2*768,i//2*290))
canvas.save(out/'sampled_training_motion_targets.jpg',quality=92)
print(json.dumps({k:v for k,v in summary.items() if k!='pairs_detail'},indent=2))
