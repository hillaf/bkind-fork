"""Render sampled cow train/val frames with a stated confidence threshold."""
import argparse,colorsys,hashlib,html,json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import load_weights,transform
from model.unsupervised_model import Model

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threshold',type=float,default=.1);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True);cfg=saved['config'];model=Model(cfg['nkpts'],pretrained=False,output_shape=(cfg['image_size']//4,)*2);load_weights(model,saved);model.eval();t=transform(cfg['image_size'])
 by_group=defaultdict(list)
 for row in map(json.loads,(a.dataset/'manifest.jsonl').read_text().splitlines()):by_group[row['split'],row['behavior']].append(row)
 colors=[tuple(round(c*255) for c in colorsys.hsv_to_rgb(i/cfg['nkpts'],.85,1)) for i in range(cfg['nkpts'])];font=ImageFont.load_default(size=18);small=ImageFont.load_default(size=12);entries=[]
 with torch.inference_mode():
  for (split,behavior),clips in sorted(by_group.items()):
   selected=sorted(clips,key=lambda v:v['destination'])[len(clips)//2];path=a.dataset/selected['destination']/'08.jpg';image=Image.open(path).convert('RGB')
   (ux,uy),logits=model(t(image).unsqueeze(0));xy=torch.stack((ux[0],uy[0]),-1).numpy();peaks=logits.flatten(2).softmax(-1).amax(-1)[0].numpy();assert np.isfinite(xy).all();visible=peaks>a.threshold
   original=image.resize((384,384),Image.Resampling.BILINEAR);overlay=original.copy();draw=ImageDraw.Draw(overlay);pixelxy=(xy+1)*.5*np.array([383,383])
   for j,(x,y) in enumerate(pixelxy):
    if not visible[j]:continue
    draw.ellipse((x-6,y-6,x+6,y+6),fill=colors[j],outline='black',width=2);draw.text((x+9,y-12),str(j),font=font,fill=colors[j],stroke_width=2,stroke_fill='black')
   canvas=Image.new('RGB',(768,464),'#121d2e');d=ImageDraw.Draw(canvas);canvas.paste(original,(0,28));canvas.paste(overlay,(384,28));d.text((8,5),split+' · '+behavior.replace('_',' '),font=font,fill='white');d.text((392,5),f'B-KinD · peak > {a.threshold:g}',font=font,fill='white')
   d.text((8,418),f'Epoch {saved["epoch"]} | shown {int(visible.sum())}/{cfg["nkpts"]} | frame 08 | raw points, no GT mapping',font=small,fill='white')
   d.text((8,440),Path(selected['destination']).name[:100],font=small,fill='white')
   folder=a.output/split;folder.mkdir(exist_ok=True);name=behavior+'.jpg';canvas.save(folder/name,quality=93)
   record=dict(split=split,behavior=behavior,source=str(path.resolve()),source_clip=selected['source_clip'],preview=str(Path(split)/name),shown_indices=np.flatnonzero(visible).tolist(),heatmap_peak=peaks.tolist(),keypoints_xy_normalized=xy.tolist())
   (folder/(behavior+'.json')).write_text(json.dumps(record,indent=2));entries.append(record)
 for split in ['train','val']:
  chosen=[e for e in entries if e['split']==split];cols=2;scale=.75;w,h=576,348;canvas=Image.new('RGB',(cols*w,((len(chosen)+cols-1)//cols)*h),'#121d2e')
  for i,e in enumerate(chosen):canvas.paste(Image.open(a.output/e['preview']).resize((w,h),Image.Resampling.LANCZOS),(i%cols*w,i//cols*h))
  canvas.save(a.output/(split+'_contact_sheet.jpg'),quality=92)
 cards=''.join(f'<article><h2>{e["split"]} · {html.escape(e["behavior"])}</h2><a href="{e["preview"]}"><img src="{e["preview"]}" loading="lazy"></a></article>' for e in entries)
 (a.output/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Cow B-KinD high-confidence previews</title><style>body{font:16px system-ui;background:#101620;color:#edf3fc;margin:24px}a{color:#8ccaff}img{width:100%}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(600px,1fr));gap:20px}h2{font-size:18px}</style><h1>Cow B-KinD · high-confidence train/val samples</h1><p>120-epoch cow checkpoint, one frame per populated behavior in each split. Original input at left; discovered keypoints with normalized-heatmap peak >0.1 at right. Same threshold as the CalMS21 filtered previews. No GT mapping, coordinate modification, smoothing or manual point removal. High confidence alone does not guarantee the point is on an animal or anatomically useful.</p><p><a href="train_contact_sheet.jpg">Training contact sheet</a> · <a href="val_contact_sheet.jpg">Validation contact sheet</a></p><main>'+cards+'</main>')
 manifest=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=hashlib.file_digest(a.checkpoint.open('rb'),'sha256').hexdigest(),epoch=saved['epoch'],threshold=a.threshold,model_mode='eval',selection='middle sorted clip per split/behavior; frame 08',entries=entries)
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps(dict(samples=len(entries),train=sum(e['split']=='train' for e in entries),val=sum(e['split']=='val' for e in entries),mean_points_shown=float(np.mean([len(e['shown_indices']) for e in entries]))),indent=2))
if __name__=='__main__':main()
