"""Animate five actual frame pairs recorded in the training run metadata."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont,ImageOps
r=Path(__file__).resolve().parents[1];dataset=r.parent/'bkind-small-662'
out=r/'outputs/infer-2491633_diagnosis/frame_gap_06s';out.mkdir(parents=True,exist_ok=True)
run=json.loads((r/'outputs/train-metadata/run.json').read_text());assert run['config']['frame_gap']==6
behaviors=['drink','feed_head_up','groom_self','pushing','scratch_neck_rail'];records=[];panels=[[],[]]
font=ImageFont.load_default(size=19);small=ImageFont.load_default(size=12)
for n,behavior in enumerate(behaviors,1):
 pairs=[v for v in run['train_pair_paths'] if Path(v[0]).parent.name.startswith(behavior+'__') and Path(v[0]).name=='00.jpg']
 pairs.sort(key=lambda v:v[0]);pair=pairs[len(pairs)//2]
 local=[dataset/'train'/Path(p).parent.name/Path(p).name for p in pair[:2]]
 assert int(local[1].stem)-int(local[0].stem)==6
 frames=[]
 for i,path in enumerate(local):
  with Image.open(path) as img:im=img.convert('RGB')
  canvas=Image.new('RGB',(384,450),'#142031');draw=ImageDraw.Draw(canvas)
  draw.text((10,8),f'{n}. '+behavior.replace('_',' '),font=font,fill='white')
  canvas.paste(ImageOps.fit(im,(384,384)),(0,36))
  draw.rectangle((0,420,384,450),fill='#1479b8' if i==0 else '#b75917')
  draw.text((10,425),f'{path.name}  |  '+('t = 0.0 s' if i==0 else 't = 0.6 s'),font=font,fill='white')
  frames.append(canvas);panels[i].append(canvas)
 name=f'{n:02d}_{behavior}.gif'
 frames[0].save(out/name,save_all=True,append_images=frames[1:],duration=[600,600],loop=0,disposal=2,optimize=False)
 records.append(dict(behavior=behavior,training_pair=pair[:2],local_frames=[str(p) for p in local],animation=name))
montage=[]
for i in range(2):
 canvas=Image.new('RGB',(1152,940),'#0b1220');draw=ImageDraw.Draw(canvas)
 draw.text((14,8),'B-KinD training pairs: 0.6-second gap (stored frames 00 and 06)',font=font,fill='white')
 for j,im in enumerate(panels[i]):canvas.paste(im,((j%3)*384,36+(j//3)*450))
 draw.text((784,515),'Each example alternates two',font=font,fill='white');draw.text((784,545),'actual training frames.',font=font,fill='white');draw.text((784,595),'No interpolated frames.',font=font,fill='white');draw.text((784,645),'20 source FPS; extraction stride 2.',font=small,fill='white')
 montage.append(canvas)
montage[0].save(out/'five_examples.gif',save_all=True,append_images=montage[1:],duration=[600,600],loop=0,disposal=2,optimize=False)
montage[0].save(out/'frame_00_preview.png');montage[1].save(out/'frame_06_preview.png')
(out/'manifest.json').write_text(json.dumps(dict(frame_gap=6,pair_separation_seconds=.6,playback_duration_ms_per_frame=600,examples=records),indent=2))
(out/'README.md').write_text('''# Five actual training pairs, 0.6 seconds apart

Examples are selected from the exact temporal pairs recorded in train-metadata/run.json,
using five different behaviors and the middle sorted clip per behavior. They alternate
stored frames 00 and 06, corresponding to 0.6 seconds at 20 source FPS / stride 2.
Each frame displays for 600 ms; looping returns from the later frame to the earlier one.
These are two-frame pair animations, not continuous video; no frames are interpolated.
Images preserve the RGB input content, with enlarged display and labels only.

Open five_examples.gif for the combined view or the individual behavior GIFs.
manifest.json records original training paths and local source frames.
''')
print(out)
for p in out.glob('*.gif'):
 with Image.open(p) as im:assert im.n_frames==2 and im.info['duration']==600
 print(p.name,p.stat().st_size)
