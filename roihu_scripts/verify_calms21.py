"""Verify prepared-frame integrity, exact upstream pairing and loader semantics.

Full image/hash verification is sequential and can be throttled independently.
"""
import argparse
from collections import Counter
import hashlib,json,shutil,subprocess,time
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--full-hash',action='store_true');p.add_argument('--pause-every',type=int,default=128);a=p.parse_args();root=a.data.resolve()
summary=json.loads((root/'summary.json').read_text());assert summary['complete']
pairs=[json.loads(v) for v in (root/'train_pairs.jsonl').read_text().splitlines()];counts=Counter();seen=set()
for v in pairs:
 assert v['frame1_index']-v['frame0_index']==6 and v['frame0_index']%7==0
 assert v['source_video'].startswith('train/')
 assert abs(v['time1_seconds']-v['time0_seconds']-.2)<1e-9
 for key in ['frame0','frame1']:
  assert v[key] not in seen;seen.add(v[key]);assert (root/v[key]).is_file()
 counts[v['video']]+=1
assert len(pairs)==25977 and len(seen)==51954
verified=0;image_bytes=0
for info in summary['source_videos']:
 assert counts[info['id']]==len(range(0,info['metadata']['nb_frames']-6,7))
 marker=json.loads((root/'train'/info['id']/'complete.json').read_text())
 assert len(marker['frames'])==2*counts[info['id']]
 for item in marker['frames']:
  path=root/'train'/info['id']/item['name'];assert path.stat().st_size==item['bytes']
  with Image.open(path) as image:assert image.size==(256,256) and image.mode=='RGB';image.verify()
  if a.full_hash:
   with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
  verified+=1;image_bytes+=item['bytes']
  if verified%a.pause_every==0:time.sleep(.05)
  if verified%10000==0:print(f'Verified {verified}/51954 PNGs',flush=True)
# Decode source frames 0,6,7 independently; verify no accidental reindexing or fps change.
first=summary['source_videos'][0];video=first['video'];width=first['metadata']['width'];height=first['metadata']['height'];nbytes=width*height*3
cmd=['ffmpeg','-nostdin','-v','error','-threads','1','-i',video,'-filter_threads','1','-vf',"select='eq(n,0)+eq(n,6)+eq(n,7)'",'-vsync','0','-frames:v','3','-pix_fmt','rgb24','-threads','1','-f','rawvideo','pipe:1']
raw=subprocess.run(cmd,capture_output=True,check=True).stdout;assert len(raw)==3*nbytes
for j,index in enumerate([0,6,7]):
 ref=Image.frombytes('RGB',(width,height),raw[j*nbytes:(j+1)*nbytes]).resize((256,256),Image.Resampling.BILINEAR)
 with Image.open(root/'train'/first['id']/f'frame_{index:08d}.png') as actual:assert np.array_equal(np.array(ref),np.array(actual))
# Lightweight dataset/transform check; no neural-network training.
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataloader.pair_manifest_dataset import PairManifestDataset
from torchvision import transforms
tr=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
ds=PairManifestDataset(root,tr,256,6);assert len(ds)==25977
for i in [0,1,len(ds)//2,len(ds)-1]:
 sample=ds[i];assert len(sample)==9
 for k in [0,1,4,5,6]:assert tuple(sample[k].shape)==(3,256,256)
 assert Path(sample[8]).stem.endswith(f'{int(Path(sample[7]).stem.split("_")[-1])+6:08d}')
# Preview representative full-view temporal pairs from each selected video.
canvas=Image.new('RGB',(1024,5*290),'#142031');draw=ImageDraw.Draw(canvas)
for j,info in enumerate(summary['source_videos']):
 pair=next(v for v in pairs if v['video']==info['id']);x=(j%2)*512;y=(j//2)*290
 for k,key in enumerate(['frame0','frame1']):
  with Image.open(root/pair[key]) as image:canvas.paste(image,(x+k*256,y+24))
 draw.text((x+6,y+5),info['id']+' | t and t + 0.2s',fill='white')
canvas.save(root/'pair_preview.jpg',quality=90)
report=dict(verified_images=verified,pairs=len(pairs),image_bytes=image_bytes,image_gib=image_bytes/2**30,
 full_sha256_verified=a.full_hash,exact_source_reference_indices=[0,6,7],source_reference_pixel_match=True,
 upstream_pair_indices_match=True,unique_frames_no_pair_overlap=True,discovery_test_videos_used=0,
 loader_samples_checked=4,validation_split='none',free_gib=shutil.disk_usage(root).free/2**30)
(root/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
