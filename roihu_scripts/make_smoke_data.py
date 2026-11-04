"""Generate moving-shape clips for an execution check, not a learning benchmark."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
for split in ['train','val']:
    out=a.output/split/('clip_'+split);out.mkdir(parents=True,exist_ok=True)
    for i in range(5):
        img=Image.new('RGB',(64,64),(30,40,50));d=ImageDraw.Draw(img)
        x=7+i*5;d.ellipse((x,18,x+20,42),fill=(200,100 if split=='train' else 180,70));img.save(out/f'frame_{i:06}.png')
print(a.output)
