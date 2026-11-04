"""Render original, discovered points, and linear-mapped predictions versus GT."""
import argparse,csv,html,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
COLORS=['#ff3030','#ffff00','#39ff14','#00ffff','#3980ff','#d740ff','#ff8c00','#ff4fa3']
NAMES=['nose','left_earbase','left_earend','right_earbase','right_earend','back_base','back_middle','tail_base']

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
 root=a.evaluation.resolve();dest=root/'visualizations';dest.mkdir(exist_ok=True)
 groups=defaultdict(list)
 for row in csv.DictReader((root/'predictions_and_errors.csv').open()):groups[row['sample_id']].append(row)
 font=ImageFont.load_default(size=16);small=ImageFont.load_default(size=12);entries=[]
 for sid,rows in sorted(groups.items()):
  path=a.dataset/'test/images'/(sid+'.png');image=Image.open(path).convert('RGB');raw=image.copy();mapped=image.copy()
  pred=json.loads((root/'original_predictions/test'/(sid+'.png.json')).read_text());xy=np.array(pred['keypoints_xy_pixels']);assert np.isfinite(xy).all()
  draw=ImageDraw.Draw(raw)
  for i,(x,y) in enumerate(xy):
   c=COLORS[i%len(COLORS)];draw.ellipse((x-14,y-14,x+14,y+14),fill=c,outline='white',width=3);draw.text((x+17,y-15),str(i),font=ImageFont.load_default(size=24),fill=c,stroke_width=3,stroke_fill='black')
  draw=ImageDraw.Draw(mapped);outside=0
  for row in rows:
   i=NAMES.index(row['bodypart']);c=COLORS[i];x,y=float(row['predicted_x']),float(row['predicted_y']);gx,gy=float(row['manual_x']),float(row['manual_y'])
   outside+=not(0<=x<image.width and 0<=y<image.height)
   draw.line((x,y,gx,gy),fill=c,width=3)
   draw.ellipse((gx-12,gy-12,gx+12,gy+12),outline='white',width=4)
   draw.line((gx-6,gy-6,gx+6,gy+6),fill=c,width=2);draw.line((gx-6,gy+6,gx+6,gy-6),fill=c,width=2)
   draw.ellipse((x-10,y-10,x+10,y+10),fill=c,outline='black',width=3)
   draw.text((x+13,y-15),str(i),font=ImageFont.load_default(size=24),fill=c,stroke_width=3,stroke_fill='black')
  canvas=Image.new('RGB',(960,414),'#121b29');dc=ImageDraw.Draw(canvas)
  for k,(title,img) in enumerate([('Input',image),('Discovered B-KinD points',raw),('Linear mapping vs GT',mapped)]):
   tile=ImageOps.contain(img,(312,312));canvas.paste(tile,(k*320+(320-tile.width)//2,27+(312-tile.height)//2));dc.text((k*320+5,5),title,font=font,fill='white')
  error=float(np.mean([float(r['error_px']) for r in rows]));dc.text((7,345),sid,font=small,fill='white');dc.text((7,365),f'Mean error: {error:.1f} px | usable GT: {len(rows)} | mapped points outside image: {outside}',font=small,fill='white');dc.text((7,386),'Right: filled dots = mapped predictions; white rings = GT; lines = errors. Raw indices are not anatomical labels.',font=small,fill='white')
  name=sid+'.jpg';canvas.save(dest/name,quality=93);entries.append(dict(sample_id=sid,image='visualizations/'+name,mean_error_px=error,usable_gt=len(rows),outside_predictions=outside))
 def sheet(selected,path):
  canvas=Image.new('RGB',(1920,math.ceil(len(selected)/2)*414),'#121b29')
  for i,item in enumerate(selected):canvas.paste(Image.open(root/item['image']),(i%2*960,i//2*414))
  canvas.save(path,quality=92)
 sheets=[]
 for i in range(0,len(entries),8):
  name=f'bkind_test_contact_sheet_{i//8+1:02}.jpg';sheet(entries[i:i+8],root/name);sheets.append(name)
 cameras={}
 for e in entries:cameras.setdefault(e['sample_id'].split('__instance-')[0],e)
 sheet(list(cameras.values()),root/'bkind_test_overview.jpg')
 legend=' · '.join(f'<span style="color:{c}">{i}: {html.escape(n.replace("_"," "))}</span>' for i,(n,c) in enumerate(zip(NAMES,COLORS)))
 cards=''.join(f'<article data-name="{html.escape(e["sample_id"])}" data-error="{e["mean_error_px"]}"><a href="{e["image"]}"><img loading="lazy" src="{e["image"]}"></a></article>' for e in entries)
 document='''<!doctype html><meta charset="utf-8"><title>B-KinD locked-test evaluation</title><style>body{font:16px system-ui;background:#101620;color:#e8edf5;margin:24px}a{color:#8ccaff}p{line-height:1.6}input,select{padding:10px;background:#263346;color:white;border:1px solid #789;margin:8px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(650px,1fr));gap:16px}article{background:#121b29;border-radius:8px}img{width:100%}</style><h1>B-KinD · linear mapping · 33 locked-test crops</h1><p>Each comparison shows the input, the eight discovered points, and the train-only linear mapping against usable GT. On the right: colored dots are predictions; white rings are GT; lines show errors. Raw discovered-point indices in the middle panel have no assigned anatomical identity.</p><p>Right-panel anatomical index/color legend: '''+legend+'''</p><p>104 original training images fitted the mapping; 33 original test images / 256 GT points were scored. Augmented copies, masks and full-scene images were excluded.</p><p><a href="bkind_test_overview.jpg">Overview</a> · <a href="REPORT.md">Evaluation report</a></p><input id="search" placeholder="Filter camera or sample"><select id="sort"><option value="name">Sample order</option><option value="worst">Largest error first</option><option value="best">Smallest error first</option></select><main>'''+cards+'''</main><script>let main=document.querySelector('main');document.querySelector('#search').oninput=e=>{let q=e.target.value.toLowerCase();main.querySelectorAll('article').forEach(a=>a.hidden=!a.dataset.name.toLowerCase().includes(q))};document.querySelector('#sort').onchange=e=>{let cards=[...main.children];cards.sort((a,b)=>e.target.value==='name'?a.dataset.name.localeCompare(b.dataset.name):e.target.value==='worst'?b.dataset.error-a.dataset.error:a.dataset.error-b.dataset.error);cards.forEach(a=>main.appendChild(a))};</script>'''
 (root/'gallery.html').write_text(document);(root/'visualization_manifest.json').write_text(json.dumps(dict(images=len(entries),source_groups=len(cameras),entries=entries,contact_sheets=sheets),indent=2))
 assert len(entries)==33
 print(f'Created {len(entries)} comparisons, {len(sheets)} contact sheets, overview and gallery')
if __name__=='__main__':main()
