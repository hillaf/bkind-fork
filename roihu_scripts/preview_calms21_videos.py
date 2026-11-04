"""Sample CalMS21 videos and render frozen B-KinD predictions, without GT scoring."""
import argparse
import colorsys
from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import load_weights,transform
from model.unsupervised_model import Model


def metadata(path):
    v=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_frames','-show_entries','format=duration','-of','json',str(path)]))
    return v['streams'][0],float(v['format']['duration'])


def read_frame(stream,size):
    data=bytearray()
    while len(data)<size:
        part=stream.read(size-len(data))
        if not part:break
        data.extend(part)
    if not data:return None
    if len(data)!=size:raise RuntimeError('Partial decoded frame')
    return bytes(data)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--videos',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=10);p.add_argument('--fps',type=int,default=10);p.add_argument('--batch-size',type=int,default=5)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True);cfg=saved['config']
    model=Model(cfg['nkpts'],pretrained=False,output_shape=(cfg['image_size']//4,)*2);load_weights(model,saved);model.eval();t=transform(cfg['image_size'])
    inventory=json.loads(a.inventory.read_text());training=sorted(a.videos/'train'/Path(v['video']).name for v in inventory['source_videos'])
    heldout=sorted((a.videos/'test').glob('*.mp4'));assert len(training)==10 and not {p.stem for p in training}&{p.stem for p in heldout}
    chosen=[('train',training[i],fraction) for i,fraction in zip([0,len(training)//2,len(training)-1],[.25,.5,.75])]
    chosen += [('test',heldout[i],fraction) for i,fraction in zip([0,len(heldout)//2,len(heldout)-1],[.25,.5,.75])]
    colors=[tuple(round(c*255) for c in colorsys.hsv_to_rgb(i/cfg['nkpts'],.85,1)) for i in range(cfg['nkpts'])]
    font=ImageFont.load_default(size=17);small=ImageFont.load_default(size=12);entries=[];gif_samples=[]
    for split,path,fraction in chosen:
        info,duration=metadata(path);width,height=info['width'],info['height'];source_fps=float(Fraction(info['r_frame_rate']));stride=round(source_fps/a.fps)
        if abs(source_fps/stride-a.fps)>1e-6:raise ValueError('Requested sampling FPS must divide source FPS')
        start_frame=round(max(0,(duration-a.seconds)*fraction)*source_fps);start=start_frame/source_fps;count=round(a.seconds*a.fps)
        folder=a.output/split;folder.mkdir(exist_ok=True);base=folder/path.stem
        panelw=512;panelh=round(height/width*panelw);outw=2*panelw;outh=panelh+72
        if outh%2:outh+=1
        decoder=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-ss',str(start),'-i',str(path),'-an','-sn','-vf',f'select=not(mod(n\\,{stride}))','-frames:v',str(count),'-fps_mode','passthrough','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        encoder=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{outw}x{outh}','-framerate',str(a.fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(base)+'.mp4'],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        all_points=[];all_peak=[];all_entropy=[];thumbnails=[];i=0
        with torch.inference_mode():
            while i<count:
                images=[]
                for _ in range(min(a.batch_size,count-i)):
                    data=read_frame(decoder.stdout,width*height*3)
                    if data is None:break
                    images.append(Image.frombytes('RGB',(width,height),data))
                if not images:break
                tensor=torch.stack([t(img) for img in images]);(ux,uy),logits=model(tensor)
                points=torch.stack((ux,uy),-1).cpu().numpy();prob=logits.flatten(2).softmax(-1);peaks=prob.amax(-1).cpu().numpy();ent=(-(prob*prob.clamp_min(1e-30).log()).sum(-1)/np.log(prob.shape[-1])).cpu().numpy()
                if not np.isfinite(points).all():raise RuntimeError('Nonfinite keypoints')
                for img,kp,peak,entropy in zip(images,points,peaks,ent):
                    original=img.resize((panelw,panelh),Image.Resampling.BILINEAR);overlay=original.copy();draw=ImageDraw.Draw(overlay)
                    xy=(kp+1)*.5*np.array([panelw-1,panelh-1])
                    for j,(x,y) in enumerate(xy):
                        draw.ellipse((x-6,y-6,x+6,y+6),fill=colors[j],outline='black',width=2)
                        draw.text((x+8,y-10),str(j),font=font,fill=colors[j],stroke_width=2,stroke_fill='black')
                    canvas=Image.new('RGB',(outw,outh),'#121d2e');canvas.paste(original,(0,26));canvas.paste(overlay,(panelw,26));dc=ImageDraw.Draw(canvas)
                    dc.text((8,5),'Original · '+split+' · '+path.stem,font=font,fill='white');dc.text((panelw+8,5),'B-KinD discovered keypoints',font=font,fill='white')
                    dc.text((8,panelh+32),f'Source time {start+i/a.fps:.2f}s | {cfg["nkpts"]} points | {a.fps} sampled FPS | no GT mapping',font=small,fill='white')
                    for j,color in enumerate(colors):dc.text((panelw+8+j*37,panelh+32),str(j),font=font,fill=color)
                    encoder.stdin.write(canvas.tobytes());all_points.append(kp);all_peak.append(peak);all_entropy.append(entropy)
                    if i==0:canvas.save(str(base)+'.poster.jpg',quality=92)
                    if i%2==0 and i<40:thumbnails.append(overlay.resize((384,round(panelh*384/panelw)),Image.Resampling.LANCZOS))
                    i+=1
        encoder.stdin.close();encode_error=encoder.stderr.read().decode();encode_status=encoder.wait();decode_error=decoder.stderr.read().decode();decode_status=decoder.wait()
        if encode_status or decode_status:raise RuntimeError(encode_error+'\n'+decode_error)
        if i!=count:raise RuntimeError(f'Decoded {i}/{count} frames')
        points=np.asarray(all_points);np.savez_compressed(str(base)+'.keypoints.npz',keypoints_xy_normalized=points,heatmap_peak=all_peak,heatmap_normalized_entropy=all_entropy,source_frame_indices=start_frame+np.arange(i)*stride,source_times_seconds=start+np.arange(i)/a.fps)
        entry=dict(split=split,source=str(path.resolve()),source_fps=source_fps,source_dimensions=[width,height],start_frame=start_frame,start_seconds=start,seconds=i/a.fps,frames=i,sampling_stride=stride,prediction_fps=a.fps,
                   video=str(Path(split)/(path.stem+'.mp4')),poster=str(Path(split)/(path.stem+'.poster.jpg')),coordinates=str(Path(split)/(path.stem+'.keypoints.npz')),mean_peak=np.mean(all_peak,axis=0).tolist(),std_keypoints_normalized=points.std(0).tolist())
        Path(str(base)+'.json').write_text(json.dumps(entry,indent=2));entries.append(entry);gif_samples.append(thumbnails)
        print(json.dumps(dict(split=split,video=path.name,frames=i,start_seconds=start)),flush=True)
    # Lightweight 4-second montage: same timeline, sampled predictions at 5 FPS.
    montage=[];h=gif_samples[0][0].height;cellh=h+32
    for frame in range(min(map(len,gif_samples))):
        canvas=Image.new('RGB',(1152,2*cellh),'#121d2e');dc=ImageDraw.Draw(canvas)
        for j,images in enumerate(gif_samples):
            x,y=j%3*384,j//3*cellh;canvas.paste(images[frame],(x,y+32));dc.text((x+7,y+8),entries[j]['split']+' · '+Path(entries[j]['source']).stem,font=small,fill='white')
        montage.append(canvas)
    montage[0].save(a.output/'prediction_montage.gif',save_all=True,append_images=montage[1:],duration=200,loop=0,disposal=2)
    montage[0].save(a.output/'prediction_overview.jpg',quality=93)
    cards=''.join(f'<article><h2>{e["split"]} · {html.escape(Path(e["source"]).stem)}</h2><p>Source interval {e["start_seconds"]:.1f}–{e["start_seconds"]+e["seconds"]:.1f}s</p><video controls preload="metadata" poster="{e["poster"]}" src="{e["video"]}"></video><p><a href="{e["video"]}">MP4</a> · <a href="{e["coordinates"]}">Coordinates and heatmap diagnostics</a></p></article>' for e in entries)
    (a.output/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>B-KinD CalMS21 previews</title><style>body{font:16px system-ui;background:#101620;color:#edf3fc;margin:24px}a{color:#8ccaff}video{width:100%;max-width:1100px}article{margin:32px 0}h2{font-size:20px}</style><h1>B-KinD · CalMS21 prediction previews</h1><p>Frozen final checkpoint after 15 epochs. Three sampled training videos and three official held-out test videos; this training run had no validation split. Ten indexed keypoints belong to the full scene and have no predefined anatomical or mouse identity assignments. Colors are fixed by keypoint index. All points are shown; none are suppressed by confidence. Clips run in real time at 10 sampled FPS. No GT mapping or cow data is used.</p>'+cards)
    manifest=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=hashlib.file_digest(a.checkpoint.open('rb'),'sha256').hexdigest(),epoch=saved['epoch'],step=saved['step'],config=cfg,device='cpu',training_videos_available=len(training),heldout_test_videos_available=len(heldout),sampled_clips=entries)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Preview gallery complete',flush=True)
if __name__=='__main__':main()
