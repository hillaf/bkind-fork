"""Render saved CalMS21 predictions with a fixed confidence threshold; no model rerun."""
import argparse,colorsys,html,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def read_frame(stream,size):
    data=bytearray()
    while len(data)<size:
        part=stream.read(size-len(data))
        if not part:break
        data.extend(part)
    if not data:return None
    if len(data)!=size:raise ValueError('Partial frame')
    return bytes(data)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previews',type=Path,required=True);p.add_argument('--threshold',type=float,default=.1);a=p.parse_args()
    root=a.previews.resolve();manifest=json.loads((root/'manifest.json').read_text());out=root/'high_confidence';out.mkdir(exist_ok=True)
    colors=[tuple(round(c*255) for c in colorsys.hsv_to_rgb(i/10,.85,1)) for i in range(10)];font=ImageFont.load_default(size=17);small=ImageFont.load_default(size=12);entries=[];thumbs=[];all_peaks=[]
    for clip in manifest['sampled_clips']:
        arrays=np.load(root/clip['coordinates']);points=arrays['keypoints_xy_normalized'];confidence=arrays['heatmap_peak'];all_peaks.append(confidence)
        source=root/clip['video'];info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height','-of','json',str(source)]))['streams'][0];w,h=info['width'],info['height'];panelw=w//2;panelh=round(clip['source_dimensions'][1]/clip['source_dimensions'][0]*panelw)
        folder=out/clip['split'];folder.mkdir(exist_ok=True);stem=Path(clip['source']).stem;base=folder/stem
        decoder=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-i',str(source),'-pix_fmt','rgb24','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        encoder=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate',str(clip['prediction_fps']),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(base)+'.mp4'],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        samples=[];counts=[]
        for i,(kp,peaks) in enumerate(zip(points,confidence)):
            data=read_frame(decoder.stdout,w*h*3)
            if data is None:raise RuntimeError('Short preview')
            frame=Image.frombytes('RGB',(w,h),data);original=frame.crop((0,26,panelw,26+panelh));overlay=original.copy();draw=ImageDraw.Draw(overlay);xy=(kp+1)*.5*np.array([panelw-1,panelh-1]);visible=peaks>a.threshold;counts.append(int(visible.sum()))
            for j,(x,y) in enumerate(xy):
                if not visible[j]:continue
                draw.ellipse((x-6,y-6,x+6,y+6),fill=colors[j],outline='black',width=2)
                draw.text((x+8,y-10),str(j),font=font,fill=colors[j],stroke_width=2,stroke_fill='black')
            canvas=Image.new('RGB',(w,h),'#121d2e');canvas.paste(original,(0,26));canvas.paste(overlay,(panelw,26));dc=ImageDraw.Draw(canvas)
            dc.text((8,5),'Original · '+clip['split']+' · '+stem,font=font,fill='white');dc.text((panelw+8,5),f'High confidence only (peak > {a.threshold:g})',font=font,fill='white')
            dc.text((8,panelh+32),f'Time {clip["start_seconds"]+i/clip["prediction_fps"]:.2f}s | shown {int(visible.sum())}/10 | unsmoothed predictions',font=small,fill='white')
            for j,c in enumerate(colors):dc.text((panelw+8+j*37,panelh+32),str(j),font=font,fill=c)
            encoder.stdin.write(canvas.tobytes())
            if i==0:canvas.save(str(base)+'.poster.jpg',quality=93)
            if i%2==0 and i<40:samples.append(overlay.resize((384,round(panelh*384/panelw)),Image.Resampling.LANCZOS))
        encoder.stdin.close();ee=encoder.stderr.read().decode();es=encoder.wait();de=decoder.stderr.read().decode();ds=decoder.wait()
        if es or ds:raise RuntimeError(ee+de)
        entry=dict(split=clip['split'],video=str(Path(clip['split'])/(stem+'.mp4')),poster=str(Path(clip['split'])/(stem+'.poster.jpg')),source_video=clip['source'],start_seconds=clip['start_seconds'],frames=len(points),threshold=a.threshold,mean_shown=float(np.mean(counts)),min_shown=min(counts),max_shown=max(counts))
        entries.append(entry);thumbs.append(samples);print(json.dumps(entry),flush=True)
    montage=[];cellh=thumbs[0][0].height+32
    for i in range(min(map(len,thumbs))):
        canvas=Image.new('RGB',(1152,cellh*2),'#121d2e');dc=ImageDraw.Draw(canvas)
        for j,images in enumerate(thumbs):
            x,y=j%3*384,j//3*cellh;canvas.paste(images[i],(x,y+32));dc.text((x+7,y+8),entries[j]['split']+' · '+Path(entries[j]['source_video']).stem,font=small,fill='white')
        montage.append(canvas)
    montage[0].save(out/'prediction_montage.gif',save_all=True,append_images=montage[1:],duration=200,loop=0,disposal=2);montage[0].save(out/'prediction_overview.jpg',quality=93)
    peaks=np.concatenate(all_peaks);summary=dict(threshold=a.threshold,threshold_source='Released extract_features.py default conf_threshold=0.1; paper qualitative displays high-confidence points only',note='Coordinates unchanged, no smoothing, hiding only. Decoded originals taken from prior preview left panels, adding one video encoding generation.',median_peak_by_index=np.median(peaks,axis=0).tolist(),fraction_shown_by_index=(peaks>a.threshold).mean(0).tolist(),mean_points_shown=float((peaks>a.threshold).sum(1).mean()),clips=entries)
    (out/'manifest.json').write_text(json.dumps(summary,indent=2))
    cards=''.join(f'<article><h2>{e["split"]} · {html.escape(Path(e["source_video"]).stem)}</h2><p>Mean displayed: {e["mean_shown"]:.2f} of 10 points</p><video controls preload="metadata" poster="{e["poster"]}" src="{e["video"]}"></video></article>' for e in entries)
    (out/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>CalMS21 high-confidence predictions</title><style>body{font:16px system-ui;background:#101620;color:#edf3fc;margin:24px}a{color:#8ccaff}video{width:100%;max-width:1100px}article{margin:28px 0}</style><h1>CalMS21 · high-confidence B-KinD points</h1><p>Same predictions as the unfiltered previews, now showing only normalized-heatmap peaks above 0.1 (the released extraction-script default). The paper shows high-confidence points for qualitative figures; this threshold is a stated display choice, not proof of reproduction. Ten points total for both mice. No anatomical labels, GT filtering, temporal smoothing or model changes.</p><p><a href="../gallery.html">Unfiltered comparisons</a> · <a href="prediction_montage.gif">Animated overview</a></p>'+cards)
    print('Filtered visualization complete')
if __name__=='__main__':main()
