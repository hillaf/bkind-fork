"""Prepare the upstream CalMS21 cable-free discovery subset, one video at a time.

Only source indices 7*i and 7*i+6 are retained. A manifest, rather than image
adjacency, defines pairs. Entire camera views are resized exactly once with
Pillow's bilinear resize, and saved as lossless RGB PNGs. No test video is decoded.
"""
import argparse
from fractions import Fraction
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from PIL import Image

IDS=['mouse001','mouse002','mouse051','mouse052','mouse055','mouse056','mouse058','mouse053','mouse054','mouse057']

def probe(path):
    data=json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,codec_name',
        '-of','json',str(path)],capture_output=True,text=True,check=True).stdout)['streams'][0]
    if Fraction(data['avg_frame_rate'])!=30 or Fraction(data['r_frame_rate'])!=30:
        raise ValueError(f'Expected constant 30 FPS: {path}: {data}')
    if data.get('nb_frames') in (None,'N/A'):raise ValueError(f'Missing frame count: {path}')
    data['nb_frames']=int(data['nb_frames']);return data

def read_exact(stream,size):
    parts=[];remaining=size
    while remaining:
        part=stream.read(remaining)
        if not part:raise RuntimeError(f'Decoder ended early; missing {remaining} bytes')
        parts.append(part);remaining-=len(part)
    return b''.join(parts)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['inventory','prepare'],default='inventory')
    p.add_argument('--reserve-gib',type=float,default=30)
    p.add_argument('--max-output-gib',type=float,default=10)
    p.add_argument('--pause-every',type=int,default=32);p.add_argument('--pause-seconds',type=float,default=.1)
    a=p.parse_args();source=a.videos.resolve();out=a.output.resolve()
    if a.reserve_gib<1 or a.max_output_gib<=0 or a.pause_every<1 or a.pause_seconds<0:raise ValueError('Invalid resource bounds')
    if source==out or source in out.parents or out in source.parents:raise ValueError('Separate output tree required')
    paths=sorted(path for path in (source/'train').glob('*.mp4') if path.name[:8] in IDS)
    if len(paths)!=10 or {p.name[:8] for p in paths}!=set(IDS):raise ValueError('Expected all 10 upstream cable-free IDs, one video per ID')
    inventory=[]
    for path in paths:
        info=probe(path);starts=list(range(0,info['nb_frames']-6,7))
        inventory.append(dict(video=str(path),relative_video=path.relative_to(source).as_posix(),id=path.stem,
            source_size_bytes=path.stat().st_size,source_mtime_ns=path.stat().st_mtime_ns,metadata=info,
            pairs=len(starts),stored_frames=2*len(starts)))
    summary=dict(source_root=str(source),selected_video_ids=sorted(IDS),source_videos=inventory,
        train_video_count=len(paths),test_video_count_used=0,frame_gap=6,pair_start_stride=7,source_fps=30,
        pair_seconds=.2,image_size=[256,256],resize='full view to square; PIL bilinear; antialias; no crop/no mask',
        encoding='lossless RGB PNG, compress_level=1',pairs=sum(v['pairs'] for v in inventory),
        stored_frames=sum(v['stored_frames'] for v in inventory),disk_reserve_gib=a.reserve_gib,
        max_output_gib=a.max_output_gib,resource_policy=dict(simultaneous_videos=1,decoder_threads=1,filter_threads=1,pause_every_frames=a.pause_every,pause_seconds=a.pause_seconds))
    if a.mode=='inventory':print(json.dumps(summary,indent=2));return
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.prepare.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another preparer is already writing this output')
    if (out/'inventory.json').exists():
        old=json.loads((out/'inventory.json').read_text())
        if old['source_videos']!=inventory:raise ValueError('Source inventory changed; use a new output')
    (out/'inventory.json').write_text(json.dumps(summary,indent=2))
    allocated=sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
    def bounds():
        free=shutil.disk_usage(out).free
        if free<a.reserve_gib*2**30:raise RuntimeError(f'Disk reserve reached: {free/2**30:.2f} GiB free')
        if allocated>a.max_output_gib*2**30:raise RuntimeError('Output size budget reached')
    bounds();all_pairs=[]
    for vi,video in enumerate(inventory,1):
        video_dir=out/'train'/video['id'];video_dir.mkdir(parents=True,exist_ok=True)
        starts=list(range(0,video['metadata']['nb_frames']-6,7));indices=sorted(n for i in starts for n in (i,i+6))
        names={n:video_dir/f'frame_{n:08d}.png' for n in indices}
        pair_rows=[dict(video=video['id'],source_video=video['relative_video'],frame0_index=i,frame1_index=i+6,
            frame0=names[i].relative_to(out).as_posix(),frame1=names[i+6].relative_to(out).as_posix(),time0_seconds=i/30,time1_seconds=(i+6)/30) for i in starts]
        marker=video_dir/'complete.json'
        if marker.exists():
            complete=json.loads(marker.read_text())
            if complete['source']!=video or len(complete['frames'])!=len(indices):raise ValueError('Completion marker mismatch')
            for item in complete['frames']:
                path=video_dir/item['name']
                if not path.is_file() or path.stat().st_size!=item['bytes']:raise ValueError(f'Completed frame missing/changed: {path}')
            all_pairs.extend(pair_rows);print(f'[{vi}/10] already complete: {video["id"]}',flush=True);continue
        # select is evaluated at the ORIGINAL frame rate; do not apply an fps filter.
        command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-threads','1','-i',video['video'],
            '-an','-sn','-dn','-filter_threads','1','-vf',"select='eq(mod(n,7),0)+eq(mod(n,7),6)'",'-vsync','0',
            '-frames:v',str(len(indices)),'-pix_fmt','rgb24','-threads','1','-f','rawvideo','pipe:1']
        frame_bytes=video['metadata']['width']*video['metadata']['height']*3
        print(f'[{vi}/10] {video["id"]}: {len(indices)} frames / {len(starts)} pairs',flush=True)
        error_file=video_dir/'decode.log';frames=[]
        with error_file.open('wb') as err:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=err)
            try:
                for j,index in enumerate(indices):
                    if j%a.pause_every==0:bounds()
                    raw=read_exact(process.stdout,frame_bytes)
                    image=Image.frombytes('RGB',(video['metadata']['width'],video['metadata']['height']),raw).resize((256,256),Image.Resampling.BILINEAR)
                    target=names[index];temp=target.with_suffix('.png.tmp');old_size=target.stat().st_size if target.exists() else 0
                    image.save(temp,format='PNG',compress_level=1);temp.replace(target)
                    size=target.stat().st_size;allocated+=size-old_size
                    with target.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
                    frames.append(dict(name=target.name,index=index,bytes=size,sha256=checksum))
                    if (j+1)%a.pause_every==0:time.sleep(a.pause_seconds)
                    if (j+1)%1000==0:print(f'  {j+1}/{len(indices)} | output {allocated/2**30:.2f} GiB | free {shutil.disk_usage(out).free/2**30:.1f} GiB',flush=True)
                process.stdout.close();status=process.wait(timeout=60)
                if status:raise RuntimeError(f'ffmpeg exited {status}; inspect {error_file}')
            except BaseException:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
        marker.write_text(json.dumps(dict(source=video,frames=frames),indent=2));all_pairs.extend(pair_rows)
    manifest=out/'train_pairs.jsonl';temp=manifest.with_suffix('.jsonl.tmp')
    temp.write_text(''.join(json.dumps(pair)+'\n' for pair in all_pairs));temp.replace(manifest)
    summary['actual_image_bytes']=sum(p.stat().st_size for p in (out/'train').rglob('*.png'))
    summary['actual_pairs']=len(all_pairs);summary['complete']=True
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(f'COMPLETE: {len(all_pairs)} pairs; {summary["actual_image_bytes"]/2**30:.3f} GiB PNGs; {out}',flush=True)
if __name__=='__main__':main()
