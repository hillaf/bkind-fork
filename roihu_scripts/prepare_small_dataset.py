"""Copy a balanced temporal RGB subset, preserving existing train/val splits."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import shutil


def source_video(name):
    match=re.match(r'^(.*?_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})',name)
    return match.group(1) if match else name.split('_cow')[0]


def choose(clips,count,rng):
    groups=defaultdict(list)
    for clip in clips:
        # Balance source videos first, then cow tracks within those videos.
        groups[source_video(clip.name)].append(clip)
    keys=sorted(groups);rng.shuffle(keys);quotas=Counter()
    for _ in range(count):
        eligible=[key for key in keys if quotas[key]<len(groups[key])]
        key=min(eligible,key=lambda k:quotas[k]);quotas[key]+=1
    selected=[]
    for key in keys:
        tracks=defaultdict(list)
        for clip in groups[key]:tracks[clip.name.rsplit('_seq',1)[0]].append(clip)
        track_keys=sorted(tracks);rng.shuffle(track_keys);tquota=Counter()
        for _ in range(quotas[key]):
            available=[t for t in track_keys if tquota[t]<len(tracks[t])]
            t=min(available,key=lambda t:tquota[t]);tquota[t]+=1
        for track in track_keys:
            members=sorted(tracks[track],key=lambda p:int(p.name.rsplit('_seq',1)[1]))
            n=tquota[track]
            # Even spacing avoids concentrating sampling in adjacent windows.
            indices=[len(members)//2] if n==1 else [round(i*(len(members)-1)/(n-1)) for i in range(n)]
            selected.extend(members[i] for i in indices)
    assert len(selected)==count and len(set(selected))==count
    return sorted(selected)


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    src=a.source.resolve();out=a.output.resolve()
    if out==src or src in out.parents or out in src.parents:raise ValueError('Source/output trees must be separate')
    if out.exists():raise ValueError(f'Output already exists: {out}')
    rng=random.Random(a.seed);plan=[];counts={};videos={}
    for split,cap in [('train',50),('val',10)]:
        split_counts={};split_videos=set()
        for behavior in sorted((src/split).iterdir()):
            if not behavior.is_dir():continue
            clips=sorted(c for c in behavior.iterdir() if c.is_dir())
            selected=choose(clips,min(cap,len(clips)),rng) if clips else []
            split_counts[behavior.name]=len(selected)
            for clip in selected:
                files=sorted(f for f in clip.iterdir() if f.suffix.lower() in {'.jpg','.jpeg'})
                assert len(files)==16,(clip,len(files))
                assert [int(f.stem) for f in files]==list(range(16)),clip
                video=source_video(clip.name);split_videos.add(video)
                plan.append((split,behavior.name,clip,files,video))
        counts[split]=split_counts;videos[split]=sorted(split_videos)
    assert sum(counts['train'].values())==569 and sum(counts['val'].values())==93
    overlap=set(videos['train'])&set(videos['val'])
    if overlap:raise ValueError(f'Source videos overlap between splits: {sorted(overlap)}')
    out.mkdir(parents=True);manifest=[];total_bytes=0
    for split,behavior,clip,files,video in plan:
        destination=out/split/(behavior+'__'+clip.name);destination.mkdir(parents=True)
        frames=[]
        for file in files:
            target=destination/file.name
            shutil.copyfile(file,target)  # Independent regular file; no links.
            assert not target.is_symlink() and target.stat().st_ino!=file.stat().st_ino
            checksum=digest(file);assert digest(target)==checksum
            size=target.stat().st_size;total_bytes+=size
            frames.append(dict(file=file.name,bytes=size,sha256=checksum))
        manifest.append(dict(split=split,behavior=behavior,source_clip=str(clip),
            destination=destination.relative_to(out).as_posix(),source_video=video,frames=frames))
    summary=dict(seed=a.seed,source=str(src),output=str(out),copy_mode='independent regular-file copies',
        clips={s:sum(v.values()) for s,v in counts.items()},per_behavior=counts,
        frames={s:16*sum(v.values()) for s,v in counts.items()},rgb_bytes=total_bytes,
        source_fps=20,source_frame_stride=2,stored_frame_interval_seconds=.1,
        frame_gap=4,pair_interval_seconds=.4,clip_span_seconds=1.5,
        pairs={s:12*sum(v.values()) for s,v in counts.items()},source_videos=videos,
        source_video_split_overlap=sorted(overlap),verification='Every RGB file SHA256 matched; no links; original frame order 00..15')
    (out/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in manifest))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    config=Path(__file__).with_name('custom.yaml').read_text().replace('frame_gap: 6','frame_gap: 4')
    (out/'bkind.yaml').write_text(config)
    (out/'README.md').write_text('''# Small B-KinD temporal training dataset

569 training clips and 93 validation clips, selected from the existing splits.
Up to 50 clips per training behavior and 10 per validation behavior; rare classes
retain all available clips. Sampling balances source videos then cow tracks and
spaces selections over each track. Behavior labels guide sampling only and are
not supplied as training targets. No source video occurs in both selected splits.

All 16 RGB JPEGs per clip are independent file copies, byte-verified with SHA256.
Masks were not copied. Filenames, order and source 224×224 resolution are retained;
the training loader resizes to 256×256. Train/val each contain one directory per
clip, directly compatible with B-KinD. Do not point the loader at the original
behavior-nested tree. Source data was not modified.

Use frame_gap=4: at 20 source FPS and extraction stride 2, pairs are 0.4 seconds
apart. Each clip spans 1.5 seconds and supplies 12 pairs: 6,828 train and 1,116 val.
This timing assumes stride 2 means every second source frame, as documented.

manifest.jsonl records every source path, copied file and hash. summary.json
contains class counts, sizes, split checks and sampling seed 42. No hard links
or symbolic links are used. Retaining all rare clips means nearby overlapping
windows may remain; the fixed original source-video split is preserved.

Submit from the BKinD repository root, after transferring this dataset to CSC:

```bash
export DATA=/scratch/project_2015338/bkind-small-662
export CONFIG="$DATA/bkind.yaml"
export BKIND_VENV=/projappl/project_2015338/bkind-venv-aarch64
export OUTPUT=/scratch/project_2015338/BKind/BKinD/outputs/small-01
sbatch --account=project_2015338 roihu_scripts/train.sbatch
```

The configuration has no local absolute data path: DATA supplies the dataset root.
''')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
