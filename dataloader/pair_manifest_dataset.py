"""Explicit temporal pairs for sparsely stored CalMS21 discovery frames."""
import json
from pathlib import Path
from dataloader.custom_dataset import CustomDataset
from dataloader.data_utils import pil_loader

class PairManifestDataset(CustomDataset):
    def __init__(self,root,transform,image_size=256,frame_gap=6):
        root=Path(root).resolve()
        summary=json.loads((root/'summary.json').read_text())
        if not summary.get('complete'):raise ValueError('Dataset preparation is incomplete')
        if summary['frame_gap']!=frame_gap:raise ValueError('Config frame gap differs from pair manifest')
        self.samples=[]
        for line in (root/'train_pairs.jsonl').read_text().splitlines():
            pair=json.loads(line)
            if pair['frame1_index']-pair['frame0_index']!=frame_gap:raise ValueError('Invalid temporal pair gap')
            first=(root/pair['frame0']).resolve();second=(root/pair['frame1']).resolve()
            if root not in first.parents or root not in second.parents:raise ValueError('Pair path escapes dataset')
            if not first.is_file() or not second.is_file():raise ValueError('Missing pair image')
            self.samples.append([str(first),str(second),[0,0,0,0],[0,0,0,0]])
        if len(self.samples)!=summary['actual_pairs']:raise ValueError('Pair count mismatch')
        self.root=str(root);self.loader=pil_loader;self.transform=transform;self.target_transform=transform
        self._image_size=[image_size,image_size];self.crop_box=False
