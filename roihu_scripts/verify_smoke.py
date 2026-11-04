"""Verify saved state and outputs from smoke_cpu.sh."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
root=a.output
train=torch.load(root/'train/last.pt',map_location='cpu',weights_only=True)
assert train['step']==2 and train['epoch']==1
for mode,step in [('finetune',1),('resume',3)]:
    saved=torch.load(root/mode/'last.pt',map_location='cpu',weights_only=True)
    assert saved['step']==step and saved['optimizer']['state']
    assert any(not torch.equal(train['state_dict'][key],value) for key,value in saved['state_dict'].items() if value.is_floating_point())
    del saved
for mode in ['train','finetune','resume']:
    run=json.loads((root/mode/'run.json').read_text())
    assert run['train_pairs']==run['val_pairs']==4
    assert run['config']['curriculum']==0
    train_paths={v[i] for v in run['train_pair_paths'] for i in [0,1]}
    val_paths={v[i] for v in run['val_pair_paths'] for i in [0,1]}
    assert not train_paths & val_paths
    for row in (root/mode/'history.jsonl').read_text().splitlines():
        record=json.loads(row);assert np.isfinite(record['train_loss']) and np.isfinite(record['val_mse'])
    assert (root/mode/'best.pt').is_file()
files=list((root/'inference').glob('*.json'));assert len(files)==5
for file in files:
    value=json.loads(file.read_text());k=np.array(value['keypoints_xy_normalized'])
    assert k.shape==(4,2) and np.isfinite(k).all() and (np.abs(k)<=1.00001).all()
assert len(list((root/'inference').glob('*.keypoints.png')))==5
print('Verified checkpoints, optimizer states, finite metrics, split separation and five inference outputs')
