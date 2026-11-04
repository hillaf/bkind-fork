"""Single-device B-KinD training, weights-only fine-tuning, resume and inference."""
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from model.unsupervised_model import Model
from dataloader.custom_dataset import CustomDataset
from dataloader.pair_manifest_dataset import PairManifestDataset
from dataloader.data_utils import pil_loader
from loss.compute_loss import computeLoss
from loss.ssim_loss import compute_ssim

DEFAULTS=dict(nkpts=8,image_size=256,frame_gap=6,batch_size=4,workers=4,epochs=30,
    lr=.001,momentum=.9,weight_decay=.0001,schedule=30,curriculum=4,
    perc_weight=[100.,1.6,2.3,1.8,2.8,100.],pretrained=True,seed=0,
    dataset_format='custom',validation_split='val')

def transform(size):
    return transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def checkpoint_load(path):
    return torch.load(path,map_location='cpu',weights_only=True)

def load_weights(model,checkpoint):
    state=checkpoint.get('state_dict',checkpoint)
    state={k.removeprefix('module.'):v for k,v in state.items()}
    model.load_state_dict(state,strict=True)

def dataset(config,split):
    if config.get('dataset_format','custom') == 'calms21_pairs':
        if split != 'train':raise ValueError('Discovery manifest contains training only; no test validation')
        return PairManifestDataset(config['data'],transform(config['image_size']),config['image_size'],config['frame_gap'])
    if config.get('dataset_format','custom') != 'custom':raise ValueError('Unknown dataset_format')
    root=Path(config['data'])/split
    if not root.is_dir():raise ValueError(f'Missing split: {root}')
    t=transform(config['image_size'])
    return CustomDataset(str(root),transform=t,target_transform=t,loader=pil_loader,
        image_size=[config['image_size']]*2,frame_gap=config['frame_gap'])

def save_checkpoint(out,model,optimizer,config,epoch,step,best,filename='last.pt'):
    temp=out/(filename+'.tmp')
    torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
        config=config,epoch=epoch,step=step,best_loss=best),temp)
    temp.replace(out/filename)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','check','download-weights','infer'])
    p.add_argument('--config',type=Path);p.add_argument('--data',type=Path)
    p.add_argument('--output',type=Path,default=Path('outputs/bkind'))
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    group=p.add_mutually_exclusive_group();group.add_argument('--resume',type=Path);group.add_argument('--finetune',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--images',type=Path)
    for key in ['epochs','batch_size','workers','image_size','frame_gap','nkpts','curriculum']:
        p.add_argument('--'+key.replace('_','-'),type=int)
    p.add_argument('--lr',type=float);p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--limit-batches',type=int,default=0,help='Limit both train and validation batches for smoke checks')
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if a.threads < 1 or a.max_steps < 0 or a.limit_batches < 0:
        p.error('threads must be positive and smoke limits nonnegative')
    torch.set_num_threads(a.threads)
    if a.mode=='download-weights':
        from model.resnet_updated import model_urls
        from model.vggforLoss import model_urls as vgg_urls
        for url in [model_urls['resnet50'],vgg_urls['vgg19']]:
            torch.hub.load_state_dict_from_url(url,progress=False,check_hash=True)
        print('ResNet50 and VGG19 weights cached');return
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: use a GPU allocation and ARM venv')
    if a.mode=='check':
        print(json.dumps(dict(torch=torch.__version__,device=str(device),cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None)));return
    config=DEFAULTS.copy()
    if a.config:
        loaded=yaml.safe_load(a.config.read_text()) or {}
        unknown=set(loaded)-set(DEFAULTS)-{'data'}
        if unknown:raise ValueError(f'Unknown config keys: {unknown}')
        config.update(loaded)
    for key in DEFAULTS:
        if hasattr(a,key) and getattr(a,key) is not None:config[key]=getattr(a,key)
    if a.data:config['data']=str(a.data.resolve())
    if a.mode=='infer':
        if not a.checkpoint or not a.images:p.error('infer requires --checkpoint and --images')
        saved=checkpoint_load(a.checkpoint)
        if 'config' in saved:config.update(saved['config'])
        model=Model(config['nkpts'],pretrained=False,output_shape=(config['image_size']//4,)*2).to(device)
        load_weights(model,saved);model.eval();a.output.mkdir(parents=True,exist_ok=True)
        paths=sorted(f for f in a.images.rglob('*') if f.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'})
        if not paths:raise ValueError('No inference images')
        t=transform(config['image_size'])
        with torch.inference_mode():
            for path in paths:
                img=Image.open(path).convert('RGB');x=t(img).unsqueeze(0).to(device)
                ux,uy=model.get_keypoints(x);points=torch.stack((ux[0],uy[0]),dim=-1).cpu().numpy()
                if not np.isfinite(points).all():raise RuntimeError('Nonfinite inference')
                xy=(points+1)/2*np.array([img.width-1,img.height-1]);draw=ImageDraw.Draw(img)
                for j,(px,py) in enumerate(xy):
                    color=['red','yellow','lime','cyan','orange','magenta','dodgerblue','white'][j%8]
                    draw.ellipse((px-5,py-5,px+5,py+5),fill=color,outline='black')
                base=a.output/path.relative_to(a.images);base.parent.mkdir(parents=True,exist_ok=True)
                img.save(str(base)+'.keypoints.png')
                Path(str(base)+'.json').write_text(json.dumps(dict(source=str(path.resolve()),keypoints_xy_normalized=points.tolist(),keypoints_xy_pixels=xy.tolist()),indent=2))
        print(f'Inferred {len(paths)} images');return
    if 'data' not in config:p.error('train requires --data or data in YAML')
    for name in ['nkpts','frame_gap','batch_size','epochs','schedule']:
        if config[name]<1:raise ValueError(f'{name} must be positive')
    if config['image_size']<64 or config['image_size']%64:raise ValueError('image_size must be a multiple of 64, at least 64')
    config['data']=str(Path(config['data']).resolve())
    if config['workers'] < 0:raise ValueError('workers must be nonnegative')
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    if config['validation_split'] not in ('val','none'):raise ValueError('validation_split must be val or none')
    if config['dataset_format']=='calms21_pairs' and config['validation_split']!='none':raise ValueError('CalMS21 paper discovery uses train only')
    train=dataset(config,'train');val=dataset(config,'val') if config['validation_split']=='val' else None
    train_paths={str(Path(s[i]).resolve()) for s in train.samples for i in [0,1]}
    val_paths={str(Path(s[i]).resolve()) for s in val.samples for i in [0,1]} if val else set()
    if train_paths & val_paths:raise ValueError('Train/validation paths overlap')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not a.resume:raise ValueError('Output already has a checkpoint; use a new output or --resume')
    model=Model(config['nkpts'],pretrained=bool(config['pretrained'] and not(a.resume or a.finetune)),
        output_shape=(config['image_size']//4,)*2).to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=config['lr'],momentum=config['momentum'],weight_decay=config['weight_decay'])
    epoch0=0;step=0;best=float('inf')
    if a.resume or a.finetune:
        saved=checkpoint_load(a.resume or a.finetune);load_weights(model,saved)
        if a.resume:
            if 'config' not in saved:raise ValueError('Legacy checkpoint: use --finetune; exact resume needs runner config')
            for key in ['nkpts','image_size','frame_gap','curriculum','perc_weight','lr','momentum','weight_decay','schedule','data','batch_size','dataset_format','validation_split']:
                if saved['config'].get(key,DEFAULTS.get(key))!=config[key]:raise ValueError(f'Resume config mismatch: {key}')
            optimizer.load_state_dict(saved['optimizer']);epoch0=saved['epoch'];step=saved.get('step',0);best=saved['best_loss']
    if epoch0 >= config['epochs']:raise ValueError('epochs must exceed the resumed epoch')
    if a.max_steps and step >= a.max_steps:raise ValueError('max-steps must exceed the resumed step')
    loss=computeLoss(SimpleNamespace(**config),device=device)
    loaders=[DataLoader(ds,batch_size=config['batch_size'],shuffle=(i==0),num_workers=config['workers'],pin_memory=device.type=='cuda') for i,ds in enumerate([train,val]) if ds is not None]
    (out/'run.json').write_text(json.dumps(dict(config=config,torch=torch.__version__,device=str(device),train_pairs=len(train),val_pairs=len(val) if val else 0,
        train_pair_paths=train.samples,val_pair_paths=val.samples if val else [],resume=str(a.resume) if a.resume else None,finetune=str(a.finetune) if a.finetune else None),indent=2))
    for epoch in range(epoch0,config['epochs']):
        for group in optimizer.param_groups:group['lr']=config['lr']*.1**(epoch//config['schedule'])
        total=0.;count=0;finished_epoch=True
        model.train()
        for bi,batch in enumerate(loaders[0]):
            x,y,mask=batch[0].to(device),batch[1].to(device),batch[2].to(device)
            kw={} if epoch<config['curriculum'] else dict(gmtr_x1=batch[4].to(device),gmtr_x2=batch[5].to(device),gmtr_x3=batch[6].to(device))
            optimizer.zero_grad(set_to_none=True);output=model(x,y,**kw)
            value=loss.update_loss(x,y,mask,output,epoch)
            if not torch.isfinite(value):raise RuntimeError('Nonfinite training loss')
            value.backward()
            if not torch.stack([torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None]).all():raise RuntimeError('Nonfinite gradient')
            optimizer.step();step+=1;total+=float(value.detach())*len(x);count+=len(x)
            print(json.dumps(dict(epoch=epoch,step=step,train_batch_loss=float(value.detach()))),flush=True)
            if (a.max_steps and step>=a.max_steps) or (a.limit_batches and bi+1>=a.limit_batches):
                finished_epoch=bi+1==len(loaders[0]);break
        model.eval();vtotal=0.;vcount=0
        with torch.no_grad():
            for bi,batch in enumerate(loaders[1] if val else []):
                x,y,mask=batch[0].to(device),batch[1].to(device),batch[2].to(device)
                output=model(x,y);value=((output[0]*mask-compute_ssim(x,y)*mask)**2).mean()
                if not torch.isfinite(value):raise RuntimeError('Nonfinite validation')
                vtotal+=float(value)*len(x);vcount+=len(x)
                if a.limit_batches and bi+1>=a.limit_batches:break
        record=dict(epoch=epoch,step=step,train_loss=total/count,val_mse=vtotal/vcount if vcount else None,completed_epoch=finished_epoch)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        improved=vcount>0 and record['val_mse']<best
        if vcount:best=min(best,record['val_mse'])
        save_checkpoint(out,model,optimizer,config,epoch+1 if finished_epoch else epoch,step,best)
        if improved:shutil.copy2(out/'last.pt',out/'best.pt')
        print(json.dumps(record),flush=True)
        if a.max_steps and step>=a.max_steps:break

if __name__=='__main__':main()
