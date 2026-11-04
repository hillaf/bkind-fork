"""Fit a train-only B-KinD linear probe and score original-crop DLC test labels.

Both prediction directories must contain JSONs from the SAME checkpoint.
Use RGB cow-crop inference only, not masks or full-scene images.
"""
import argparse,csv,json,math
from pathlib import Path
import numpy as np


def read_predictions(root,expected):
    found={}
    for path in sorted(root.rglob('*.json')):
        value=json.loads(path.read_text())
        if 'keypoints_xy_normalized' not in value:continue
        # Source stem is authoritative; it is recorded by the inference runner.
        source=Path(value['source'])
        if source.parent.name!='images':continue
        sid=source.stem
        if sid not in expected:continue
        if sid in found:raise ValueError(f'Duplicate RGB prediction: {sid}')
        keypoints=np.asarray(value['keypoints_xy_normalized'],dtype=float)
        if keypoints.ndim!=2 or keypoints.shape[1]!=2 or not np.isfinite(keypoints).all():
            raise ValueError(f'Invalid keypoints: {path}')
        found[sid]=keypoints.reshape(-1)
    missing=expected-set(found)
    if missing:raise ValueError(f'{len(missing)} missing predictions in {root}; example: {sorted(missing)[0]}')
    return found


def read_annotations(root,split):
    samples=json.loads((root/split/'annotations.json').read_text())['samples']
    return {s['sample_id']:s for s in samples}


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stats(rows):
    e=np.array([r['error_px'] for r in rows]);n=np.array([r['normalized_error'] for r in rows])
    return dict(samples=len({r['sample_id'] for r in rows}),evaluated_keypoints=len(rows),
        mean_error_px=float(e.mean()),median_error_px=float(np.median(e)),rmse_px=float(np.sqrt((e**2).mean())),
        mean_normalized_error=float(n.mean()),pck_0_05=float((n<=.05).mean()),pck_0_10=float((n<=.1).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-predictions',type=Path,required=True);p.add_argument('--test-predictions',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True,help='Existing AutoLink and DLC comparison CSV')
    p.add_argument('--dlc-predictions',type=Path,required=True,help='Recorded DLC test CSV for exact GT identity check')
    p.add_argument('--checkpoint-label',required=True,help='Identify the common checkpoint used for both inference sets')
    a=p.parse_args();names=['nose','left_earbase','left_earend','right_earbase','right_earend','back_base','back_middle','tail_base']
    train=read_annotations(a.dataset,'train');test=read_annotations(a.dataset,'test')
    if set(train)&set(test):raise ValueError('Train/test sample overlap')
    tr=read_predictions(a.train_predictions,set(train));te=read_predictions(a.test_predictions,set(test))
    ids=sorted(train);test_ids=sorted(test);X=np.stack([tr[s] for s in ids]);XT=np.stack([te[s] for s in test_ids])
    if X.shape[1]!=XT.shape[1]:raise ValueError('Train/test keypoint counts differ')
    X=np.c_[X,np.ones(len(X))];XT=np.c_[XT,np.ones(len(XT))]
    coefficients=[];diagnostics=[]
    for name in names:
        valid=[];target=[]
        for i,sid in enumerate(ids):
            points={k['bodypart']:k for k in train[sid]['primary_keypoints']}
            gt=points[name]
            if gt.get('annotated') and gt.get('usable_for_training'):
                valid.append(i);target.append([gt['crop_x'],gt['crop_y']])
        b,_,rank,singular=np.linalg.lstsq(X[valid],np.asarray(target,dtype=float),rcond=None)
        coefficients.append(b);diagnostics.append(dict(bodypart=name,train_labels=len(valid),rank=int(rank),features_with_intercept=X.shape[1]))
    coefficients=np.asarray(coefficients);predicted=np.einsum('nf,jfc->njc',XT,coefficients);rows=[]
    for i,sid in enumerate(test_ids):
        sample=test[sid];points={k['bodypart']:k for k in sample['primary_keypoints']}
        # This evaluation protocol uses 640x640 original crops, matching DLC.
        from PIL import Image
        with Image.open(a.dataset/'test/images'/sample['crop_file_name']) as image:width,height=image.size
        for j,name in enumerate(names):
            gt=points[name]
            if not(gt.get('annotated') and gt.get('usable_for_training')):continue
            px,py=predicted[i,j];gx,gy=gt['crop_x'],gt['crop_y'];error=float(np.hypot(px-gx,py-gy))
            rows.append(dict(method='B-KinD (Linear mapping)',sample_id=sid,bodypart=name,predicted_x=float(px),predicted_y=float(py),
                manual_x=gx,manual_y=gy,error_px=error,normalized_error=error/math.hypot(width,height)))
    reference=list(csv.DictReader(a.dlc_predictions.open()));first=reference[0]['model']
    reference={(r['sample_id'],r['bodypart']):r for r in reference if r['model']==first and r['error_px'] not in ('','nan')}
    if {(r['sample_id'],r['bodypart']) for r in rows}!=set(reference):raise ValueError('DLC test sample/landmark keys differ')
    for row in rows:
        ref=reference[row['sample_id'],row['bodypart']]
        if not np.allclose([row['manual_x'],row['manual_y']],[float(ref['manual_x']),float(ref['manual_y'])],atol=1e-7):
            raise ValueError('DLC GT coordinate mismatch')
    a.output.mkdir(parents=True,exist_ok=True);np.save(a.output/'linear_coefficients.npy',coefficients)
    write_csv(a.output/'predictions_and_errors.csv',rows)
    summary=dict(method='B-KinD (Linear mapping)',**stats(rows));write_csv(a.output/'metrics_overall.csv',[summary])
    per=[dict(method=summary['method'],bodypart=name,**stats([r for r in rows if r['bodypart']==name])) for name in names]
    write_csv(a.output/'metrics_per_keypoint.csv',per)
    comparison=list(csv.DictReader(a.comparison.open()));comparison=[r for r in comparison if r['method']!=summary['method']]
    comparison.append({k:summary[k] for k in comparison[0]});write_csv(a.output/'comparison_all.csv',comparison)
    report='# B-KinD linear-mapping evaluation\n\nTrain-only ordinary least squares with intercept; validation/test labels were not used for fitting or tuning. Predictions use original crop coordinates. Test GT identities and coordinates match the recorded DLC evaluation.\n\n| Method | Mean error (px) | RMSE (px) | PCK@0.05 | PCK@0.10 |\n|---|---:|---:|---:|---:|\n'
    for r in comparison:report+=f"| {r['method']} | {float(r['mean_error_px']):.2f} | {float(r['rmse_px']):.2f} | {100*float(r['pck_0_05']):.2f}% | {100*float(r['pck_0_10']):.2f}% |\n"
    report+='\nPer-keypoint B-KinD scores:\n\n| Keypoint | GT count | Mean error (px) | PCK@0.05 | PCK@0.10 |\n|---|---:|---:|---:|---:|\n'
    for r in per:report+=f"| {r['bodypart']} | {r['evaluated_keypoints']} | {r['mean_error_px']:.2f} | {100*r['pck_0_05']:.2f}% | {100*r['pck_0_10']:.2f}% |\n"
    (a.output/'REPORT.md').write_text(report)
    (a.output/'run.json').write_text(json.dumps(dict(checkpoint_label=a.checkpoint_label,train_predictions=str(a.train_predictions.resolve()),
        test_predictions=str(a.test_predictions.resolve()),fit_images=len(train),score_images=len(test),diagnostics=diagnostics,metrics=summary,
        note='The inference JSONs do not record checkpoint hashes; caller must ensure both sets use the same checkpoint.'),indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
