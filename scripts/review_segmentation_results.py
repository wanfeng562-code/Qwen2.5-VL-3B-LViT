#!/usr/bin/env python3
"""CPU-only post-test review; refuses to run before the complete six-run summary."""
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (666,667,668)
ARMS = ('legacy','qwen')
SOURCES = {'legacy':'provided_legacy_text','qwen':'qwen_predicted_text'}
OUT = ROOT/'outputs/visual_review/segmentation/seed666'


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def project_path(value):
    p=(ROOT/value).resolve()
    if not p.is_relative_to(ROOT):raise ValueError(f'Outside project: {value}')
    return p


def scores(pred,gt):
    p=int(pred.sum()); g=int(gt.sum()); intersection=int((pred & gt).sum())
    if p+g == 0: return {'dice':1.,'iou':1.}
    return {'dice':2*intersection/(p+g),'iou':intersection/(p+g-intersection)}


def mask(path):
    im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if im is None: raise ValueError(f'Could not read GT {path}')
    return cv2.resize((im>0).astype(np.uint8),(224,224),interpolation=cv2.INTER_NEAREST)>0


def prediction(path):
    im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if im is None or im.shape!=(224,224) or im.dtype!=np.uint8 or not set(np.unique(im)).issubset({0,255}):
        raise ValueError(f'Invalid 224x224 uint8 0/255 prediction: {path}')
    return im>0


def csv_write(path, rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


def check_ready():
    # This gate is deliberately before test-manifest or test-output reads.
    status_path=ROOT/'outputs/stage234/formal_experiments/status.json'
    if not status_path.is_file() or load_json(status_path).get('status')!='completed':
        raise RuntimeError('Formal queue including joint inference has not completed; review is not ready')
    summary_path=ROOT/'outputs/stage234/formal_experiments/summary.json'
    if not summary_path.is_file():
        raise RuntimeError('Six-run formal test summary is not complete; test review is not ready')
    summary=load_json(summary_path)
    if summary.get('status')!='complete' or summary.get('dataset')!='monuseg' or summary.get('test_count_per_run')!=14:
        raise ValueError('Unexpected or incomplete formal summary')
    records=summary['records']
    if len(records)!=6 or {(r['arm'],r['seed']) for r in records}!={(a,s) for a in ARMS for s in SEEDS}:
        raise ValueError('Expected exactly six distinct arm/seed results')
    return status_path,summary_path,records


def load_histories(records):
    result={}
    for row in records:
        arm,seed=row['arm'],row['seed'];base=ROOT/f'outputs/lvit/monuseg/{arm}_seed{seed}'
        lines=(base/'history.jsonl').read_text(encoding='utf-8').splitlines()
        history=[json.loads(line) for line in lines if line.strip()]
        if len(history)!=row['epochs'] or [x['epoch'] for x in history]!=list(range(1,len(history)+1)):
            raise ValueError(f'History epoch mismatch: {arm} {seed}')
        if not (1<=row['best_epoch']<=len(history)):
            raise ValueError(f'Invalid best epoch: {arm} {seed}')
        if any(not math.isfinite(float(x['train_loss'])) or not math.isfinite(float(x['validation']['dice'])) for x in history):
            raise ValueError(f'Nonfinite history: {arm} {seed}')
        result[(arm,seed)]={'path':base/'history.jsonl','history':history,'best_epoch':row['best_epoch']}
    return result


def history_figure(histories,out):
    fig,axes=plt.subplots(3,2,figsize=(13,11),sharex=False)
    colors={'legacy':'#2463a6','qwen':'#e07a2d'}
    for index,seed in enumerate(SEEDS):
        for arm in ARMS:
            h=histories[(arm,seed)];rows=h['history'];epochs=[x['epoch'] for x in rows]
            label='Legacy text' if arm=='legacy' else 'Qwen text'
            axes[index,0].plot(epochs,[x['train_loss'] for x in rows],label=label,color=colors[arm])
            vals=[x['validation']['dice'] for x in rows]
            axes[index,1].plot(epochs,vals,label=label,color=colors[arm])
            k=h['best_epoch']-1
            axes[index,1].scatter([epochs[k]],[vals[k]],color=colors[arm],marker='o',s=32,zorder=4)
        axes[index,0].set_title(f'Seed {seed}: training loss')
        axes[index,1].set_title(f'Seed {seed}: validation Dice (dots: selected epoch)')
        for ax in axes[index]:
            ax.set_xlabel('Epoch');ax.grid(alpha=.22);ax.legend(fontsize=8)
        axes[index,0].set_ylabel('Loss');axes[index,1].set_ylabel('Dice (fraction)')
    fig.suptitle('MoNuSeg: training and validation histories, each run shown to its own stopping epoch',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(out/'training_curves.png',dpi=160);fig.savefig(out/'training_curves.pdf');plt.close(fig)


def main():
    status_path,summary_path,records=check_ready()
    histories=load_histories(records)
    manifest=ROOT/'manifests/legacy_subset_v1/monuseg/test.csv'
    with manifest.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if len(rows)!=14 or len({r['sample_id'] for r in rows})!=14 or any(r['dataset']!='monuseg' or r['split']!='test' for r in rows):
        raise ValueError('Expected 14 unique MoNuSeg test samples')
    ids={r['sample_id'] for r in rows};digest_manifest=digest(manifest);results={}
    for arm in ARMS:
        base=ROOT/f'outputs/lvit/monuseg/{arm}_seed666/test'
        metrics_path=base/'test_metrics.json';m=load_json(metrics_path)
        if m.get('dataset')!='monuseg' or m.get('split')!='test' or m.get('source')!=SOURCES[arm] or m.get('seed')!=666 or m.get('sample_count')!=14:
            raise ValueError(f'Metric identity mismatch: {arm}')
        if m.get('manifest_sha256')!=digest_manifest or m.get('code_matches_training') is not True or m.get('checkpoint_epoch')!=m.get('checkpoint_best_epoch'):
            raise ValueError(f'Manifest, code identity, or best checkpoint mismatch: {arm}')
        checkpoint=project_path(m['checkpoint'])
        if digest(checkpoint)!=m['checkpoint_sha256']:raise ValueError(f'Checkpoint hash mismatch: {arm}')
        reported={x['sample_id']:x for x in m['samples']}
        if len(m['samples'])!=14 or set(reported)!=ids:raise ValueError(f'Per-sample mismatch: {arm}')
        present={p.stem for p in (base/'predictions').glob('*.png')}
        if present!=ids:raise ValueError(f'Prediction set mismatch: {arm}')
        results[arm]={'metrics':m,'path':metrics_path,'samples':reported,'prediction_dir':base/'predictions'}
    output=[];pairs=[]
    for row in rows:
        sid=row['sample_id'];gt=mask(project_path(row['mask_path']));values={}
        for arm in ARMS:
            entry=results[arm];sample=entry['samples'][sid];path=entry['prediction_dir']/(sid+'.png')
            if sample['image_path']!=row['image_path']:raise ValueError(f'Image mismatch: {sid}')
            calculated=scores(prediction(path),gt)
            for key in ('dice','iou'):
                if not math.isclose(calculated[key],sample[key],rel_tol=1e-8,abs_tol=1e-8):
                    raise ValueError(f'{arm}/{sid}/{key} mismatch with saved metric')
            values[arm]=calculated
        output.append({'sample_id':sid,'dataset':'monuseg','split':'test','baseline_dice':values['legacy']['dice'],
                       'baseline_iou':values['legacy']['iou'],'qwen_dice':values['qwen']['dice'],
                       'qwen_iou':values['qwen']['iou'],'dice_delta':values['qwen']['dice']-values['legacy']['dice']})
        pairs.append({'sample_id':sid,'baseline_path':str((results['legacy']['prediction_dir']/(sid+'.png')).relative_to(ROOT)),
                      'qwen_path':str((results['qwen']['prediction_dir']/(sid+'.png')).relative_to(ROOT))})
    for arm in ARMS:
        for key in ('dice','iou'):
            vals=[r[('baseline_' if arm=='legacy' else 'qwen_')+key] for r in output]
            if not math.isclose(sum(vals)/14,results[arm]['metrics'][key],rel_tol=1e-8,abs_tol=1e-8):
                raise ValueError(f'{arm} aggregate {key} mismatch')
    ranking=sorted(output,key=lambda r:(r['baseline_dice'],r['sample_id']))
    primary=[ranking[i]['sample_id'] for i in (0,6,13)]
    deltas=sorted(output,key=lambda r:(r['dice_delta'],r['sample_id']))
    supplemental=[deltas[0]['sample_id'],deltas[-1]['sample_id']]
    if OUT.exists():raise FileExistsError(f'Refusing to overwrite existing review: {OUT}')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.seed666-stage-',dir=OUT.parent) as td:
        stage=Path(td);csv_write(stage/'prediction_pairs.csv',pairs);csv_write(stage/'sample_metrics.csv',output)
        selected=[r for r in rows if r['sample_id'] in set(supplemental)]
        csv_write(stage/'supplemental_manifest.csv',selected)
        (stage/'primary_labels.json').write_text(json.dumps(dict(zip(primary,['Lowest baseline Dice','Lower median baseline Dice','Highest baseline Dice'])))+'\n')
        min_delta,max_delta=deltas[0]['dice_delta'],deltas[-1]['dice_delta']
        min_name='Largest decrease' if min_delta<0 else ('Smallest increase' if min_delta>0 else 'Smallest change')
        max_name='Largest increase' if max_delta>0 else ('Smallest decrease' if max_delta<0 else 'Largest change')
        delta_labels={supplemental[0]:f'{min_name}: delta Dice={min_delta:+.4f}',supplemental[1]:f'{max_name}: delta Dice={max_delta:+.4f}'}
        if supplemental[0]==supplemental[1]: delta_labels[supplemental[0]]='Both delta extremes (same case)'
        (stage/'supplemental_labels.json').write_text(json.dumps(delta_labels)+'\n')
        render=ROOT/'scripts/render_experiment_review.py'
        common=['--predictions',str(stage/'prediction_pairs.csv'),'--seed','666','--cohort-size','14','--per-sheet','3']
        subprocess.run([sys.executable,str(render),'--manifest',str(manifest),'--output-dir',str(stage/'primary'),'--select-comparison','--labels-json',str(stage/'primary_labels.json'),*common],check=True)
        subprocess.run([sys.executable,str(render),'--manifest',str(stage/'supplemental_manifest.csv'),'--output-dir',str(stage/'delta_examples'),'--labels-json',str(stage/'supplemental_labels.json'),*common],check=True)
        plotted=load_json(stage/'primary/index.json')
        if plotted['ids']!=primary:raise ValueError('Rendered primary IDs do not match predeclared ranking')
        for folder in ('primary','delta_examples'):
            index_path=stage/folder/'index.json'
            index=load_json(index_path)
            for key in ('manifest','predictions'):
                if index.get(key): index[key]=index[key].replace(str(stage),str(OUT))
            index_path.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        history_figure(histories,stage)
        receipt={'status':'complete','dataset':'monuseg','split':'test','seed':666,'sample_count':14,
                 'comparison':'provided legacy text versus Qwen-generated text; not an image-only ablation',
                 'primary_rule':'baseline Dice@224 ascending, sample_id tie break: worst, lower median, best',
                 'primary_ids':primary,'supplemental_rule':'Qwen minus baseline Dice@224 ascending, sample_id tie break: minimum and maximum delta',
                 'supplemental_ids':supplemental,'supplemental_deltas':[min_delta,max_delta],
                 'same_case_for_both_delta_extremes':supplemental[0]==supplemental[1],
                 'inference_grid':'224x224; OpenCV nearest-neighbor GT; binary predictions threshold >=0.5 in original evaluation',
                 'formal_queue_status_sha256':digest(status_path),'formal_summary_sha256':digest(summary_path),'test_manifest_sha256':digest_manifest,
                 'metrics_sha256':{a:digest(results[a]['path']) for a in ARMS},
                 'history_sha256':{f'{a}_{s}':digest(histories[(a,s)]['path']) for a in ARMS for s in SEEDS},
                 'prediction_sha256':{a:{sid:digest(results[a]['prediction_dir']/(sid+'.png')) for sid in sorted(ids)} for a in ARMS}}
        (stage/'review_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        os.rename(stage,OUT)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
