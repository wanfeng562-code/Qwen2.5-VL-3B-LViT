"""Save actual validation predictions and independently recompute their metrics."""
import os,sys,json,csv
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from src.experiment.common import atomic_json
from src.phase46.engine import evaluate_checkpoint
from src.phase45.attributes import rows_for

def main():
    torch.set_num_threads(8)
    summary=json.loads((ROOT/'outputs/phase46/confirmation/summary.json').read_text())
    assert summary['status'] in ('stopped_initial_gate','stopped_validation_gate','validation_passed')
    out=ROOT/'outputs/phase46/visual_review';out.mkdir(exist_ok=True)
    rows=rows_for(('val',));lookup={r['sample_id']:r for r in rows}
    selected={r['arm']:r for r in summary['records'] if r['seed']==666};metrics={};all_metrics={}
    for rec in summary['records']:
        arm,seed=rec['arm'],rec['seed']
        d=out/(arm if seed==666 else f'{arm}_seed{seed}')
        if (d/'val_metrics.json').exists():result=json.loads((d/'val_metrics.json').read_text())
        else:result=evaluate_checkpoint(rec['run']+'/best.pt','val',str(d.relative_to(ROOT)),save_predictions=True)
        assert abs(result['dice']-rec['best_validation_dice'])<1e-12
        values=[]
        for row in rows:
            gt=cv2.resize((cv2.imread(str(ROOT/row['mask_path']),0)>0).astype(np.uint8),(224,224),interpolation=cv2.INTER_NEAREST)>0
            png=cv2.imread(str(d/'predictions'/f"{row['sample_id']}.png"),0);assert png.shape==(224,224) and set(np.unique(png))<={0,255}
            pred=png>0;inter=int((gt&pred).sum());den=int(gt.sum()+pred.sum());union=int((gt|pred).sum())
            dice=2*inter/den if den else 1.;iou=inter/union if union else 1.
            saved=next(x for x in result['samples'] if x['sample_id']==row['sample_id'])
            assert abs(dice-saved['dice'])<1e-12 and abs(iou-saved['iou'])<1e-12
            values.append({'sample_id':row['sample_id'],'dice':dice,'iou':iou,'fp':int((pred&~gt).sum()),'fn':int((~pred&gt).sum())})
        all_metrics[f'{arm}_{seed}']=values
        if seed==666:metrics[arm]=values
    picks=[x['sample_id'] for i,x in enumerate(sorted(metrics['candidate'],key=lambda x:(x['dice'],x['sample_id']))) if i in (0,2,5)]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    canvas=Image.new('RGB',(1600,1470),'white');draw=ImageDraw.Draw(canvas)
    draw.text((20,15),'VALIDATION ONLY | seed 666 | 6 images | Qwen visual candidate versus matched constant input',font=font,fill='black')
    for index,sid in enumerate(picks):
        row=lookup[sid];im=Image.open(ROOT/row['image_path']).convert('RGB');y=80+index*460
        gt=cv2.resize((cv2.imread(str(ROOT/row['mask_path']),0)>0).astype(np.uint8),(224,224),interpolation=cv2.INTER_NEAREST)
        draw.text((20,y-30),f"{['Lowest','Lower median','Highest'][index]} candidate validation Dice | {sid}",font=font,fill='black')
        for col,arm in enumerate(('image','gt','candidate','constant')):
            label={'image':'Original image','gt':'GT (green)','candidate':'Qwen '+summary['selection']['candidate'],'constant':'Matched constant control'}[arm]
            pic=np.array(im).astype(float)
            if arm!='image':
                mask=gt if arm=='gt' else cv2.imread(str(out/arm/'predictions'/f'{sid}.png'),0)>0
                mask=cv2.resize(mask.astype(np.uint8),im.size,interpolation=cv2.INTER_NEAREST)>0
                color=np.array([0,255,0] if arm=='gt' else [255,0,255]);pic[mask]=pic[mask]*.55+color*.45
            canvas.paste(Image.fromarray(pic.astype(np.uint8)).resize((380,380),Image.Resampling.LANCZOS),(10+col*400,y+35))
            if arm in metrics:label+=f" Dice={next(x['dice'] for x in metrics[arm] if x['sample_id']==sid):.4f}"
            draw.text((10+col*400,y+7),label,font=font,fill='black')
    canvas.save(out/'comparison_validation.png')
    seeds=sorted({r['seed'] for r in summary['records']});fig,axes=plt.subplots(len(seeds),2,figsize=(15,4.3*len(seeds)),squeeze=False)
    colors={'candidate':'#1f77b4','constant':'#ff7f0e','legacy':'#2ca02c','shuffled':'#d62728'}
    for i,seed in enumerate(seeds):
        for rec in (r for r in summary['records'] if r['seed']==seed):
            history=[json.loads(x) for x in (ROOT/rec['run']/'history.jsonl').read_text().splitlines()]
            axes[i,0].plot([x['epoch'] for x in history],[x['train_loss'] for x in history],label=rec['arm'],color=colors[rec['arm']])
            axes[i,1].plot([x['epoch'] for x in history],[100*x['validation']['dice'] for x in history],label=rec['arm'],color=colors[rec['arm']])
        axes[i,0].set_title(f'Seed {seed}: training loss');axes[i,1].set_title(f'Seed {seed}: validation Dice (%)')
        for ax in axes[i]:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=130);plt.close(fig)
    table=[]
    for sid in lookup:
        row={'sample_id':sid}
        for arm,values in metrics.items():row.update({arm+'_'+k:v for k,v in next(x for x in values if x['sample_id']==sid).items() if k!='sample_id'})
        row['candidate_minus_constant_pp']=100*(row['candidate_dice']-row['constant_dice']);table.append(row)
    with (out/'per_image_validation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)
    atomic_json(out/'review.json',{'status':'completed','split':'val','sample_count':6,'displayed_seed':666,'metrics':metrics,'all_seed_metrics':all_metrics,'independently_rescored_pngs':6*len(all_metrics),'selected_ids':picks,
                                'selection_rule':'candidate lowest/lower median/highest validation Dice, ID tie break','png_independent_scores':'passed','test_evaluated':False})
    print('VALIDATION_REVIEW_COMPLETE',list(metrics))

if __name__=='__main__':main()
