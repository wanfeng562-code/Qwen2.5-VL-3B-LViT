"""Actual validation predictions and plots after the confirmation gate decision."""
import os,sys,json,csv,hashlib
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from src.experiment.common import atomic_json,sha256_file
from src.phase45.engine import evaluate_checkpoint
from src.phase45.attributes import rows_for

def score(pred,truth):
    inter=int((pred&truth).sum());n=int(pred.sum()+truth.sum());u=int((pred|truth).sum())
    return (2*inter/n if n else 1.,inter/u if u else 1.)

def main():
    torch.set_num_threads(8)
    summary=json.loads((ROOT/'outputs/phase45/confirmation/summary.json').read_text());assert summary['status']=='stopped_degenerate_text'
    out=ROOT/'outputs/phase45/visual_review';out.mkdir(parents=True,exist_ok=True)
    rows=rows_for(('val',));lookup={r['sample_id']:r for r in rows}
    selected=json.loads((ROOT/'outputs/phase45/screening/summary.json').read_text());metrics={}
    for arm,record in selected.items():
        record={**record,'run':f'outputs/phase45/runs/{arm}_seed666'}
        d=out/arm
        if (d/'val_metrics.json').exists():m=json.loads((d/'val_metrics.json').read_text())
        else:m=evaluate_checkpoint(record['run']+'/best.pt','val',str(d.relative_to(ROOT)),save_predictions=True)
        assert abs(m['dice']-record['best_validation_dice'])<1e-12
        values=[]
        for row in rows:
            raw=cv2.imread(str(ROOT/row['mask_path']),cv2.IMREAD_GRAYSCALE)
            gt=cv2.resize((raw>0).astype(np.uint8),(224,224),interpolation=cv2.INTER_NEAREST)>0
            pred=cv2.imread(str(d/'predictions'/f"{row['sample_id']}.png"),cv2.IMREAD_GRAYSCALE)
            assert pred.shape==(224,224) and set(np.unique(pred))<={0,255}
            dice,iou=score(pred>0,gt)
            saved=next(x for x in m['samples'] if x['sample_id']==row['sample_id'])
            assert abs(dice-saved['dice'])<1e-12 and abs(iou-saved['iou'])<1e-12
            values.append({'sample_id':row['sample_id'],'dice':dice,'iou':iou,'false_positive_pixels':int(((pred>0)&~gt).sum()),'false_negative_pixels':int(((pred==0)&gt).sum())})
        metrics[arm]=values
    order=sorted(metrics['F'],key=lambda x:(x['dice'],x['sample_id']))
    picks=[order[i]['sample_id'] for i in (0,(len(order)-1)//2,len(order)-1)]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    canvas=Image.new('RGB',(1600,1470),'white');draw=ImageDraw.Draw(canvas)
    draw.text((20,15),'VALIDATION ONLY | seed 666 | 6 images | masks evaluated on 224x224 grid',font=font,fill='black')
    for rownum,sid in enumerate(picks):
        row=lookup[sid];image=Image.open(ROOT/row['image_path']).convert('RGB');y=80+rownum*460
        gray=cv2.imread(str(ROOT/row['mask_path']),cv2.IMREAD_GRAYSCALE)
        gt=cv2.resize((gray>0).astype(np.uint8),(224,224),interpolation=cv2.INTER_NEAREST)
        draw.text((20,y-30),f"{'Lowest' if rownum==0 else 'Lower median' if rownum==1 else 'Highest'} F validation Dice | {sid}",font=font,fill='black')
        for col,arm in enumerate(('image','gt','F','F_constant')):
            label={'image':'Original image','gt':'GT (green)','F':'F: Qwen attribute text','F_constant':'F: constant text control'}[arm]
            pic=np.array(image).astype(float)
            if arm!='image':
                mask=gt if arm=='gt' else cv2.imread(str(out/arm/'predictions'/f'{sid}.png'),cv2.IMREAD_GRAYSCALE)>0
                mask=cv2.resize(np.array(mask,dtype=np.uint8),image.size,interpolation=cv2.INTER_NEAREST)>0
                color=np.array([0,255,0] if arm=='gt' else [255,0,255]);pic[mask]=pic[mask]*.55+color*.45
            canvas.paste(Image.fromarray(pic.astype(np.uint8)).resize((380,380),Image.Resampling.LANCZOS),(10+col*400,y+35))
            if arm in ('F','F_constant'):label+=f" Dice={next(x['dice'] for x in metrics[arm] if x['sample_id']==sid):.4f}"
            draw.text((10+col*400,y+7),label,font=font,fill='black')
    canvas.save(out/'comparison_validation.png')
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    colors={'P':'#e68623','E':'#2266aa','F':'#338855','F_constant':'#aa4499'}
    for arm in selected:
        hist=[json.loads(x) for x in (ROOT/f'outputs/phase45/runs/{arm}_seed666/history.jsonl').read_text().splitlines()]
        axes[0].plot([x['epoch'] for x in hist],[x['train_loss'] for x in hist],color=colors[arm],label=arm,alpha=.8)
        axes[1].plot([x['epoch'] for x in hist],[100*x['validation']['dice'] for x in hist],color=colors[arm],label=arm,alpha=.8)
    axes[0].set_title('Seed 666: training loss');axes[1].set_title('Seed 666: validation Dice (%)')
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Phase 4.5 screening only | 4 actual runs | no seed 667/668 training or test evaluation')
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/'training_curves.png',dpi=130);fig.savefig(out/'training_curves.pdf');plt.close(fig)
    csvrows=[]
    for sid in lookup:
        row={'sample_id':sid}
        for arm in selected:
            item=next(x for x in metrics[arm] if x['sample_id']==sid)
            row.update({arm+'_'+k:v for k,v in item.items() if k!='sample_id'})
        row['F_minus_constant_dice_pp']=100*(row['F_dice']-row['F_constant_dice']);csvrows.append(row)
    with (out/'per_image_validation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csvrows)
    receipt={'status':'completed','split':'val','sample_count':6,'seed':666,'screened_models':4,'selection':'F Dice ascending: worst/lower median/best; ties by ID','selected_ids':picks,'metrics':metrics,'test_visualizations':False,'three_seed_training_performed':False,'prediction_png_independent_score_check':'passed'}
    atomic_json(out/'review.json',receipt);print('REVIEW_COMPLETE',json.dumps({k:v for k,v in receipt.items() if k!='metrics'}))

if __name__=='__main__':main()
