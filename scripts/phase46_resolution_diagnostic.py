"""GT-only representational ceiling, never a deployable prediction or model score."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.experiment.common import atomic_json
from src.experiment.data import load_mask
from src.phase45.attributes import rows_for

def ceiling(mask,grid):
    # Each grid cell is assigned one binary label; choose the optimal top-k
    # occupancy cells for this individual image using its GT, not a predictor.
    block=224//grid;assert block*grid==224
    counts=mask.reshape(grid,block,grid,block).sum((1,3)).reshape(-1)
    gt_total=mask.sum();order=np.sort(counts)[::-1];tp=np.concatenate([[0],np.cumsum(order)])
    den=np.arange(len(tp))*block*block+gt_total
    values=np.divide(2*tp,den,out=np.ones_like(tp,dtype=float),where=den>0)
    best=int(np.argmax(values));return {'oracle_dice':float(values[best]),'selected_cells':best,'foreground_pixels':int(gt_total)}

def main():
    records=[]
    for row in rows_for(('train','val')):
        gt=load_mask(ROOT/row['mask_path'])[0].numpy()
        records.append({'sample_id':row['sample_id'],'split':row['split'],'grids':{str(g):ceiling(gt,g) for g in (14,28,56,112,224)}})
    means={split:{str(g):float(np.mean([r['grids'][str(g)]['oracle_dice'] for r in records if r['split']==split])) for g in (14,28,56,112,224)} for split in ('train','val')}
    result={'status':'completed','method':'per-image GT oracle top-k binary constant cells at a specified grid',
            'limitation':'This ceiling applies only to piecewise-constant predictions on that grid. It is not an upper bound for bilinear features fused with the high-resolution LViT branch, not a model result, and not evidence of Qwen improvement.',
            'mean_oracle_dice':means,'records':records,'test_accessed':False}
    atomic_json(ROOT/'outputs/phase46/spatial_resolution_diagnostic.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'}))

if __name__=='__main__':main()
