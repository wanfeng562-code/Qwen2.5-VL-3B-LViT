import os,sys,json,argparse
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from src.experiment.common import atomic_json,seed_everything,sha256_file
from src.experiment.loss import WeightedDiceBCE
from src.phase45.engine import setup_data,run_identity,evaluate
from src.phase45.model import build_model
from src.phase45.data import ManifestDataset,make_loader

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(8)
    ck=torch.load(ROOT/a.checkpoint,map_location='cpu',weights_only=False);config=ck['config'];seed_everything(config['seed'])
    rows,hashes,features=setup_data(config);assert ck['identity']==run_identity(config,hashes,features)
    model=build_model(config['fusion']).cuda().eval();model.load_state_dict(ck['model_state'],strict=True)
    ids=[r['sample_id'] for r in rows['val']];correct={s:features.lookup[s] for s in ids};results=[]
    for shift in range(6):
        features.lookup.update({s:correct[ids[(i+shift)%6]] for i,s in enumerate(ids)})
        ds=ManifestDataset(rows['val'],features)
        loader=make_loader(ds,2,config['seed'],device='cuda')
        result=evaluate(model,loader,WeightedDiceBCE(),'cuda')
        results.append({'shift':shift,**result})
    assert abs(results[0]['dice']-ck['best_validation_dice'])<1e-12
    result={'checkpoint_sha256':sha256_file(ROOT/a.checkpoint),'split':'val','count':6,'correct_dice_percent':results[0]['dice_percent'],
            'permuted_mean_dice_percent':float(np.mean([x['dice_percent'] for x in results[1:]])),'permuted_drop_pp':float(results[0]['dice_percent']-np.mean([x['dice_percent'] for x in results[1:]])),
            'results':results,'input_features_bitwise_equal':all(torch.equal(correct[s],correct[ids[0]]) for s in ids)}
    atomic_json(ROOT/a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)

if __name__=='__main__':main()
