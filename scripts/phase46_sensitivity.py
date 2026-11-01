import os,sys,json,argparse
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from src.experiment.common import atomic_json,seed_everything,sha256_file
from src.experiment.loss import WeightedDiceBCE
from src.phase46.engine import setup_data,run_identity,evaluate
from src.phase46.model import build_model
from src.phase46.data import ManifestDataset,make_loader

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(8);ck=torch.load(ROOT/a.checkpoint,map_location='cpu',weights_only=False);cfg=ck['config'];seed_everything(cfg['seed'])
    rows,hashes,store=setup_data(cfg);assert ck['identity']==run_identity(cfg,hashes,store)
    model=build_model(cfg).cuda().eval();model.load_state_dict(ck['model_state'],strict=True)
    ids=[x['sample_id'] for x in rows['val']];original={sid:store.lookup[sid] for sid in ids};key='text' if cfg['architecture']=='S' else 'vision'
    results=[];tensors=[]
    hook=model.register_forward_hook(lambda m,args,out:tensors.append(out.detach().cpu()))
    for shift in range(6):
        for i,sid in enumerate(ids):store.lookup[sid]={**original[sid],key:original[ids[(i+shift)%6]][key]}
        tensors.clear();ds=ManifestDataset(rows['val'],store);loader=make_loader(ds,2,cfg['seed'],device='cuda')
        result=evaluate(model,loader,WeightedDiceBCE(),'cuda');prediction=torch.cat(tensors)
        if shift==0:baseline=prediction.clone()
        result.update(shift=shift,probability_max_abs_change=float((prediction-baseline).abs().max()),
                      binary_changed_fraction=float(((prediction>=.5)!=(baseline>=.5)).float().mean()))
        results.append(result)
    hook.remove();assert abs(results[0]['dice']-ck['best_validation_dice'])<1e-12
    summary={'checkpoint_sha256':sha256_file(ROOT/a.checkpoint),'split':'val','count':6,'correct_dice_percent':results[0]['dice_percent'],
             'permuted_mean_dice_percent':float(np.mean([x['dice_percent'] for x in results[1:]])),
             'permuted_drop_pp':float(results[0]['dice_percent']-np.mean([x['dice_percent'] for x in results[1:]])),
             'mean_binary_changed_fraction':float(np.mean([x['binary_changed_fraction'] for x in results[1:]])),
             'any_prediction_changed':any(x['probability_max_abs_change']>0 for x in results[1:]),'results':results,'test_accessed':False}
    atomic_json(ROOT/a.output,summary);print(json.dumps({k:v for k,v in summary.items() if k!='results'}))

if __name__=='__main__':main()
