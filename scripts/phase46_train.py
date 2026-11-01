import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,argparse,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.phase46.engine import train,configure
from src.phase46.budget import claim,finish,locked

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output-dir',required=True);p.add_argument('--resume');a=p.parse_args()
    torch.set_num_threads(8)
    config=json.loads((ROOT/a.config).read_text());config['seed']=a.seed;config=configure(config)
    run_id=Path(a.output_dir).name
    with locked('gpu.lock'):
        claim(run_id,'segmentation',config,bool(a.resume))
        try:
            result=train(config,a.output_dir,a.resume);finish(run_id,'completed',result)
        except BaseException as e:
            finish(run_id,'failed',{'error':type(e).__name__,'message':str(e)});raise

if __name__=='__main__':main()
