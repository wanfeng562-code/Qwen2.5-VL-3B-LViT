"""Validate held-out data construction against all existing validation modes."""
import os,sys,json
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.phase46.engine import configure,setup_data
from src.experiment.common import atomic_json,sha256_file
from scripts.phase46_final_data import conditions

def main():
    torch.set_num_threads(8)
    cfg=configure(json.loads((ROOT/'configs/phase46/V_mid.json').read_text()))
    dev=torch.load(ROOT/cfg['feature_cache'],map_location='cpu',weights_only=False)
    val_ids=[r['sample_id'] for r in dev['records'] if r['split']=='val'];indices=[dev['sample_ids'].index(sid) for sid in val_ids]
    val={'sample_ids':val_ids,'features':dev['features'][indices],'legacy_text_features':dev['legacy_text_features'][indices]}
    checks=[]
    for mode in ('correct','constant','shuffled','legacy'):
        for seed in (666,667,668):
            _,_,old=setup_data({**cfg,'seed':seed,'feature_mode':mode})
            new,mapping=conditions(dev,val,mode,seed)
            assert all(torch.equal(new[sid][key],old.lookup[sid][key]) for sid in val_ids for key in ('vision','text'))
            checks.append({'mode':mode,'seed':seed,'six_images_conditions_bitwise_equal':True})
    result={'status':'passed','check_count':len(checks),'checks':checks,
            'script_sha256':sha256_file(ROOT/'scripts/phase46_final_data.py'),'test_accessed':False}
    atomic_json(ROOT/'outputs/phase46/final_data_check.json',result);print(json.dumps(result))

if __name__=='__main__':main()
