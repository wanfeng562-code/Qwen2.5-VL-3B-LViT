"""Independent artifact audit; does not train or generate descriptions."""
import sys,json,csv,hashlib,statistics,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from src.experiment.common import atomic_json,sha256_file
from src.phase45.engine import configure,setup_data

def model_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    torch.set_num_threads(8)
    summary=json.loads((ROOT/'outputs/phase45/confirmation/summary.json').read_text())
    assert summary['status']=='stopped_degenerate_text'
    assert not summary['three_seed_training_performed'] and not summary['records']
    freeze=json.loads((ROOT/'docs/phase45_freeze.json').read_text())
    checks=[]
    for group in ('legacy_tracked_sha256','phase45_core_sha256'):
        for name,expected in freeze[group].items():assert sha256_file(ROOT/name)==expected,(group,name)
        checks.append({'id':group,'status':'passed','files':len(freeze[group])})
    ledger=json.loads((ROOT/'outputs/phase45/registry.json').read_text())
    assert len(ledger['runs'])<=18 and len({x['run_id'] for x in ledger['runs']})==len(ledger['runs'])
    assert all(x['status']=='completed' for x in ledger['runs'])
    checks.append({'id':'formal_budget','status':'passed','count':len(ledger['runs']),'limit':18})
    artifacts={}
    for entry in ledger['runs']:
        d=ROOT/'outputs/phase45/runs'/entry['run_id'];result=json.loads((d/'result.json').read_text());rc=json.loads((d/'run_config.json').read_text())
        assert rc['config']==entry['config']
        h=[json.loads(x) for x in (d/'history.jsonl').read_text().splitlines()]
        assert [x['epoch'] for x in h]==list(range(1,result['epochs_completed']+1))
        best=max(h,key=lambda x:x['validation']['dice'])
        assert best['epoch']==result['best_epoch'] and best['validation']['dice']==result['best_validation_dice']
        assert all(x['train_sample_count']==24 and x['validation']['sample_count']==6 for x in h)
        assert rc['counts']=={'train':24,'val':6}
        assert all(x['text_gradient_l2_min']>0 for x in h)
        assert (rc['config']['max_epochs'],rc['config']['min_epochs'],rc['config']['patience'])==(2000,50,50)
        if result['stop_reason']=='validation_patience':assert result['epochs_completed']-result['best_epoch']==50
        ck=torch.load(d/'best.pt',map_location='cpu',weights_only=False)
        assert ck['epoch']+1==result['best_epoch'] and ck['identity']==rc['identity']
        assert ck['config']==rc['config']
        statehash=model_hash(ck['model_state']);del ck;gc.collect()
        artifacts[entry['run_id']]={'best_epoch':best['epoch'],'epochs':len(h),'dice_percent':100*best['validation']['dice'],'initial_model_sha256':rc['initial_model_sha256'],'best_model_tensors_sha256':statehash,'best_checkpoint_sha256':sha256_file(d/'best.pt'),'feature_mode':rc['config']['feature_mode'],'mapping':rc['feature_mapping']}
        checks.append({'id':'training_'+entry['run_id'],'status':'passed','epochs':len(h),'best_epoch':best['epoch']})
    assert set(artifacts)=={'P_seed666','E_seed666','F_seed666','F_constant_seed666'}
    for a,b in (('P_seed666','E_seed666'),('F_seed666','F_constant_seed666')):
        assert artifacts[a]['initial_model_sha256']==artifacts[b]['initial_model_sha256']
    checks.append({'id':'matched_initial_weights','status':'passed','pairs':[['P','E'],['F','F_constant']]})
    selected=summary['selection']['candidate']
    config=json.loads((ROOT/f'configs/phase45/{selected}.json').read_text())
    equivalence=[]
    for seed in (666,667,668):
        proper=configure({**config,'seed':seed,'feature_mode':'correct'})
        rows,_,correct=setup_data(proper);_,_,shuffled=setup_data({**proper,'feature_mode':'shuffled'})
        assert all(torch.equal(correct.lookup[k],shuffled.lookup[k]) for k in correct.lookup)
        mapping=shuffled.mapping
        assert all(k!=v['source_sample_id'] and mapping[v['source_sample_id']]['split']==v['split'] for k,v in mapping.items())
        saved=next(x for x in summary['input_equivalence'] if x['seed']==seed)
        assert mapping==saved['mapping'] and correct.sha256==saved['feature_cache_sha256']
        equivalence.append({'seed':seed,'input_features_bitwise_equal':True,'training_performed':seed==666,
                            'reassigned_ids':len(mapping),'unchanged_text_pairs':sum(x['same_raw_text'] for x in mapping.values()),
                            'counts':{k:len(v) for k,v in rows.items()}})
    checks.append({'id':'independent_feature_equivalence_not_training','status':'passed','per_seed':equivalence})
    cache=json.loads((ROOT/'outputs/phase45/preflight/checks.json').read_text())['cache_checks']
    assert all(x['maximum_feature_difference_between_images']==0 for x in cache.values())
    assert not (ROOT/'outputs/phase45/test_authorization.json').exists()
    assert not any('test' in p.name for p in (ROOT/'cache/phase45').iterdir())
    assert not any(p.parent.name=='test' for p in (ROOT/'outputs/phase45').rglob('*'))
    checks.append({'id':'test_closed_on_evidence_stop','status':'passed'})
    lora=json.loads((ROOT/'outputs/phase45/lora_gate.json').read_text())
    assert not lora['eligible'] and not any(x['kind']=='lora' for x in ledger['runs'])
    sensitivity=json.loads((ROOT/'outputs/phase45/F_sensitivity.json').read_text())
    assert len(sensitivity['results'])==6 and sensitivity['input_features_bitwise_equal']
    assert all(x['dice']==sensitivity['results'][0]['dice'] for x in sensitivity['results'])
    assert abs(lora['F_vs_constant_pp']-(artifacts['F_seed666']['dice_percent']-artifacts['F_constant_seed666']['dice_percent']))<1e-10
    assert abs(lora['permutation_drop_pp'])<1e-10
    checks.append({'id':'lora_gate_and_budget','status':'passed','gate':lora})
    result={'status':'passed','checks':checks,'check_count':len(checks),'formal_runs':len(ledger['runs']),'artifacts':artifacts,'equivalence':equivalence,
            'three_seed_training_performed':False,'three_seed_validation_gate':'not_run_early_evidence_stop','test_generated':False,'test_evaluated':False}
    atomic_json(ROOT/'outputs/phase45/final_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','checks')},ensure_ascii=False))

if __name__=='__main__':main()
