"""Audit completed phase46 artifacts independently of the training controller."""
import sys,json,gc,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.experiment.common import atomic_json,sha256_file
from src.phase46.engine import setup_data,run_identity

def read(path):return json.loads((ROOT/path).read_text())

def main():
    torch.set_num_threads(8);checks=[]
    freeze=read('docs/phase46_start.json')
    for name,expected in freeze['old_tracked_sha256'].items():assert sha256_file(ROOT/name)==expected,('old file changed',name)
    checks.append({'check':'old_files_unchanged','files':len(freeze['old_tracked_sha256'])})
    for name,expected in read('docs/phase46_core_freeze.json').items():assert sha256_file(ROOT/name)==expected,('phase46 core changed',name)
    checks.append({'check':'new_core_unchanged'})
    ledger=read('outputs/phase46/registry.json');old=read('outputs/phase45/registry.json')
    assert len(ledger['runs'])<=14 and len(old['runs'])==4 and len(ledger['runs'])+len(old['runs'])<=18
    assert all(x['status']=='completed' for x in ledger['runs'])
    checks.append({'check':'formal_budget','new':len(ledger['runs']),'old':len(old['runs']),'combined_limit':18})
    records={}
    unchanged=['dataset','protocol','max_epochs','min_epochs','patience','batch_size','learning_rate','optimizer','weight_decay','scheduler','scheduler_T_0','scheduler_T_mult','scheduler_eta_min','dice_weight','bce_weight','precision','geometric_augmentation']
    protocol=read('configs/phase45/F.json')
    for entry in ledger['runs']:
        name=entry['run_id'];d=ROOT/'outputs/phase46/runs'/name
        rc=json.loads((d/'run_config.json').read_text());result=json.loads((d/'result.json').read_text())
        assert rc['config']==entry['config'] and rc['counts']=={'train':24,'val':6}
        assert all(rc['config'][key]==protocol[key] for key in unchanged)
        history=[json.loads(line) for line in (d/'history.jsonl').read_text().splitlines()]
        assert [x['epoch'] for x in history]==list(range(1,result['epochs_completed']+1))
        best=max(history,key=lambda x:x['validation']['dice'])
        assert best['epoch']==result['best_epoch'] and best['validation']['dice']==result['best_validation_dice']
        assert all(x['train_sample_count']==24 and x['validation']['sample_count']==6 and x['text_gradient_l2_min']>0 for x in history)
        if result['stop_reason']=='validation_patience':assert result['epochs_completed']-result['best_epoch']==50
        rows,hashes,features=setup_data(rc['config']);identity=run_identity(rc['config'],hashes,features);assert identity==rc['identity']
        ck=torch.load(d/'best.pt',map_location='cpu',weights_only=False)
        assert ck['identity']==identity and ck['epoch']+1==result['best_epoch'] and ck['config']==rc['config']
        digest=hashlib.sha256()
        for key,tensor in sorted(ck['model_state'].items()):digest.update(key.encode());digest.update(tensor.cpu().contiguous().numpy().tobytes())
        records[name]={'dice_percent':100*result['best_validation_dice'],'best_epoch':result['best_epoch'],'epochs':result['epochs_completed'],
                       'initial_model_sha256':rc['initial_model_sha256'],'best_checkpoint_sha256':sha256_file(d/'best.pt'),
                       'model_tensors_sha256':digest.hexdigest(),'config_sha256':identity['config_sha256'],'feature_cache_sha256':features.sha256}
        del ck,features;gc.collect();checks.append({'check':'training_'+name,'status':'passed'})
    summary=read('outputs/phase46/confirmation/summary.json');assert summary['status'] in ('stopped_initial_gate','stopped_validation_gate','validation_passed')
    means={}
    if summary['status']!='stopped_initial_gate':
        assert len(summary['records'])==12
        for seed in (666,667,668):
            paired=[records[Path(r['run']).name] for r in summary['records'] if r['seed']==seed]
            assert len(paired)==4 and len({r['initial_model_sha256'] for r in paired})==1
        means={arm:statistics.mean(records[Path(r['run']).name]['dice_percent'] for r in summary['records'] if r['arm']==arm) for arm in ('candidate','legacy','constant','shuffled')}
        for arm,value in means.items():assert abs(value-summary['validation_gate']['mean_dice_percent'][arm])<1e-10
        passed=means['candidate']>means['constant']+1e-10 and means['candidate']>means['shuffled']+1e-10 and means['candidate']>=means['legacy']-1e-10
        assert passed==summary['validation_gate']['passed']
        checks.append({'check':'paired_initial_weights_and_three_seed_gate','status':'passed','means':means})
    else:
        paired=[records[Path(r['run']).name] for r in summary['records']]
        assert len(paired)==2 and paired[0]['initial_model_sha256']==paired[1]['initial_model_sha256']
    assert not (ROOT/'outputs/phase46/test_authorization.json').exists()
    assert not any('test' in p.name for p in (ROOT/'cache/phase46').iterdir())
    checks.append({'check':'test_not_accessed'})
    result={'status':'passed','check_count':len(checks),'checks':checks,'formal_runs':len(ledger['runs']),'records':records,
            'experiment_status':summary['status'],'validation_means_percent':means,'test_evaluated':False}
    atomic_json(ROOT/'outputs/phase46/development_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','records')}))

if __name__=='__main__':main()
