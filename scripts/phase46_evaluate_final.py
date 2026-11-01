"""One locked final re-evaluation of all selected/control checkpoints."""
import os,sys,json,csv,statistics,fcntl
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.experiment.common import atomic_json,sha256_file,seed_everything,gpu_memory
from src.experiment.data import ManifestDataset,make_loader
from src.experiment.loss import WeightedDiceBCE
from src.phase46.engine import setup_data,run_identity,evaluate,load_checkpoint
from src.phase46.model import build_model
from scripts.phase46_final_data import conditions

def read(p):return json.loads((ROOT/p).read_text())

def main():
    torch.set_num_threads(8)
    auth=read('outputs/phase46/test_authorization.json');assert auth['validation_gate_passed']
    for name,expected in auth['evaluation_scripts'].items():assert sha256_file(ROOT/name)==expected
    assert sha256_file(ROOT/'outputs/phase46/confirmation/summary.json')==auth['validation_summary_sha256']
    assert sha256_file(ROOT/'cache/phase46/V_mid_development.pt')==auth['development_cache_sha256']
    manifest=ROOT/'manifests/legacy_subset_v1/monuseg/test.csv';assert sha256_file(manifest)==auth['test_manifest_sha256']
    with manifest.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==14 and all(r['split']=='test' for r in rows)
    for row in rows:
        for key in ('image','mask'):
            p=(ROOT/row[key+'_path']).resolve();assert p.is_relative_to(ROOT.resolve()) and sha256_file(p)==row[key+'_sha256']
    heldout=torch.load(ROOT/'cache/phase46/V_mid_test.pt',map_location='cpu',weights_only=False)
    assert heldout['sample_ids']==[r['sample_id'] for r in rows]
    assert heldout['metadata']['manifest_sha256']==auth['test_manifest_sha256']
    assert heldout['metadata']['development_cache_sha256']==auth['development_cache_sha256']
    assert heldout['metadata']['authorization_sha256']==sha256_file(ROOT/'outputs/phase46/test_authorization.json')
    assert heldout['metadata']['extractor_sha256']==auth['evaluation_scripts']['scripts/phase46_extract_final.py']
    for row,record in zip(rows,heldout['records']):assert all(row[k]==record[k] for k in ('sample_id','split','image_sha256','image_path'))
    dev=torch.load(ROOT/'cache/phase46/V_mid_development.pt',map_location='cpu',weights_only=False)
    output=ROOT/'outputs/phase46/final_test';output.mkdir(parents=True,exist_ok=True)
    results=[]
    with (output/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for record in auth['records']:
            arm,seed=record['arm'],record['seed'];checkpoint=record['run']+'/best.pt'
            expected=auth['checkpoints'][checkpoint];assert sha256_file(ROOT/checkpoint)==expected
            cfg=read(record['run']+'/run_config.json')['config']
            lookup,mapping=conditions(dev,heldout,cfg['feature_mode'],seed)
            directory=output/f'{arm}_seed{seed}';directory.mkdir(exist_ok=True)
            identity={'authorization_sha256':sha256_file(ROOT/'outputs/phase46/test_authorization.json'),
                      'test_cache_sha256':sha256_file(ROOT/'cache/phase46/V_mid_test.pt'),'checkpoint_sha256':expected,
                      'test_manifest_sha256':auth['test_manifest_sha256'],'mapping':mapping}
            saved=directory/'metrics.json'
            if saved.exists():
                result=json.loads(saved.read_text());assert result['final_identity']==identity
                assert len(list((directory/'predictions').glob('*.png')))==14
            else:
                seed_everything(seed,cfg['deterministic'])
                _,hashes,features=setup_data(cfg);run_id=run_identity(cfg,hashes,features)
                model=build_model(cfg).cuda().eval();load_checkpoint(checkpoint,model,expected_identity=run_id)
                loader=make_loader(ManifestDataset(rows,SimpleNamespace(lookup=lookup),cfg['preload']),cfg['batch_size'],seed,workers=cfg['workers'],device='cuda')
                torch.cuda.reset_peak_memory_stats()
                result=evaluate(model,loader,WeightedDiceBCE(cfg['dice_weight'],cfg['bce_weight']),'cuda',prediction_dir=directory/'predictions')
                assert result['sample_count']==14
                result.update(arm=arm,seed=seed,final_identity=identity,scope=auth['scope'],**gpu_memory('cuda'))
                atomic_json(saved,result);del model,features;torch.cuda.empty_cache()
            results.append(result)
            atomic_json(output/'progress.json',{'status':'running','finished':len(results),'total':12})
            print('EVALUATED',arm,seed,flush=True)
        aggregates={}
        for arm in ('candidate','constant','shuffled','legacy'):
            part=[r for r in results if r['arm']==arm];assert {r['seed'] for r in part}=={666,667,668}
            aggregates[arm]={metric:{'mean_percent':100*statistics.mean(r[metric] for r in part),'sample_sd_percent':100*statistics.stdev(r[metric] for r in part)} for metric in ('dice','iou')}
        old={seed:read(f'outputs/lvit/monuseg/legacy_seed{seed}/test/test_metrics.json') for seed in (666,667,668)}
        seed_deltas={str(r['seed']):100*(r['dice']-old[r['seed']]['dice']) for r in results if r['arm']=='candidate'}
        d=aggregates['candidate']['dice']['mean_percent'];i=aggregates['candidate']['iou']['mean_percent'];rule=auth['success']
        gates={'dice_target':d>=rule['mean_test_dice_percent_at_least'],'iou_target':i>=rule['mean_test_iou_percent_at_least'],
               'beats_fixed':d-aggregates['constant']['dice']['mean_percent']>=rule['beats_fixed_and_shuffled_pp'],
               'beats_shuffled':d-aggregates['shuffled']['dice']['mean_percent']>=rule['beats_fixed_and_shuffled_pp'],
               'not_below_matched_legacy':d>=aggregates['legacy']['dice']['mean_percent'],
               'two_seeds_improved':sum(value>0 for value in seed_deltas.values())>=2}
        pairs=[]
        for row in rows:
            sid=row['sample_id'];item={'sample_id':sid}
            for arm in aggregates:
                part=[r for r in results if r['arm']==arm]
                item[arm+'_dice_percent']=100*statistics.mean(next(x['dice'] for x in r['samples'] if x['sample_id']==sid) for r in part)
            for arm in ('constant','shuffled','legacy'):item['candidate_minus_'+arm+'_pp']=item['candidate_dice_percent']-item[arm+'_dice_percent']
            item['old_legacy_dice_percent']=100*statistics.mean(next(x['dice'] for x in r['samples'] if x['sample_id']==sid) for r in old.values())
            item['candidate_minus_old_pp']=item['candidate_dice_percent']-item['old_legacy_dice_percent'];pairs.append(item)
        with (output/'paired_per_image.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(pairs[0]),lineterminator='\n');writer.writeheader();writer.writerows(pairs)
        summary={'status':'completed','scope':auth['scope'],'aggregates':aggregates,'seed_deltas_vs_old_pp':seed_deltas,'success_gates':gates,
                 'target_achieved':all(gates.values()),'results':results,'per_image':pairs,'no_test_based_selection':True}
        atomic_json(output/'summary.json',summary);atomic_json(output/'status.json',{k:v for k,v in summary.items() if k not in ('results','per_image')})
        print(json.dumps({k:v for k,v in summary.items() if k not in ('results','per_image')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
