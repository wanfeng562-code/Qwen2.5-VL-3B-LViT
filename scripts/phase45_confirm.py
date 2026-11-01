"""Predefined decisions and matched confirmation. Never invokes test generation."""
import os,sys,json,subprocess,time,fcntl,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiment.common import atomic_json
from src.phase45.engine import configure

def call(argv,log):
    with log.open('a') as f:
        child=subprocess.Popen([sys.executable,'-u',*argv],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'8','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
        atomic_json(ROOT/'outputs/phase45/confirmation/status.json',{'status':'running','argv':argv,'pid':child.pid,'started_at':time.time()})
        code=child.wait()
    if code:raise RuntimeError(f'Child failed ({code}), inspect {log}')

def main():
    out=ROOT/'outputs/phase45/confirmation';out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        screen=json.loads((ROOT/'outputs/phase45/screening/summary.json').read_text())
        sensitivity=ROOT/'outputs/phase45/F_sensitivity.json'
        if not sensitivity.exists():call(['scripts/phase45_sensitivity.py','--checkpoint','outputs/phase45/runs/F_seed666/best.pt','--output','outputs/phase45/F_sensitivity.json'],out/'sensitivity.log')
        sens=json.loads(sensitivity.read_text());attrs=json.loads((ROOT/'cache/phase45/attributes_base_development.scores.json').read_text())
        margin=100*(screen['F']['best_validation_dice']-screen['F_constant']['best_validation_dice'])
        gate={'F_vs_constant_pp':margin,'F_vs_constant_passed':margin>=.3,'permutation_drop_pp':sens['permuted_drop_pp'],'permutation_passed':sens['permuted_drop_pp']>=.2,
              'attribute_accuracy':attrs['val']['field_accuracy'],'attribute_accuracy_below80':attrs['val']['field_accuracy']<.8}
        gate['eligible']=all(gate[k] for k in ('F_vs_constant_passed','permutation_passed','attribute_accuracy_below80'))
        atomic_json(ROOT/'outputs/phase45/lora_gate.json',gate)
        if gate['eligible']:
            atomic_json(out/'status.json',{'status':'lora_gate_open','action':'Perform the pre-registered LoRA branch before selecting candidate'});print('LORA_GATE_OPEN',flush=True);return
        maximum=max(screen[k]['best_validation_dice'] for k in ('P','E','F'))
        selected=next(k for k in ('P','E','F') if maximum-screen[k]['best_validation_dice']<=.001+1e-12)
        config=json.loads((ROOT/f'configs/phase45/{selected}.json').read_text())
        encoding='legacy' if selected=='P' else 'complete'
        selection={'candidate':selected,'seed':666,'scores':{k:screen[k]['best_validation_dice'] for k in ('P','E','F')},'tie_margin_pp':.1,'priority':['P','E','F'],'lora_gate':gate}
        target=ROOT/'configs/phase45/selected.json'
        if target.exists():assert json.loads(target.read_text())==config
        else:atomic_json(target,config)
        atomic_json(out/'selection.json',selection)
        # Evidence-stop: there is no image-text correspondence to perturb.
        # This is reported separately from the three-seed validation gate.
        import torch
        from src.phase45.engine import setup_data
        proofs=[]
        for seed in (666,667,668):
            proper=configure({**config,'seed':seed,'feature_mode':'correct'})
            rows,hashes,correct=setup_data(proper)
            _,_,shuffled=setup_data({**proper,'feature_mode':'shuffled'})
            proofs.append({'seed':seed,'training_images_compared':len(rows['train']),'validation_images_compared':len(rows['val']),
                           'all_input_features_bitwise_equal':all(torch.equal(correct.lookup[k],shuffled.lookup[k]) for k in correct.lookup),
                           'feature_cache_sha256':correct.sha256,'mapping':shuffled.mapping})
        if all(x['all_input_features_bitwise_equal'] for x in proofs):
            stopped={'status':'stopped_degenerate_text','selection':selection,'formal_confirmation_runs':0,'three_seed_training_performed':False,
                     'reason':'All image-specific feature tensors are identical. Correct and shuffled modes are numerically the same experiment for each paired seed; the required +0.5pp informative-text gain cannot be established.',
                     'stop_basis':'User-approved evidence-does-not-support-further-expansion rule; this is earlier than the planned three-seed validation gate.',
                     'input_equivalence':proofs,'records':[],'test_generated':False,'test_evaluated':False}
            atomic_json(out/'summary.json',stopped);atomic_json(out/'status.json',{k:v for k,v in stopped.items() if k not in ('input_equivalence','records')})
            print('STOPPED_DEGENERATE_TEXT',flush=True);return
        cache=f'cache/phase45/provided_legacy_{encoding}_development.pt'
        if not (ROOT/cache).exists():call(['scripts/phase45_cache.py','--source','provided_legacy_text','--encoding',encoding,'--output',cache],out/'legacy_cache.log')
        configurations={
            'candidate':config,
            'legacy':{**config,'source':'provided_legacy_text','feature_cache':cache,'feature_mode':'correct'},
            'constant':{**config,'feature_mode':'constant'},
            'shuffled':{**config,'feature_mode':'shuffled'},
        }
        records=[]
        for arm,cfg in configurations.items():
            cp=ROOT/f'configs/phase45/confirm_{arm}.json'
            if cp.exists():assert json.loads(cp.read_text())==cfg
            else:atomic_json(cp,cfg)
            for seed in (666,667,668):
                run_name=f'{selected}_seed666' if arm=='candidate' and seed==666 else f'confirm_{arm}_seed{seed}'
                reused=arm=='candidate' and seed==666
                if arm=='constant' and seed==666 and selected=='F':run_name='F_constant_seed666';reused=True
                run=ROOT/'outputs/phase45/runs'/run_name;run.mkdir(parents=True,exist_ok=True)
                if (run/'result.json').exists():
                    rc=json.loads((run/'run_config.json').read_text())['config'];assert rc==configure({**cfg,'seed':seed})
                else:
                    argv=['scripts/phase45_train.py','--config',str(cp.relative_to(ROOT)),'--seed',str(seed),'--output-dir',str(run.relative_to(ROOT))]
                    if (run/'last.pt').exists():argv+=['--resume',str(run/'last.pt')]
                    call(argv,run/'run.log')
                result=json.loads((run/'result.json').read_text());assert result['status']=='completed'
                records.append({'arm':arm,'seed':seed,'run':str(run.relative_to(ROOT)),'reused_screening':reused,**result})
                atomic_json(out/'progress.json',{'selection':selection,'records':records})
                print('CONFIRMED',arm,seed,result['best_validation_dice'],flush=True)
        means={arm:100*statistics.mean(x['best_validation_dice'] for x in records if x['arm']==arm) for arm in configurations}
        validation_gate={'mean_dice_percent':means,'beats_constant':means['candidate']>means['constant']+1e-10,'beats_shuffled':means['candidate']>means['shuffled']+1e-10,'not_below_legacy':means['candidate']>=means['legacy']-1e-10}
        validation_gate['passed']=all(validation_gate[k] for k in ('beats_constant','beats_shuffled','not_below_legacy'))
        summary={'status':'validation_passed' if validation_gate['passed'] else 'stopped_at_validation_gate','selection':selection,'records':records,'validation_gate':validation_gate,'test_generated':False,'test_evaluated':False}
        atomic_json(out/'summary.json',summary);atomic_json(out/'status.json',{'status':summary['status'],'validation_gate':validation_gate})
        print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
