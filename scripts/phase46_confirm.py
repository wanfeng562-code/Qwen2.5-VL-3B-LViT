"""Locked selection and matched controls; cannot launch any test evaluation."""
import os,sys,json,time,subprocess,fcntl,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiment.common import atomic_json,sha256_file
from src.phase46.engine import configure

def call(argv,log,out):
    with log.open('a') as f:
        p=subprocess.Popen([sys.executable,'-u',*argv],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'8','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
        atomic_json(out/'status.json',{'status':'running','argv':argv,'pid':p.pid,'started_at':time.time()})
        code=p.wait()
    if code:raise RuntimeError(f'Child failed ({code}); inspect {log}')

def main():
    out=ROOT/'outputs/phase46/confirmation';out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert json.loads((ROOT/'outputs/phase46/screening/status.json').read_text())['status']=='completed'
        screen=json.loads((ROOT/'outputs/phase46/screening/summary.json').read_text());maximum=max(v['best_validation_dice'] for v in screen.values())
        selected=next(k for k in ('S','V_mid','V_merged') if k in screen and maximum-screen[k]['best_validation_dice']<=.001+1e-12)
        config=json.loads((ROOT/f'configs/phase46/{selected}.json').read_text())
        selection={'candidate':selected,'seed':666,'scores':{k:v['best_validation_dice'] for k,v in screen.items()},'tie_margin_pp':.1,'priority':['S','V_mid','V_merged']}
        for path,value in [(out/'selection.json',selection),(ROOT/'configs/phase46/selected.json',config)]:
            if path.exists():assert json.loads(path.read_text())==value
            else:atomic_json(path,value)
        def run(arm,seed):
            cfg={**config,'feature_mode':{'candidate':'correct','constant':'constant','legacy':'legacy','shuffled':'shuffled'}[arm]}
            cp=ROOT/f'configs/phase46/confirm_{arm}.json'
            if cp.exists():assert json.loads(cp.read_text())==cfg
            else:atomic_json(cp,cfg)
            name=f'{selected}_seed666' if arm=='candidate' and seed==666 else f'confirm_{arm}_seed{seed}'
            directory=ROOT/'outputs/phase46/runs'/name;directory.mkdir(parents=True,exist_ok=True)
            if (directory/'result.json').exists():assert json.loads((directory/'run_config.json').read_text())['config']==configure({**cfg,'seed':seed})
            else:
                argv=['scripts/phase46_train.py','--config',str(cp.relative_to(ROOT)),'--seed',str(seed),'--output-dir',str(directory.relative_to(ROOT))]
                if (directory/'last.pt').exists():argv+=['--resume',str(directory/'last.pt')]
                call(argv,directory/'run.log',out)
            result=json.loads((directory/'result.json').read_text());assert result['status']=='completed'
            return {'arm':arm,'seed':seed,'run':str(directory.relative_to(ROOT)),**result}
        proper=run('candidate',666);constant=run('constant',666)
        sensitivity=out/'seed666_sensitivity.json'
        if not sensitivity.exists():call(['scripts/phase46_sensitivity.py','--checkpoint',proper['run']+'/best.pt','--output',str(sensitivity.relative_to(ROOT))],out/'sensitivity.log',out)
        sens=json.loads(sensitivity.read_text());initial_gate={'beats_constant':proper['best_validation_dice']>constant['best_validation_dice'],
            'correct_minus_constant_pp':100*(proper['best_validation_dice']-constant['best_validation_dice']),'any_prediction_changed':sens['any_prediction_changed'],
            'permuted_drop_pp':sens['permuted_drop_pp'],'mean_binary_changed_fraction':sens['mean_binary_changed_fraction']}
        initial_gate['passed']=initial_gate['beats_constant'] and initial_gate['any_prediction_changed'];atomic_json(out/'initial_gate.json',initial_gate)
        records=[proper,constant]
        if not initial_gate['passed']:
            stopped={'status':'stopped_initial_gate','selection':selection,'initial_gate':initial_gate,'records':records,'test_evaluated':False}
            atomic_json(out/'summary.json',stopped);atomic_json(out/'status.json',stopped);print(json.dumps(stopped));return
        for arm,seeds in [('candidate',(667,668)),('legacy',(666,667,668)),('constant',(667,668)),('shuffled',(666,667,668))]:
            for seed in seeds:
                records.append(run(arm,seed));atomic_json(out/'progress.json',{'selection':selection,'records':records,'initial_gate':initial_gate})
                print('COMPLETE',arm,seed,records[-1]['best_validation_dice'],flush=True)
        means={arm:100*statistics.mean(x['best_validation_dice'] for x in records if x['arm']==arm) for arm in ('candidate','legacy','constant','shuffled')}
        gate={'mean_dice_percent':means,'beats_constant':means['candidate']>means['constant']+1e-10,'beats_shuffled':means['candidate']>means['shuffled']+1e-10,'not_below_legacy':means['candidate']>=means['legacy']-1e-10}
        gate['passed']=all(gate[k] for k in ('beats_constant','beats_shuffled','not_below_legacy'))
        summary={'status':'validation_passed' if gate['passed'] else 'stopped_validation_gate','selection':selection,'initial_gate':initial_gate,'records':records,'validation_gate':gate,'test_evaluated':False}
        atomic_json(out/'summary.json',summary);atomic_json(out/'status.json',{'status':summary['status'],'validation_gate':gate});print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
