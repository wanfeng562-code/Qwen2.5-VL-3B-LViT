import os,sys,json,subprocess,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiment.common import atomic_json,sha256_file
from src.phase46.engine import configure

def main():
    out=ROOT/'outputs/phase46/screening';out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert json.loads((ROOT/'outputs/phase46/preflight/checks.json').read_text())['status']=='passed'
        for name,expected in json.loads((ROOT/'docs/phase46_core_freeze.json').read_text()).items():assert sha256_file(ROOT/name)==expected
        enabled=json.loads((ROOT/'cache/phase46/candidate_gate.json').read_text())['enabled'];results={}
        for name in ('S','V_mid','V_merged'):
            if not enabled[name]:continue
            run=ROOT/f'outputs/phase46/runs/{name}_seed666';run.mkdir(parents=True,exist_ok=True)
            config=json.loads((ROOT/f'configs/phase46/{name}.json').read_text())
            if (run/'result.json').exists():
                assert json.loads((run/'run_config.json').read_text())['config']==configure(config)
                results[name]=json.loads((run/'result.json').read_text());assert results[name]['status']=='completed';continue
            argv=[sys.executable,'-u','scripts/phase46_train.py','--config',f'configs/phase46/{name}.json','--seed','666','--output-dir',str(run.relative_to(ROOT))]
            if (run/'last.pt').exists():argv+=['--resume',str(run/'last.pt')]
            with (run/'run.log').open('a') as f:
                child=subprocess.Popen(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'8','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
                atomic_json(out/'status.json',{'status':'running','candidate':name,'pid':child.pid,'started_at':time.time()})
                code=child.wait()
            if code:
                atomic_json(out/'status.json',{'status':'failed','candidate':name,'exit_code':code});raise RuntimeError(f'{name} failed; inspect its run.log')
            results[name]=json.loads((run/'result.json').read_text());atomic_json(out/'summary.json',results)
            print('COMPLETE',name,results[name],flush=True)
        atomic_json(out/'summary.json',results);atomic_json(out/'status.json',{'status':'completed','runs':len(results),'skipped':[k for k,v in enabled.items() if not v]})

if __name__=='__main__':main()
