import os,sys,json,subprocess,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiment.common import atomic_json,sha256_file

def main():
    out=ROOT/'outputs/phase45/screening';out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert json.loads((ROOT/'outputs/phase45/preflight/checks.json').read_text())['status']=='passed'
        results={}
        for name in ('P','E','F','F_constant'):
            run=ROOT/f'outputs/phase45/runs/{name}_seed666';run.mkdir(parents=True,exist_ok=True)
            if (run/'result.json').exists():
                results[name]=json.loads((run/'result.json').read_text());assert results[name]['status']=='completed';continue
            argv=[sys.executable,'-u','scripts/phase45_train.py','--config',f'configs/phase45/{name}.json','--seed','666','--output-dir',str(run.relative_to(ROOT))]
            if (run/'last.pt').exists():argv+=['--resume',str(run/'last.pt')]
            with (run/'run.log').open('a') as f:
                child=subprocess.Popen(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'8','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
                atomic_json(out/'status.json',{'status':'running','candidate':name,'pid':child.pid,'started_at':time.time()})
                code=child.wait()
            if code:
                atomic_json(out/'status.json',{'status':'failed','candidate':name,'exit_code':code});raise RuntimeError(f'{name} failed; see its run.log')
            results[name]=json.loads((run/'result.json').read_text())
            print('COMPLETE',name,results[name],flush=True)
        atomic_json(out/'summary.json',results);atomic_json(out/'status.json',{'status':'completed','runs':4})

if __name__=='__main__':main()
