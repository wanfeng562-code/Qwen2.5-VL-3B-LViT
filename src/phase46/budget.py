import contextlib,fcntl,json,os,time
from src.experiment.common import ROOT,atomic_json,sha256_text

@contextlib.contextmanager
def locked(name):
    p=ROOT/'outputs/phase46'/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        yield

def claim(run_id,kind,config,resume=False):
    with locked('registry.lock'):
        old=json.loads((ROOT/'outputs/phase45/registry.json').read_text());assert len(old['runs'])==4
        p=ROOT/'outputs/phase46/registry.json'
        ledger=json.loads(p.read_text()) if p.exists() else {'limit':14,'prior_phase45_runs':4,'combined_limit':18,'runs':[]}
        identity=sha256_text(json.dumps(config,sort_keys=True))
        existing=[x for x in ledger['runs'] if x['run_id']==run_id]
        if existing:
            row=existing[0];assert row['config_sha256']==identity and resume and row['status']!='completed'
            row.update(status='running',resumed_at=time.time(),pid=os.getpid())
        else:
            assert len(ledger['runs'])<14 and len(ledger['runs'])+len(old['runs'])<18
            ledger['runs'].append({'run_id':run_id,'kind':kind,'config':config,'config_sha256':identity,'status':'running','started_at':time.time(),'pid':os.getpid(),'slot':len(ledger['runs'])+1})
        atomic_json(p,ledger)

def finish(run_id,status,result=None):
    assert status in ('completed','failed','stopped')
    with locked('registry.lock'):
        p=ROOT/'outputs/phase46/registry.json';ledger=json.loads(p.read_text())
        row=next(x for x in ledger['runs'] if x['run_id']==run_id)
        row.update(status=status,ended_at=time.time(),result=result);atomic_json(p,ledger)
