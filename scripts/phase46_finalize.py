"""Wait for the existing training queue, then execute the audited terminal branch."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment.common import atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-pid', type=int, required=True)
    args = parser.parse_args()
    out = ROOT / 'outputs/phase46/finalization'
    out.mkdir(parents=True, exist_ok=True)
    stages = []
    started = time.time()

    def status(state, **kwargs):
        atomic_json(out/'status.json',dict(status=state,updated_at=time.time(),started_at=started,stages=stages,**kwargs))

    def child(script, *argv):
        status('running', stage=script)
        with (out/(script+'.log')).open('a') as log:
            process = subprocess.Popen([sys.executable,'-u','scripts/'+script+'.py',*argv],cwd=ROOT,
                env={**os.environ,'OMP_NUM_THREADS':'8','CUBLAS_WORKSPACE_CONFIG':':4096:8'},stdout=log,stderr=subprocess.STDOUT)
            status('running',stage=script,child_pid=process.pid)
            result = process.wait()
        if result:
            raise RuntimeError(f'{script} exited {result}; inspect {out/(script+".log")}')
        stages.append(dict(script=script,completed_at=time.time()))

    with (out/'finalization.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            proc = Path('/proc')/str(args.training_pid)
            if proc.exists():
                assert b'scripts/phase46_confirm.py' in (proc/'cmdline').read_bytes(), 'PID is not the expected controller'
            def training_alive():
                try:
                    if (proc/'stat').read_text().split()[2]=='Z':return False
                    assert b'scripts/phase46_confirm.py' in (proc/'cmdline').read_bytes(), 'Training PID was reused'
                    return True
                except FileNotFoundError:return False
            while training_alive():
                status('waiting_existing_training',training_pid=args.training_pid)
                time.sleep(10)
            summary_path = ROOT/'outputs/phase46/confirmation/summary.json'
            assert summary_path.exists(), 'Controller exited without a terminal summary; do not restart automatically'
            summary = json.loads(summary_path.read_text())
            assert summary['status'] in ('validation_passed','stopped_validation_gate')
            assert len(summary['records'])==12
            # No new training or recovery is launched by this controller.
            if not (ROOT/'outputs/phase46/test_authorization.json').exists():
                child('phase46_audit')
            else:
                audit=json.loads((ROOT/'outputs/phase46/development_audit.json').read_text())
                assert audit['status']=='passed' and audit['experiment_status']=='validation_passed'
            child('phase46_review')
            benchmark=ROOT/'outputs/phase46/benchmark.json'
            if benchmark.exists():
                b=json.loads(benchmark.read_text());assert b['status']=='completed'
                assert b['script_sha256']==sha256_file(ROOT/'scripts/phase46_benchmark.py')
            else:
                child('phase46_benchmark')
            if summary['validation_gate']['passed']:
                replay=ROOT/'outputs/phase46/validation_replay_extraction_receipt.json'
                if not replay.exists():child('phase46_extract_final','--mode','validation_replay')
                child('phase46_freeze_final')
                cache=ROOT/'cache/phase46/V_mid_test.pt'
                if not cache.exists():child('phase46_extract_final','--mode','test')
                else:
                    receipt=json.loads((ROOT/'outputs/phase46/test_extraction_receipt.json').read_text())
                    assert receipt['status']=='completed' and receipt['count']==14
                child('phase46_evaluate_final')
                child('phase46_final_audit')
                child('phase46_final_visuals')
            else:
                assert not (ROOT/'outputs/phase46/test_authorization.json').exists()
                assert not (ROOT/'cache/phase46/V_mid_test.pt').exists()
            child('phase46_report')
            status('ready_for_primary_agent_review',validation_status=summary['status'],test_evaluated=summary['validation_gate']['passed'])
        except Exception:
            status('failed',error=traceback.format_exc())
            raise


if __name__=='__main__':
    main()
