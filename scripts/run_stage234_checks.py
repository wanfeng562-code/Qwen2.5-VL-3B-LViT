"""One-shot guarded queue: finish live Qwen pilot, then real LViT check and v2 calibration."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/stage234/readiness_queue'
out.mkdir(parents=True,exist_ok=True)
state=json.loads((ROOT/'outputs/stage234/qwen_pilot_v1/process.json').read_text())
pid=state['pid']
while True:
    proc=Path('/proc')/str(pid)
    if not proc.exists():break
    cmd=(proc/'cmdline').read_bytes()
    if not cmd:break
    if b'generate_qwen_descriptions.py' not in cmd:raise RuntimeError('PID reused; refusing wait assumption')
    print('WAITING_QWEN_PILOT',pid,flush=True)
    time.sleep(10)
receipt=json.loads((ROOT/'cache/qwen/pilot_v1.receipt.json').read_text())
assert receipt['status']=='complete' and receipt['count']==60
jobs=[
 ('lvit_real_check',['scripts/check_lvit.py','--config','configs/training_monuseg_legacy_v1.json','--output-dir','outputs/lvit_checks/legacy_full','--mode','full','--steps','100']),
 ('qwen_calibration_v2',['scripts/generate_qwen_descriptions.py','--selection','configs/qwen_calibration_v2.json','--config','configs/qwen_generation_v2.json','--output','cache/qwen/calibration_v2.jsonl','--mode','both'])
]
try:
 for label,argv in jobs:
  logdir=ROOT/'outputs/stage234'/label;logdir.mkdir(parents=True,exist_ok=True)
  print('START',label,flush=True)
  with (logdir/'run.log').open('w') as f:
   child=subprocess.Popen([sys.executable,'-u',*argv],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   (logdir/'process.json').write_text(json.dumps({'pid':child.pid,'argv':argv}))
   code=child.wait()
  assert code==0,f'{label} failed exit {code}'
  print('COMPLETED',label,flush=True)
 (out/'result.json').write_text(json.dumps({'status':'complete','jobs':[j[0] for j in jobs]}))
except Exception as exc:
 (out/'result.json').write_text(json.dumps({'status':'failed','error':repr(exc)}))
 raise
