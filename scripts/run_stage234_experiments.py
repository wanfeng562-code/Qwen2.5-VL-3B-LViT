"""Serial, restartable phase 2/4 MoNuSeg experiment queue after frozen pilot.

Test evaluation happens only after all six training runs finish. A failure stops
the queue and records the failed job; no parameter or prompt tuning is automatic.
"""
import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/stage234/formal_experiments'
SEEDS=(666,667,668)
ARMS=('legacy','qwen')
QUEUE_LOCK_FD=None
CONTRACTS={}


def read(path):return json.loads((ROOT/path).read_text())


def sha(path):
    h=hashlib.sha256()
    with (ROOT/path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic(path,obj):
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w') as handle:
        json.dump(obj,handle,indent=2,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temp,path)


def process_start_ticks(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None


def reject_active_job(folder):
    path=folder/'process.json'
    if not path.exists():return
    previous=json.loads(path.read_text());pid=int(previous['pid'])
    if pid<=0:raise RuntimeError(f'Invalid recorded job PID: {path}')
    try:
        raw=(Path('/proc')/str(pid)/'cmdline').read_bytes()
    except FileNotFoundError:return
    if not raw:return  # A terminated zombie no longer computes or writes output.
    actual=[part.decode(errors='replace') for part in raw.split(b'\0') if part]
    expected=[sys.executable,'-u',*previous['argv']]
    ticks=process_start_ticks(pid)
    matching=actual==expected and (not previous.get('start_ticks') or ticks==previous['start_ticks'])
    detail='recorded child is still running' if matching else 'recorded PID is alive with a different command/start time; audit stale process receipt'
    raise RuntimeError(f'Refusing duplicate job start: {folder.name}: {detail}; PID={pid}; cmdline={actual}')


def validate_plain_cache():
    plain='cache/qwen/monuseg_plain_v2.jsonl'
    receipt=read('cache/qwen/monuseg_plain_v2.receipt.json')
    freeze=read('docs/stage234_freeze.json')
    assert receipt['status']=='complete' and receipt['count']==44
    assert receipt['config_sha256']==freeze['generation_config_sha256']
    assert receipt.get('model_sha256'),'missing actual Qwen model fingerprint'
    if receipt.get('cache_sha256'):assert receipt['cache_sha256']==sha(plain)
    rows={}
    for split in ('train','val','test'):
        with (ROOT/f'manifests/legacy_subset_v1/monuseg/{split}.csv').open() as handle:
            for row in csv.DictReader(handle):rows[row['sample_id']]=row
    done={}
    for line in (ROOT/plain).read_text().splitlines():
        record=json.loads(line);sid=record['sample_id']
        assert sid in rows and sid not in done,'unexpected/duplicate plain sample'
        for key in ('dataset','split','image_path','image_sha256'):assert record[key]==rows[sid][key]
        assert record['config_sha256']==receipt['config_sha256'] and record['model_sha256']==receipt['model_sha256']
        assert isinstance(record.get('plain_text'),str) and record['plain_text'].strip() and record['plain_text'].split('\n')[0].strip()
        done[sid]=record
    assert set(done)==set(rows) and len(done)==44
    return receipt


def current_contract(arm,seed):
    key=(arm,seed)
    if key not in CONTRACTS:
        from src.experiment.engine import configure,setup_data,run_identity
        config=read(f'configs/training_monuseg_{arm}_v1.json');config['seed']=seed
        config=configure(config);rows,hashes,features=setup_data(config)
        CONTRACTS[key]=(config,run_identity(config,hashes,features),rows)
    return CONTRACTS[key]


def completed_training(base,arm,seed):
    if not (ROOT/base/'result.json').exists():return False
    result=read(base+'/result.json')
    if result['status']!='completed':return False
    config,identity,rows=current_contract(arm,seed)
    run_config=read(base+'/run_config.json')
    assert run_config['config']==config and run_config['identity']==identity,'completed training differs from current frozen contract'
    assert result['test_evaluated'] is False and result['epochs_completed']>=1
    expected_steps=result['epochs_completed']*((len(rows['train'])+config['batch_size']-1)//config['batch_size'])
    assert result['global_step']==expected_steps
    import torch
    for name,epoch in [('best.pt',result['best_epoch']),('last.pt',result['epochs_completed'])]:
        checkpoint=torch.load(ROOT/base/name,map_location='cpu',weights_only=False)
        assert checkpoint['identity']==identity and checkpoint['config']==config
        assert checkpoint['epoch']+1==epoch
        if name=='best.pt':assert checkpoint['epoch']==checkpoint['best_epoch']
        else:assert checkpoint['global_step']==expected_steps
        del checkpoint
    return True


def completed_test(base,arm,seed):
    path=base+'/test/test_metrics.json'
    if not (ROOT/path).exists():return False
    result=read(path);config,identity,rows=current_contract(arm,seed)
    assert result['dataset']==config['dataset'] and result['source']==config['source'] and result['seed']==seed
    assert result['sample_count']==len(rows['test'])==14 and result['manifest_sha256']==identity['manifest_sha256']['test']
    assert result['checkpoint_sha256']==sha(base+'/best.pt') and result['code_matches_training'] is True
    assert result['checkpoint_epoch']==result['checkpoint_best_epoch']==read(base+'/result.json')['best_epoch']
    assert len(result['samples'])==14 and {row['sample_id'] for row in result['samples']}=={row['sample_id'] for row in rows['test']}
    return True


def completed_joint(joint,image,checkpoint):
    path=joint+'/pipeline_result.json'
    if not (ROOT/path).exists():return False
    result=read(path)
    assert result['status']=='complete' and result['mask_input'] is False
    assert result['input_image_sha256']==sha(image) and result['checkpoint_sha256']==sha(checkpoint)
    assert result['generation_config_sha256']==read('docs/stage234_freeze.json')['generation_config_sha256']
    receipt=validate_plain_cache()
    assert result['qwen_model_sha256']==receipt['model_sha256']
    segmented=read(joint+'/segmentation/inference.json')
    assert segmented==result['segmentation'] and segmented['training_source']=='qwen_predicted_text'
    assert segmented['image_sha256']==result['input_image_sha256'] and segmented['checkpoint_sha256']==result['checkpoint_sha256']
    assert (ROOT/joint/'segmentation/mask_224.png').is_file() and (ROOT/joint/'segmentation/probability_224.npy').is_file()
    return True


def wait_pilot():
    state=read('outputs/stage234/qwen_pilot_v2/process.json')
    proc=Path('/proc')/str(state['pid'])
    while proc.exists():
        cmd=(proc/'cmdline').read_bytes()
        if not cmd:break
        if b'generate_qwen_descriptions.py' not in cmd:raise RuntimeError('Pilot PID has been reused')
        print('WAITING_PILOT_V2',state['pid'],flush=True);time.sleep(15)
    receipt=read('cache/qwen/pilot_v2.receipt.json')
    assert receipt['status']=='complete' and receipt['count']==60
    freeze=read('docs/stage234_freeze.json')
    assert receipt['config_sha256']==freeze['generation_config_sha256']


def run(label,argv,monitor_gpu=False):
    folder=OUT/label;folder.mkdir(exist_ok=True,parents=True)
    reject_active_job(folder)
    if QUEUE_LOCK_FD is None:raise RuntimeError('Queue lock must be held before starting a child')
    print('START',label,flush=True)
    atomic(OUT/'status.json',{'status':'running','job':label,'argv':argv,'started_at':time.time()})
    env=os.environ.copy();env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1',STAGE234_QUEUE_LOCK_FD=str(QUEUE_LOCK_FD))
    monitor=None
    start=time.perf_counter()
    if monitor_gpu:
        from validate_qwen import GPUMonitor
        monitor=GPUMonitor(folder/'gpu_memory.csv');monitor.start()
    try:
        with (folder/'run.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-u',*argv],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(QUEUE_LOCK_FD,))
            atomic(folder/'process.json',{'pid':process.pid,'argv':argv,'start_ticks':process_start_ticks(process.pid),'started_at':time.time()})
            code=process.wait()
        receipt={'exit_code':code,'wall_seconds':time.perf_counter()-start,'argv':argv}
        if monitor is not None:receipt['gpu']=monitor.close();monitor=None
        atomic(folder/'receipt.json',receipt)
        if code:raise RuntimeError(f'{label} failed with exit {code}; inspect {folder}/run.log')
        print('COMPLETED',label,receipt['wall_seconds'],flush=True)
    finally:
        if monitor is not None:monitor.close()


def verify_pair():
    configs={arm:read(f'configs/training_monuseg_{arm}_v1.json') for arm in ARMS}
    a={k:v for k,v in configs['legacy'].items() if k not in ('source','feature_cache')}
    b={k:v for k,v in configs['qwen'].items() if k not in ('source','feature_cache')}
    assert a==b,'Training budgets/settings must match'
    import torch
    stores={arm:torch.load(ROOT/configs[arm]['feature_cache'],map_location='cpu',weights_only=False) for arm in ARMS}
    assert stores['legacy']['sample_ids']==stores['qwen']['sample_ids']
    for key in ('encoder','protocol_sha256','manifest_sha256'):
        assert stores['legacy']['metadata'][key]==stores['qwen']['metadata'][key],key
    freeze=read('docs/stage234_freeze.json')
    assert stores['qwen']['metadata']['generation_config_sha256']==[freeze['generation_config_sha256']]
    receipt=validate_plain_cache()
    assert stores['qwen']['metadata']['generation_model_sha256']==[receipt['model_sha256']]
    assert stores['qwen']['metadata']['descriptions_sha256']==sha('cache/qwen/monuseg_plain_v2.jsonl')
    assert len(stores['qwen']['sample_ids'])==44
    evidence={'status':'passed','seeds':list(SEEDS),'same_configuration_except':['source','feature_cache'],'sample_count':44,'same_text_encoder':True,'same_protocol_and_manifests':True,'test_selection':'best validation Dice, tie earliest','generation_frozen_before_test':freeze['frozen_before_test_generation'],'config_sha256':{arm:sha(f'configs/training_monuseg_{arm}_v1.json') for arm in ARMS},'feature_cache_sha256':{arm:sha(configs[arm]['feature_cache']) for arm in ARMS}}
    atomic(OUT/'comparison_contract.json',evidence)


def summarize():
    records=[]
    for arm in ARMS:
        for seed in SEEDS:
            base=f'outputs/lvit/monuseg/{arm}_seed{seed}'
            training=read(base+'/result.json');evaluation=read(base+'/test/test_metrics.json')
            assert evaluation['sample_count']==14 and evaluation['seed']==seed
            record={'arm':arm,'seed':seed,'dice_percent':evaluation['dice_percent'],'iou_percent':evaluation['iou_percent'],'epochs':training['epochs_completed'],'best_epoch':training['best_epoch'],'train_seconds':training['training_elapsed_seconds'],'peak_allocated_gib':training['peak_allocated_bytes']/1024**3,'checkpoint_sha256':evaluation['checkpoint_sha256'],'evaluation_path':base+'/test/test_metrics.json','train_result_path':base+'/result.json'}
            records.append(record)
    aggregates={}
    for arm in ARMS:
        subset=[r for r in records if r['arm']==arm]
        aggregates[arm]={k:{'mean':statistics.mean(x[k] for x in subset),'sample_std':statistics.stdev(x[k] for x in subset)} for k in ('dice_percent','iou_percent','train_seconds')}
    paired=[]
    for seed in SEEDS:
        legacy=next(r for r in records if r['arm']=='legacy' and r['seed']==seed)
        qwen=next(r for r in records if r['arm']=='qwen' and r['seed']==seed)
        paired.append({'seed':seed,'dice_percentage_points':qwen['dice_percent']-legacy['dice_percent'],'iou_percentage_points':qwen['iou_percent']-legacy['iou_percent']})
    summary={'status':'complete','dataset':'monuseg','protocol':'legacy_subset_v1','records':records,'aggregate':aggregates,'paired_differences':paired,'comparison_limitation':'provided legacy text provenance is unverified; results compare text-input conditions, not an image-only ablation','test_count_per_run':14,'seeds':list(SEEDS)}
    atomic(OUT/'summary.json',summary)
    print('EXPERIMENT_RESULTS',json.dumps(summary),flush=True)


def main():
    global QUEUE_LOCK_FD
    OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (OUT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        QUEUE_LOCK_FD=lock.fileno()
        try:
            for path in OUT.glob('*/process.json'):reject_active_job(path.parent)
            assert read('outputs/lvit_checks/legacy_full/checks.json')['status']=='passed'
            assert read('outputs/lvit_checks/engine_smoke/verification.json')['status']=='passed'
            wait_pilot()
            plain='cache/qwen/monuseg_plain_v2.jsonl'
            if not (ROOT/'cache/qwen/monuseg_plain_v2.receipt.json').exists():
                run('descriptions_all_splits',['scripts/generate_qwen_descriptions.py','--dataset','monuseg','--config','configs/qwen_generation_v2.json','--mode','plain','--output',plain,'--reuse-plain','cache/qwen/pilot_v2.jsonl'])
            validate_plain_cache()
            feature='cache/text_features/monuseg_qwen_predicted_text_v1.pt'
            if not (ROOT/feature).exists():
                run('qwen_features',['scripts/cache_text_features.py','--dataset','monuseg','--source','qwen_predicted_text','--descriptions',plain,'--output',feature,'--device','cpu'])
            verify_pair()
            run('qwen_data_check',['scripts/check_lvit.py','--config','configs/training_monuseg_qwen_v1.json','--output-dir','outputs/lvit_checks/qwen_data_cpu','--mode','data','--device','cpu'])
            for seed in SEEDS:
                for arm in ARMS:
                    base=f'outputs/lvit/monuseg/{arm}_seed{seed}'
                    if completed_training(base,arm,seed):
                        print('REUSE_COMPLETED_TRAINING',base,flush=True);continue
                    argv=['scripts/train_lvit.py','--config',f'configs/training_monuseg_{arm}_v1.json','--seed',str(seed),'--output-dir',base]
                    if (ROOT/base/'last.pt').exists():argv+=['--resume',base+'/last.pt']
                    run(f'train_{arm}_{seed}',argv,monitor_gpu=True)
            # No test scores are inspected until every training setting has run.
            for seed in SEEDS:
                for arm in ARMS:
                    base=f'outputs/lvit/monuseg/{arm}_seed{seed}'
                    assert completed_training(base,arm,seed)
                    if completed_test(base,arm,seed):continue
                    run(f'test_{arm}_{seed}',['scripts/train_lvit.py','--evaluate-only','--checkpoint',base+'/best.pt','--split','test','--output-dir',base+'/test','--save-predictions'])
                    assert completed_test(base,arm,seed)
            summarize()
            with (ROOT/'manifests/legacy_subset_v1/monuseg/test.csv').open() as f:
                sample=sorted(csv.DictReader(f),key=lambda r:r['sample_id'])[0]
            joint='outputs/stage234/joint_inference'
            checkpoint='outputs/lvit/monuseg/qwen_seed666/best.pt'
            if not completed_joint(joint,sample['image_path'],checkpoint):
                run('joint_inference',['scripts/infer_qwen_lvit.py','--image',sample['image_path'],'--dataset','monuseg','--checkpoint',checkpoint,'--output-dir',joint])
            assert completed_joint(joint,sample['image_path'],checkpoint)
            atomic(OUT/'status.json',{'status':'completed','summary':'outputs/stage234/formal_experiments/summary.json','joint_inference':joint+'/pipeline_result.json'})
            print('ALL_COMPUTE_COMPLETE_VISUAL_AND_FINAL_AUDIT_REQUIRED',flush=True)
        except Exception as exc:
            atomic(OUT/'failure.json',{'error':repr(exc),'at':time.time()})
            atomic(OUT/'status.json',{'status':'failed','error':repr(exc)})
            raise


if __name__=='__main__':main()
