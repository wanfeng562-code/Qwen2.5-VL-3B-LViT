"""New image -> frozen Qwen -> frozen BERT -> trained LViT, without a mask input."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def resolve(path):
    return (ROOT/path).resolve()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True)
    p.add_argument('--dataset',default='monuseg',choices=['monuseg','cov19','mosmeddata'])
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--generation-config',default='configs/qwen_generation_v2.json')
    p.add_argument('--output-dir',required=True)
    args=p.parse_args()
    start=time.perf_counter()
    image,checkpoint,out=resolve(args.image),resolve(args.checkpoint),resolve(args.output_dir)
    assert image.is_file() and checkpoint.is_file()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'pipeline_result.json').exists():raise FileExistsError('Choose a fresh output directory for a new timed inference')
    import torch
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['config']['source']!='qwen_predicted_text' or saved['config']['dataset']!=args.dataset:
        raise ValueError('Joint inference requires a Qwen-description-trained checkpoint for this dataset')
    if saved['epoch']!=saved['best_epoch']:
        raise ValueError('Joint inference requires the validation-selected checkpoint')
    selected_seed=saved['config']['seed']
    features=resolve(saved['config']['feature_cache'])
    assert digest(features)==saved['identity']['feature_cache_sha256'],'Training feature cache changed'
    metadata=torch.load(features,map_location='cpu',weights_only=False)['metadata']
    del saved
    config=json.loads(resolve(args.generation_config).read_text())
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    assert metadata['generation_config_sha256']==[config_hash],'Prompt/preprocessor differs from training'
    env=os.environ.copy();env.update(OMP_NUM_THREADS='8',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    inherited_fd=os.environ.get('STAGE234_QUEUE_LOCK_FD')
    pass_fds=()
    if inherited_fd is not None:
        inherited_fd=int(inherited_fd);os.fstat(inherited_fd)
        pass_fds=(inherited_fd,)
    def run(argv,label):
        with (out/(label+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-u',*argv],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,pass_fds=pass_fds)
    qwen_start=time.perf_counter()
    run(['scripts/generate_qwen_descriptions.py','--image',str(image),'--dataset',args.dataset,'--config',args.generation_config,'--mode','plain','--output',str(out/'description.jsonl')],'qwen')
    qwen_seconds=time.perf_counter()-qwen_start
    records=[json.loads(line) for line in (out/'description.jsonl').read_text().splitlines()]
    assert len(records)==1
    record=records[0]
    assert record['image_sha256']==digest(image)
    assert record['config_sha256']==config_hash and record['dataset']==args.dataset and record['split']=='new_image'
    assert record['plain_text'].strip() and record['plain_text'].split('\n')[0].strip()
    assert metadata['generation_model_sha256']==[record['model_sha256']],'Qwen weights differ from training descriptions'
    generation_receipt=json.loads((out/'description.receipt.json').read_text())
    assert generation_receipt['status']=='complete' and generation_receipt['count']==1
    assert generation_receipt['config_sha256']==config_hash and generation_receipt['model_sha256']==record['model_sha256']
    assert generation_receipt['cache_sha256']==digest(out/'description.jsonl')
    text_path=out/'description.txt';text_path.write_text(record['plain_text'],encoding='utf-8')
    lvit_start=time.perf_counter()
    run(['scripts/infer_lvit.py','--image',str(image),'--description-file',str(text_path),'--checkpoint',str(checkpoint),'--output-dir',str(out/'segmentation')],'lvit')
    lvit_seconds=time.perf_counter()-lvit_start
    segmentation=json.loads((out/'segmentation/inference.json').read_text())
    assert segmentation['image_sha256']==record['image_sha256'] and segmentation['checkpoint_sha256']==digest(checkpoint)
    assert segmentation['training_source']=='qwen_predicted_text'
    assert (out/'segmentation/mask_224.png').is_file() and (out/'segmentation/probability_224.npy').is_file()
    result={'status':'complete','input_image':str(image),'input_image_sha256':record['image_sha256'],'mask_input':False,'description':record['plain_text'],'checkpoint_sha256':digest(checkpoint),'selected_seed':selected_seed,'qwen_process_wall_seconds':qwen_seconds,'qwen_generation_seconds':record['timing']['plain']['generate_seconds'],'qwen_records_generated_this_invocation':generation_receipt['plain_generated_this_invocation'],'qwen_records_existing_on_entry':generation_receipt['records_existing_on_entry'],'bert_lvit_process_wall_seconds':lvit_seconds,'end_to_end_wall_seconds':time.perf_counter()-start,'generation_config_sha256':config_hash,'qwen_model_sha256':record['model_sha256'],'segmentation':segmentation}
    atomic_json(out/'pipeline_result.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
