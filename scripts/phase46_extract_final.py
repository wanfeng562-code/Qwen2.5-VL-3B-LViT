"""Extract selected visual features after the frozen validation authorization."""
import os,sys,json,time,hashlib,argparse,csv
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from src.experiment.common import atomic_json,atomic_torch_save,sha256_file,sha256_text
from src.experiment.text import load_descriptions
from src.phase45.text import CompleteEncoder
from src.phase45.attributes import rows_for

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['validation_replay','test'],required=True);args=parser.parse_args()
    torch.set_num_threads(8);torch.manual_seed(666)
    cfg=json.loads((ROOT/'configs/phase46/contract.json').read_text())
    if args.mode=='test':
        auth=json.loads((ROOT/'outputs/phase46/test_authorization.json').read_text());assert auth['validation_gate_passed']
        assert sha256_file(Path(__file__))==auth['evaluation_scripts']['scripts/phase46_extract_final.py']
        manifest=ROOT/'manifests/legacy_subset_v1/monuseg/test.csv'
        with manifest.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        assert len(rows)==14 and all(r['split']=='test' for r in rows)
        output=ROOT/'cache/phase46/V_mid_test.pt';assert not output.exists(),'Do not overwrite final test features'
    else:
        rows=rows_for(('val',))[:1];output=None
    for row in rows:assert sha256_file(ROOT/row['image_path'])==row['image_sha256']
    files={x.name:sha256_file(x) for x in sorted((ROOT/cfg['model_path']).iterdir()) if x.is_file() and x.suffix in ('.json','.safetensors','.txt')}
    identity=json.loads((ROOT/'cache/phase46/extraction_identity.json').read_text())
    assert sha256_text(json.dumps(files,sort_keys=True))==identity['model_sha256']
    start=time.perf_counter()
    processor=AutoProcessor.from_pretrained(ROOT/cfg['model_path'],local_files_only=True,use_fast=False,min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels'])
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT/cfg['model_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0}).eval().requires_grad_(False)
    torch.cuda.synchronize();load_seconds=time.perf_counter()-start
    capture={};hook=model.visual.blocks[15].register_forward_hook(lambda m,a,result:capture.update(mid=result.detach()))
    values=[];records=[]
    for row in rows:
        messages=[{'role':'user','content':[{'type':'image','image':Image.open(ROOT/row['image_path']).convert('RGB'),'min_pixels':cfg['min_pixels'],'max_pixels':cfg['max_pixels']},{'type':'text','text':'Describe the image.'}]}]
        before=time.perf_counter();images,videos=process_vision_info(messages)
        batch=processor(text=[processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)],images=images,videos=videos,return_tensors='pt').to('cuda')
        torch.cuda.synchronize();preprocess_seconds=time.perf_counter()-before;before=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        grid=batch.image_grid_thw;assert grid.shape==(1,3) and int(grid[0,0])==1
        h,w=map(int,grid[0,1:].tolist())
        with torch.inference_mode():
            model.visual(batch.pixel_values.to(model.visual.dtype),grid_thw=grid)
            window,_=model.visual.get_window_index(grid)
            mid=capture.pop('mid').reshape(-1,4,1280)[torch.argsort(window)]
            mid=mid.reshape(h//2,w//2,2,2,1280).permute(4,0,2,1,3).reshape(1,1280,h,w).float()
            feature=F.adaptive_avg_pool2d(mid,(28,28))[0].half().cpu()
        torch.cuda.synchronize();seconds=time.perf_counter()-before
        record={k:row[k] for k in ('sample_id','dataset','split','image_path','image_sha256')}
        record.update(grid_thw=grid.tolist(),preprocess_seconds=preprocess_seconds,extract_seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),pixel_values_sha256=hashlib.sha256(batch.pixel_values.cpu().numpy().tobytes()).hexdigest())
        if args.mode=='validation_replay':
            original=torch.load(ROOT/'cache/phase46/visual_samples'/f"{row['sample_id']}.pt",map_location='cpu',weights_only=False)
            assert torch.equal(feature,original['features']['mid']) and record['pixel_values_sha256']==original['pixel_values_sha256']
            record['original_cache_bitwise_equal']=True
        values.append(feature);records.append(record);print('EXTRACTED',row['sample_id'],flush=True)
    hook.remove();del model;torch.cuda.empty_cache()
    if args.mode=='test':
        encoder=CompleteEncoder(device='cpu');texts=load_descriptions(rows,'provided_legacy_text')
        legacy,encoded=encoder.encode([texts[r['sample_id']] for r in rows])
        dev=torch.load(ROOT/'cache/phase46/V_mid_development.pt',map_location='cpu',weights_only=False)
        assert encoder.fingerprint==dev['metadata']['encoder']
        payload={'sample_ids':[r['sample_id'] for r in rows],'features':torch.stack(values),'legacy_text_features':legacy,'records':records,'legacy_text_records':encoded,
                 'metadata':{'source':'phase46_V_mid','manifest_sha256':sha256_file(manifest),'development_cache_sha256':sha256_file(ROOT/'cache/phase46/V_mid_development.pt'),
                             'authorization_sha256':sha256_file(ROOT/'outputs/phase46/test_authorization.json'),'extraction_identity':identity,
                             'extractor_sha256':sha256_file(Path(__file__)),'encoder':encoder.fingerprint}}
        atomic_torch_save(output,payload)
    receipt={'status':'completed','mode':args.mode,'count':len(rows),'model_load_seconds':load_seconds,'wall_seconds':time.perf_counter()-start,'records':records,'script_sha256':sha256_file(Path(__file__))}
    atomic_json(ROOT/f'outputs/phase46/{args.mode}_extraction_receipt.json',receipt)
    print('COMPLETE',args.mode,flush=True)

if __name__=='__main__':main()
