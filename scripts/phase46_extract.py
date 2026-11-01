"""Image-only frozen Qwen features and train-example-conditioned attributes."""
import os,sys,json,time,hashlib
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from src.experiment.common import atomic_json,atomic_torch_save,sha256_file,sha256_text
from src.phase45.attributes import parse,score

def main():
    torch.set_num_threads(8);torch.manual_seed(666)
    cp=ROOT/'configs/phase46/contract.json';cfg=json.loads(cp.read_text())
    selection=json.loads((ROOT/'configs/phase46/development_images.json').read_text())['records']
    assert len(selection)==30 and {x['split'] for x in selection}=={'train','val'}
    assert all(set(x)=={'sample_id','dataset','split','image_path','image_sha256','width','height'} for x in selection)
    out=ROOT/'cache/phase46';out.mkdir(parents=True,exist_ok=True)
    identity={'contract_sha256':sha256_file(cp),'script_sha256':sha256_file(Path(__file__)),
              'model_sha256':json.loads((ROOT/'cache/phase45/attributes_base_development.metadata.json').read_text())['model_sha256']}
    model_files={x.name:sha256_file(x) for x in sorted((ROOT/cfg['model_path']).iterdir()) if x.is_file() and x.suffix in ('.json','.safetensors','.txt')}
    assert sha256_text(json.dumps(model_files,sort_keys=True))==identity['model_sha256'],'Qwen weights/config drift'
    marker=out/'extraction_identity.json'
    if marker.exists():assert json.loads(marker.read_text())==identity
    else:atomic_json(marker,identity)
    started=time.perf_counter()
    processor=AutoProcessor.from_pretrained(ROOT/cfg['model_path'],local_files_only=True,use_fast=False,min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels'])
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT/cfg['model_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0}).eval().requires_grad_(False)
    model.generation_config.temperature=1.;model.generation_config.top_p=1.;model.generation_config.top_k=50
    torch.cuda.synchronize();load_seconds=time.perf_counter()-started
    capture={}
    hook=model.visual.blocks[15].register_forward_hook(lambda m,a,result:capture.update(mid=result.detach()))
    def content(row,example=False):
        assert sha256_file(ROOT/row['image_path'])==row['image_sha256']
        return {'type':'image','image':Image.open(ROOT/row['image_path']).convert('RGB'),
                'min_pixels':cfg['min_pixels'],'max_pixels':cfg['example_max_pixels'] if example else cfg['max_pixels']}
    def inputs(messages):
        imgs,vids=process_vision_info(messages)
        return processor(text=[processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)],images=imgs,videos=vids,return_tensors='pt').to('cuda')
    feature_records=[]
    for row in selection:
        target=out/'visual_samples'/f"{row['sample_id']}.pt";target.parent.mkdir(exist_ok=True)
        if target.exists():
            old=torch.load(target,map_location='cpu',weights_only=False);assert old['identity']==identity and old['image_sha256']==row['image_sha256'];feature_records.append({k:v for k,v in old.items() if k!='features'});continue
        batch=inputs([{'role':'user','content':[content(row),{'type':'text','text':'Describe the image.'}]}])
        grid=batch.image_grid_thw;assert grid.shape==(1,3) and int(grid[0,0])==1
        h,w=map(int,grid[0,1:].tolist());merge=model.visual.spatial_merge_size;assert merge==2
        torch.cuda.synchronize();before=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            merged=model.visual(batch.pixel_values.to(model.visual.dtype),grid_thw=grid)
            window,_=model.visual.get_window_index(grid)
            mid=capture.pop('mid').reshape(-1,merge*merge,1280)[torch.argsort(window)]
            mid=mid.reshape(h//merge,w//merge,merge,merge,1280).permute(4,0,2,1,3).reshape(1,1280,h,w).float()
            merged=merged.reshape(h//merge,w//merge,2048).permute(2,0,1).unsqueeze(0).float()
            features={'mid':F.adaptive_avg_pool2d(mid,(28,28))[0].half().cpu(),'merged':F.adaptive_avg_pool2d(merged,(28,28))[0].half().cpu()}
        torch.cuda.synchronize()
        record={**row,'identity':identity,'grid_thw':grid.tolist(),'extract_seconds':time.perf_counter()-before,
                'pixel_values_sha256':hashlib.sha256(batch.pixel_values.cpu().numpy().tobytes()).hexdigest(),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'features':features}
        assert all(torch.isfinite(v).all() for v in features.values())
        atomic_torch_save(target,record);feature_records.append({k:v for k,v in record.items() if k!='features'})
        print('FEATURE',row['split'],row['sample_id'],flush=True)
    hook.remove();capture.clear()
    generation=out/'fewshot_attributes.jsonl'
    records=[json.loads(x) for x in generation.read_text().splitlines()] if generation.exists() else []
    assert len({x['sample_id'] for x in records})==len(records)
    for record in records:assert record['identity']==identity
    done={x['sample_id'] for x in records}
    for row in selection:
        if row['sample_id'] in done:continue
        examples=[next(x for x in cfg['examples'][category] if x['sample_id']!=row['sample_id']) for category in ('low','medium','high')]
        messages=[{'role':'system','content':[{'type':'text','text':'Use the labeled training examples to understand the nuclei foreground categories. Answer for the final image only.'}]}]
        for example in examples:
            messages.append({'role':'user','content':[content(example,True),{'type':'text','text':cfg['query_prompt']}]})
            messages.append({'role':'assistant','content':[{'type':'text','text':json.dumps(example['answer'])}]})
        messages.append({'role':'user','content':[content(row),{'type':'text','text':cfg['query_prompt']}]})
        batch=inputs(messages);torch.cuda.synchronize();before=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():generated=model.generate(**batch,max_new_tokens=64,do_sample=False,use_cache=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-before
        tokens=generated[0,batch.input_ids.shape[1]:]
        raw=processor.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
        record={**row,'identity':identity,'raw':raw,**parse(raw),'example_ids':[x['sample_id'] for x in examples],
                'generate_seconds':seconds,'input_tokens':int(batch.input_ids.shape[1]),'output_tokens':len(tokens),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        with generation.open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
        records.append(record);print('TEXT',row['split'],row['sample_id'],raw,flush=True)
    assert len(records)==30
    atomic_json(out/'fewshot_scores.json',score(generation))
    atomic_json(out/'extraction_receipt.json',{'status':'completed','identity':identity,'model_load_seconds':load_seconds,'wall_seconds':time.perf_counter()-started,'feature_records':feature_records,'count':30,'test_accessed':False})
    print('COMPLETE',json.dumps(score(generation)),flush=True)

if __name__=='__main__':main()
