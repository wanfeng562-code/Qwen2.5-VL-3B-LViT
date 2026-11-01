"""Frozen offline image-only Qwen generation with content-keyed resumable caches.

No mask or legacy description is read. Pilot selection may have been stratified
using training/validation labels upstream; only image fields reach this script.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

ROOT = Path(__file__).resolve().parents[1]
IMAGE_FIELDS = ('sample_id', 'dataset', 'split', 'image_path', 'image_sha256', 'width', 'height')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def structure(raw, width=1000, height=1000):
    errors=[]
    try:
        obj=json.loads(raw)
        if not isinstance(obj,dict) or set(obj)!= {'regions'} or not isinstance(obj['regions'],list):
            return [],False,['invalid_top_level']
        regions=obj['regions']
        if len(regions)>3:errors.append('too_many_regions')
        ids=[]
        for i,r in enumerate(regions):
            if not isinstance(r,dict) or set(r)!= {'id','description','box'}:
                errors.append(f'region_{i}_keys');continue
            if isinstance(r['id'],bool) or not isinstance(r['id'],int):errors.append(f'region_{i}_id')
            ids.append(str(r['id']))
            if not isinstance(r['description'],str) or not r['description'].strip():errors.append(f'region_{i}_description')
            b=r['box']
            if not isinstance(b,list) or len(b)!=4 or not all(not isinstance(x,bool) and isinstance(x,(int,float)) and math.isfinite(x) for x in b):
                errors.append(f'region_{i}_box');continue
            if not (0<=b[0]<=width and 0<=b[2]<=width and 0<=b[1]<=height and 0<=b[3]<=height):errors.append(f'region_{i}_bounds')
            if not(b[0]<b[2] and b[1]<b[3]):errors.append(f'region_{i}_degenerate')
        if len(set(ids))!=len(ids):errors.append('duplicate_ids')
        return regions,not errors,errors
    except (ValueError,TypeError) as e:
        return [],False,[type(e).__name__]


def adapt_structure(raw,width,height):
    """Explicit envelope-only adapter; no semantic or coordinate repair."""
    text=raw.strip();actions=[]
    match=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```',text,flags=re.DOTALL|re.IGNORECASE)
    if match:
        text=match.group(1).strip();actions.append('strip_single_full_json_code_fence')
    try:
        obj=json.loads(text)
    except ValueError as exc:
        return [],False,actions,[type(exc).__name__]
    if isinstance(obj,list):
        obj={'regions':obj};actions.append('wrap_top_level_array_in_regions')
    regions,valid,errors=structure(json.dumps(obj),width,height)
    return regions,valid,actions,errors


def read_records(args):
    if args.selection:
        data=json.loads((ROOT/args.selection).read_text())
        rows=data['records'] if isinstance(data,dict) else data
    elif args.image:
        from PIL import Image
        p=Path(args.image).resolve(strict=True)
        with Image.open(p) as im: w,h=im.size
        rows=[{'sample_id':args.sample_id or 'new_image_'+digest(p)[:16], 'dataset':args.dataset,'split':'new_image','image_path':str(p),'image_sha256':digest(p),'width':w,'height':h}]
    else:
        rows=[]
        for split in ('train','val','test'):
            with (ROOT/f'manifests/legacy_subset_v1/{args.dataset}/{split}.csv').open(newline='') as f:
                rows.extend({k:r[k] for k in IMAGE_FIELDS} for r in csv.DictReader(f))
    result=[{k:r[k] for k in IMAGE_FIELDS} for r in rows]
    assert len({r['sample_id'] for r in result})==len(result),'duplicate samples'
    if args.selection:assert all(r['split'] in ('train','val') for r in result),'test prohibited in pilot'
    return result


def main():
    invocation_started=time.perf_counter()
    generator_source=Path(__file__).read_bytes()
    generator_hash=hashlib.sha256(generator_source).hexdigest()
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group()
    source.add_argument('--selection')
    source.add_argument('--image')
    p.add_argument('--dataset',choices=['monuseg','cov19','mosmeddata'],default='monuseg')
    p.add_argument('--sample-id')
    p.add_argument('--config',default='configs/qwen_generation_v1.json')
    p.add_argument('--mode',choices=['plain','both'],default='both')
    p.add_argument('--output',required=True)
    p.add_argument('--reuse-plain',help='Prior pilot cache; matching config, model and image hashes required')
    args=p.parse_args()
    config=json.loads((ROOT/args.config).read_text())
    model_dir=ROOT/config['model_directory']
    # Full content hashes identify actual local weights, tokenizer and processor.
    model_files={str(f.relative_to(model_dir)):digest(f) for f in sorted(model_dir.iterdir()) if f.is_file() and f.suffix in ('.json','.safetensors','.txt')}
    model_hash=canonical_hash(model_files)
    config_hash=canonical_hash(config)
    contract_path=ROOT/'configs/qwen_output_contract_v1.json'
    contract_hash=canonical_hash(json.loads(contract_path.read_text()))
    rows=read_records(args)
    out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    source_archive=out.with_suffix('.source-'+generator_hash+'.py')
    if source_archive.exists():
        assert digest(source_archive)==generator_hash,'generator source archive changed'
    else:
        temporary=source_archive.with_name(source_archive.name+'.tmp')
        with temporary.open('wb') as handle:
            handle.write(generator_source);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,source_archive)
    metadata=out.with_suffix('.metadata.json')
    expected={'config':config,'config_sha256':config_hash,'model_sha256':model_hash,'model_files':model_files,'mode':args.mode,'samples':[r['sample_id'] for r in rows]}
    if metadata.exists():
        previous=json.loads(metadata.read_text())
        for k in expected:assert previous[k]==expected[k],f'cache metadata mismatch: {k}'
    else:
        atomic_json(metadata,expected)
    done={}
    if out.exists():
        for line in out.read_text().splitlines():
            r=json.loads(line)
            assert r['config_sha256']==config_hash and r['model_sha256']==model_hash
            assert r['sample_id'] not in done
            done[r['sample_id']]=r
    assert set(done)<={r['sample_id'] for r in rows}
    reuse={}
    if args.reuse_plain:
        for line in (ROOT/args.reuse_plain).read_text().splitlines():
            r=json.loads(line)
            if r['config_sha256']==config_hash and r['model_sha256']==model_hash and r.get('plain_text'):
                reuse[r['sample_id']]=r
    pending=[]
    for r in rows:
        assert digest(ROOT/r['image_path'])==r['image_sha256'],f'image changed {r["sample_id"]}'
        if r['sample_id'] in done:
            assert done[r['sample_id']]['image_sha256']==r['image_sha256']
            for key in ('dataset','split','image_path'):
                assert done[r['sample_id']][key]==r[key],f'cached provenance differs: {key}'
            assert isinstance(done[r['sample_id']].get('plain_text'),str) and done[r['sample_id']]['plain_text'].strip()
            if args.mode=='both':
                assert 'structured_raw' in done[r['sample_id']] and 'regions' in done[r['sample_id']]
        else:pending.append(r)
    if not pending:
        assert set(done)=={r['sample_id'] for r in rows},'incomplete cache cannot be finalized'
        receipt_path=out.with_suffix('.receipt.json')
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            assert receipt['status']=='complete' and receipt['count']==len(rows)
            assert receipt['config_sha256']==config_hash and receipt['model_sha256']==model_hash
        else:
            # The previous process may have committed its final record and died
            # before writing a receipt. Do not invent model loading/GPU timings.
            receipt={'status':'complete','count':len(done),'model_load_seconds':None,
                     'wall_seconds':None,'gpu':None,'config_sha256':config_hash,'model_sha256':model_hash,
                     'receipt_recovered_from_complete_cache':True}
        receipt.update(cache_bytes=out.stat().st_size,cache_sha256=digest(out),
                       recovery_validation_seconds=time.perf_counter()-invocation_started,
                       records_existing_on_entry=len(done),records_written_this_invocation=0,
                       plain_generated_this_invocation=0,plain_reused_this_invocation=0,
                       receipt_writer_sha256=generator_hash,
                       receipt_writer_source_archive=str(source_archive),
                       record_generator_sha256=sorted({r.get('generator_sha256','unknown') for r in done.values()}))
        atomic_json(receipt_path,receipt)
        print('CACHE_COMPLETE',json.dumps(receipt),flush=True);return
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from validate_qwen import GPUMonitor
    torch.set_num_threads(8)
    torch.manual_seed(666)
    torch.cuda.set_device(0)
    started=time.perf_counter()
    monitor=GPUMonitor(out.with_suffix('.gpu.csv'));monitor.start()
    try:
        processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True,use_fast=False,min_pixels=config['min_pixels'],max_pixels=config['max_pixels'])
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0},low_cpu_mem_usage=True).eval().requires_grad_(False)
        model.generation_config.temperature=1.0
        model.generation_config.top_p=1.0
        model.generation_config.top_k=50
        torch.cuda.synchronize();load_seconds=time.perf_counter()-started
        print('MODEL_LOADED',round(load_seconds,3),'PENDING',len(pending),flush=True)

        def generate(image,prompt,limit):
            t=time.perf_counter()
            pixel_coordinates=config['coordinate_system']=='absolute_resized_pixels_xyxy'
            image_content={'type':'image','image':image}
            if pixel_coordinates:
                image_content.update(min_pixels=config['min_pixels'],max_pixels=config['max_pixels'])
            messages=[{'role':'user','content':[image_content,{'type':'text','text':prompt}]}]
            imgs,vids=process_vision_info(messages)
            coordinate_width,coordinate_height=imgs[0].size
            prompt=prompt.replace('[[IMAGE_WIDTH]]',str(coordinate_width)).replace('[[IMAGE_HEIGHT]]',str(coordinate_height))
            messages[0]['content'][1]['text']=prompt
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[text],images=imgs,videos=vids,padding=True,return_tensors='pt').to('cuda:0')
            actual_height=int(inputs['image_grid_thw'][0,1])*processor.image_processor.patch_size
            actual_width=int(inputs['image_grid_thw'][0,2])*processor.image_processor.patch_size
            if pixel_coordinates:
                assert (actual_width,actual_height)==(coordinate_width,coordinate_height),'Unexpected second resize invalidates coordinate frame'
            torch.cuda.synchronize();prep=time.perf_counter()-t
            torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=limit,do_sample=False,use_cache=True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            trimmed=output[0,inputs['input_ids'].shape[1]:]
            decoded=processor.decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
            stats={'preprocess_seconds':prep,'generate_seconds':elapsed,'input_tokens':inputs['input_ids'].shape[1],'output_tokens':len(trimmed),'hit_token_limit':len(trimmed)>=limit,'image_grid_thw':inputs['image_grid_thw'].tolist(),'model_input_width':actual_width,'model_input_height':actual_height,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
            return decoded,stats

        existing_count=len(done);plain_generated=plain_reused=0
        with out.open('a') as stream:
            for i,r in enumerate(pending):
                info=config['datasets'][r['dataset']]
                with Image.open(ROOT/r['image_path']) as im:image=im.convert('RGB')
                assert image.size==(int(r['width']),int(r['height']))
                record={**r,'width':image.width,'height':image.height,'config_sha256':config_hash,'model_sha256':model_hash,'timing':{},'generator_sha256':generator_hash,'generator_source_archive':str(source_archive)}
                cached=reuse.get(r['sample_id'])
                if cached and cached['image_sha256']==r['image_sha256']:
                    record['plain_text']=cached['plain_text'];record['timing']['plain']=cached['timing']['plain'];record['plain_reused_from']=args.reuse_plain
                    plain_reused+=1
                else:
                    record['plain_text'],record['timing']['plain']=generate(image,config['plain_prompt'].format(**info),config['plain_max_new_tokens'])
                    plain_generated+=1
                assert record['plain_text'],'Empty plain text'
                if args.mode=='both':
                    raw,stats=generate(image,config['structured_prompt'].format(**info),config['structured_max_new_tokens'])
                    record['structured_raw']=raw;record['timing']['structured']=stats
                    record['coordinate_system']=config['coordinate_system']
                    pixel_coordinates=config['coordinate_system']=='absolute_resized_pixels_xyxy'
                    w,h=(stats['model_input_width'],stats['model_input_height']) if pixel_coordinates else (1000,1000)
                    record['coordinate_width']=w;record['coordinate_height']=h
                    regions,record['structure_valid'],record['structure_errors']=structure(raw,w,h)
                    regions,record['adapter_valid'],record['adapter_actions'],record['adapter_errors']=adapt_structure(raw,w,h)
                    record['output_contract_sha256']=contract_hash
                    record['regions_raw']=regions
                    # Only explicit envelope repair; invalid coordinates are never clipped.
                    record['regions']=[{**r,'box':[r['box'][0]*1000/w,r['box'][1]*1000/h,r['box'][2]*1000/w,r['box'][3]*1000/h]} for r in regions] if record['adapter_valid'] else []
                stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
                done[r['sample_id']]=record
                print(json.dumps({'completed':len(done),'total':len(rows),'sample_id':r['sample_id'],'plain':record['plain_text'],'structure_valid':record.get('structure_valid'),'timing':record['timing']}),flush=True)
        assert set(done)=={r['sample_id'] for r in rows}
        receipt={'status':'complete','count':len(done),'model_load_seconds':load_seconds,'wall_seconds':time.perf_counter()-started,'cache_bytes':out.stat().st_size,'cache_sha256':digest(out),'config_sha256':config_hash,'model_sha256':model_hash,'versions':{k:importlib.metadata.version(k) for k in ('torch','transformers','qwen-vl-utils','Pillow')},'gpu':monitor.close(),
                 'records_existing_on_entry':existing_count,'records_written_this_invocation':len(pending),
                 'plain_generated_this_invocation':plain_generated,'plain_reused_this_invocation':plain_reused,
                 'receipt_writer_sha256':generator_hash,
                 'receipt_writer_source_archive':str(source_archive),
                 'record_generator_sha256':sorted({r.get('generator_sha256','unknown') for r in done.values()})}
        monitor=None
        atomic_json(out.with_suffix('.receipt.json'),receipt)
        print('COMPLETE',json.dumps(receipt),flush=True)
    finally:
        if monitor is not None:monitor.close()


if __name__=='__main__':
    main()
