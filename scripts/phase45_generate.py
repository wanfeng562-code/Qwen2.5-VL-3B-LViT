"""Image-only attribute generation. Test inputs require a validation authorization."""
import os,sys,json,time,argparse,gc
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiment.common import atomic_json,sha256_file,sha256_text
from src.phase45.attributes import parse,score

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--selection',default='configs/phase45/development_images.json');p.add_argument('--adapter');args=p.parse_args()
    cp=ROOT/'configs/phase45/attribute_contract.json';config=json.loads(cp.read_text());config_hash=sha256_file(cp)
    selection=json.loads((ROOT/args.selection).read_text())['records']
    allowed={'sample_id','dataset','split','image_path','image_sha256','width','height'}
    assert all(set(x)==allowed for x in selection),'Only image fields can enter the generator'
    if any(x['split']=='test' for x in selection):
        assert json.loads((ROOT/'outputs/phase45/test_authorization.json').read_text())['validation_gate_passed']
    assert len({x['sample_id'] for x in selection})==len(selection)
    directory=ROOT/config['model_path']
    model_files={x.name:sha256_file(x) for x in sorted(directory.iterdir()) if x.is_file() and x.suffix in ('.json','.safetensors','.txt')}
    model_hash=sha256_text(json.dumps(model_files,sort_keys=True))
    adapter_hash=sha256_text(json.dumps({x.name:sha256_file(x) for x in sorted((ROOT/args.adapter).iterdir()) if x.suffix in ('.json','.safetensors')},sort_keys=True)) if args.adapter else None
    out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    identity={'contract_sha256':config_hash,'model_sha256':model_hash,'adapter_sha256':adapter_hash,'sample_ids':[x['sample_id'] for x in selection], 'generator_sha256':sha256_file(Path(__file__))}
    meta=out.with_suffix('.metadata.json')
    if meta.exists():assert json.loads(meta.read_text())==identity
    else:atomic_json(meta,identity)
    records=[json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
    assert len({x['sample_id'] for x in records})==len(records)
    done={x['sample_id']:x for x in records};pending=[]
    for row in selection:
        assert sha256_file(ROOT/row['image_path'])==row['image_sha256']
        if row['sample_id'] in done:
            assert all(done[row['sample_id']][k]==v for k,v in row.items())
            assert done[row['sample_id']]['identity']==identity
        else:pending.append(row)
    if pending:
        import torch
        from PIL import Image
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        from validate_qwen import GPUMonitor
        torch.set_num_threads(8);torch.manual_seed(666)
        monitor=GPUMonitor(out.with_suffix('.gpu.csv'));monitor.start();started=time.perf_counter()
        try:
            processor=AutoProcessor.from_pretrained(directory,local_files_only=True,use_fast=False,min_pixels=config['min_pixels'],max_pixels=config['max_pixels'])
            model=Qwen2_5_VLForConditionalGeneration.from_pretrained(directory,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0}).eval().requires_grad_(False)
            if args.adapter:
                from peft import PeftModel
                model=PeftModel.from_pretrained(model,ROOT/args.adapter,is_trainable=False).eval()
            model.generation_config.temperature=1.;model.generation_config.top_p=1.;model.generation_config.top_k=50
            torch.cuda.synchronize();load_seconds=time.perf_counter()-started
            for row in pending:
                before=time.perf_counter()
                im=Image.open(ROOT/row['image_path']).convert('RGB')
                messages=[{'role':'user','content':[{'type':'image','image':im,'min_pixels':config['min_pixels'],'max_pixels':config['max_pixels']},{'type':'text','text':config['prompt']}]}]
                imgs,vids=process_vision_info(messages)
                inputs=processor(text=[processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)],images=imgs,videos=vids,padding=True,return_tensors='pt').to('cuda')
                torch.cuda.synchronize();prep=time.perf_counter()-before;before=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True)
                torch.cuda.synchronize();seconds=time.perf_counter()-before
                tokens=generated[0,inputs.input_ids.shape[1]:]
                raw=processor.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
                record={**row,'identity':identity,'source':'qwen_predicted_attributes','raw':raw,**parse(raw),'timing':{'preprocess_seconds':prep,'generate_seconds':seconds,'output_tokens':len(tokens),'hit_limit':len(tokens)>=64,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}}
                with out.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                print(row['split'],row['sample_id'],raw,flush=True)
            wall=time.perf_counter()-started
        finally:
            gpu_monitor=monitor.close()
        receipt={'status':'complete','count':len(selection),'model_load_seconds':load_seconds,'wall_seconds':wall,'identity':identity,'gpu_monitor':gpu_monitor}
        atomic_json(out.with_suffix('.receipt.json'),receipt)
    if {x['split'] for x in selection}=={'train','val'}:atomic_json(out.with_suffix('.scores.json'),score(out))
    print('COMPLETE',str(out),flush=True)

if __name__=='__main__':main()
