import os,sys,json,copy,hashlib,gc
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,cv2
from src.experiment.common import atomic_json,sha256_file,seed_everything,rng_state,restore_rng
from src.experiment.model import build_model as old_model
from src.experiment.data import load_image,load_mask
from src.experiment.loss import WeightedDiceBCE
from src.phase45.model import build_model
from src.phase45.data import FeatureStore,manifest_rows
from src.phase45.attributes import parse,label,rows_for
from src.phase45.engine import update_batch,make_optimizer,configure

def main():
    torch.set_num_threads(8);seed_everything(666)
    output=ROOT/'outputs/phase45/preflight';output.mkdir(parents=True,exist_ok=True)
    rows,hashes=manifest_rows('monuseg',verify_assets=True);all_rows=rows['train']+rows['val']
    contract=json.loads((ROOT/'configs/phase45/attribute_contract.json').read_text())
    assert parse('{"coverage":"low","location":"upper_left"}')['valid']
    for raw in ['{}','{"coverage":"wrong","location":"upper_left"}','{"coverage":"low","location":"upper_left","mask":true}']:
        x=parse(raw);assert not x['valid'] and x['coverage']==x['location']=='unknown'
    try:rows_for(('test',));raise AssertionError('Test guard failed')
    except AssertionError as e:assert 'gated' in str(e)
    q=np.zeros((8,8),np.uint8);q[:4,:4]=1
    assert label(q,contract)['location']=='upper_left'
    for transformed,expected in [(np.fliplr(q),'upper_right'),(np.flipud(q),'lower_left'),(np.rot90(q,2),'lower_right')]:
        assert label(transformed,contract)['location']==expected
        assert label(transformed,contract)['coverage']==label(q,contract)['coverage']
    caches={enc:torch.load(ROOT/f'cache/phase45/base_{enc}_development.pt',map_location='cpu',weights_only=False) for enc in ('legacy','complete')}
    cache_checks={}
    for enc,c in caches.items():
        f=c['features'];assert f.shape==(30,10,768) and torch.isfinite(f).all()
        diff=float((f-f[0]).abs().max())
        cache_checks[enc]={'unique_raw_texts':len({x['raw_text'] for x in c['records']}),'maximum_feature_difference_between_images':diff,'maximum_train_mean_difference':float((f[:24].mean(0)-f[0]).abs().max())}
    images=torch.stack([load_image(ROOT/x['image_path']) for x in rows['train'][:2]]).cuda()
    targets=torch.stack([load_mask(ROOT/x['mask_path']) for x in rows['train'][:2]]).cuda()
    checks={}
    for fusion in ('legacy','cross_attention'):
        seed_everything(666);old=old_model();old_state={k:v.clone() for k,v in old.state_dict().items()};del old
        seed_everything(666);model=build_model(fusion).cuda()
        same=[k for k in model.state_dict() if k in old_state]
        assert all(torch.equal(model.state_dict()[k].cpu(),old_state[k]) for k in same)
        config=configure(json.loads((ROOT/'configs/phase45/F.json').read_text()));config['fusion']=fusion
        text=caches['complete' if fusion=='cross_attention' else 'legacy']['features'][:2].cuda()
        if fusion=='legacy':
            old=old_model().cuda().eval();old.load_state_dict(old_state)
            model.eval()
            with torch.no_grad():assert torch.equal(model(images,text),old(images,text))
            del old
        del old_state
        model.train();opt,sched=make_optimizer(model,config);loss=WeightedDiceBCE()
        batch={'image':images,'mask':targets,'text':text}
        first=update_batch(model,opt,loss,batch,'cuda');assert first['text_gradient_l2']>0
        state={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'optimizer':copy.deepcopy(opt.state_dict()),'rng':rng_state()}
        path=output/f'{fusion}_restore.pt';torch.save(state,path)
        second=update_batch(model,opt,loss,batch,'cuda');expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        opt_expected=copy.deepcopy(opt.state_dict());saved=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        restored=update_batch(model,opt,loss,batch,'cuda')
        assert second['loss']==restored['loss']
        assert all(torch.equal(v.cpu(),expected[k]) for k,v in model.state_dict().items())
        for key,value in opt.state_dict()['state'].items():
            for n,t in value.items():assert torch.equal(t.cpu(),opt_expected['state'][key][n].cpu()) if torch.is_tensor(t) else t==opt_expected['state'][key][n]
        checks[fusion]={'shared_initial_tensors_equal':len(same),'first_update':first,'exact_restored_next_loss':restored['loss'],'model_and_optimizer_exact':True,'checkpoint_roundtrip_sha256':sha256_file(path)}
        path.unlink();del model,opt,sched,state,saved,expected,opt_expected;gc.collect();torch.cuda.empty_cache()
    # Independently replay preprocessing for every image, without loading Qwen.
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    processor=AutoProcessor.from_pretrained(ROOT/contract['model_path'],local_files_only=True,use_fast=False,min_pixels=contract['min_pixels'],max_pixels=contract['max_pixels'])
    fingerprints=[]
    for row in all_rows:
        message=[{'role':'user','content':[{'type':'image','image':Image.open(ROOT/row['image_path']).convert('RGB'),'min_pixels':contract['min_pixels'],'max_pixels':contract['max_pixels']},{'type':'text','text':contract['prompt']}]}]
        imgs,vids=process_vision_info(message)
        inputs=processor(text=[processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True)],images=imgs,videos=vids,return_tensors='pt')
        fingerprints.append({'sample_id':row['sample_id'],'pixel_values_sha256':hashlib.sha256(inputs.pixel_values.numpy().tobytes()).hexdigest(),'grid':inputs.image_grid_thw.tolist()})
    assert len({x['pixel_values_sha256'] for x in fingerprints})==30
    result={'status':'passed','counts':{k:len(v) for k,v in rows.items()},'test_guard':'passed','attribute_flip_and_rotation':'passed','cache_checks':cache_checks,'models':checks,'processor_replay_distinct_inputs':fingerprints,'limits':'processor hashes independently replayed after generation; not a trace captured during original generation'}
    atomic_json(output/'checks.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('models','processor_replay_distinct_inputs')}),flush=True)

if __name__=='__main__':main()
