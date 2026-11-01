"""Real-data wiring, feature ordering and exact restart checks before training."""
import os,sys,json,copy,gc,hashlib
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.experiment.common import atomic_json,sha256_file,seed_everything,rng_state,restore_rng
from src.experiment.data import load_image,load_mask
from src.experiment.loss import WeightedDiceBCE
from src.phase45.model import build_model as old_model
from src.phase46.engine import configure,setup_data,update_batch,make_optimizer
from src.phase46.model import build_model

def main():
    torch.set_num_threads(8);seed_everything(666)
    out=ROOT/'outputs/phase46/preflight';out.mkdir(parents=True,exist_ok=True)
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel as Vision
    dummy=SimpleNamespace(spatial_merge_size=2,window_size=112,patch_size=14,spatial_merge_unit=4)
    order_checks=[]
    for h,w in [(8,12),(72,72)]:
        grid=torch.tensor([[1,h,w]])
        window,_=Vision.get_window_index(dummy,grid)
        original=torch.arange(h*w).reshape(h,w)
        grouped=original.reshape(h//2,2,w//2,2).permute(0,2,1,3).reshape(-1,4)
        windowed=grouped[window]
        restored=windowed[torch.argsort(window)].reshape(h//2,w//2,2,2).permute(0,2,1,3).reshape(h,w)
        assert torch.equal(restored,original)
        order_checks.append({'grid':[h,w],'all_positions_restored':True})
    images=json.loads((ROOT/'configs/phase46/development_images.json').read_text())['records']
    train_ids={r['sample_id'] for r in images if r['split']=='train'}
    generated=[json.loads(x) for x in (ROOT/'cache/phase46/fewshot_attributes.jsonl').read_text().splitlines()]
    assert len(generated)==30 and all(len(x['example_ids'])==3 and set(x['example_ids'])<=train_ids and x['sample_id'] not in x['example_ids'] for x in generated)
    checks={};gate=json.loads((ROOT/'cache/phase46/candidate_gate.json').read_text())
    for name,enabled in gate['enabled'].items():
        if not enabled:continue
        config=configure(json.loads((ROOT/f'configs/phase46/{name}.json').read_text()))
        rows,_,store=setup_data(config)
        image=torch.stack([load_image(ROOT/r['image_path']) for r in rows['train'][:2]]).cuda()
        target=torch.stack([load_mask(ROOT/r['mask_path']) for r in rows['train'][:2]]).cuda()
        condition={k:torch.stack([store.lookup[r['sample_id']][k] for r in rows['train'][:2]]).cuda() for k in next(iter(store.lookup.values()))}
        assert not torch.equal(condition['vision'][0],condition['vision'][1])
        seed_everything(666);model=build_model(config).cuda().eval()
        baseline=old_model('legacy').cuda().eval();baseline.load_state_dict(model.base.state_dict())
        gate_initial=model.gate.detach().clone();model.gate.data.zero_()
        with torch.inference_mode():assert torch.equal(model(image,condition),baseline(image,condition['text']))
        model.gate.data.copy_(gate_initial);del baseline
        injected=[]
        handle=model.base.outc.register_forward_pre_hook(lambda m,args:injected.append(list(args[0].shape)))
        with torch.inference_mode():
            output=model(image,condition)
            changed=model(image,{**condition,'vision':condition['vision'].flip(0)})
        handle.remove();assert injected==[[2,64,224,224]]*2
        delta=float((output-changed).abs().max());assert delta>0
        model.train();optimizer,scheduler=make_optimizer(model,config);criterion=WeightedDiceBCE();batch={'image':image,'mask':target,'text':condition}
        first=update_batch(model,optimizer,criterion,batch,'cuda')
        gradients={k:float(p.grad.detach().norm()) for k,p in model.named_parameters() if k.startswith('adapter') or k=='gate'}
        assert all(v>0 for v in gradients.values())
        state={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'optimizer':copy.deepcopy(optimizer.state_dict()),'rng':rng_state()}
        checkpoint=out/(name+'_restart.pt');torch.save(state,checkpoint)
        second=update_batch(model,optimizer,criterion,batch,'cuda')
        expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};opt_expected=copy.deepcopy(optimizer.state_dict())
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        third=update_batch(model,optimizer,criterion,batch,'cuda')
        assert second['loss']==third['loss'] and all(torch.equal(v.cpu(),expected[k]) for k,v in model.state_dict().items())
        for key,value in optimizer.state_dict()['state'].items():
            for k,v in value.items():assert torch.equal(v.cpu(),opt_expected['state'][key][k].cpu()) if torch.is_tensor(v) else v==opt_expected['state'][key][k]
        checks[name]={'gate_zero_exact_legacy_forward':True,'injection_shape':injected[0],'actual_input_permutation_max_output_change':delta,
                      'adapter_gradient_norms':gradients,'first_update_loss':first['loss'],'restored_next_loss':third['loss'],
                      'exact_model_optimizer_restore':True,'feature_sha256':store.sha256,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        checkpoint.unlink();del model,optimizer,scheduler,state,saved,expected,opt_expected,condition,image,target;gc.collect();torch.cuda.empty_cache()
    freeze={str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((ROOT/'src/phase46').glob('*.py'))}
    atomic_json(ROOT/'docs/phase46_core_freeze.json',freeze)
    atomic_json(out/'checks.json',{'status':'passed','coordinates':order_checks,'train_examples_only_no_self':True,'models':checks,'preflight_updates_per_model':3,'test_accessed':False})
    print('PREFLIGHT_PASSED',json.dumps(checks),flush=True)

if __name__=='__main__':main()
