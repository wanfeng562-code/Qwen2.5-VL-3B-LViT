"""Post-run independent encoding and fusion checks; no optimizer updates."""
import os,sys,json,math
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.phase45.text import CompleteEncoder
from src.phase45.model import build_model
from src.phase45.engine import setup_data
from src.experiment.common import atomic_json,seed_everything
from src.experiment.data import load_image,load_mask
from src.experiment.loss import WeightedDiceBCE

def main():
    torch.set_num_threads(8)
    encoder=CompleteEncoder(device='cpu')
    assert not encoder.model.training and not any(p.requires_grad for p in encoder.model.parameters())
    texts=['Nuclei coverage high concentration unknown',
           'Hyperchromatic nuclei have irregular boundaries and heterogeneous chromatin, with concentration upper left',
           ' '.join(['nuclei']*40+['low'])]
    features,records=encoder.encode(texts);checks=[]
    batch=encoder.tokenizer(texts,padding=True,return_tensors='pt',return_special_tokens_mask=True)
    all_special=batch.pop('special_tokens_mask')
    with torch.inference_mode():all_hidden=encoder.model(**batch).last_hidden_state
    for i,text in enumerate(texts):
        special=all_special[i];hidden=all_hidden[i]
        tokens=encoder.tokenizer.convert_ids_to_tokens(batch['input_ids'][i].tolist());groups=[]
        for pos,token in enumerate(tokens):
            if special[pos] or not batch['attention_mask'][i,pos]:continue
            if token.startswith('##') and groups:groups[-1].append(pos)
            else:groups.append([pos])
        words=torch.stack([hidden[g].float().mean(0) for g in groups]);n=len(words)
        independent=torch.stack([words[math.floor(j*n/10):math.ceil((j+1)*n/10)].mean(0) for j in range(10)])
        assert torch.allclose(features[i],independent,atol=2e-6,rtol=2e-6)
        assert records[i]['available_merged_tokens']==n and not records[i]['silent_truncation']
        checks.append({'input':text,'wordpieces':tokens,'merged_tokens':records[i]['merged_tokens'],
                       'merged_count':n,'independent_pool_max_abs_error':float((features[i]-independent).abs().max())})
    changed,_=encoder.encode([texts[-1][:-3]+'high'])
    tail_difference=float((features[-1]-changed[0]).abs().max());assert tail_difference>1e-5
    try:encoder.encode([' '.join(['nuclei']*100)]);raise RuntimeError('Overflow was silently accepted')
    except AssertionError as exc:assert 'silent' in str(exc)
    config=json.loads((ROOT/'configs/phase45/F.json').read_text())
    rows,_,store=setup_data(config);seed_everything(666)
    model=build_model('cross_attention').cuda().eval()
    image=torch.stack([load_image(ROOT/r['image_path']) for r in rows['train'][:2]]).cuda()
    target=torch.stack([load_mask(ROOT/r['mask_path']) for r in rows['train'][:2]]).cuda()
    text=torch.stack([store.lookup[r['sample_id']] for r in rows['train'][:2]]).cuda()
    shapes=[]
    hook=model.downVit.text_attention.register_forward_pre_hook(lambda m,args:shapes.append([list(x.shape) for x in args]))
    prediction=model(image,text);loss=WeightedDiceBCE()(prediction,target);loss.backward();hook.remove()
    assert shapes==[[[2,196,64],[2,10,64]]]
    grads={name:float(p.grad.detach().norm()) for name,p in model.named_parameters() if name.startswith('downVit.text_attention') and p.grad is not None}
    assert grads and all(math.isfinite(v) and v>0 for v in grads.values())
    assert abs(float(model.downVit.text_attention.gate.detach())-.1)<1e-6
    assert model.downVit.text_attention.attention.num_heads==4
    assert not hasattr(model.downVit,'CTBN3')
    with torch.inference_mode():perturbed=model(image,text.roll(1,dims=2))
    delta=float((prediction.detach()-perturbed).abs().max());assert delta>0
    result={'status':'passed','encoder_frozen':True,'encoder_version':encoder.fingerprint,
            'cases':checks,'late_word_change_max_feature_difference':tail_difference,'overflow_rejected':True,
            'actual_fusion_input_shapes':shapes,'attention_parameter_gradient_norms':grads,
            'synthetic_text_perturbation_output_max_difference':delta,'optimizer_steps':0,
            'limitation':'Synthetic perturbation is a wiring check, not evidence of useful image-specific text or better segmentation.'}
    atomic_json(ROOT/'outputs/phase45/encoding_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('encoder_version','cases','attention_parameter_gradient_norms')}))

if __name__=='__main__':main()
