"""Freeze a new development-only comparison before any formal training."""
import os,sys,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiment.common import atomic_json,sha256_file
from src.phase45.attributes import rows_for

def main():
    out=ROOT/'configs/phase46';out.mkdir(parents=True,exist_ok=True)
    rows=rows_for(('train','val'),verify=True)
    refs=json.loads((ROOT/'outputs/phase45/attribute_references.json').read_text())['records']
    train=[r for r in refs if r['split']=='train']
    examples={}
    for category in ['low','medium','high']:
        pool=sorted([x for x in train if x['coverage']==category],key=lambda x:(x['foreground_fraction'],x['sample_id']))
        center=(len(pool)-1)/2
        examples[category]=[{'sample_id':pool[i]['sample_id'],'image_path':pool[i]['image_path'],'image_sha256':pool[i]['image_sha256'],
                             'answer':{k:pool[i][k] for k in ('coverage','location')}} for i in sorted(range(len(pool)),key=lambda i:(abs(i-center),i))]
    contract={'version':'phase46_train_examples_and_frozen_visual_v1','sample_count':30,'splits':['train','val'],
              'min_pixels':200704,'max_pixels':1003520,'example_max_pixels':200704,'max_new_tokens':64,
              'examples_per_query':3,'exclude_query_from_examples':True,'example_selection':'class medians from train only, then distance to median; never use query label',
              'examples':examples,'query_prompt':json.loads((ROOT/'configs/phase45/attribute_contract.json').read_text())['prompt'],
              'visual_layers':['block15_before_merger','final_merged'],'cached_grid':[28,28],
              'visual_dtype':'float32 compute from BF16 tower, float16 storage','model_path':'weights/Qwen2.5-VL-3B-Instruct',
              'processor_input':'same query resolution as phase45; examples capped separately',
              'text_training_gate':'more than one sentence and validation field accuracy above phase45 2/12; score unknown as incorrect',
              'screening':['S_fewshot_text','V_mid_visual','V_merged_visual'],'tie_margin_pp':.1,
              'selection':'best seed666 validation Dice, within 0.1pp prefer S then V_mid then V_merged',
              'formal_training_limit':14,'previous_phase45_formal_runs':4,'combined_limit':18,
              'training_protocol':'unchanged FP32 Adam, no augmentation, 24/6/14, max2000/min50/patience50',
              'test_policy':'closed until 3-seed matched validation gates; original-split final re-evaluation only',
              'success':{'mean_test_dice_percent_at_least':78.5412,'mean_test_iou_percent_at_least':63.4906,
                         'beats_fixed_and_shuffled_pp':.5,'not_below_matched_legacy':True,'improves_old_baseline_at_least_seeds':2}}
    p=out/'contract.json'
    if p.exists():assert json.loads(p.read_text())==contract
    else:atomic_json(p,contract)
    atomic_json(out/'development_images.json',{'records':[{k:r[k] for k in ('sample_id','dataset','split','image_path','image_sha256','width','height')} for r in rows]})
    freeze={'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'contract_sha256':sha256_file(p),'previous_phase45_runs':4,'new_limit':14,
            'old_tracked_sha256':{name:sha256_file(ROOT/name) for name in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if name}}
    fp=ROOT/'docs/phase46_start.json'
    if not fp.exists():atomic_json(fp,freeze)
    print(json.dumps({'status':'prepared','train':24,'val':6,'formal_limit':14,'test_access':False}))

if __name__=='__main__':main()
