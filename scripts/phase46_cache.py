import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.phase45.data import manifest_rows
from src.phase45.text import CompleteEncoder
from src.experiment.text import load_descriptions
from src.experiment.common import atomic_json,atomic_torch_save,sha256_file

def main():
    torch.set_num_threads(8);rows,hashes=manifest_rows('monuseg',verify_assets=True);all_rows=rows['train']+rows['val']
    receipt=json.loads((ROOT/'cache/phase46/extraction_receipt.json').read_text());assert receipt['status']=='completed'
    output=ROOT/'cache/phase46';encoder=CompleteEncoder(device='cpu')
    descriptions=load_descriptions(all_rows,'provided_legacy_text')
    legacy,legacy_records=encoder.encode([descriptions[r['sample_id']] for r in all_rows])
    samples={r['sample_id']:torch.load(output/'visual_samples'/f"{r['sample_id']}.pt",map_location='cpu',weights_only=False) for r in all_rows}
    raw=[json.loads(x) for x in (output/'fewshot_attributes.jsonl').read_text().splitlines()];byid={r['sample_id']:r for r in raw}
    assert len(raw)==len(byid)==30
    text,text_records=encoder.encode([byid[r['sample_id']]['plain_text'] for r in all_rows])
    base=json.loads((ROOT/'configs/phase45/F.json').read_text());score=json.loads((output/'fewshot_scores.json').read_text())
    enabled={'S':len({r['plain_text'] for r in raw})>1 and score['val']['field_accuracy']>2/12,'V_mid':True,'V_merged':True}
    for name,key,channels in [('S',None,0),('V_mid','mid',1280),('V_merged','merged',2048)]:
        records=[]
        for i,row in enumerate(all_rows):
            source=samples[row['sample_id']];assert source['image_sha256']==row['image_sha256'] and source['identity']==receipt['identity']
            rec={'sample_id':row['sample_id'],'split':row['split'],'image_sha256':row['image_sha256']}
            if name=='S':rec.update(text_records[i])
            else:rec.update(grid_thw=source['grid_thw'],visual_sample_sha256=sha256_file(output/'visual_samples'/f"{row['sample_id']}.pt"))
            records.append(rec)
        f=text if name=='S' else torch.stack([samples[r['sample_id']]['features'][key] for r in all_rows])
        path=output/f'{name}_development.pt';assert not path.exists(),'Keep original cache immutable'
        meta={'source':'phase46_'+name,'kind':'text' if name=='S' else 'visual','channels':channels,'encoder':encoder.fingerprint,
              'manifest_sha256':hashes,'protocol_sha256':sha256_file(ROOT/'configs/data_protocol_v1.json'),
              'extraction_identity':receipt['identity'],'enabled_for_screening':enabled[name]}
        atomic_torch_save(path,{'sample_ids':[r['sample_id'] for r in all_rows],'features':f,'legacy_text_features':legacy,'records':records,'legacy_text_records':legacy_records,'metadata':meta})
        config={**base,'source':'phase46_'+name,'architecture':name,'feature_cache':str(path.relative_to(ROOT)),'fusion':'cross_attention' if name=='S' else 'legacy'}
        atomic_json(ROOT/f'configs/phase46/{name}.json',config)
        print(name,'enabled',enabled[name],'shape',tuple(f.shape),'unique',len({f[i].numpy().tobytes() for i in range(len(f))}),flush=True)
    atomic_json(output/'candidate_gate.json',{'enabled':enabled,'fewshot_scores':score,'test_accessed':False})

if __name__=='__main__':main()
