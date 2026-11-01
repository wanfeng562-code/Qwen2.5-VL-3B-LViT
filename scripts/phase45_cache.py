import sys,json,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.experiment.common import atomic_json,atomic_torch_save,sha256_file,sha256_text
from src.experiment.text import FrozenBertEncoder,load_descriptions
from src.phase45.text import CompleteEncoder
from src.phase45.data import manifest_rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',choices=['qwen_attributes','provided_legacy_text'],required=True);p.add_argument('--encoding',choices=['legacy','complete'],required=True);p.add_argument('--descriptions');p.add_argument('--output',required=True);args=p.parse_args()
    output=ROOT/args.output;assert not output.exists(),'Never overwrite frozen feature caches'
    torch.set_num_threads(8);started=time.perf_counter()
    rows,hashes=manifest_rows('monuseg',verify_assets=True);all_rows=rows['train']+rows['val']
    if args.source=='qwen_attributes':
        records=[json.loads(x) for x in (ROOT/args.descriptions).read_text().splitlines()]
        assert len(records)==len(all_rows)
        byid={x['sample_id']:x for x in records};assert len(byid)==len(records)
        for row in all_rows:
            assert all(byid[row['sample_id']][k]==row[k] for k in ('sample_id','image_sha256','split','image_path'))
        descriptions={sid:r['plain_text'] for sid,r in byid.items()}
        identities={json.dumps(x['identity'],sort_keys=True) for x in records};assert len(identities)==1
        generation=json.loads(identities.pop())
    else:descriptions=load_descriptions(all_rows,'provided_legacy_text');generation=None
    encoder=(CompleteEncoder if args.encoding=='complete' else FrozenBertEncoder)(device='cpu')
    load_seconds=time.perf_counter()-started;before=time.perf_counter()
    features,encoded=encoder.encode([descriptions[x['sample_id']] for x in all_rows])
    encode_seconds=time.perf_counter()-before
    for row,rec in zip(all_rows,encoded):rec.update(sample_id=row['sample_id'],split=row['split'],image_sha256=row['image_sha256'],raw_text=descriptions[row['sample_id']],raw_text_sha256=sha256_text(descriptions[row['sample_id']]))
    meta={'source':args.source,'encoder':encoder.fingerprint,'manifest_sha256':hashes,'protocol_sha256':sha256_file(ROOT/'configs/data_protocol_v1.json'),
          'generation_identity':generation,'encoding':args.encoding,'load_seconds':load_seconds,'encode_seconds':encode_seconds,'total_seconds':time.perf_counter()-started,
          'descriptions_sha256':sha256_file(ROOT/args.descriptions) if args.descriptions else None,'splits':['train','val'],'sample_count':30}
    atomic_torch_save(output,{'schema_version':1,'sample_ids':[x['sample_id'] for x in all_rows],'features':features,'records':encoded,'metadata':meta})
    atomic_json(output.with_suffix('.json'),{**meta,'cache_sha256':sha256_file(output),'cache_bytes':output.stat().st_size})
    print('CACHE_COMPLETE',args.output,features.shape,flush=True)

if __name__=='__main__':main()
