import csv,json
import torch
from src.experiment.common import ROOT,project_path,sha256_file,sha256_text
from src.experiment.data import ManifestDataset,make_loader,read_protocol
from .attributes import rows_for

def manifest_rows(dataset,protocol_path='configs/data_protocol_v1.json',verify_assets=False):
    assert dataset=='monuseg'
    parts=rows_for(('train','val'),verify=verify_assets)
    rows={s:[x for x in parts if x['split']==s] for s in ('train','val')}
    hashes={s:sha256_file(ROOT/f'manifests/legacy_subset_v1/monuseg/{s}.csv') for s in rows}
    return rows,hashes

class FeatureStore:
    def __init__(self,path,all_rows,expected_source=None,expected_manifests=None,mode='correct',seed=666):
        self.path=project_path(path);payload=torch.load(self.path,map_location='cpu',weights_only=False)
        self.metadata=payload['metadata'];self.source=self.metadata['source'];self.sha256=sha256_file(self.path)
        assert expected_source is None or self.source==expected_source
        assert self.metadata['manifest_sha256']==expected_manifests
        expected={x['sample_id'] for x in all_rows}
        assert set(payload['sample_ids'])==expected and len(payload['sample_ids'])==len(expected)
        f=payload['features'];assert f.shape==(len(expected),10,768) and f.dtype==torch.float32 and torch.isfinite(f).all()
        raw=dict(zip(payload['sample_ids'],f.unbind(0)));records={x['sample_id']:x for x in payload['records']}
        for row in all_rows:
            rec=records[row['sample_id']]
            assert rec['image_sha256']==row['image_sha256'] and rec['split']==row['split']
        self.original=raw;self.lookup={};self.mapping={}
        train_ids=[x['sample_id'] for x in all_rows if x['split']=='train']
        self.training_mean=torch.stack([raw[x] for x in train_ids]).mean(0)
        for split in sorted({x['split'] for x in all_rows}):
            ids=[x['sample_id'] for x in all_rows if x['split']==split]
            shift=1+seed%(len(ids)-1) if len(ids)>1 else 0
            for i,sid in enumerate(ids):
                mapped=ids[(i+shift)%len(ids)] if mode=='shuffled' else sid
                assert mode in ('correct','constant','shuffled')
                self.lookup[sid]=self.training_mean if mode=='constant' else raw[mapped]
                self.mapping[sid]={'source_sample_id':mapped if mode!='constant' else None,'same_raw_text':records[sid]['raw_text']==records[mapped]['raw_text'] if mode!='constant' else None,'split':split}
