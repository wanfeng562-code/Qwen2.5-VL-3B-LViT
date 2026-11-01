import torch
from src.experiment.common import ROOT,project_path,sha256_file
from src.experiment.data import ManifestDataset,make_loader,read_protocol
from src.phase45.data import manifest_rows

class FeatureStore:
    def __init__(self,path,all_rows,expected_source=None,expected_manifests=None,mode='correct',seed=666):
        self.path=project_path(path);payload=torch.load(self.path,map_location='cpu',weights_only=False)
        self.metadata=payload['metadata'];self.source=self.metadata['source'];self.sha256=sha256_file(self.path)
        assert self.source==expected_source and self.metadata['manifest_sha256']==expected_manifests
        ids=payload['sample_ids'];assert len(ids)==len(set(ids))==len(all_rows)
        assert set(ids)=={r['sample_id'] for r in all_rows}
        records={r['sample_id']:r for r in payload['records']}
        for row in all_rows:assert records[row['sample_id']]['image_sha256']==row['image_sha256'] and records[row['sample_id']]['split']==row['split']
        f=payload['features'].float();legacy=payload['legacy_text_features'].float()
        assert legacy.shape==(len(ids),10,768) and torch.isfinite(f).all() and torch.isfinite(legacy).all()
        visual=self.metadata['kind']=='visual'
        assert f.shape[1:]==((self.metadata['channels'],28,28) if visual else (10,768))
        raw=dict(zip(ids,f.unbind(0)));texts=dict(zip(ids,legacy.unbind(0)))
        train=[r['sample_id'] for r in all_rows if r['split']=='train']
        feature_mean=torch.stack([raw[k] for k in train]).mean(0)
        text_mean=torch.stack([texts[k] for k in train]).mean(0)
        self.lookup={};self.mapping={};self.original=raw
        assert mode in ('correct','constant','shuffled','legacy')
        for split in ('train','val'):
            subset=[r['sample_id'] for r in all_rows if r['split']==split];shift=1+seed%(len(subset)-1)
            for i,sid in enumerate(subset):
                other=subset[(i+shift)%len(subset)] if mode=='shuffled' else sid
                feature=feature_mean if mode in ('constant','legacy') else raw[other]
                if visual:
                    self.lookup[sid]={'vision':feature,'text':texts[sid] if mode=='legacy' else text_mean}
                else:self.lookup[sid]={'text':texts[sid] if mode=='legacy' else feature}
                self.mapping[sid]={'split':split,'source_sample_id':other if mode in ('correct','shuffled') else None,
                                   'unchanged_feature':torch.equal(raw[sid],raw[other]) if mode=='shuffled' else None,
                                   'text_policy':'per_image_legacy' if mode=='legacy' else 'train_legacy_mean' if visual else mode}
