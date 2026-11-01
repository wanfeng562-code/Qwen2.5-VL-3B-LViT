"""Exact development-compatible conditions for a separately gated held-out split."""
import torch

def conditions(development,heldout,mode,seed):
    assert mode in ('correct','constant','shuffled','legacy')
    train_ids=[r['sample_id'] for r in development['records'] if r['split']=='train']
    assert len(train_ids)==24
    index={sid:i for i,sid in enumerate(development['sample_ids'])}
    feature_mean=torch.stack([development['features'][index[sid]].float() for sid in train_ids]).mean(0)
    text_mean=torch.stack([development['legacy_text_features'][index[sid]].float() for sid in train_ids]).mean(0)
    ids=heldout['sample_ids'];assert len(ids)==len(set(ids))
    vision=heldout['features'].float();texts=heldout['legacy_text_features'].float()
    assert texts.shape==(len(ids),10,768) and vision.shape==(len(ids),1280,28,28)
    assert torch.isfinite(vision).all() and torch.isfinite(texts).all()
    shift=1+seed%(len(ids)-1);lookup={};mapping={}
    for i,sid in enumerate(ids):
        j=(i+shift)%len(ids) if mode=='shuffled' else i
        lookup[sid]={'vision':feature_mean if mode in ('constant','legacy') else vision[j],
                     'text':texts[i] if mode=='legacy' else text_mean}
        mapping[sid]={'source_sample_id':ids[j] if mode in ('correct','shuffled') else None,
                      'feature_mode':mode,'mean_source':'24_development_train_images_only',
                      'unchanged_feature':torch.equal(vision[i],vision[j]) if mode=='shuffled' else None}
    return lookup,mapping
