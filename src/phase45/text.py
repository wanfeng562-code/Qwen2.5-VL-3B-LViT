import torch
import torch.nn.functional as F
from src.experiment.text import FrozenBertEncoder,encoder_fingerprint

class CompleteEncoder(FrozenBertEncoder):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.fingerprint={**self.fingerprint,'version':'phase45_bert_full64_wordpiece_mean_pool10_v1',
                          'max_sequence_length_including_special':64,'positions':10,
                          'text_preparation':'raw_text_no_filler','aggregation':'last_hidden_state_mean_contiguous_hashhash_then_adaptive_avg_pool10',
                          'short_feature_padding':'adaptive_pool_no_padding'}
    @torch.inference_mode()
    def encode(self,texts):
        assert all(isinstance(x,str) and x.strip() for x in texts)
        lengths=[len(self.tokenizer.encode(t,add_special_tokens=True)) for t in texts]
        assert max(lengths)<=64, 'Refuse silent raw text truncation'
        batch=self.tokenizer(texts,padding=True,truncation=False,return_tensors='pt',return_special_tokens_mask=True)
        special=batch.pop('special_tokens_mask')
        hidden=self.model(**{k:v.to(self.device) for k,v in batch.items()}).last_hidden_state
        features=[];records=[]
        from src.experiment.common import sha256_text
        for i,text in enumerate(texts):
            tokens=self.tokenizer.convert_ids_to_tokens(batch['input_ids'][i].tolist());groups=[];words=[]
            for pos,token in enumerate(tokens):
                if special[i,pos] or not batch['attention_mask'][i,pos]:continue
                if token.startswith('##') and groups:groups[-1].append(pos);words[-1]+=token[2:]
                else:groups.append([pos]);words.append(token)
            assert groups
            merged=torch.stack([hidden[i,g].float().mean(0) for g in groups])
            value=F.adaptive_avg_pool1d(merged.T.unsqueeze(0),10)[0].T.cpu()
            features.append(value)
            records.append({'raw_text':text,'raw_text_sha256':sha256_text(text),'merged_tokens':words,'available_merged_tokens':len(words),'raw_wordpiece_count_with_special':lengths[i],'silent_truncation':False,'pooling':'all_valid_merged_positions_to_10'})
        return torch.stack(features),records
