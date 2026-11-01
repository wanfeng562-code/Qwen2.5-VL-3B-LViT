import types
import torch
from torch import nn
from src.experiment.model import build_model as original_model

MODEL_VERSION='phase45_lvit_v1'

class TextAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm=nn.LayerNorm(64)
        self.text_norm=nn.LayerNorm(64)
        self.attention=nn.MultiheadAttention(64,4,dropout=0,batch_first=True)
        self.gate=nn.Parameter(torch.tensor(.1))
    def forward(self,x,text):
        t=self.text_norm(text)
        delta=self.attention(self.query_norm(x),t,t,need_weights=False)[0]
        return x+torch.tanh(self.gate)*delta

def first_layer_forward(self,x,skip_x,text,reconstruct=False):
    assert not reconstruct and self.dim==64
    x=self.embeddings(x)
    x=self.text_attention(x,text)
    return self.Encoder_blocks(x)

def build_model(fusion='legacy'):
    model=original_model()
    if fusion=='cross_attention':
        # Isolate new-layer initialization from subsequent training RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed()+4500)
            model.downVit.text_attention=TextAttention()
        del model.downVit.CTBN3
        model.downVit.forward=types.MethodType(first_layer_forward,model.downVit)
    else:assert fusion=='legacy'
    return model
