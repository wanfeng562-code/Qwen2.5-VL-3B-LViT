import torch
import torch.nn.functional as F
from torch import nn
from src.phase45.model import build_model as text_model

MODEL_VERSION='phase46_qwen_dense_residual_v1'

class ConditionedModel(nn.Module):
    def __init__(self,architecture):
        super().__init__();self.architecture=architecture
        self.base=text_model('cross_attention' if architecture=='S' else 'legacy')
        self.context=None
        if architecture!='S':
            channels={'V_mid':1280,'V_merged':2048}[architecture]
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed()+4600)
                self.adapter=nn.Sequential(nn.GroupNorm(1,channels),nn.Conv2d(channels,64,1,bias=False),
                                           nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,64,3,padding=1))
                self.gate=nn.Parameter(torch.tensor(.1))
            self.base.outc.register_forward_pre_hook(self._inject)

    def _inject(self,module,args):
        assert self.context is not None
        x=args[0];delta=self.adapter(self.context)
        delta=F.interpolate(delta,size=x.shape[-2:],mode='bilinear',align_corners=False)
        return (x+torch.tanh(self.gate)*delta,)

    def forward(self,image,condition):
        assert self.context is None
        if self.architecture!='S':self.context=condition['vision']
        try:return self.base(image,condition['text'])
        finally:self.context=None

    def conditioning_gate(self):
        gate=self.base.downVit.text_attention.gate if self.architecture=='S' else self.gate
        return float(torch.tanh(gate).detach())

def build_model(config):return ConditionedModel(config['architecture'])
