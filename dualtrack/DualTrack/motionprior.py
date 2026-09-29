"""Native large MotionPrior behind the Final dual-tick interaction modules."""
import os, sys
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .io import ROOT, locked_path, safe_torch_load
SOURCE=str(ROOT.parent/'motionprior')
if SOURCE not in sys.path: sys.path.insert(0,SOURCE)
from motionprior.models.motion_prior import MotionPrior, MotionPriorConfig
from motionprior.models.semoco_token_interface import FrozenSeMoCoTokenInterface
from motionprior.models.transformer import _apply_rope
from motionprior.models.q0_factorization import factorized_q0_sample
from motionprior.contracts.semoco import SEMOCO_CONTRACT
from joint_motion.generation import _sample

@dataclass
class PriorCache:
    held: torch.Tensor
    conditions: torch.Tensor
    full: list
    semantic: list

class MotionPriorTower(nn.Module):
    def __init__(self, prior):
        super().__init__(); self.prior=prior; self.checkpoint_blocks=False

    @classmethod
    def load(cls,device):
        p=safe_torch_load(locked_path('motionprior'),allow_numpy_arrays=True)
        assert p['step']==861853
        assert p['contract_snapshot']['checkpoint_sha256']==SEMOCO_CONTRACT.checkpoint_sha256
        interface=FrozenSeMoCoTokenInterface.from_contract(device='cpu')
        prior=MotionPrior(interface,MotionPriorConfig(**p['resolved_config']['model']))
        # Check every embedded codec tensor against the independently pinned codec.
        expected=interface.state_dict()
        for key,value in expected.items():
            assert torch.equal(value,p['model']['token_interface.'+key]),key
        prior.load_state_dict(p['model'],strict=True)
        return cls(prior.to(device).eval())

    def embed(self,codes): return self.prior.frame_embedding(codes).hidden
    def bos(self,batch,device): return torch.full((batch,1,16),1024,device=device,dtype=torch.long)


    def _temporal(self, temporal, x):
        mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        return temporal(x, mask)

    def contexts(self,held,condition):
        if held.shape[1]>self.prior.config.context_frames: raise ValueError('MotionPrior context exceeds 150 ticks')
        e=self.prior.frame_embedding(held)
        return self._temporal(self.prior.temporal,e.hidden+condition),self._temporal(self.prior.semantic_temporal,e.semantic+condition)

    def depth_context(self,semantic,full):
        return self.prior.depth_context_norm(full+self.prior.semantic_to_depth(semantic))

    def logits(self,semantic,full,previous_q0,targets):
        return (self.prior.q0_logits(semantic,previous_q0),*self.prior.depth.forward_parallel(self.depth_context(semantic,full),targets))

    @torch.no_grad()
    def sample(self,semantic,full,previous_q0,sampling,generator):
        _,transition,destination=self.prior._q0_outputs(semantic,previous_q0)
        def sampler(scores,*,temperature,top_k): return _sample(scores,sampling,generator)
        q0=factorized_q0_sample(transition,destination,previous_q0,vocab_size=1024,sampler=sampler,temperature=0.,top_k=None)
        packet=q0[:,None];depth=self.depth_context(semantic,full)
        for k in range(1,16): packet=torch.cat((packet,_sample(self.prior.depth.next_logits(depth,packet,k),sampling,generator)[:,None]),-1)
        return packet

    def _cached(self,temporal,x,caches=None):
        caches=caches or [None]*len(temporal.transformer.blocks);result=[]
        for block,cache in zip(temporal.transformer.blocks,caches):
            a=block.attention;z=block.attention_norm(x);b,t,_=z.shape
            q,k,v=a.qkv(z).view(b,t,3,a.num_heads,a.head_dim).unbind(2)
            q,k,v=q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
            offset=0 if cache is None else cache[0].shape[2];pos=torch.arange(offset,offset+t,device=x.device)
            q,k=_apply_rope(q,pos),_apply_rope(k,pos)
            if cache is not None:k,v=torch.cat((cache[0],k),2),torch.cat((cache[1],v),2)
            allowed=torch.arange(k.shape[2],device=x.device)[None,:]<=pos[:,None]
            h=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed[None,None],dropout_p=0.)
            x=x+a.output(h.transpose(1,2).reshape(b,t,a.d_model));x=x+block.ffn(block.ffn_norm(x));result.append((k,v))
        return temporal.transformer.norm(x),result

    @torch.no_grad()
    def prefill(self,held,conditions):
        e=self.prior.frame_embedding(held)
        full,fc=self._cached(self.prior.temporal,e.hidden+conditions)
        semantic,sc=self._cached(self.prior.semantic_temporal,e.semantic+conditions)
        return full,semantic,PriorCache(held,conditions,fc,sc)

    @torch.no_grad()
    def step(self,previous,condition,cache):
        held=self.bos(condition.shape[0],condition.device) if previous is None else previous[:,None];condition=condition[:,None]
        if cache is None:return self.prefill(held,condition)
        history=torch.cat((cache.held,held),1);conds=torch.cat((cache.conditions,condition),1)
        if history.shape[1]>150:
            full,semantic,new=self.prefill(history[:,-150:],conds[:,-150:]);return full[:,-1:],semantic[:,-1:],new
        e=self.prior.frame_embedding(held)
        full,fc=self._cached(self.prior.temporal,e.hidden+condition,cache.full)
        semantic,sc=self._cached(self.prior.semantic_temporal,e.semantic+condition,cache.semantic)
        return full,semantic,PriorCache(history,conds,fc,sc)
