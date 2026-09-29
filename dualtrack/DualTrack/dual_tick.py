"""Two intact temporal backbones, causal history exchange and same-tick fusion."""
from dataclasses import dataclass
import torch
from torch import nn

from joint_motion.config import JointMotionConfig, validate_qwen_0_6b_talker
from joint_motion.model import JointBatch
from joint_motion.losses import speech_packet_loss, motion_packet_loss
from joint_motion.generation import SamplingConfig, _sample, _speech_depth_packet, _speech_packet_embedding
from .motionprior import MotionPriorTower
from .joint_tick import make_joint_batch, SpeechRngStream


STARTUP_POLICY = "native_bos_no_cross_input_v1"


class ZeroResidual(nn.Sequential):
    def __init__(self, input_dim, output_dim, width=512):
        super().__init__(nn.LayerNorm(input_dim), nn.Linear(input_dim,width), nn.SiLU(),
                         nn.Linear(width,output_dim,bias=False))
        nn.init.zeros_(self[-1].weight)


class TickFusion(nn.Module):
    """Pointwise joint state, with native hidden states retained as residuals."""
    def __init__(self,qwen_dim=1024,motion_dim=768,width=512):
        super().__init__()
        self.qwen_norm=nn.LayerNorm(qwen_dim)
        self.motion_norm=nn.LayerNorm(motion_dim)
        self.trunk=nn.Sequential(nn.Linear(qwen_dim+motion_dim,width),nn.SiLU())
        self.to_speech=nn.Linear(width,qwen_dim,bias=False)
        self.to_motion=nn.Linear(width,motion_dim,bias=False)
        nn.init.zeros_(self.to_speech.weight)
        nn.init.zeros_(self.to_motion.weight)

    def forward(self,qwen,motion,enabled=True):
        joint=torch.cat((qwen,motion),dim=-1)
        if not enabled:
            return joint,qwen,motion
        shared=self.trunk(torch.cat((self.qwen_norm(qwen),self.motion_norm(motion)),dim=-1))
        return joint,qwen+self.to_speech(shared),motion+self.to_motion(shared)


@dataclass
class DualTickBatch:
    joint: JointBatch
    speech_codes: torch.Tensor
    motion_codes: torch.Tensor
    flan_text: torch.Tensor
    text_valid: torch.Tensor
    loss_start: int = 0
    speech_history_codes: torch.Tensor | None = None
    motion_history_codes: torch.Tensor | None = None
    generated_packet_mask: torch.Tensor | None = None

    def validate_history(self, config):
        values=(self.speech_history_codes,self.motion_history_codes,self.generated_packet_mask)
        if all(x is None for x in values):
            if self.joint.generated_history_mask is not None:
                raise ValueError('generated joint history requires explicit dual histories')
            return
        if any(x is None for x in values):raise ValueError('both histories and generation mask required')
        mask=self.generated_packet_mask
        if mask.shape!=(len(self.speech_codes),) or mask.dtype!=torch.bool:
            raise ValueError('invalid generated packet mask')
        for history,target,size in ((self.speech_history_codes,self.speech_codes,2048),
                                    (self.motion_history_codes,self.motion_codes,1024)):
            if history.shape!=target.shape or history.dtype!=torch.long or torch.any((history<0)|(history>=size)):
                raise ValueError('invalid history token shape, dtype or vocabulary')
            if not torch.equal(history[~mask],target[~mask]):raise ValueError('unmarked history differs from true targets')
        start=int(self.joint.motion_bos_mask[0].nonzero()[0]);t=len(mask)
        if self.joint.generated_history_mask is None or not torch.equal(self.joint.generated_history_mask[0,start+1:start+t+1],mask):
            raise ValueError('dual/joint generated masks differ')
        if not torch.equal(self.joint.motion_input_codes[0,start+1:start+t+1],self.motion_history_codes):
            raise ValueError('joint/Lite motion histories differ')
        if not torch.equal(self.joint.speech_packet_targets[0,start:start+t],self.speech_codes) or not torch.equal(self.joint.motion_packet_targets[0,start:start+t],self.motion_codes):
            raise ValueError('true targets changed while replacing history')


@dataclass
class DualStates:
    qwen_raw: torch.Tensor
    motion_raw: torch.Tensor
    joint_hidden: torch.Tensor
    speech_hidden: torch.Tensor
    motion_hidden: torch.Tensor
    motion_full: torch.Tensor | None = None
    previous_q0: torch.Tensor | None = None


@dataclass
class DualOutput:
    loss: torch.Tensor
    speech_loss: object
    motion_loss: object
    states: DualStates
    speech_q0_logits: torch.Tensor
    motion_logits: tuple
    speech_depth_logits: torch.Tensor | None = None


def make_dual_batch(model,prompt,speech,motion,flan_text,text_valid,*,has_eos=False,loss_start=0):
    device=next(model.parameters()).device
    joint=make_joint_batch(model,prompt,speech,motion,has_eos=has_eos)
    joint.validate(model.joint_config)
    if not 0 <= loss_start < len(speech):
        raise ValueError('loss_start must leave at least one target tick')
    flan_text=flan_text.to(device=device,dtype=joint.base_input_embeddings.dtype)
    text_valid=text_valid.to(device=device,dtype=torch.bool)
    if flan_text.ndim==2:
        flan_text=flan_text[None]
    if text_valid.ndim==1:
        text_valid=text_valid[None]
    if (flan_text.ndim!=3 or flan_text.shape[0]!=1 or text_valid.shape!=flan_text.shape[:2]
        or not text_valid.any() or not torch.isfinite(flan_text).all()):
        raise ValueError("requires one valid Flan text prefix")
    return DualTickBatch(joint,speech.long().to(device),motion.long().to(device),flan_text,text_valid,int(loss_start))


class DualTickModel(nn.Module):
    def __init__(self,talker,semoco_lite):
        super().__init__()
        self.joint_config=JointMotionConfig()
        validate_qwen_0_6b_talker(talker,self.joint_config)
        self.talker=talker
        self.semoco_lite=semoco_lite
        if semoco_lite.prior.config.d_model!=768:
            raise ValueError("requires MotionPrior large 768")
        self.motion_to_speech_input=ZeroResidual(768,1024)
        self.speech_to_motion_input=ZeroResidual(1024,768)
        self.fusion=TickFusion()
        self.qwen_top_layers=0
        self.lite_top_layers=0
        self.exchange_enabled=True
        self._freeze_policy()

    @classmethod
    def from_qwen(cls,qwen):
        # Strict full state_dict loading occurs in FullLiteMotionTower.load.
        model=cls(qwen.model.talker,MotionPriorTower.load(qwen.device))
        return model.to(device=qwen.device,dtype=next(qwen.model.talker.parameters()).dtype)

    def _freeze_policy(self):
        self.requires_grad_(False)
        self.talker.eval()
        self.semoco_lite.eval()


    def train(self,mode=True):
        super().train(mode)
        self._freeze_policy()
        return self

    def motion_input_delta(self,embedding):
        if self.exchange_enabled:
            return self.motion_to_speech_input(embedding)
        return embedding.new_zeros(*embedding.shape[:-1],1024)

    def speech_input_delta(self,embedding):
        if self.exchange_enabled:
            return self.speech_to_motion_input(embedding)
        return embedding.new_zeros(*embedding.shape[:-1],768)

    def speech_bos(self):
        device=next(self.parameters()).device
        ids=torch.full((1,1),self.joint_config.speech_codec_bos_token_id,device=device,dtype=torch.long)
        return self.talker.get_input_embeddings()(ids)

    def states_from_raw(self,qwen,motion):
        joint,speech_h,motion_h=self.fusion(qwen,motion,self.exchange_enabled)
        return DualStates(qwen,motion,joint,speech_h,motion_h)

    def teacher_forced_states(self,batch):
        j=batch.joint
        j.validate(self.joint_config)
        batch.validate_history(self.joint_config)
        motion=self.semoco_lite.embed(j.motion_input_codes)
        motion=motion*j.motion_input_mask[...,None]
        # Only real previous packets cross modalities; native BOS stays local.
        active=j.motion_input_mask[...,None]
        x=j.base_input_embeddings+self.motion_input_delta(motion)*active
        qwen=self.talker.model(inputs_embeds=x,attention_mask=j.attention_mask,position_ids=j.position_ids,
            use_cache=False,return_dict=True).last_hidden_state
        start=int(j.motion_bos_mask[0].nonzero()[0])
        qwen=qwen[:,start:]
        speech_history=batch.speech_codes if batch.speech_history_codes is None else batch.speech_history_codes
        motion_history=batch.motion_codes if batch.motion_history_codes is None else batch.motion_history_codes
        previous_speech=_speech_packet_embedding(self,speech_history).squeeze(1)[None]
        previous_speech=torch.cat((self.speech_bos(),previous_speech),dim=1)
        held=torch.cat((self.semoco_lite.bos(1,qwen.device),motion_history[None]),dim=1)
        speech_condition=self.speech_input_delta(previous_speech[:,1:])
        speech_condition=torch.cat((speech_condition.new_zeros(1,1,768),speech_condition),dim=1)
        full,semantic=self.semoco_lite.contexts(held,speech_condition)
        state=self.states_from_raw(qwen,semantic)
        state.motion_full=full;state.previous_q0=held[...,0]
        return state

    def forward(self,batch):
        state=self.teacher_forced_states(batch)
        left=batch.loss_start
        codes=batch.speech_codes[left:]
        t=len(codes)
        q0_logits=self.talker.codec_head(state.speech_hidden)
        pieces=[state.speech_hidden[0,left:left+t,None]]
        for i in range(15):
            emb=self.talker.get_input_embeddings() if i==0 else self.talker.code_predictor.get_input_embeddings()[i-1]
            pieces.append(emb(codes[:,i:i+1]))
        depth=self.talker.code_predictor.forward_finetune(inputs_embeds=torch.cat(pieces,1),labels=None,use_cache=False).logits
        start=int(batch.joint.motion_bos_mask[0].nonzero()[0])
        q0_loss_mask=batch.joint.speech_q0_mask[:,start:].clone()
        q0_loss_mask[:,:left]=False
        speech_loss=speech_packet_loss(q0_logits,batch.joint.speech_q0_targets[:,start:],
            q0_loss_mask,depth,codes,self.joint_config.speech_depth_loss_scale)
        motion_targets=batch.motion_codes[left:][None]
        motion_logits=self.semoco_lite.logits(state.motion_hidden[:,left:left+t],state.motion_full[:,left:left+t],state.previous_q0[:,left:left+t],motion_targets)
        motion_loss=motion_packet_loss(motion_logits,motion_targets,
            torch.ones(1,t,device=codes.device,dtype=torch.bool),self.joint_config.motion_loss_weights,
            self.joint_config.motion_depth_loss_scale)
        return DualOutput(speech_loss.total+motion_loss.total,speech_loss,motion_loss,state,q0_logits,tuple(motion_logits),depth)


@dataclass
class DualTick:
    index: int
    states: DualStates
    speech_packet: torch.Tensor
    motion_packet: torch.Tensor


class DualTickStream:
    """Two persistent caches; one causal update of each backbone per next()."""
    def __init__(self,model,prompt,flan_text,text_valid,*,max_ticks=300,speech_seed=0,motion_seed=1,
                 speech_sampling=None,motion_sampling=None,repetition_penalty=1.05):
        if not 1<=max_ticks<=300 or repetition_penalty<=0:
            raise ValueError("invalid generation budget or repetition penalty")
        self.model=model.eval()
        self.device=next(model.parameters()).device
        self.prompt={k:v.to(self.device) for k,v in prompt.items()}
        if self.prompt['inputs_embeds'].shape[0]!=1 or not self.prompt['attention_mask'].bool().all():
            raise ValueError("one unpadded Qwen prompt required")
        self.text=flan_text.to(self.device,dtype=next(model.parameters()).dtype)
        self.text_valid=text_valid.to(self.device,dtype=torch.bool)
        if self.text.ndim==2:self.text=self.text[None]
        if self.text_valid.ndim==1:self.text_valid=self.text_valid[None]
        if self.text.shape[0]!=1 or self.text_valid.shape!=self.text.shape[:2] or not self.text_valid.any():
            raise ValueError("one valid Lite text prefix required")

        self.max_ticks=max_ticks
        self.speech_sampling=speech_sampling or SamplingConfig(do_sample=False)
        self.motion_sampling=motion_sampling or SamplingConfig(do_sample=False)
        self.repetition_penalty=repetition_penalty
        self.speech_rng=SpeechRngStream(self.device,speech_seed)
        self.motion_rng=torch.Generator(device=self.device).manual_seed(motion_seed)
        self.qwen_cache=None
        self.motion_cache=None
        self.qwen_attention=None
        self.previous_speech=None
        self.previous_motion=None
        self.q0_history=[]
        self.index=0
        self.qwen_forward_count=0
        self.motion_forward_count=0
        self.stop_reason=None
        self.states=None

    def __iter__(self):return self

    @torch.no_grad()
    def advance_states(self):
        model=self.model
        tower=model.semoco_lite
        if self.index==0:
            x=self.prompt['inputs_embeds'].clone()
            self.qwen_attention=torch.ones(x.shape[:2],device=x.device,dtype=torch.long)
            positions,_=model.talker.get_rope_index(self.qwen_attention)
            previous_speech=model.speech_bos()
        else:
            previous_speech=_speech_packet_embedding(model,self.previous_speech)
            motion=tower.embed(self.previous_motion[:,None])
            x=previous_speech+model.motion_input_delta(motion)
            trailing=self.prompt['trailing_text_hidden']
            k=self.index-1
            x=x+(trailing[:,k:k+1] if k<trailing.shape[1] else self.prompt['tts_pad_embed'])
            position=self.qwen_attention.shape[1]
            self.qwen_attention=torch.cat((self.qwen_attention,torch.ones(1,1,device=x.device,dtype=torch.long)),1)
            positions=torch.full((3,1,1),position,device=x.device,dtype=torch.long)
        output=model.talker.model(inputs_embeds=x,attention_mask=self.qwen_attention,position_ids=positions,
            past_key_values=self.qwen_cache,use_cache=True,return_dict=True)
        self.qwen_cache=output.past_key_values
        qwen=output.last_hidden_state[:,-1]
        self.qwen_forward_count+=1
        condition=(qwen.new_zeros(1,768) if self.index==0
                   else model.speech_input_delta(previous_speech)[:,0])
        full,semantic,self.motion_cache=tower.step(self.previous_motion,condition,self.motion_cache)
        self.motion_forward_count+=1
        self.states=model.states_from_raw(qwen,semantic[:,-1])
        self.states.motion_full=full[:,-1]
        self.states.previous_q0=self.motion_cache.held[:,-1,0]
        return self.states

    @torch.no_grad()
    def __next__(self):
        if self.stop_reason is not None or self.index>=self.max_ticks:
            if self.stop_reason is None:self.stop_reason='max_ticks'
            raise StopIteration
        state=self.advance_states()
        model=self.model
        with self.speech_rng.use():
            scores=model.talker.codec_head(state.speech_hidden).float()
            if self.q0_history and self.repetition_penalty!=1:
                ids=torch.tensor(self.q0_history,device=self.device).unique()
                values=scores[:,ids]
                scores[:,ids]=torch.where(values<0,values*self.repetition_penalty,values/self.repetition_penalty)
            eos=model.joint_config.speech_codec_eos_token_id
            eos_score=scores[:,eos].clone()
            scores[:,model.joint_config.speech_depth_vocab_size:]=-torch.inf
            if self.index>=self.speech_sampling.min_speech_ticks:scores[:,eos]=eos_score
            q0=_sample(scores,self.speech_sampling)
            if q0.item()==eos:
                self.stop_reason='speech_eos'
                raise StopIteration
            speech=_speech_depth_packet(model,state.speech_hidden,q0,self.speech_sampling)
        packet=model.semoco_lite.sample(state.motion_hidden,state.motion_full,state.previous_q0,self.motion_sampling,self.motion_rng)
        result=DualTick(self.index,state,speech[0].clone(),packet[0].clone())
        self.previous_speech=speech
        self.previous_motion=packet
        self.q0_history.append(int(q0.item()))
        self.index+=1
        return result


def dual_tick_inventory(model):
    return {'architecture':'dual_temporal_joint_tick_motionprior',
            'qwen_temporal_layers':28,'motion_temporal_layers':4,'semantic_temporal_layers':2,
            'motion_depth_layers':3,'history_exchange':True,'same_tick_state_fusion':True,
            'startup_policy':STARTUP_POLICY,'motion_base_step':861853}
