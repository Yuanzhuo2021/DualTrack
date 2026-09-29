from __future__ import annotations
from contextlib import contextmanager
import torch
from joint_motion.model import JointBatch, build_motion_teacher_forcing
from .qwen import isolated_rng

@torch.no_grad()
def native_joint_prompt(qwen, text, reference, language="english"):
    """Capture the official prompt before any temporal forward or speech sampling."""
    result = {}
    class PromptReady(Exception):
        pass
    def capture(module, args, kwargs):
        for name in ("inputs_embeds", "attention_mask", "trailing_text_hidden", "tts_pad_embed"):
            result[name] = kwargs[name].detach().clone()
        raise PromptReady()
    handle = qwen.model.talker.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        try:
            qwen.model.generate(**qwen.generation_arguments(text, reference, language, max_ticks=2))
        except PromptReady:
            pass
    finally:
        handle.remove()
    if not result or result["inputs_embeds"].shape[0] != 1 or not result["attention_mask"].bool().all():
        raise ValueError("joint tick prompt requires a single unpadded sample")
    return result


@torch.no_grad()
def make_joint_batch(model, prompt, speech, motion, *, has_eos=False):
    """Text prefix + previous S/M, explicitly shifted to current S/M targets."""
    if speech.ndim != 2 or speech.shape != motion.shape or speech.shape[-1] != 16:
        raise ValueError("paired codes must both have shape [T,16]")
    if len(speech) == 0:
        raise ValueError("empty paired sequence")
    device = next(model.parameters()).device
    speech, motion = speech.long().to(device), motion.long().to(device)
    prefill = prompt["inputs_embeds"].to(device)
    t = len(speech)
    parts = [model.talker.get_input_embeddings()(speech[:,0])]
    parts.extend(model.talker.code_predictor.get_input_embeddings()[i-1](speech[:,i]) for i in range(1,16))
    emb = torch.stack(parts,1).sum(1)[None]
    tail = prompt["tts_pad_embed"].to(device).expand(1,t,-1).clone()
    n = min(t,prompt["trailing_text_hidden"].shape[1])
    tail[:,:n] = prompt["trailing_text_hidden"][:,:n].to(device)
    base = torch.cat((prefill,emb+tail),1)
    start = prefill.shape[1]-1
    tf = build_motion_teacher_forcing(motion[None],base.shape[1],start)
    sp_target = torch.zeros(1,base.shape[1],16,device=device,dtype=torch.long)
    sp_target[:,start:start+t] = speech
    sp_mask = torch.zeros(base.shape[:2],device=device,dtype=torch.bool)
    sp_mask[:,start:start+t] = True
    q0 = sp_target[...,0].clone()
    q0mask = sp_mask.clone()
    q0[:,start+t] = model.joint_config.speech_codec_eos_token_id
    if has_eos:
        q0mask[:,start+t] = True
    attention = torch.ones(base.shape[:2],device=device,dtype=torch.long)
    positions,_ = model.talker.get_rope_index(attention)
    return JointBatch(base,attention,tf.input_codes,tf.input_mask,tf.bos_mask,
        q0,q0mask,sp_target,sp_mask,tf.targets,tf.target_mask,positions)


class SpeechRngStream:
    """Restore RNG on every yield boundary; caller work cannot perturb speech."""
    def __init__(self, device, seed):
        self.device = torch.device(device)
        with isolated_rng(self.device,seed):
            self.cpu = torch.get_rng_state()
            self.cuda = torch.cuda.get_rng_state(self.device) if self.device.type=="cuda" else None

    @contextmanager
    def use(self):
        devices = [self.device.index or 0] if self.device.type=="cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.cpu)
            if self.cuda is not None:
                torch.cuda.set_rng_state(self.cuda,self.device)
            try:
                yield
            finally:
                self.cpu = torch.get_rng_state()
                if self.cuda is not None:
                    self.cuda = torch.cuda.get_rng_state(self.device)
