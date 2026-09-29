"""Text + reference voice -> synchronized speech and UMR499 motion."""
from pathlib import Path
import argparse,json,os
import numpy as np
import torch
import soundfile as sf
from runtime import load_model
from DualTrack.joint_tick import native_joint_prompt
from DualTrack.dual_tick import DualTickStream
from joint_motion.generation import SamplingConfig

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text',required=True)
    p.add_argument('--language',choices=['english','spanish','chinese','japanese'],default='english')
    p.add_argument('--reference-audio',type=Path,required=True)
    p.add_argument('--qwen-dir',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--ticks',type=int,default=150)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--greedy',action='store_true')
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    if a.qwen_dir:os.environ['DUALTRACK_QWEN_DIR']=str(a.qwen_dir.resolve())
    torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    qwen,model=load_model(a.device)
    from data.umr_schema import SLICE_FOOT_CONTACT
    wave,sr=sf.read(a.reference_audio,dtype='float32',always_2d=True)
    with torch.inference_mode():
        reference=qwen.reference(wave.mean(1),sr)
        prompt=native_joint_prompt(qwen,a.text,reference,language=a.language)
        config=SamplingConfig(do_sample=not a.greedy,min_speech_ticks=2)
        # This model's generation state does not consume the legacy Flan input.
        stream=DualTickStream(model,prompt,torch.zeros(1,1024),torch.ones(1,dtype=torch.bool),max_ticks=a.ticks,speech_seed=a.seed,motion_seed=a.seed+1,speech_sampling=config,motion_sampling=config)
        speech=[];motion=[]
        for tick in stream:speech.append(tick.speech_packet);motion.append(tick.motion_packet)
        if not speech:raise RuntimeError('No packets generated')
        speech=torch.stack(speech);motion=torch.stack(motion)
        generated,sr=qwen.decode(speech)
        interface=model.semoco_lite.prior.token_interface
        raw=interface.decode(motion[None]);features=raw*interface.norm_std+interface.norm_mean
        features[:,SLICE_FOOT_CONTACT]=raw[:,SLICE_FOOT_CONTACT]
    a.out_dir.mkdir(parents=True,exist_ok=True)
    sf.write(a.out_dir/'speech.wav',generated,sr,subtype='FLOAT')
    np.savez_compressed(a.out_dir/'motion.npz',features=features[0].T.cpu().numpy(),fps=np.float32(25),motion_codes=motion.cpu().numpy(),speech_codes=speech.cpu().numpy(),token_hz=np.float32(12.5))
    report=dict(text=a.text,language=a.language,ticks=len(speech),stop_reason=stream.stop_reason,seed=a.seed,motion_format='UMR499 at 25 Hz; foot contacts are logits',conditioning='reference-speaker embedding')
    (a.out_dir/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
