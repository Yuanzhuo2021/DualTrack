"""Teacher-forced speech/motion losses on one cached aligned pair."""
from pathlib import Path
import argparse,json,os
import numpy as np
import torch
from runtime import load_model
from DualTrack.io import safe_torch_load
from DualTrack.dual_tick import make_dual_batch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition',type=Path,required=True,help='Cached PT with prompt, flan_text, text_valid')
    p.add_argument('--tokens',type=Path,required=True,help='NPZ with speech_codes and motion_codes [T,16]')
    p.add_argument('--qwen-dir',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.qwen_dir:os.environ['DUALTRACK_QWEN_DIR']=str(a.qwen_dir.resolve())
    qwen,model=load_model(a.device);condition=safe_torch_load(a.condition)
    with np.load(a.tokens,allow_pickle=False) as z:
        speech=torch.from_numpy(z['speech_codes'].copy()).long().to(a.device)
        motion=torch.from_numpy(z['motion_codes'].copy()).long().to(a.device)
    if speech.shape!=motion.shape or speech.ndim!=2 or speech.shape[1]!=16 or not 0<len(speech)<model.semoco_lite.prior.config.context_frames:
        raise ValueError('Expected matching [T,16] arrays with 1 <= T <= 149')
    with torch.inference_mode():
        batch=make_dual_batch(model,condition['prompt'],speech,motion,condition['flan_text'],condition['text_valid'],has_eos=False)
        out=model(batch)
    report=dict(ticks=len(speech),speech_loss=float(out.speech_loss.total),motion_loss=float(out.motion_loss.total),total_loss=float(out.loss),protocol='teacher-forced paired token loss; no natural EOS target; not perceptual or FGD metrics')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
