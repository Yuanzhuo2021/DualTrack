"""Teacher-forced token NLL and accuracy, using the model context limit."""
from pathlib import Path
import argparse,json
import torch
import torch.nn.functional as F
from runtime import load_model
from motion_token_io import read_tokens

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args();model=load_model(a.device)
    target=torch.from_numpy(read_tokens(a.input)).long()[None].to(a.device)
    if target.shape[1]==0:raise ValueError('Empty token sequence')
    losses=torch.zeros(16,device=a.device);correct=torch.zeros_like(losses)
    with torch.inference_mode():
        held=torch.cat((torch.full_like(target[:,:1],1024),target[:,:-1]),1)
        for start in range(0,target.shape[1],model.config.context_frames):
            # Disjoint teacher-forced windows; true previous token retained at each boundary.
            end=min(start+model.config.context_frames,target.shape[1]);t=target[:,start:end]
            out=model(held[:,start:end],torch.ones(t.shape[:2],device=a.device,dtype=torch.bool),targets=t)
            for k,logits in enumerate(out.logits):
                losses[k]+=F.cross_entropy(logits.reshape(-1,1024),t[:,:,k].reshape(-1),reduction='sum')
                correct[k]+=(logits.argmax(-1)==t[:,:,k]).sum()
    n=target.shape[1]
    report=dict(ticks=n,protocol='disjoint teacher-forced windows; context reset per window',nll_per_codebook=(losses/n).tolist(),accuracy_per_codebook=(correct/n).tolist(),mean_nll=float(losses.mean()/n))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
