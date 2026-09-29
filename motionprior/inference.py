"""Generate motion tokens from BOS or a prefix and decode to 25 Hz UMR499."""
from pathlib import Path
import argparse, json
import numpy as np
import torch
from runtime import load_model, decode_features

from motion_token_io import read_tokens

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--prefix',type=Path,help='Optional NPZ; otherwise generate from BOS')
    p.add_argument('--ticks',type=int,default=96)
    p.add_argument('--temperature',type=float,default=0.0)
    p.add_argument('--top-k',type=int,default=20)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=load_model(a.device,a.checkpoint)
    prefix=torch.from_numpy(read_tokens(a.prefix)).long()[None].to(a.device) if a.prefix else torch.full((1,1,16),1024,device=a.device,dtype=torch.long)
    with torch.inference_mode():
        tokens=model.generate(prefix,a.ticks,temperature=a.temperature,top_k=a.top_k)
        features=decode_features(model.token_interface,tokens)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,motion_codes=tokens[0].cpu().numpy(),features=features[0].cpu().numpy(),fps=np.float32(25),token_hz=np.float32(12.5))
    print(json.dumps(dict(output=str(a.out),ticks=tokens.shape[1],frames=features.shape[1])))
if __name__=='__main__':main()
