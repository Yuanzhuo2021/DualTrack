"""Load frozen base models and the exact 13000-step inference delta."""
from pathlib import Path
import torch
import json
from DualTrack.io import ROOT, safe_torch_load,sha256_file
from DualTrack.qwen import QwenSpeechTower
from DualTrack.dual_tick import DualTickModel

def load_model(device='cpu'):
    path=ROOT/'checkpoints/step_013000.pt'
    if sha256_file(path)!='fb3b704fbb5a0afb1325592379165b2a4cc409579b7fc3dbd1390b3fc41221bb':
        raise ValueError('Expected the pinned DualTrack 13000 checkpoint')
    payload=safe_torch_load(path)
    if payload.get('step')!=13000:
        raise ValueError('Wrong DualTrack checkpoint')
    original = json.loads((ROOT/'base_locks.json').read_text())['original_checkpoint_lock_digests']
    if payload['base_locks_sha256']!=original['base'] or payload['architecture_lock_sha256']!=original['architecture']:
        raise ValueError('Base/architecture lock mismatch')
    qwen=QwenSpeechTower.load(device,dtype=torch.float32)
    model=DualTickModel.from_qwen(qwen)
    owned=dict(model.named_parameters());state=payload['state_dict']
    for name,tensor in state.items():
        if name not in owned or owned[name].shape!=tensor.shape or not torch.isfinite(tensor).all():
            raise ValueError('Invalid delta parameter: '+name)
    result=model.load_state_dict(state,strict=False)
    if result.unexpected_keys:raise ValueError(result.unexpected_keys)
    return qwen,model.requires_grad_(False).eval()
