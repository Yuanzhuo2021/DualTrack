"""Load and decode the exact MotionPrior 861853 prior."""
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'tokenizer'))
from checkpoint_io import load_checkpoint, sha256_file
from motionprior.models.motion_prior import MotionPrior, MotionPriorConfig
from motionprior.models.semoco_token_interface import FrozenSeMoCoTokenInterface
from motionprior.contracts.semoco import SEMOCO_CONTRACT

def load_model(device='cpu', checkpoint=None):
    checkpoint = Path(checkpoint or ROOT / 'checkpoints/step861853.pt')
    expected = 'a0a651d0699662fdfb925646e11ece5be8b027d66cac914047657e5157323442'
    if sha256_file(checkpoint) != expected:
        raise ValueError('Expected the pinned MotionPrior 861853 checkpoint')
    payload = load_checkpoint(checkpoint)
    if payload['step'] != 861853 or payload['contract_snapshot']['checkpoint_sha256'] != SEMOCO_CONTRACT.checkpoint_sha256:
        raise ValueError('Motion prior/tokenizer contract mismatch')
    interface = FrozenSeMoCoTokenInterface.from_contract(device='cpu')
    for name, value in interface.state_dict().items():
        if not torch.equal(value, payload['model']['token_interface.' + name]):
            raise ValueError('Embedded codec differs: ' + name)
    model = MotionPrior(interface, MotionPriorConfig(**payload['resolved_config']['model']))
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).requires_grad_(False).eval()

@torch.inference_mode()
def decode_features(interface, tokens):
    from data.umr_schema import SLICE_FOOT_CONTACT
    raw = interface.decode(tokens)
    features = raw * interface.norm_std + interface.norm_mean
    features[:, SLICE_FOOT_CONTACT] = raw[:, SLICE_FOOT_CONTACT]
    return features.transpose(1, 2)
