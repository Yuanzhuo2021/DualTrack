"""Resolve inference assets with checkpoint hash checks."""
from pathlib import Path
from functools import lru_cache
import json,os,sys
ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT.parent
for path in (BUNDLE/'tokenizer', BUNDLE/'motionprior', ROOT/'third_party/Qwen3-TTS'):
    sys.path.insert(0,str(path))
from checkpoint_io import load_checkpoint,sha256_file

def safe_torch_load(path, **kwargs):
    return load_checkpoint(path)

@lru_cache(maxsize=None)
def locked_path(name):
    locks=json.loads((ROOT/'base_locks.json').read_text())
    qwen=Path(os.environ.get('DUALTRACK_QWEN_DIR', str(BUNDLE/'external/Qwen3-TTS-12Hz-0.6B-Base')))
    paths={'motionprior':BUNDLE/'motionprior/checkpoints/step861853.pt',
           'semoco_tokenizer':BUNDLE/'tokenizer/checkpoints/best.pt',
           'qwen_model':qwen/'model.safetensors',
           'qwen_speech_tokenizer':qwen/'speech_tokenizer/model.safetensors'}
    path=paths[name]
    if not path.is_file():
        raise FileNotFoundError(f'{path}: download release weights, or set --qwen-dir / DUALTRACK_QWEN_DIR for the Qwen base')
    if sha256_file(path)!=locks['artifacts'][name]['sha256']:
        raise ValueError('Checkpoint hash mismatch: '+str(path))
    return path
