"""Restricted tensor/NumPy checkpoint loading; never enables arbitrary pickle."""
from pathlib import Path
import hashlib
import importlib
import numpy as np
import torch

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def load_checkpoint(path):
    core = importlib.import_module('numpy._core.multiarray' if hasattr(np, '_core') else 'numpy.core.multiarray')
    allowed = [core._reconstruct, np.ndarray, np.dtype,
               type(np.dtype('float32')), type(np.dtype('float64'))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True, mmap=True)
