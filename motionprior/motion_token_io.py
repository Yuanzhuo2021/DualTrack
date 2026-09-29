import numpy as np

def read_tokens(path):
    with np.load(path, allow_pickle=False) as z:
        key = next((k for k in ('motion_codes','motion_tokens','semoco_motion_tokens','tokens_structured','tokens') if k in z), None)
        if key is None:
            raise ValueError('NPZ has no supported motion token array')
        values = z[key].copy()
    if values.ndim != 2 or values.shape[1] != 16 or not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values >= 1024)):
        raise ValueError('Expected integer motion tokens [T,16] in [0,1023]')
    return values
