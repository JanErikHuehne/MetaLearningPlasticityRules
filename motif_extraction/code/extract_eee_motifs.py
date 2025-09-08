import os 
import json 
import numpy as np
from tqdm import tqdm 
from utils import mock_network, get_triplets








con_ee, con_ie = mock_network()
nas_path = "/home/ge84yes/data" #str(os.environ['nas_path'])
os.makedirs(nas_path + "/motifs", exist_ok=True)
with open(f'{nas_path}/good.txt', "r") as f:
    samples = f.readlines()



for sample in samples:
    
    with open(f'{nas_path}/{sample.strip()}', 'r') as f:
        df = json.load(f)

    weights_ee = df['long_run']['weights_ee']  # dict: index -> list

    # Convert dict -> aligned arrays
    idx = np.fromiter((int(k) for k in weights_ee.keys()), count=len(weights_ee), dtype=np.int64)
    # Extract the scalar we care about (values[0][0]) **only** when it passes your rule
    def pick(v):
        # v is a list; keep v[0][0] if len(v)==2 OR (len(v)==1 and v[0][-1]==0), else NaN
        if len(v) == 2:
            return v[0][0]
        v0 = v[0]
        return v0[0] if v0[-1] == 0 else np.nan

    val0 = np.fromiter((pick(v) for v in weights_ee.values()), count=len(weights_ee), dtype=float)

    # Keep only valid entries
    mask = ~np.isnan(val0)
    idx_kept = idx[mask]
    ii = con_ee.i[idx_kept]
    jj = con_ee.j[idx_kept]
    vv = val0[mask]
    # Build matrix in one go
    matrix_ee = np.zeros((800, 800), dtype=float)
    matrix_ee[ii, jj] = vv
    binary_matrix_ee = (matrix_ee > 0).astype(np.uint8)
    result = np.vstack(get_triplets(binary_matrix_ee))
    out_name = os.path.splitext(os.path.basename(sample.strip()))[0] + "_motifs.npy"
    np.save(os.path.join(nas_path, "motifs", out_name), result)
    