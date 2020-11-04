from joblib import delayed, Parallel
import numpy as np
from evaluate import auc_eval
from tqdm import tqdm
def par_sim(fun, n, *args, backend='loky'):
    result = Parallel(n_jobs=-1, backend=backend)(delayed(fun)(*args) for _ in range(n))
    return result

def par_addnoise(fun, sources, leadfield, settings, noise_trials=None, backend='loky'):
    result = Parallel(n_jobs=-1, backend=backend)(delayed(fun)(np.matmul(leadfield, source[0]), settings, noise_trials=noise_trials) for source in sources)
    return result

def par_auc(sources, y_ests, pos, return_idx=50, backend='loky'):
    result = Parallel(n_jobs=-1, backend=backend)(delayed(auc_eval)(source[0][:, return_idx], y_est, source[1], pos, plotme=False) 
        for source, y_est in tqdm(zip(sources, y_ests)))
    
    # result = []
    # for source, y_est in zip(sources, y_ests):
    #     res = auc_eval(source[0][:, return_idx], y_est, source[1], pos, plotme=False)
    #     result.append( res )

    return result
