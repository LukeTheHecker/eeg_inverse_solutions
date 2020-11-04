import numpy as np
import scipy.ndimage as ndimage



def loreta_sourceCov(n_dip, sigma=5):
    sourceCov = np.identity(n_dip)
    # sourceCovSmooth = loreta_smooth(sourceCov)
    sourceCovSmooth = ndimage.gaussian_filter(sourceCov, sigma=(sigma, sigma), order=0)
    return sourceCovSmooth

def loreta_smooth(identity):
    for i in range(identity.shape[0]-1):
        identity[i+1, i] = 0.5
        identity[i, i+1] = 0.5
    return identity