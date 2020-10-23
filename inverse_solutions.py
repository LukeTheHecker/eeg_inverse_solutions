from types import AsyncGeneratorType
import numpy as np
from copy import deepcopy
from scipy.stats import pearsonr
from source_covs import *

def exhaustive_dipole_search(x, leadfield, pos):
    y = np.zeros((pos.shape[0]))
    error = np.zeros((pos.shape[0]))
    for i in range(pos.shape[0]):
        y_test = deepcopy(y)
        y_test[i] = 1
        forward_projection = np.sum(y_test*leadfield, axis=1)
        error[i] = 1 - pearsonr(forward_projection, x)[0]
    
    y_best = deepcopy(y)
    y_best[np.argmin(error)] = 1

    print(error)
    return y_best

def loreta(x, leadfield, sensorNoise=None, sigma=10):
    ''' Calculate the loreta estimate for the eeg inverse problem. ''' 
    if sensorNoise is None:
        sensorNoise = np.zeros((leadfield.shape[0]))

    sourceCov = loreta_sourceCov(leadfield.shape[1], sigma=sigma)
    w = np.matmul( np.matmul(sourceCov, leadfield.T), (np.linalg.inv(sensorNoise + np.matmul(np.matmul(leadfield, sourceCov), leadfield.T))) )
    y_est = np.sum(w*x, axis=1)

    return y_est

def minimum_norm_estimate(x, leadfield, sensorNoise=None):
    ''' Calculate the minimum norm estimate for the eeg inverse problem.''' 
    if sensorNoise is None:
        sensorNoise = np.zeros((leadfield.shape[0]))

    sourceCov = np.identity(leadfield.shape[1])
    w = np.matmul( np.matmul(sourceCov, leadfield.T), (np.linalg.inv(sensorNoise + np.matmul(np.matmul(leadfield, sourceCov), leadfield.T))) )
    y_est = np.sum(w*x, axis=1)
    
    return y_est

def sourceCovEstimate(x, leadfield, sourceCov, sensorNoise=None):
    ''' Calculate the minimum norm estimate for the eeg inverse problem given a specific source covariance matrix.''' 
    if sensorNoise is None:
        sensorNoise = np.zeros((leadfield.shape[0]))

    w = np.matmul( np.matmul(sourceCov, leadfield.T), (np.linalg.inv(sensorNoise + np.matmul(np.matmul(leadfield, sourceCov), leadfield.T))) )
    y_est = np.sum(w*x, axis=1)
    
    return y_est

def dipfit(x, leadfield, sensorNoise=None):
    ''' Calculate the minimum norm estimate for the eeg inverse problem.''' 
    if sensorNoise is None:
        sensorNoise = np.zeros((leadfield.shape[0]))

    sourceCov = np.zeros((leadfield.shape[0]))
    sourceCov[100, 100] = 1
    w = np.matmul( np.matmul(sourceCov, leadfield.T), (np.linalg.inv(sensorNoise + np.matmul(np.matmul(leadfield, sourceCov), leadfield.T))) )
    y_est = np.sum(w*x, axis=1)
    
    return y_est

def minimum_norm_estimate_2(x, leadfield, reg=True, alpha=2e-10):
    # Implementation from:
    # Komssi et al. 2004, doi:10.1016/j.clinph.2003.10.034

    N = leadfield.shape[0]
    K = np.dot(leadfield, leadfield.T)
    
    if reg:
        regularizationTerm = alpha*np.identity(K.shape[0])
        K_inv = np.linalg.pinv(K) + regularizationTerm
    else:
        K_inv = np.linalg.pinv(K)
    
    w = K_inv * x
    y_est = np.sum(np.matmul(w.T , leadfield), axis=0)
    return y_est

def minimum_norm_estimate_3(x, leadfield, sensorNoise, tikhonov=0.05):
    ''' Based on gramfort et al https://www.sciencedirect.com/science/article/pii/S1053811920309150'''
    K_mne = np.matmul(leadfield.T, np.linalg.inv(np.matmul(leadfield, leadfield.T) + tikhonov**2 * sensorNoise))
    y_est = np.matmul(K_mne, x)
    return y_est

def sloreta(x, leadfield, sensorNoise, tikhonov=0.05):
    ''' Based on gramfort et al https://www.sciencedirect.com/science/article/pii/S1053811920309150'''
    K_mne = np.matmul(leadfield.T, np.linalg.inv(np.matmul(leadfield, leadfield.T) + tikhonov**2 * sensorNoise))
    W_diag = 1 / np.diag(np.matmul(K_mne, leadfield))

    W_slor = np.zeros((leadfield.shape[1], leadfield.shape[1]))

    for i in range(leadfield.shape[1]):
        W_slor[i, i] = W_diag[i]

    K_slor = np.matmul(W_slor, K_mne)
    y_est = np.matmul(K_slor, x)

    return y_est

def dspm(x, leadfield, sensorNoise, tikhonov=0.05):
    ''' Based on gramfort et al https://www.sciencedirect.com/science/article/pii/S1053811920309150
    Todo: Create a real noise covariance matrix!
    '''
    
    noiseCov = np.identity(len(sensorNoise))
    noiseCov
    K_mne = np.matmul(leadfield.T, np.linalg.inv(np.matmul(leadfield, leadfield.T) + tikhonov**2 * sensorNoise))
    W_diag = np.sqrt(1 / np.diag(np.matmul(np.matmul(K_mne, noiseCov), K_mne.T)))

    W_dspm = np.zeros((leadfield.shape[1], leadfield.shape[1]))
    for i in range(leadfield.shape[1]):
        W_dspm[i, i] = W_diag[i]

    K_dspm = np.matmul(W_dspm, K_mne)
    y_est = np.matmul(K_dspm, x)

    return y_est
