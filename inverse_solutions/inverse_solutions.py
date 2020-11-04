from types import AsyncGeneratorType
import numpy as np
from copy import deepcopy
from scipy.stats import pearsonr
from source_covs import *
from util import *
from mne.inverse_sparse import gamma_map

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

    W_slor = np.diag(W_diag)

    W_slor = np.sqrt(W_slor)

    K_slor = np.matmul(W_slor, K_mne)
    y_est = np.matmul(K_slor, x)

    return y_est

def dspm(x, leadfield, sensorNoise, tikhonov=0.05):
    ''' Based on https://www.sciencedirect.com/science/article/pii/S1053811920309150
    Todo: Create a real noise covariance matrix!
    '''
    
    noiseCov = np.identity(len(sensorNoise))
    noiseCov
    K_mne = np.matmul(leadfield.T, np.linalg.inv(np.matmul(leadfield, leadfield.T) + tikhonov**2 * sensorNoise))
    W_diag = 1 / np.diag(np.matmul(np.matmul(K_mne, noiseCov), K_mne.T))

    W_dspm = np.diag(W_diag)
    
    W_dspm = np.sqrt(W_dspm)

    K_dspm = np.matmul(W_dspm, K_mne)
    y_est = np.matmul(K_dspm, x)

    return y_est

def eloreta(x, leadfield, tikhonov=0.05, stopCrit=0.005):
    D, C = calc_eloreta_D(leadfield, tikhonov, stopCrit=stopCrit)
    
    K_elor = np.matmul( np.matmul(np.linalg.inv(D), leadfield.T), np.linalg.inv( np.matmul( np.matmul( leadfield, np.linalg.inv(D) ), leadfield.T) + (tikhonov**2 * C) ) )

    y_est = np.matmul(K_elor, x)
    return y_est

def calc_eloreta_D(leadfield, tikhonov, stopCrit=0.005):
    ''' Algorithm that optimizes weight matrix D as described in 
        Assessing interactions in the brain with exactlow-resolution electromagnetic tomography; Pascual-Marqui et al. 2011 and
        https://www.sciencedirect.com/science/article/pii/S1053811920309150
        '''
    numberOfElectrodes, numberOfVoxels = leadfield.shape
    # initialize weight matrix D with identity and some empirical shift (weights are usually quite smaller than 1)
    D = np.identity(numberOfVoxels)
    H = centeringMatrix(numberOfElectrodes)
    print('Optimizing eLORETA weight matrix W...')
    cnt = 0
    while True:
        old_D = deepcopy(D)
        print(f'\trep {cnt+1}')
        C = np.linalg.pinv( np.matmul( np.matmul(leadfield, np.linalg.inv(D)), leadfield.T ) + (tikhonov * H) )
        for v in range(numberOfVoxels):
            leadfield_v = np.expand_dims(leadfield[:, v], axis=1)
            D[v, v] = np.sqrt( np.matmul(np.matmul(leadfield_v.T, C), leadfield_v) )
        
        averagePercentChange = np.abs(1 - np.mean(np.divide(np.diagonal(D), np.diagonal(old_D))))
        print(f'averagePercentChange={100*averagePercentChange:.2f} %')
        if averagePercentChange < stopCrit:
            print('\t...converged...')
            break
        cnt += 1
    print('\t...done!')
    return D, C

def mne_elor(evoked, fwd, noiseCovariance, return_idx=50):
    inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noiseCovariance, fixed=True, verbose=False)
    stc = np.abs(mne.minimum_norm.apply_inverse(evoked, inv, method='eLORETA', verbose=False))
    y_eloreta = stc.data[:, return_idx]
    return y_eloreta

def mne_mne(evoked, fwd, noiseCovariance, return_idx=50):
    inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noiseCovariance, fixed=True, verbose=False)
    stc = np.abs(mne.minimum_norm.apply_inverse(evoked, inv, method='MNE', verbose=False))
    y_eloreta = stc.data[:, return_idx]
    return y_eloreta

def mne_lcmv(evoked, fwd, noiseCovariance, dataCovariance, return_idx=50):
 
    filters = mne.beamformer.make_lcmv(evoked.info, fwd, dataCovariance, \
                reg=0.05, noise_cov=noiseCovariance, weight_norm='nai', \
                verbose=0)
                
    stc = mne.beamformer.apply_lcmv(evoked, filters, verbose=0)
    y_lcmv = stc.data[:, return_idx]
    return y_lcmv

def mne_gamma_map(evoked, fwd, noiseCovariance, return_idx=50, alpha=1e-3):

    numberOfDipoles = int(fwd['sol']['data'].shape[1] / 3)
    numberOfTimepoints = evoked._data.shape[1]
    
    evoked.set_eeg_reference('average', projection=True)
    
    stc = gamma_map(
        evoked, fwd, noiseCovariance, alpha, xyz_same_gamma=False, loose=0, return_residual=False,
        return_as_dipoles=False, verbose=0)
        
    y_est = dipoles_to_stc(stc, numberOfDipoles, numberOfTimepoints)

    return y_est[:, return_idx]

def mne_mxne(evoked, fwd, noiseCovariance, return_idx=50, alpha=40., l1_ratio=0.03, depth=0.9, \
        weights_min=8):

    inverse_operator = mne.minimum_norm.make_inverse_operator(evoked.info, \
        fwd, noiseCovariance, loose=0, depth=depth, verbose=0)

    stc_elor = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2=1. / 9., \
        method='dSPM', verbose=0)
    
    numberOfDipoles = int(fwd['sol']['data'].shape[1] / 3)
    numberOfTimepoints = evoked._data.shape[1]

    # try:
    while True:
        try:
            stc = mne.inverse_sparse.tf_mixed_norm( \
                evoked, fwd, noiseCovariance, alpha=alpha, l1_ratio=l1_ratio, loose=0,
                depth=depth, maxit=200, tol=1e-6, weights=stc_elor, weights_min=weights_min,
                debias=True, wsize=16, tstep=4, window=0.05, return_as_dipoles=False,
                return_residual=False, verbose=0)
            y_est = dipoles_to_stc(stc, numberOfDipoles, numberOfTimepoints)
            y_est = np.nan_to_num(y_est[:, return_idx])
            if np.max(y_est) == 0:
                print(f'max source is zero, decreasing alpha to {int(round(alpha*0.9))}')
                alpha = int(round(alpha*0.9))
                continue
            else:
                break
        except:
            print(f'reducing weights to {weights_min/2}')
            weights_min /= 2
            continue
            
    return y_est
