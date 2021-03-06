from mne.cov import regularize
import numpy as np
from skimage.restoration import inpaint
import mne
from scipy.stats import pearsonr
import pickle as pkl
import os
import matplotlib.pyplot as plt
from scipy.io import savemat

import sim

def get_events(onoff, annotations, srate, round_to, triglens, triglabels, mode='checkers'):
    ''' Extract Events from annotations using trigger lengths '''

    a = annotations[0]
    e = np.zeros((a.shape[0], a.shape[1]+1))
    e[:,0:3] = a
    sample_factor = 1000 / srate
    if mode=='checkers':
        # how it used to be:
        for i in range(e.shape[0]):
            if e[i, 2] == onoff[0]:
                j = i + 1
                while j < e.shape[0]:
                    if e[j, 2] == onoff[1]:
                        if round_to == 1:
                            e[i, 3] = (e[j , 0] - e[i,0]) * sample_factor
                        else:
                            e[i, 3] = (np.round((e[j,0] - e[i,0])/round_to)*round_to) * sample_factor

                        if e[i, 3] in triglens:
                            idx = triglens.index(e[i, 3])
                            e[i,2] = triglabels[idx]
                        break
                    j+=1

    elif mode=='lattices':
        # Iterate through all events "e"
        for i in range(e.shape[0]-1):
            # If On-trigger found
            if e[i, 2] == onoff[0] and e[i+1, 2] == 1:
                j = i + 1
                # serch for Off-trigger
                while j < e.shape[0]-1:
                    # If Off-trigger found
                    if e[j, 2] == onoff[1] and e[j+1, 2] == 2:
                        # Store the trigger length in ms at On-trigger 
                        if round_to == 1:
                            e[i, 3] = (e[j , 0] - e[i,0]) * sample_factor
                        else:
                            e[i, 3] = (np.round((e[j,0] - e[i,0])/round_to)*round_to) * sample_factor
                        # If trigger length is of interest
                        if e[i, 3] in triglens:
                            # get index in list of trigger lengths
                            idx = triglens.index(e[i, 3])
                            k = i - 1
                            # go back to stimulus onset in order to place the stimulus code correctly
                            while True:
                                if e[k, 2] == onoff[0] and e[k+1, 2] == 1:
                                    e[k, 2] = triglabels[idx]
                                    break
                                k -= 1
                            else:
                                print(f'{e[i, 3]} not in triglens')
                        break
                    j += 1
    return np.ndarray.astype(e[:,0:3],'int32')

def get_raw_data(pth):
    ''' Load some  raw data based on brain vision header (.vhdr) file. It's only
    done to have 1) channel positions and 2) the data structure.'''
    
    # Load
    raw = mne.io.read_raw_brainvision(pth, preload=True)

    # Read montage
    # montage = mne.channels.read_montage(kind='standard_1020',
    #                                     ch_names=raw.ch_names, transform=True)
    raw.set_montage('standard_1020')

    # Remove VEOG
    try:
        mne.io.Raw.drop_channels(raw, "VEOG")
    except:
        print('no VEOG anyway')
    # Common Average Reference
    raw.set_eeg_reference('average', projection=True)

    # filter
    raw.filter(0.1, 30, n_jobs=4)

    # define artifacts
    reject = dict(eeg=100.0*1e-6)

    # Read Events
    annotations = mne.events_from_annotations(raw)

    # Calculate event identity
    onoff = [3, 3]
    srate = 500
    round_to = 1
    triglabels = [10, 10, 10, 10, 10]
    triglens = [2, 3, 4, 5, 6]
    mode = 'checkers'
    event = get_events(onoff, annotations, srate, round_to, triglens, triglabels, mode=mode)

    # Pick only stimulus onsets:
    event_id = {'Stimulus': event[3, 2]}

    # epoch
    epochs = mne.Epochs(raw, event, event_id=event_id, tmin=-0.06, tmax=1,
                        proj=True, baseline=(-0.06, 0.04),
                        preload=True, reject=reject)

    epochs.set_montage('standard_1020')
    # Calc ERP
    evoked = epochs['Stimulus'].average()
    evoked.set_montage('standard_1020')
    
    return raw, epochs, evoked

def create_source_model(subject, subjects_dir, pth, res='low'):

    if res == 'low':
        spacing = 'ico4'
    elif res == 'high':
        spacing = 'ico5'
    else:
        return

    src = mne.setup_source_space(subject, spacing=spacing, surface='white',
                                      subjects_dir=subjects_dir, add_dist=False,
                                      n_jobs=-1, verbose=None)
    # src.plot()

    src.save('{}\\{}Res-src.fif'.format(pth,res), overwrite=True)
    return src

def vec_to_sevelev_newlayout(x):
    ''' convert a vector consisting of 32 electrodes to a 7x11 matrix using 
    inpainting '''
    x = np.squeeze(x)
    w = 11
    h = 7
    elcpos = np.empty((h, w))
    elcpos[:] = np.nan
    elcpos[0, 4] = x[0]
    elcpos[1, 3] = x[1]
    elcpos[1, 2] = x[2]
    elcpos[2, 0] = x[3]
    elcpos[2, 2] = x[4]
    elcpos[2, 4] = x[5]
    elcpos[3, 3] = x[6]
    elcpos[3, 1] = x[7]
    elcpos[4, 0] = x[8]
    elcpos[4, 2] = x[9]
    elcpos[4, 4] = x[10]
    
    elcpos[5, 5] = x[11]
    elcpos[5, 3] = x[12]
    elcpos[5, 2] = x[13]
    elcpos[6, 4] = x[14]
    elcpos[6, 5] = x[15]
    elcpos[6, 6] = x[16]

    elcpos[5, 7] = x[17]
    elcpos[5, 8] = x[18]
    elcpos[4, 10] = x[19]
    elcpos[4, 8] = x[20]
    elcpos[4, 6] = x[21]
    elcpos[3, 5] = x[22]
    elcpos[3, 7] = x[23]
    elcpos[3, 9] = x[24]
    
    elcpos[2, 10] = x[25] # FT10
    elcpos[2, 8] = x[26]
    elcpos[2, 6] = x[27]
    elcpos[1, 7] = x[28]
    elcpos[1, 8] = x[29]
    elcpos[0, 6] = x[30]
    # elcpos[1, 5] = 5 Fz was reference
    # elcpos[6, 2] = 28 PO9 deleted
    # elcpos[6, 8] = 32 PO10 deleted

    mask = np.zeros((elcpos.shape))
    mask[np.isnan(elcpos)] = 1
        
    
    return inpaint.inpaint_biharmonic(elcpos, mask, multichannel=False)

def sevelev_to_vec_newlayout(x):
    x = np.squeeze(x)
    if len(x.shape) == 2:
        x_out = np.zeros((1, 31))
        x = np.expand_dims(x, axis=0)
    else:
        x_out = np.zeros((x.shape[0], 31))

    for i in range(x.shape[0]):
        tmp = np.squeeze(x[i, :])
        #print(tmp.shape)
        x_out[i,:] = tmp[[0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 0], [4, 3, 2, 0, 2, 4, 3, 1, 0, 2, 4, 5, 3, 2, 4, 5, 6, 7, 8, 10, 8, 6, 5, 7, 9, 10, 8, 6, 7, 8, 6]]
    return 

def project(leadfield, y):
    return np.sum(y*leadfield, axis=1)

def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def get_adjacency_matrix(pos, tris):
    adjMat = np.identity(pos.shape[0])

    for i, conn in enumerate(tris):
        adjMat[conn[0], conn[1]] = 1
        adjMat[conn[1], conn[0]] = 1

        adjMat[conn[0], conn[2]] = 1
        adjMat[conn[2], conn[0]] = 1
        
        adjMat[conn[1], conn[2]] = 1
        adjMat[conn[2], conn[1]] = 1

    return adjMat

def get_laplacian_adjacency_matrix(adjMat):
    numberOfDipoles = adjMat.shape[0]
    adjMat2 = np.zeros((numberOfDipoles, numberOfDipoles))
    for i in range(numberOfDipoles):
        adjMat2[i, i] = np.sum(adjMat[i, :])


    return adjMat - adjMat2

def get_W_sigma(adjMatLap, sigma, upperBound=8):
    # W_sigma = np.exp(sigma * adjMatLap)
    i = 0
    W_sigma = (sigma**i / np.math.factorial(i)) * adjMatLap**i
    for i in range(1, upperBound):
        W_sigma += (sigma**i / np.math.factorial(i)) * adjMatLap**i
    return W_sigma

    
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def eval_estimate(y_true, y_pred):
    error = np.mean(np.square(y_true-y_pred))
    error_normed = np.mean(np.square(y_true/np.max(y_true)-y_pred/np.max(y_pred)))
    corr = pearsonr(y_true, y_pred)[0]
    print(f'error={error:.3f}({error_normed:.3f}), r={corr:.2f}')
    return error, corr

def centeringMatrix(n):
    ''' Centering matrix, which when multiplied with a vector subtract the mean of the vector.'''
    C = np.identity(n) - (1/n) * np.ones((n, n))
    return C

def find_indices_close_to_source(simSettings, pos):
    ''' Finds the dipole indices that are closest to the active sources. 
    Parameters:
    -----------
    simSettings : dict, retrieved from the simulate_source function
    pos : numpy.ndarray, list of all dipole positions in XYZ coordinates

    Return:
    -------
    ordered_indices : numpy.ndarray, ordered list of dipoles that are near active sources in ascending order with respect to their distance to         the next source.'''
    numberOfDipoles = pos.shape[0]
    pos_indices = np.arange(numberOfDipoles)
    src_center_indices = simSettings['scr_center_indices']
    numberOfSources = len(src_center_indices)
    amplitudes = simSettings['amplitudes']
    diamters = simSettings['diameters']
    sourceMask = simSettings['sourceMask']
    sourceIndices = np.array([i[0] for i in np.argwhere(sourceMask==1)])
    numberOfNans = 0
    min_distance_to_source = np.zeros((numberOfDipoles))
    for i in range(numberOfDipoles):
        if sourceMask[i] == 1:
            min_distance_to_source[i] = np.nan
            numberOfNans +=1
        elif sourceMask[i] == 0:
            distances = np.sqrt(np.sum((pos[sourceIndices, :] - pos[i, :])**2, axis=1))
            min_distance_to_source[i] = np.min(distances)
        else:
            print('source mask has invalid entries')
    
    # min_distance_to_source = min_distance_to_source[~np.isnan(min_distance_to_source)]
    ordered_indices = np.argsort(min_distance_to_source)
    # ordered_indices[np.where(~np.isnan(min_distance_to_source[ordered_indices]]
    return ordered_indices[:-numberOfNans]

def sens_spec(y, y_hat):
    nTruePositives = len(np.where(np.logical_and(y == 1, y_hat == 1))[0])
    nTrueNegatives = len(np.where(np.logical_and(y == 0, y_hat == 0))[0])
    nFalsePositives = len(np.where(np.logical_and(y == 0, y_hat == 1))[0])
    nFalseNegatives = len(np.where(np.logical_and(y == 1, y_hat == 0))[0])
    
    # print(f'nTruePositives={nTruePositives}')
    try:
        sensitivity = nTruePositives / (nTruePositives + nFalseNegatives)
    except ZeroDivisionError:
        sensitivity = 0

    try:
        specificity = nTrueNegatives / (nTrueNegatives + nFalsePositives)
    except ZeroDivisionError:
        specificity = 0

    return sensitivity, specificity

def add_noise(x, snr):
    rms_x = rms(x)
    rms_noise = rms_x / snr
    noise = np.random.randn(len(x)) * rms_noise
    return x + noise

def get_pulse(x):
    ''' Returns a pulse of length x'''
    sr = 1
    freq = (1/x) / 2
    time = np.arange(x)

    signal = np.sin(2*np.pi*freq*time)
    return signal

def repeat_newcol(x, n):
    ''' Repeat a list/numpy.ndarray x in n columns.'''
    out = np.zeros((len(x), n))
    for i in range(n):
        out[:,  i] = x
    return np.squeeze(out)


def get_chan_pos_list(info, montage_type='standard_1020'):
    ''' Returns a list of channel names + respective position in 
    the same order as they occur in the info structure'''
    if type(montage_type) == str:
        montage = mne.channels.make_standard_montage(montage_type)
    else:
        montage = montage_type
    n_chan = len(montage.ch_names)
    chan_assign = {}
    for i in range(3, n_chan+3):
        chan_assign[montage.dig[i]['ident']] = montage.ch_names[i-3]
    # print(chan_assign)
    ###
    chan_list = [list() for i in range(len(info.ch_names))]

    for i in range(3, len(info['dig'])):
        current_identitiy = info['dig'][i]['ident']
        # print(f'Current Identitiy: {current_identitiy}')
        chan_list[i-3].append(chan_assign[current_identitiy])
        # print(f'Channel name: {chan_assign[current_identitiy]}')
        chan_list[i-3].append(info['dig'][i]['r'])
        pos = info['dig'][i]['r']
        # print(f'Channel position: {pos}\n')

    chan_list_ordererd = chan_list.copy()
    # Loop through channel names in raw data
    for j, chan in enumerate(info['ch_names']):
        # Find the correct row in chan_list
        for row in chan_list:
            if row[0] == chan:
                # And copy that row to the correctly ordered list
                chan_list_ordererd[j] = row
    # print(chan_list_ordererd)
    return np.array(chan_list_ordererd)

def create_noise_segments(path, numberOfTrials, durOfTrial, sampleFreq, filtfreqs):
    # from sim import get_actual_noise
    # path_store = os.path.dirname(path)
    fn = f'{path}/noise_dur{durOfTrial}_sf{sampleFreq}_filt{filtfreqs}.pkl'
    trials = sim.get_actual_noise(path, numberOfTrials, durOfTrial, sampleFreq=sampleFreq, 
            filtfreqs=filtfreqs)
    print(f'saving trials..')

    plt.figure()
    plt.plot(np.mean(trials, axis=0).T)
    plt.title('noise trials')


    with open(fn, 'wb') as f:
        pkl.dump(trials, f)

def get_noise_trials(settings):
    path = settings["path"]
    durOfTrial = settings["durOfTrial"]
    sampleFreq = settings["sampleFreq"]
    filtfreqs = settings["filtfreqs"]

    fn = f'{path}/noise_dur{durOfTrial}_sf{sampleFreq}_filt{filtfreqs}.pkl'
    if not os.path.isfile(fn):
            print('No noise trials for these settings found! Creating some..')
            noise_trials = create_noise_segments(path, 200, durOfTrial, sampleFreq, filtfreqs)

    with open(fn, 'rb') as f:
        noise_trials = pkl.load(f)
    return noise_trials

def data_to_mne(eegData, settings, info, scalingFactor=1e6):
    info = mne.create_info(info.ch_names, settings['sampleFreq'], ch_types='eeg')
    epochs = [mne.EpochsArray(data*scalingFactor, info, tmin=0.0, verbose=0) for data in eegData]

    for epoch in epochs:
        epoch.set_eeg_reference(ref_channels='average', verbose=0)
        epoch.set_montage('standard_1020')


    
    evokeds = [epoch.average() for epoch in epochs]

    for evoked in evokeds:
        evoked.set_eeg_reference(ref_channels='average', verbose=0, projection=True)
        evoked.set_montage('standard_1020')
    return epochs, evokeds

def get_covariances(epochs, noise_baseline, regularize=True):
    if regularize:
        noiseCovariances = [mne.cov.regularize(mne.compute_covariance(epoch, tmin=noise_baseline[0], tmax=noise_baseline[1], method='empirical', verbose=0, n_jobs=-1), \
            epoch.info, rank=None, verbose=0) for epoch in epochs]
        dataCovariances = [mne.cov.regularize(mne.compute_covariance(epoch, method='empirical', verbose=0, n_jobs=-1), \
            epoch.info, rank=None, verbose=0) for epoch in epochs]    
    else:
        noiseCovariances = [mne.compute_covariance(epoch, tmin=noise_baseline[0], tmax=noise_baseline[1], method='empirical', verbose=0, n_jobs=-1) for epoch in epochs]
        dataCovariances = [mne.compute_covariance(epoch, method='empirical', verbose=0, n_jobs=-1) for epoch in epochs]

    return noiseCovariances, dataCovariances

def brainstorm_to_mne_space(y_bst, neighborMatrix=None):
    if neighborMatrix is None:
        fn = 'C:/Users/Lukas/Documents/projects/eeg_inverse_solutions/assets/neighborMatrix.pkl'

        with open(fn, 'rb') as f:
            neighborMatrix = pkl.load(f)

    numberOfDipoles = len(neighborMatrix)
    y_est = np.zeros((numberOfDipoles))
    for i in range(numberOfDipoles):
        y_est[i] = np.mean(y_bst[neighborMatrix[i]])
    return y_est

def epochs_covs_to_fif(epochs, dataCovariances, noiseCovariances, pth_dest):
    
    for i, epoch in enumerate(epochs):
        if np.mod(i, 100) == 0:
            print(f'saved {i} epochs')
        savemat(pth_dest + 'sim_' + str(i) + '.mat', dict(data=epoch.average()._data))
        # epoch.average().save(pth_dest + 'sim_' + str(i) + '-epo.fif')
        dataCovariances[i].save(pth_dest + 'dataCov_' + str(i) + '-cov.fif')
        noiseCovariances[i].save(pth_dest + 'noiseCov_' + str(i) + '-cov.fif')

    print(f'\tdone.')

def dipoles_to_stc(stc, numberOfDipoles, numberOfTimepoints):
    dipolesPerHemisphere = (len(stc.vertices[0]), len(stc.vertices[1]))
    y_est = np.zeros((numberOfDipoles, numberOfTimepoints))
    if dipolesPerHemisphere[0] != 0:
        y_est[stc.vertices[0], :] = stc.data[0:dipolesPerHemisphere[0], :]
    if dipolesPerHemisphere[1] != 0:
        y_est[stc.vertices[1], :] = stc.data[-dipolesPerHemisphere[1]:, :]
    return y_est