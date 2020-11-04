import mne
import numpy as np
import random
import pickle as pkl
import os
import matplotlib.pyplot as plt
from util import *

def simulate_source(pos, settings):
    n_sources = settings['n_sources']
    diameters = settings['diam']
    amplitudes = settings['amplitude'] * 1e-9  # comes in nAm
    shape = settings['shape']
    durOfTrial = settings['durOfTrial']
    sampleFreq = settings['sampleFreq']

    if durOfTrial > 0:
        if durOfTrial < 0.5 :
            print(f'durOfTrial should be either 0 or at least 0.5 seconds!')
            return
        
        signalLen = int(sampleFreq*durOfTrial)
        pulselen = sampleFreq/10
        pulse = get_pulse(pulselen)
        signal = np.zeros((signalLen))
        start = int(np.floor((signalLen - pulselen) / 2))
        end = int(np.ceil((signalLen - pulselen) / 2))
        signal[start:-end] = pulse
        signal /= np.max(signal)
    else: 
        sampleFreq = 0
        signal = 1




    sourceMask = np.zeros((pos.shape[0]))
    # If n_sources is a range:
    if isinstance(n_sources, (tuple, list)):
        n_sources = random.randrange(*n_sources)
  
    if isinstance(diameters, (tuple, list)):
        diameters = [random.randrange(*diameters) for _ in range(n_sources)]
    else:
        diameters = [diameters for _ in range(n_sources)]

    if isinstance(amplitudes, (tuple, list)):
        amplitudes = [random.randrange(*amplitudes) for _ in range(n_sources)]
    else:
        amplitudes = [amplitudes for _ in range(n_sources)]
    
    src_centers = np.random.choice(np.arange(0, pos.shape[0]), 
                n_sources, replace=False)
    
    source = np.zeros((pos.shape[0]))
    for i, src_center in enumerate(src_centers):
        # Smoothing and amplitude assignment
        dists = np.sqrt(np.sum((pos - pos[src_center, :])**2, axis=1))
        d = np.where(dists<diameters[i]/2)
        if shape == 'gaussian':
            source[:] += gaussian(dists, 0, diameters[i]/2) * amplitudes[i]
        elif shape == 'flat':
            source[d] += amplitudes[i]
        else:
            raise(BaseException, "shape must be of type >string< and be either >gaussian< or >flat<.")
        sourceMask[d] = 1

    # if durOfTrial > 0:
    n = np.clip(int(sampleFreq * durOfTrial), a_min=1, a_max=None)
    sourceOverTime = repeat_newcol(source, n)
    source = np.squeeze(sourceOverTime * signal)

    


    simSettings = dict(scr_center_indices=src_centers, amplitudes=amplitudes, diameters=diameters, shape=shape, sourceMask=sourceMask)
    return source, simSettings

def get_actual_noise(path, numberOfSegments, durOfSegment, sampleFreq=100, filtfreqs=(0.1, 30)):
    ''' Loads all eeg (.vhdr) data sets, does a little preprocessing (filtering, resampling) and then extracts random segments of them. 
        Segments have the following properties:
        * re-referenced to common average
        * baseline corrected to first 10 datapoints
    '''
    segmentSize = int(durOfSegment*sampleFreq)
    
    fileList = np.array(os.listdir(path))
    vhdr_indices = np.where([i.endswith('.vhdr') for i in fileList])[0]
    fileList=fileList[vhdr_indices]

    dataSets = []
    for i, fn in enumerate(fileList):
        raw = mne.io.read_raw_brainvision(path + '/' + fn, preload=True, verbose=0)
        raw.filter(*filtfreqs, verbose=0)
        raw.resample(sampleFreq)
        dataSets.append( raw._data ) 

    numberOfChannels = dataSets[0].shape[0]
    segments = np.zeros((numberOfSegments, numberOfChannels, segmentSize))
    
    from util import rms

    for i in range(numberOfSegments):
        dataSet = dataSets[np.random.choice(np.arange(len(dataSets)))]
        segmentStartIndex = np.random.choice( np.arange(dataSet.shape[1] - segmentSize) )
        segment = dataSet[:, segmentStartIndex:segmentStartIndex+segmentSize]
        # Common Average reference
        segment = np.array( [seg  - np.mean(segment, axis=0) for seg in segment] )
        # Baseline Correction
        segment = np.array([seg - np.mean(seg[0:10]) for seg in segment])
        # RMS scaling so each trial is about equally 'loud'
        trial_rms = np.mean([rms(chan) for chan in segment])
        segments[i, :, :] = segment / trial_rms
    return segments

def add_real_noise(x, settings, noise_trials=None):
    ''' Takes an EEG signal 'x' and adds real noise.
    Parameters:
    -----------
    snr : float/int, signal to noise ratio (plain ratio, not in dB!)
    path : str, location of raw eeg data files to load
    numberOfTrials : int, number of trials to average (does not affect snr but rather the structure of the noise!)
    durOfTrial : float/int, duration in seconds
    sampleFreq : int, sampling frequency of the data
    filtfreqs : tuple/list, (lower frequency, upper frequency), the limits of the bandpass filter
    '''
    snr = settings["snr"]
    path = settings["path"]
    numberOfTrials = settings["numberOfTrials"]
    durOfTrial = settings["durOfTrial"]
    sampleFreq = settings["sampleFreq"]
    filtfreqs = settings["filtfreqs"]

    if noise_trials is None:
        noise_trials = get_noise_trials(settings)
        
    

    choice = np.random.choice(np.arange(len(noise_trials)), numberOfTrials)
    noise_trials = np.array(noise_trials[choice])
    # Now scale each noise_trial such that it has the provided SNR when added to the signal
    trials = np.zeros((noise_trials.shape))
    for i, noise_trial in enumerate(noise_trials):
        trial_sd = np.max([np.std(tr) for tr in noise_trial])
        peak_x = np.max( [np.max(np.abs(chan)) for chan in x] )
        noise_scaler = peak_x / (snr * trial_sd)
        trials[i, :, :] = noise_trial * noise_scaler + x
    return trials
    # mean_noise = np.mean(noise_trials, axis=0)
    # # rms_noise = np.mean( [np.std(chan) for chan in mean_noise] )
    # # rms_x = np.max( [np.max(np.abs(chan)) for chan in x] )
    # # noise_scaler = rms_x / (snr * rms_noise)
    # x_noise = x + mean_noise # * noise_scaler

    # '''
    # peak_x / (std_noise * scaler) = snr
    # peak_x = snr/ (std_noise * scaler)
    # std_noise * scaler  / peak_x = 1/snr
    # std_noise * scaler = peak_x / snr
    # scaler = peak_x / snr * std_noise

    # '''
    # print(f'peak x = {rms_x}')
    # print(f'mean noise_stdpeak = {rms_noise}')
    # print(f'scale noise by {noise_scaler} should yield rms_noise of {rms_noise*noise_scaler}')

    # plt.figure()
    # plt.plot(mean_noise.T)
    # plt.title('noise')

    # plt.figure()
    # plt.plot(x.T)
    # plt.title('x')

    # plt.figure()
    # plt.plot(x_noise.T)
    # plt.title('both')

    # return x_noise

