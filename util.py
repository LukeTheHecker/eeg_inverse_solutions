import numpy as np
from skimage.restoration import inpaint
import mne

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
                                      n_jobs=4, verbose=None)
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