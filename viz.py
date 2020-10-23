import mne
import numpy as np
import matplotlib.pyplot as plt

def quickplot(data, pth_res, del_below=0.1, backend='matplotlib', title='generic title', figure=None):
    ''' quickly plot a source '''
    data = np.squeeze(data)
    mask_below_thr = data < (np.max(data) * del_below)
    data[mask_below_thr] = 0
    # Read some dummy object to assign the voxel values to
    if len(data) == 20484:
        try:
            a = mne.read_source_estimate(pth_res + "\\sourcetemplate-lh.stc")
        except:
            a = mne.read_source_estimate(pth_res + "/sourcetemplate-lh.stc")


    else:
        try:
            a = mne.read_source_estimate(pth_res + "\\ResSourceEstimate-lh.stc")
        except:
            a = mne.read_source_estimate(pth_res + "/ResSourceEstimate-lh.stc")

    # assign precomputed voxel values
    for i in range(a.data.shape[1]):
        a.data[:, i] = data

    # its a template average
    a.subject = "fsaverage"
    # Use stc-object plot function
    clim = {'kind': 'percent',
            'lims': (20, 50, 100)
            }
    views = ['lat', 'med', 'ros', 'cau', 'dor']
    fleft = a.plot(hemi='lh', initial_time=0.5, surface="white", backend=backend, title=title+'_lh' , clim=clim, transparent=True, figure=figure)
    fright = a.plot(hemi='rh', initial_time=0.5, surface="white", backend=backend, title=title+'_rh' , clim=clim, transparent=True, figure=figure)
    return fleft, fright