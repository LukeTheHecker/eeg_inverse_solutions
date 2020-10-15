from util import create_source_model, get_raw_data
import os
import mne
import pickle as pkl
import numpy as np

fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')
src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

pth_res = 'assets/'
pth = 'assets/CL_GC_01.vhdr'
raw, epochs, evoked = get_raw_data(pth)
raw.set_eeg_reference(projection=True)  # needed for inverse modeling

res = 'low'
# Source Model
src = create_source_model(subject, subjects_dir, pth_res, res=res)
# Forward Model
fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=-1) 
mne.write_forward_solution(pth_res+'\\{}-fwd.fif'.format(subject),fwd, 
                           overwrite=True)
# Fixed Orientations
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                         use_cps=True)

###############################################################################
# Create Container for Source Estimate which is needed to plot data later on  #
noise_cov = mne.compute_covariance(epochs, tmax=0., method=['empirical'], 
                                   rank=None, verbose=True)

inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov,
                                     loose=0.2, depth=0.8)

stc, residual = mne.minimum_norm.apply_inverse(evoked, inv, 0.05,
                              method="dSPM", pick_ori="normal",
                              return_residual=True, verbose=True)

stc.save(pth_res+"\\ResSourceEstimate".format(), ftype='stc')

        


# leadfield = fwd['sol']['data']
leadfield = fwd_fixed['sol']['data']

# save leadfield
fn = "{}\\leadfield.pkl".format(pth_res)
with open(fn, 'wb') as f:
    pkl.dump([leadfield], f)
# Load source space file
source = mne.read_source_spaces(pth_res+"/lowRes-src.fif")
pos_left = mne.vertex_to_mni(source[0]['vertno'], hemis=0, subject='fsaverage')
pos_right = mne.vertex_to_mni(source[0]['vertno'], hemis=1, subject='fsaverage')
pos = np.concatenate([pos_left, pos_right], axis=0)

# save pos
fn = "{}\\pos.pkl".format(pth_res)
with open(fn, 'wb') as f:
    pkl.dump([pos], f)
