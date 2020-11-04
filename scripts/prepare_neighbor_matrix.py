from scipy.io import savemat, loadmat
import mne
import pickle as pkl
import os
import numpy as np
pth_res = 'assets/'

# Load MNE position file in head coordinates: https://mne.tools/dev/auto_tutorials/source-modeling/plot_source_alignment.html
with open(pth_res + '/pos.pkl', 'rb') as file:  
    pos = pkl.load(file)[0]

# Transform from head space to MNI space
subject = 'fsaverage'
fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
trans = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')
trans = mne.read_trans(trans)
pos_mne_mni = mne.head_to_mni(pos, subject=subject, mri_head_t=trans, verbose=0) / 1000


# Load Brainstorm Vertice Positions in MNI space:
pth_bst_pos = 'C:/Users/Lukas/Documents/projects/eeg_inverse_solutions/matlab/results/model/mni_positions.mat'
pos_bst_mni = loadmat(pth_bst_pos)['mni_positions']



def k_neighbor_connectivity(pos_bst_mni, pos_mne_mni, k=3):
    ''' 
    Parameters:
    -----------
    pos_bst_mni : str, path of the brainstorm vertex position file. Positions are required to 
        be in MNI space
    pos_mne_mni : str, path of the MNE headmodel vertex position file. Positions need to be in 
        MNI space.
    k : int, number of neighbors to include into the neighbor matrix
    Return:
    -------
    neihborMatrix : list of lists, each of which contains the k nearest neighbors for to the brainstorm position in the mne positions. 

    '''
    neighborMatrix = [list() for _ in range(pos_mne_mni.shape[0])]
    for i in range(len(neighborMatrix)):
        referencePos = pos_mne_mni[i, :]
        distances = np.sqrt(np.sum(np.square(pos_bst_mni - referencePos), axis=1))
        indices = np.argsort(distances)[:k]
        neighborMatrix[i] = indices
    return neighborMatrix

neighborMatrix = k_neighbor_connectivity(pos_bst_mni, pos_mne_mni, k=3)
print(f'neighbor Matrix of length {len(neighborMatrix)} is saved')
fn = pth_res + 'neighborMatrix.pkl'
with open(fn, 'wb') as f:
    pkl.dump(neighborMatrix, f)