import numpy as np
from viz import quickplot
import matplotlib.pyplot as plt
import mne
import pickle as pkl
from sim import *
from inverse_solutions import *
from source_covs import *
from util import *
from eval import *
pth_res = 'assets'

# Load data
with open(pth_res + '/leadfield.pkl', 'rb') as file:
    leadfield = pkl.load(file)[0]
with open(pth_res + '/pos.pkl', 'rb') as file:  
    pos = pkl.load(file)[0]
with open(pth_res + '/info.pkl', 'rb') as file:  
    info = pkl.load(file)
chanPos = get_chan_pos_list(info, montage_type='standard_1020')

# Simulate
settings = {"n_sources": 1,  # number of sources
            "diam": (20, 60),  # diameter of source patches in mm
            "amplitude": 9.5,  # amplitude of source patches
            "shape": 'gaussian',
            "signalDur": 1,
            "SNR": 1,
            "sfreq": 100
            }
numberOfSimulations = 100

y, simSettings = [simulate_source(pos, settings) for _ in range(numberOfSimulations)]
x = add_real_noise(np.matmul(leadfield, y), settings['SNR'], numberOfTrials=1, durOfTrial=settings['signalDur'], sampleFreq=settings['sfreq'], filtfreqs=(1, 30))

# plt.figure()
# plt.plot(x.T)
# print('')