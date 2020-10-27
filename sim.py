import enum
import numpy as np
import random
from util import *

def simulate_source(leadfield, pos, settings):
    # Simulate 'niter' different forward solutions
    # n_sources = (1, 16)
    # n_sources = (1, 5)
    # prob_sources = tuple(gaussian(np.arange(n_sources[0],
    #                                         np.max(n_sources)+1), 3, 4)) 
    # prob_sources /= np.sum(prob_sources)
    

    x = np.zeros((leadfield.shape[0]))
    y = np.zeros((leadfield.shape[1]))
    
    sim = generate_source(leadfield, pos, settings)
    x = np.sum(sim*leadfield, axis=1)
    y = sim
    
    return x, y


def generate_source(leadfield, pos, settings):
    # Generate a source configuration based on settings
    sim = np.zeros((leadfield.shape[1]))
    
    n_sources = settings['n_sources']
    diameter = settings["diam"]
    amplitude = settings["amplitude"]
    shape = settings["shape"]

    # If n_sources is a range:
    if isinstance(n_sources, (tuple, list)):
        n_sources = random.randrange(*n_sources)
  
    if isinstance(diameter, (tuple, list)):
        diameter = [random.randrange(*diameter) for _ in range(n_sources)]
    else:
        diameter = [diameter for _ in range(n_sources)]

    if isinstance(amplitude, (tuple, list)):
        amplitude = [random.randrange(*amplitude) for _ in range(n_sources)]
    else:
        amplitude = [amplitude for _ in range(n_sources)]
    
    src_centers = np.random.choice(np.arange(0, pos.shape[0]), 
                n_sources, replace=False)
    
    source = np.zeros((pos.shape[0]))
    for i, src_center in enumerate(src_centers):
        # Smoothing and amplitude assignment
        dists = np.sqrt(np.sum((pos - pos[src_center, :])**2, axis=1))
        d = np.where(dists<diameter[i]/2)
        if shape == 'gaussian':
            source[:] += gaussian(dists, 0, diameter[i]/2) * amplitude[i]
        elif shape == 'same':
            source[d] += amplitude[i]
        else:
            raise(ValueError, "shape must be of type >string< and be either >gaussian< or >same<.")
    return source