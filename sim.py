import numpy as np


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
    #print(n_sources)
    src_center = np.random.choice(np.arange(0, pos.shape[0]), 1,
                                   replace=False)
    src_diam = settings["diam"]
    src_amp = settings["amplitude"]
    # Smoothing and amplitude assignment
    
    #breakpoint()
    
    dists = np.sqrt(np.sum((pos - pos[src_center, :])**2, axis=1))
    d = np.where(dists<src_diam/2)
    sim[d] = src_amp
    return sim