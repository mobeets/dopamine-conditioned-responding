import numpy as np
from numpy.random import default_rng

def TD(Z, R, bs=None, ms=None, gamma=0.9, alpha=0.1, w_memory=1, save_index=0, isi_index=0, eps=0.0, eps_z=0.0, seed=123):
    """
    performs TD learning on representations Z and rewards R,
        with optional slope and bias perturbations on the RPE calculation
        to simulate DA perturbations
    returns a dictionary containing RPEs, values, and weights at each time step

    Z (array: (T,K)) is representation at each time step
    R (array: (T,)) is reward at each time step
    bs (array: (T,)) controls offset of DA transformation (default: 0)
    ms (array: (T,)) controls slope of DA transformation (default: 1)
    gamma (float) is discount factor (default: 0.9)
    alpha (float) is learning rate (default: 0.1)
    w_memory (float) is degree to which weights are preserved across time steps (default: 1, aka no forgetting)
    save_index (int) will save data whenever Z[:,save_index] == 1 (default: 0)
    isi_index (int) is the index of the ISI feature in the state representation, used to count trial indices (default: 0)
    eps (float) is std dev of noise added to RPE at each time step (default: 0)
    eps_z (float) is std dev of noise added to Z at each time step (default: 0)
    seed (int) is random seed for noise generation
    """
    Znext_noNoise = Z[1:].copy()
    if eps_z > 0.0:
        rng = default_rng(seed+1) if seed is not None else default_rng()
        Z = Z + eps_z * rng.standard_normal(size=Z.shape)
    Zcur = Z[:-1]
    Znext = Z[1:]
    Rc = R[:-1]
    bc = bs[:-1] if bs is not None else np.zeros_like(Rc)
    mc = ms[:-1] if ms is not None else np.ones_like(Rc)

    w = np.zeros(Z.shape[1])
    ws = [w.copy()]
    rpe = np.nan
    rpes = []
    values = []
    weights = []
    rng = default_rng(seed) if seed is not None else default_rng()
    nse = eps * rng.standard_normal(size=len(Z))
    trial_index = 0

    data = {'rpes': [], 'values': [], 'weights': [], 'trial_index': []}
    for t, (z, z_next, z_nn, r, b, m) in enumerate(zip(Zcur, Znext, Znext_noNoise, Rc, bc, mc)):
        # calculate transformed RPE
        rpe = r + gamma*z_next.dot(w) - z.dot(w) + nse[t]
        rpe_trans = m*(rpe-b)

        # if representation matches save_index, save rpe, value, and weights
        if z_nn[isi_index] == 1: # CS onset
            trial_index += 1
        if z_nn[save_index] == 1: # CS onset
            data['rpes'].append(rpe_trans) # rpe from current trial
            data['values'].append(z_next.dot(w)) # value on current trial
            data['weights'].append(w.copy())
            data['trial_index'].append(trial_index)
        # learning update
        w = w_memory*w + (alpha * rpe_trans * z)
        
    # convert all lists to arrays
    for key, values in data.items():
        data[key] = np.array(values)
    return data

def make_trial(iti, isi, iti_min, add_bias=True):
    """
    Creates a single trace conditioning trial with given parameters.
    iti (int): inter-trial interval (ITI) in time steps
    isi (int): inter-stimulus interval (ISI) in time steps
    iti_min (int): minimum ITI in time steps
    add_bias (bool): whether to add a bias term to the state representation (default: True)
    Returns:
        Z: 2D array of shape (T, K) representing the state at each time step, where T is the total number of time steps in the trial and K is the number of features in the state representation
        R: 1D array of shape (T,) representing the reward at each time step
    """
    K = iti_min + 1 + isi
    # true ITI
    Z_iti = np.zeros((iti - iti_min, K))
    if iti > iti_min:
        Z_iti[:,-1] = 1.
    # early ITI
    Z_pre = np.hstack([np.zeros((iti_min, isi)), np.eye(iti_min), np.zeros((iti_min, 1))])
    Z_isi = np.hstack([np.eye(isi), np.zeros((isi,K-isi))])
    
    Z = np.vstack([Z_pre, Z_iti, Z_isi])
    if add_bias:
        Z = np.hstack([Z, np.ones((len(Z),1))]) # add bias term
    R = np.zeros(len(Z))
    R[-1] = 1.
    return Z, R

def create_experiment(ntrials, iti_min, iti_p, isi, r_prob=1.0, add_bias=True, seed=None, reward_size=1, cue_is_visible=True, stack=False):
    """
    Creates a trace conditioning experiment with given parameters.
    ntrials (int): number of trials in the experiment
    iti_min (int): minimum inter-trial interval (ITI) in time steps
    iti_p (float): probability parameter for geometric distribution of ITI lengths
    isi (int): inter-stimulus interval (ISI) in time steps
    r_prob (float): probability of reward delivery on each trial (default: 1.0)
    add_bias (bool): whether to add a bias term to the state representation (default: True)
    seed (int): random seed for reproducibility (default: None)
    reward_size (float): magnitude of reward delivered on rewarded trials (default: 1)
    cue_is_visible (bool): whether the cue is visible during the ITI (default: True)
    stack (bool): whether to stack trials into a single array (default: False)
    """
    Z = []
    R = []
    rng = default_rng(seed)
    itis = iti_min + rng.geometric(p=iti_p, size=ntrials)
    rewards = reward_size*(rng.random(size=ntrials) <= r_prob).astype(int)
    for itic, r in zip(itis, rewards):
        Zc, Rc = make_trial(iti=itic, isi=isi, iti_min=iti_min, add_bias=add_bias)
        Rc[Rc > 0] = r
        if not cue_is_visible:
            Zc[:,:isi] = 0*Zc[:,:isi]
        Z.append(Zc)
        R.append(Rc)
    if stack:
        Z = np.vstack(Z)
        R = np.hstack(R)
    return Z, R
