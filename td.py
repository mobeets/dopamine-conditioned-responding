import numpy as np
from numpy.random import default_rng

def make_trial(iti, isi, iti_min, add_bias=True):
    """
    columns: ISIs, ITI_MINs, ITI
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

def create_experiment_multiple_cues(ntrials, iti_min, iti_p, isis_per_cue, r_probs_per_cue=(1.0,), cue_probs=None, add_bias=True, seed=None, reward_size=1, visibility_per_cue=(True,), stack=False):
    Z = []
    R = []
    rng = default_rng(seed)
    
    # sample ITIs
    itis = iti_min + rng.geometric(p=iti_p, size=ntrials)

    # sample cue order pseudorandomly
    ncues = len(isis_per_cue)
    if cue_probs is None:
        cue_probs = np.ones(ncues) / ncues
    n_trials_per_cue = (ntrials * np.array(cue_probs)).astype(int)
    cues = np.hstack([[cue]*n for cue,n in enumerate(n_trials_per_cue)]).astype(int)
    inds = rng.permutation(len(cues))
    cues = cues[inds]

    # get reward per trial
    r_probs = np.hstack([r_probs_per_cue[cue] for cue in cues])
    rewards = reward_size*(rng.random(size=ntrials) <= r_probs).astype(int)

    for cue, itic, r in zip(cues, itis, rewards):
        isi = isis_per_cue[cue]
        Zc, Rc = make_trial(iti=itic, isi=isi, iti_min=iti_min, add_bias=add_bias)
        
        Z_cue = Zc[:,:isi]
        if not visibility_per_cue[cue]:
            Z_cue = 0*Z_cue
        Z_iti = Zc[:,isi:]

        # stack up reps for all cues
        Z_cues = []
        for i, cisi in enumerate(isis_per_cue):
            if i != cue:
                Z_cur = np.zeros((len(Z_cue), cisi))
            else:
                Z_cur = Z_cue
            Z_cues.append(Z_cur)
        Z_cues = np.hstack(Z_cues)
        Zc = np.hstack([Z_cues, Z_iti])

        Rc[Rc > 0] = r
        Z.append(Zc)
        R.append(Rc)
    if stack:
        Z = np.vstack(Z)
        R = np.hstack(R)
    return Z, R
