#%% imports

import numpy as np
from sklearn.model_selection import KFold

#%% helper functions

def zscore_if_possible(y, mu=None, sd=None):
    changed_shape = False
    if len(y.shape) == 1:
        changed_shape = True
        y = y[:,None]
        mu = mu[:,None] if mu is not None else mu
        sd = sd[:,None] if sd is not None else sd
    sds = np.nanstd(y, axis=0) if sd is None else sd
    sds[sds == 0] = 1 # if no sdev, we won't divide by it
    mu = np.nanmean(y, axis=0) if mu is None else mu
    z = (y - mu)/sds
    return z[:,0] if changed_shape else z

#%% functions

def get_feature_from_trials(trials, field_name, window=None, trial_types=None, mode='mean', trial_type_key='trial_type'):
    trial_keeper = lambda trial: (trial_types is None) or (trial[trial_type_key] in trial_types)

    data = [trial[field_name] for trial in trials if trial_keeper(trial)]
    if len(data) == 0:
        return np.array([])
    else:
        data = np.vstack(data)
    if window is not None:
        data = data[:,window[0]:window[1]]
    if mode == 'mean':
        X = data.mean(axis=1)
    elif mode == 'max':
        X = data.max(axis=1)
    elif mode == 'nanmean':
        X = np.nanmean(data, axis=1)
    else:
        raise Exception(f"Unrecognized {mode=}. Options are 'max' or 'mean'.")
    return X

def get_data_per_session(session, da_window, lick_window, trial_types=None, zscore_spikes=False, use_mean_spikes=True, subtract_baseline_spikes=False, trial_type_key='trial_type'):
    # collect DA, licking, and session index data
    X = get_feature_from_trials(session, 'DA', da_window, trial_types, 'mean' if use_mean_spikes else 'max', trial_type_key=trial_type_key)
    if subtract_baseline_spikes:
        Xb = get_feature_from_trials(session, 'baseline_DA_rate', None, trial_types, 'mean', trial_type_key=trial_type_key)
        # Xb is per bin, just like X
        X = X - Xb # *(da_window[1]-da_window[0])
    Y = get_feature_from_trials(session, 'licks', lick_window, trial_types, 'mean', trial_type_key=trial_type_key)
    X = zscore_if_possible(X) if zscore_spikes else X
    return X, Y
