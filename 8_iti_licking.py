#%% imports

import os.path
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

from td import create_experiment, TD
from figure_helpers import plt, handle_fig
from data_helpers import build_design_matrix

EXPERIMENT_PARAMS = {
    'ntrials': 10000, # number of trials to simulate
    'iti_min': 20, # minimum ITI in time steps
    'iti_p': 0.125, # probability of ITI increasing by 1 time step (geometric distribution)
    'isi': 8, # ISI in time steps
    'add_bias': True, # whether to add a bias term to the state representation
    'seed': None, # random seed for experiment generation
}
TD_PARAMS = {
    'alpha': 0.004, # learning rate
    'gamma': 0.9, # discount factor
    'w_memory': 1, # degree to which weights are preserved across time steps
    'eps': 0.05, # noise added to RPEs during TD learning
    'seed': None, # random seed for TD learning
    'eps_z': 0.0, # noise added to state representation during TD learning
}

#%%

def exp_smooth_agent(xs, alpha=0.9, eps=0.0, seed=123):
    """
    xs is an array of values (e.g., CS rpes, values)
    alpha is the smoothing factor (0 < alpha < 1)
    eps is the std dev of noise to add to the outcome
    returns the outcome of exponentially smoothing xs and adding noise
    """
    ys = np.zeros_like(xs)
    ys[0] = xs[0]
    for t in range(1, len(ys)):
        ys[t] = (1-alpha)*ys[t-1] + alpha*xs[t]

    # add noise to responses
    if eps > 0:
        rng = default_rng(seed)
        nse = eps * rng.standard_normal(size=len(ys))
        ys += nse
    return ys

#%%

experiment_params = EXPERIMENT_PARAMS.copy()
Z, R = create_experiment(experiment_params['ntrials'], experiment_params['iti_min'], experiment_params['iti_p'], experiment_params['isi'], r_prob=1.0, add_bias=experiment_params['add_bias'], reward_size=1, seed=experiment_params['seed'], stack=True)

# td_params = TD_PARAMS.copy()
# data = TD(Z, R, gamma=td_params['gamma'], alpha=td_params['alpha'], w_memory=td_params['w_memory'], save_index=0, eps=td_params['eps'], seed=td_params['seed'], eps_z=td_params['eps_z'])

data = TD(Z, R, gamma=td_params['gamma'], alpha=td_params['alpha'], w_memory=td_params['w_memory'], save_index=0, eps=td_params['eps'], seed=td_params['seed'], eps_z=td_params['eps_z'])

activation = (1 + 5*data['rpes'])*data['values']
# average per trial
activation = np.array([activation[data['trial_index']==i].mean() for i in range(1, data['trial_index'].max()+1)])
values = np.array([data['values'][data['trial_index']==i].mean() for i in range(1, data['trial_index'].max()+1)])
rpes = np.array([data['rpes'][data['trial_index']==i].mean() for i in range(1, data['trial_index'].max()+1)])

plt.plot(values, label='value')
plt.plot(rpes, label='rpe', alpha=0.4)
plt.plot(activation, label='activation', alpha=0.4)
# plt.xlim([79900, 80000])
# plt.xlim([0, 200])
plt.legend()
plt.show()

data = TD(Z, R, gamma=td_params['gamma'], alpha=td_params['alpha'], w_memory=td_params['w_memory'], save_index=Z.shape[1]-2, eps=td_params['eps'], seed=td_params['seed'], eps_z=td_params['eps_z'])

activation = (1 + 5*data['rpes'])*data['values']
# average per trial
activation = np.array([activation[data['trial_index']==i].mean() for i in range(1, data['trial_index'].max()+1)])
values = np.array([data['values'][data['trial_index']==i].mean() for i in range(1, data['trial_index'].max()+1)])
rpes = np.array([data['rpes'][data['trial_index']==i].mean() for i in range(1, data['trial_index'].max()+1)])

plt.plot(values, label='value')
plt.plot(rpes, label='rpe', alpha=0.4)
plt.plot(activation, label='activation', alpha=0.4)
# plt.xlim([79900, 80000])
# plt.xlim([0, 200])
plt.legend()

# t = 50; plt.plot(Z[:t,-2], label='rpe')

# licks = exp_smooth_agent(activation, alpha=0.9, eps=0.1, seed=123)


#%%
