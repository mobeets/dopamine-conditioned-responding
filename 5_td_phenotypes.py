#%% imports

import os.path
import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import KFold
from tqdm import tqdm

from td import create_experiment, create_experiment_multiple_cues
from figure_helpers import plt, handle_fig
from regression_model import zscore_if_possible, get_data_per_session

#%% experiment and agent helper functions

def TD_with_trial_data(Z, R, gamma, alpha=0.1, isi_ind=1, w_memory=1, eps=0.0, seed=None, eps_z=0.0):
    Znext_noNoise = Z[1:].copy()
    if eps_z > 0.0:
        rng = default_rng(seed+1) if seed is not None else default_rng()
        Z = Z + eps_z * rng.standard_normal(size=Z.shape)
    Zcur = Z[:-1]
    Znext = Z[1:]
    
    Rc = R[:-1]
    w = np.zeros(Z.shape[1])
    rpe = np.nan
    data = {'CS_rpes': [], 'values': [], 'weights': []}

    rng = default_rng(seed)
    nse = eps * rng.standard_normal(size=len(Z))

    for z, z_next, z_nn, r, eps_t in zip(Zcur, Znext, Znext_noNoise, Rc, nse):
        rpe = r + gamma*z_next.dot(w) - z.dot(w) + eps_t

        if z_nn[isi_ind] == 1: # CS onset
            data['CS_rpes'].append(rpe) # CS rpe from current trial
            data['values'].append(z_next.dot(w)) # CS value on current trial
            data['weights'].append(w.copy())
        
        w = w_memory*w + alpha*rpe*z
    
    # convert all lists to arrays
    for key, values in data.items():
        data[key] = np.array(values)
    return data

EXPERIMENT_PARAMS = {
    'ntrials': 10000,
    'iti_min': 5,
    'iti_p': 0.125,
    'isi': 8,
    'add_bias': True,
    'seed': None, 
}
TD_PARAMS = {
    'alpha': 0.004,
    'gamma': 0.9,
    'w_memory': 1,
    'eps': 0.05, # noise added to RPEs during TD learning
    'seed': None,
    'eps_z': 0.0, # noise added to state representation during TD learning
}

def train(experiment_params=None, td_params=None, **overrides):
    """
    Apply temporal difference learning to classical conditioning experiment
    """
    if experiment_params is None:
        experiment_params = EXPERIMENT_PARAMS.copy()
    else:
        experiment_params = experiment_params.copy()
    if td_params is None:
        td_params = TD_PARAMS.copy()
    else:
        td_params = td_params.copy()
    for key, value in overrides.items():
        if key in experiment_params:
            assert key not in td_params, f"Parameter {key} is defined in both experiment_params and td_params. Please specify only one."
            experiment_params[key] = value
        elif key in td_params:
            td_params[key] = value
        else:
            raise ValueError(f"Unknown parameter: {key}")

    Z, R = create_experiment(experiment_params['ntrials'], experiment_params['iti_min'], experiment_params['iti_p'], experiment_params['isi'], r_prob=1.0, add_bias=experiment_params['add_bias'], reward_size=1, seed=experiment_params['seed'], stack=True)

    data = TD_with_trial_data(Z, R, gamma=td_params['gamma'], alpha=td_params['alpha'], w_memory=td_params['w_memory'], isi_ind=0, eps=td_params['eps'], seed=td_params['seed'], eps_z=td_params['eps_z'])
    return data

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

def exp_smooth_agent_multiple(xss, alpha=0.9, weights=(1.0), eps=0.0, seed=123):
    """
    xss is a list of arrays of values (e.g., CS rpes, values)
    alpha is a smoothing factor (0 < alpha < 1)
    weights is a list of weights for each array in xss
    eps is the std dev of noise to add to the outcome
    returns the outcome of exponentially smoothing each xs, taking the weighted sum, and then adding noise
    """
    x = 0
    for weight, xs in zip(weights, xss):
        x += (weight * xs)
    xs = x
    return exp_smooth_agent(xs, alpha=alpha, eps=eps, seed=seed)

def run_agents(data, agent_names=None, eps=0.0, alpha=0.9, seed=123):
    agents = {
        'CS Value': {'licks': exp_smooth_agent(data['values'], alpha=alpha, eps=eps, seed=seed)},
        'CS DA': {'licks': exp_smooth_agent(data['CS_rpes'], alpha=alpha, eps=eps, seed=seed)},
        'CS DA*V': {'licks': exp_smooth_agent(data['CS_rpes'] * data['values'], alpha=alpha, eps=eps, seed=seed)},
        'CS DA+V': {'licks': exp_smooth_agent_multiple((data['CS_rpes'], data['values']), alpha=alpha, weights=(1.0, 2.0), eps=eps, seed=seed)},
    }

    if agent_names is not None:
        for agent in agent_names:
            if agent not in agents:
                raise ValueError(f"Agent '{agent}' not found in available agents: {list(agents.keys())}")
        agents = {name: agents[name] for name in agent_names}
    return agents

#%% regression helper functions

def build_design_matrix(X, K):
    """
    Build a design matrix for regression with K time steps of history.
    X: 2D array of shape (T, D) where T is the number of time steps and D is the number of features
    K: int, number of time steps of history to include in the design matrix
    Returns:
        Z: 2D array of shape (T-K+1, D*K) where T-K+1 is the number of time steps after K and D*K is the number of features
    """
    X = np.asarray(X)
    T, D = X.shape
    Z = np.zeros((T - K + 1, D * K))
    for t in range(K - 1, T):
        window = X[t - K + 1:t + 1]  # shape (K, D)
        # Transpose to (D, K) and then flatten so we get feature 0's full window first, then feature 1's
        Z[t - K + 1, :] = window.T.flatten()
    return Z

def shuffle_trials(y, y_ref):
    """
    shuffle trials in y, preserving NaN positions from y_ref
    """
    rng = default_rng(123)
    y_shuffled = y.copy()
    ix_valid = ~np.isnan(y_ref)
    y_valid = y_shuffled[ix_valid]
    rng.shuffle(y_valid)
    y_shuffled[ix_valid] = y_valid
    return y_shuffled

def simulate_agents(nruns, data_keys, all_run_params, agent_names=None):
    clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig = plt.figure(figsize=(3*len(data_keys),3))
    results = []
    # plot progress bar
    for n in tqdm(range(nruns)):
        run_params = {key: np.random.choice(value) for key, value in all_run_params.items()}
        history_length = run_params['ridge_history_length']
        alpha = run_params['ridge_alpha'] # for Ridge regression
        eps_agent = run_params['eps_agent'] # noise added to agent's licks
        data = train(alpha=run_params['td_alpha'], w_memory=run_params['td_w_memory'], eps_z=run_params.get('td_eps_z', 0.0), gamma=run_params.get('td_gamma', 0.9))
        agents = run_agents(data, agent_names=agent_names, alpha=run_params['agent_alpha'], eps=eps_agent)

        results.append({'run_params': run_params, 'agents': agents, 'data': data})
    return results

#%% generate agents

def generate_agent_data(add_rep_noise, nruns=500, do_overwrite=False):
    data_keys = ('CS_rpes',)
    agent_names = ('CS Value', 'CS DA') # names of agents to run

    all_run_params = {
        'ridge_history_length': [5], # history length for Ridge regression
        'ridge_alpha': [1.0], # Ridge regression regularization parameter
        'agent_alpha': np.arange(0.1, 1, 0.1), # smoothing factor for agent's licks
        'eps_agent': [0.1, 0.2, 0.3, 0.4], # noise added to agent's licks
        # 'eps_agent': [0.0, 0.1, 0.2, 0.3, 0.5, 0.9], # noise added to agent's licks
        'td_eps_z': [0.2 if add_rep_noise else 0.0], # noise added to state representation during TD learning
        'td_gamma': [0.5, 0.6, 0.7, 0.8, 0.9],
        'td_alpha': [0.001, 0.004, 0.01, 0.02, 0.05, 0.1], # TD learning rate
        'td_w_memory': [1-1e-6, 1-1e-5, 1-1e-4],} # TD forgetting factor

    results = simulate_agents(nruns=nruns, data_keys=data_keys, all_run_params=all_run_params, agent_names=agent_names)
    filename = "data/regression_weights_agents_nomdl_with_noise.npz" if add_rep_noise else "data/regression_weights_agents_nomdl.npz"
    if not os.path.exists(filename) or do_overwrite:
        np.savez(filename, results=results)
    else:
        print(f"File {filename} already exists. Set do_overwrite=True to overwrite.")

#%% plot correlation between licking and CS DA history

do_save = False
add_rep_noise = False
history_length = 5

filename = f"data/regression_weights_agents_nomdl_with_noise.npz" if add_rep_noise else "data/regression_weights_agents_nomdl.npz"
if not os.path.exists(filename):
    generate_agent_data(add_rep_noise)
results = np.load(filename, allow_pickle=True)['results']

# Calculate correlations between licking and CS DA at different lags
Rhos = [[], []]
for result in results:
    data = result['data']
    X = np.column_stack([data[key] for key in data_keys]).copy() # shape (T, D)
    Z_all = build_design_matrix(X, history_length)
    ix = np.any(np.isnan(Z_all), axis=1)

    for i, (agent_name, agent) in enumerate(result['agents'].items()):
        y = agent['licks'].copy()
        y_target = y[history_length-1:] # shape (T-K+1,)
        Z = Z_all[~ix].copy()
        y_target = y_target[~ix]
        rhos = np.array([np.corrcoef(Z[:,k], y_target)[0,1] for k in range(Z.shape[1])])
        Rhos[i].append(rhos)
Rhos = [np.vstack(r) for r in Rhos]

# Plot mean correlations per agent
nx = history_length
fig = plt.figure(figsize=(3,3))
x = np.arange(nx)[::-1]
plot_quantile_band(x, Rhos[0], 'C0', label='H1')
plot_quantile_band(x, Rhos[1], '#3BAA58', label='H2')
plt.xlabel('# Trials back ($\\tau$)')
plt.ylabel('Corr. between Licks(t) and CS RPE(t-$\\tau$)')
plt.xticks(np.arange(0, nx, 1))
plt.legend(fontsize=8)
plt.tight_layout()
plt.xlim([-0.1, nx-0.9])
plt.ylim([-0.05, 0.65])
plt.plot(plt.xlim(), [0, 0], 'k--', alpha=0.5, zorder=-1)
handle_fig(fig, f"regression_weights_agents_nomdl_with_noise" if add_rep_noise else "regression_weights_agents_nomdl", do_save=do_save)

# Print d' of regression weights between agents
rhos_1 = Rhos[0][:,-1]
rhos_2 = Rhos[1][:,-1]
# print d-prime of rhos_1 vs rhos_2 on bootstrapped samples
nboot = 1000
dprimes = []
rng = default_rng(123)
for i in range(nboot):
    ix1 = rng.choice(np.arange(len(rhos_1)), size=len(rhos_1), replace=True)
    ix2 = rng.choice(np.arange(len(rhos_2)), size=len(rhos_2), replace=True)
    mu1 = np.mean(rhos_1[ix1])
    mu2 = np.mean(rhos_2[ix2])
    sd1 = np.std(rhos_1[ix1])
    sd2 = np.std(rhos_2[ix2])
    pooled_sd = np.sqrt((sd1**2 + sd2**2) / 2)
    dprime = (mu1 - mu2) / pooled_sd
    dprimes.append(dprime)
dprimes = np.array(dprimes)
mu_dp = np.mean(dprimes)
se_dp = np.std(dprimes)
ci_lower = np.percentile(dprimes, 2.5)
ci_upper = np.percentile(dprimes, 97.5)
print(f"d': {mu_dp:.3f} ± {se_dp:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

#%% get correlations between licking and CS DA history in empirical data

do_save = False
conds = [
    {'D': D2, 'trial_type': 'A', 'title': D2.exp_name},
    {'D': D3, 'trial_type': 'A', 'title': D3.exp_name},
    {'D': D4, 'trial_type': 'B', 'title': D4.exp_name},
    {'D': D5b, 'trial_type': 'A', 'title': D5b.exp_name},
    {'D': D5a, 'trial_type': 'A', 'title': D5a.exp_name},
    {'D': D1, 'trial_type': 'A', 'title': D1.exp_name},
    ]

history_length = 5
nboots = 100

Rhos = []
Rhos_shuf = []
for i, cond in enumerate(conds):
    D = cond['D']
    trial_type = cond['trial_type']
    trial_types = D.trial_types[trial_type][:1]
    da_window = D.da_bin_window
    lick_window = D.lick_bin_windows[trial_type]

    lick_to_hz = 1000 / D.bin_size
    day_indices = None # if 'Qian' not in D.exp_name else (3,4)
    # day_indices = None if 'Qian' not in D.exp_name else (0,1,)
    use_mean_spikes = False if ('Qian' in D.exp_name or 'Amo' in D.exp_name) else True

    X = []
    Xshuf = []
    y = []
    for j, session in enumerate(D.Sessions):
        if day_indices is not None and session[0]['day_index'] not in day_indices:
            continue
        Xc, yc = get_data_per_session(session, da_window, lick_window, trial_types=trial_types, zscore_spikes=False, use_mean_spikes=use_mean_spikes, subtract_baseline_spikes=False)

        # session-wise z-scoring
        Xc = zscore_if_possible(Xc)
        yc = zscore_if_possible(yc)
        
        X_pad = np.nan * np.ones(history_length)
        X.append(X_pad)
        y.append(X_pad)
        Xshuf.append(X_pad)
        X.append(Xc)
        y.append(yc)
        Xshuf.append(shuffle_trials(Xc, yc))

    X = np.hstack(X)
    y = np.hstack(y)
    Z_all = build_design_matrix(X[:,None], history_length)
    ix = np.any(np.isnan(Z_all), axis=1)
    y_target = y[history_length-1:] # shape (T-K+1,)
    Z = Z_all[~ix]
    y = y_target[~ix]

    Xshuf = np.hstack(Xshuf)
    Z_all_shuf = build_design_matrix(Xshuf[:,None], history_length)
    ix_shuf = np.any(np.isnan(Z_all_shuf), axis=1)
    Z_shuf = Z_all_shuf[~ix_shuf]
    y_shuf = y_target[~ix_shuf]

    rhos_all = []
    rhos_all_shuf = []
    for _ in range(nboots):
        Zc = Z.copy()
        yc = y.copy()
        # sample random 90% of trials
        rng = default_rng()
        n_samples = int(0.9 * len(yc))
        sample_idx = rng.choice(len(yc), size=n_samples, replace=False)
        Zc = Zc[sample_idx]
        yc = yc[sample_idx]
        # compute correlations
        rhos = np.array([np.corrcoef(Zc[:,k], yc)[0,1] for k in range(Zc.shape[1])])
        rhos_all.append(rhos)

        # now do the same for shuffled data
        Zc_shuf = Z_shuf.copy()
        yc_shuf = y_shuf.copy()
        sample_idx_shuf = rng.choice(len(yc_shuf), size=n_samples, replace=False)
        Zc_shuf = Zc_shuf[sample_idx_shuf]
        yc_shuf = yc_shuf[sample_idx_shuf]
        rhos_shuf = np.array([np.corrcoef(Zc_shuf[:,k], yc_shuf)[0,1] for k in range(Zc_shuf.shape[1])])
        rhos_all_shuf.append(rhos_shuf)
    rhos_all = np.vstack(rhos_all)
    Rhos.append(rhos_all)
    rhos_all_shuf = np.vstack(rhos_all_shuf)
    Rhos_shuf.append(rhos_all_shuf)

fig = plot_correlations([rhos[:,-1] for rhos in Rhos], ylbl="Corr. between Licks(t) and CS DA(t)", xticklabels=[cond['title'] for cond in conds], zeroline=True)
handle_fig(fig, f"regression_weights_all_experiments", do_save=do_save)
