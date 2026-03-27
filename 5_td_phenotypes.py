#%% imports

import os.path
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

from td import create_experiment, TD
from figure_helpers import plt, handle_fig
from data_helpers import build_design_matrix

#%% experiment and agent helper functions

EXPERIMENT_PARAMS = {
    'ntrials': 10000, # number of trials to simulate
    'iti_min': 5, # minimum ITI in time steps
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

    data = TD(Z, R, gamma=td_params['gamma'], alpha=td_params['alpha'], w_memory=td_params['w_memory'], save_index=0, eps=td_params['eps'], seed=td_params['seed'], eps_z=td_params['eps_z'])
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
        'CS DA': {'licks': exp_smooth_agent(data['rpes'], alpha=alpha, eps=eps, seed=seed)},
        'CS DA*V': {'licks': exp_smooth_agent(data['rpes'] * data['values'], alpha=alpha, eps=eps, seed=seed)},
        'CS DA+V': {'licks': exp_smooth_agent_multiple((data['rpes'], data['values']), alpha=alpha, weights=(1.0, 2.0), eps=eps, seed=seed)},
    }

    if agent_names is not None:
        for agent in agent_names:
            if agent not in agents:
                raise ValueError(f"Agent '{agent}' not found in available agents: {list(agents.keys())}")
        agents = {name: agents[name] for name in agent_names}
    return agents

#%% regression helper functions

def simulate_agents(nruns, all_run_params, agent_names=None):
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

def generate_agent_data(add_rep_noise, nruns=500, do_overwrite=False, try_to_load=True):
    filename = "data/regression_weights_agents_nomdl_with_noise.npz" if add_rep_noise else "data/regression_weights_agents_nomdl.npz"

    if try_to_load and os.path.exists(filename):
        results = np.load(filename, allow_pickle=True)['results']
        return results
        
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

    results = simulate_agents(nruns=nruns, all_run_params=all_run_params, agent_names=agent_names)
    
    if not os.path.exists(filename) or do_overwrite:
        np.savez(filename, results=results)
    else:
        print(f"File {filename} already exists. Set do_overwrite=True to overwrite.")
    return results

#%% plot correlation between licking and CS DA history

do_save = True
history_length = 5
data_keys = ('rpes',)

for add_rep_noise in [False, True]:
    results = generate_agent_data(add_rep_noise)

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

    def plot_quantile_band(x, Y, color='C0', label=None):
        for q in [10,20,30,40]:
            qlb, qub = np.percentile(Y, [q, 100-q], axis=0)
            plt.fill_between(x, qlb, qub, color=color, alpha=0.2, linewidth=0)
        plt.plot(x, np.median(Y, axis=0), color=color, linewidth=2, label=label)

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
