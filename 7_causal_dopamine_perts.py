#%% imports

import numpy as np
from numpy.random import default_rng
from td import create_experiment, TD
from figure_helpers import plt, handle_fig

#%% Lee 2020, multiple experiments
# key observations:
# - DA inhibition during reward reduces anticipatory licking
# - DA inhibition during cue reduces licking but less than during reward
# - DA inhibition during cue on random half of trials reduces licking on those trials compared to non-inhibited trials

ntrials = 1000
iti_min = 5
iti_p = 0.25
isi = 8
gamma = 0.9
w_memory = 1
eps_z = 0.001
eps = 0.0
add_bias = True
nexperiments = 100
seeds = np.arange(nexperiments)
n_final_trials = 50

pert_params = { # (b, m) where y = m*(rpe - b)
    'CS+': (0.0, 1.0),
    'CS+ (cue inhibition)': (0.1, 1.0),
    'CS+ (rew. inhibition)': (0.1, 1.0),
    'CS+ (cue inhibition*)': (0.1, 1.0),
}
clrs = {
    'CS+': 'k',
    'CS+ (cue inhibition)': '#0893c2',
    'CS+ (rew. inhibition)': '#38c7b9',
    'CS+ (cue inhibition*)': '#0893c2',
}
models = [
    {'name': 'Value', 'alpha': 0.05},
    {'name': 'DA', 'alpha': 0.05},
]

results = []
for seed in seeds:
    Z, R = create_experiment(ntrials, iti_min, iti_p, isi, r_prob=1.0, add_bias=add_bias, reward_size=1, seed=seed)
    X = np.hstack([i*np.ones(z.shape[0]) for i, z in enumerate(Z)]) # trial indices

    # define when opto stim begins
    pert_trial = int(ntrials/2)
    pert_timestep = len(np.vstack(Z[:pert_trial]))

    Z = np.vstack(Z)
    R = np.hstack(R)
    rng = default_rng(seed + 123)

    for j, model in enumerate(models):
        alpha = model['alpha']
        for i, (mode, (b_pert, m_pert)) in enumerate(pert_params.items()):
            # set experimental parameters before and after perturbation
            bs = np.zeros_like(R)
            ms = np.ones_like(R)
            
            all_pert_trials = np.arange(pert_trial, ntrials)
            if '*' in mode:
                # select random half of trials to perturb
                rng.shuffle(all_pert_trials)
                pert_trials = sorted(all_pert_trials[:int(len(all_pert_trials)/2)])
                non_pert_trials = np.setdiff1d(all_pert_trials, pert_trials)
                pert_indices = np.hstack([np.where(X == t)[0] for t in pert_trials])
                bs[pert_indices] = b_pert
                ms[pert_indices] = m_pert
            else:
                pert_trials = all_pert_trials
                non_pert_trials = []
                bs[pert_timestep:] = b_pert
                ms[pert_timestep:] = m_pert

            if '(cue' in mode or '(rew' in mode:        
                if '(cue' in mode:
                    # will stim only during ISI
                    ixc = Z[:,:1].sum(axis=1) > 0
                    
                    ix = False * np.ones_like(Z[:,0], dtype=bool)
                    inds = np.where(ixc)[0]
                    ix[inds] = True
                    ix[inds-1] = True

                elif '(rew' in mode:
                    # will stim only during/following reward delivery
                    ixc = Z[:,(isi-2):-2].sum(axis=1) > 0
                    ix = False * np.ones_like(Z[:,0], dtype=bool)
                    inds = np.where(ixc)[0]
                    ix[inds] = True

                # set non-perturbed timesteps to control
                bs[~ix] = 0
                ms[~ix] = 1

            td_data = TD(Z, R, bs, ms, gamma=gamma, alpha=alpha, w_memory=w_memory, save_index=0, eps_z=eps_z, eps=eps)

            if model['name'] == 'Value':
                p_action = td_data['values']
            elif model['name'] == 'DA':
                p_action = td_data['rpes']
            assert len(p_action) == ntrials

            final_lick_probs = p_action[-n_final_trials:].mean()
            lick_probs_pert_trials = p_action[pert_trials]
            lick_probs_non_pert_trials = p_action[non_pert_trials]
            result = {'model': model['name'], 'mode': mode, 'seed': seed, 'final_lick_prob': final_lick_probs, 'lick_probs_pert_trials': lick_probs_pert_trials, 'lick_probs_non_pert_trials': lick_probs_non_pert_trials}
            results.append(result)

#%% plot Lee 2020 results for blockwise perturbations

do_save = True

mode_names = {'CS+': 'Control', 'CS+ (cue inhibition)': 'Cue Inhibition', 'CS+ (rew. inhibition)': 'Reward Inhib.', 'CS+ (cue inhibition*)': 'Cue Inhib.*'}
for model in models:
    fig = plt.figure(figsize=(2,3))
    xlbls = []
    xi = 0
    for i, mode in enumerate(pert_params.keys()):
        if '*' in mode:
            continue
        xlbls.append(mode_names[mode])
        ys = np.hstack([result['final_lick_prob'] for result in results if result['model'] == model['name'] and result['mode'] == mode])

        xs = np.ones_like(ys) * (xi) + 0.1 * (np.random.rand(len(ys)) - 0.5)
        mu = ys.mean()
        se = ys.std() / np.sqrt(len(ys))
        plt.plot(xs, ys, '.', color='k', alpha=0.1, zorder=-1)
        plt.plot([xi, xi], [mu - se, mu + se], 'k-', linewidth=1)
        plt.plot(xi, mu, 'o', markersize=10, color='k', markeredgecolor='white')
        xi += 1
    
    plt.xticks(np.arange(len(xlbls)), xlbls, rotation=45, ha='right')
    plt.ylabel('Anticipatory lick rate (a.u.)')
    plt.xlim([-0.5, len(xlbls)-0.5])
    plt.tight_layout()
    handle_fig(fig, f"DA_perts_Lee_summary_{model['name']}", do_save=do_save)

#%% plot Lee 2020 results on random trials

do_save = True

for model in models:
    fig = plt.figure(figsize=(1.5,2.5))
    for mode in pert_params.keys():
        if '*' not in mode:
            continue
        ys_pert = np.hstack([np.mean(result['lick_probs_pert_trials']) for result in results if result['model'] == model['name'] and result['mode'] == mode])
        ys_non_pert = np.hstack([np.mean(result['lick_probs_non_pert_trials']) for result in results if result['model'] == model['name'] and result['mode'] == mode])

        for i, ys in enumerate([ys_non_pert, ys_pert]):
            xs = np.ones_like(ys) * (i) + 0.1 * (np.random.rand(len(ys)) - 0.5)
            plt.plot(xs, ys, '.', color='k', alpha=0.1, zorder=-1)
            mu = ys.mean()
            se = ys.std() / np.sqrt(len(ys))
            plt.plot(i, mu, 'o', markersize=10, color='k', markeredgecolor='white')
            plt.plot([i, i], [mu - se, mu + se], 'k-', linewidth=1, zorder=-1)
    plt.xticks([0,1], ['Off', 'On'], rotation=45)
    plt.xlabel('Inhibition')
    plt.xlim([-0.5, 1.5])
    plt.ylim([0, 0.2] if model['name'] == 'DA' else [0, 0.6])
    plt.ylabel('Anticipatory lick rate (a.u.)')
    plt.tight_layout()
    handle_fig(fig, f"DA_perts_Lee_random_summary_{model['name']}", do_save=do_save)

#%% Morrens 2020, multiple experiments

ntrials = 200
seed = 500
iti_min = 5
iti_p = 0.25
isi = 8
gamma = 0.9
w_memory = 1
eps_z = 0.01
eps = 0.0
add_bias = True
nexperiments = 100
seeds = np.arange(nexperiments)
n_final_trials = 50

pert_params = { # (b, m) where y = m*(rpe - b)
    'CS+': (0.0, 1.0),
    'CS–': (0.0, 1.0),
    'CS+ (cue stim)': (-0.1, 1.0),
    'CS– (cue stim)': (-0.1, 1.0),
}
clrs = {
    'CS+': 'k',
    'CS–': 'gray',
    'CS+ (cue stim)': 'red',
    'CS– (cue stim)': 'gray',
}
models = [
    {'name': 'Value', 'alpha': 0.1},
    {'name': 'DA', 'alpha': 0.1},
]

results = []
for seed in seeds:
    Z, R = create_experiment(ntrials, iti_min, iti_p, isi, r_prob=1.0, add_bias=add_bias, reward_size=1, seed=seed)
    X = np.hstack([i*np.ones(z.shape[0]) for i, z in enumerate(Z)]) # trial indices

    Z = np.vstack(Z)
    R = np.hstack(R)
    rng = default_rng(seed + 123)

    c = 0
    for j, model in enumerate(models):
        alpha = model['alpha']
        for i, (mode, (b_pert, m_pert)) in enumerate(pert_params.items()):
            bs = b_pert * np.ones_like(R)
            ms = m_pert * np.ones_like(R)

            if '(cue stim)' in mode or '(rew stim)' in mode:
                if '(cue stim)' in mode:
                    # will stim only during ISI
                    ixc = Z[:,:1].sum(axis=1) > 0
                    inds = np.where(ixc)[0]
                    ix = False * np.ones_like(Z[:,0], dtype=bool)
                    ix[inds] = True
                    ix[inds-1] = True
                elif '(rew stim)' in mode:
                    # will stim only during/following reward delivery
                    ixc = Z[:,(isi-2):-2].sum(axis=1) > 0
                    inds = np.where(ixc)[0]
                    ix = False * np.ones_like(Z[:,0], dtype=bool)
                    ix[inds] = True
                # set non-perturbed timesteps to control
                bs[~ix] = 0
                ms[~ix] = 1

            Zc = Z.copy()
            Rc = R.copy()
            if 'CS–' in mode:
                Rc = 0*Rc
            td_data = TD(Zc, Rc, bs, ms, gamma=gamma, alpha=alpha, w_memory=w_memory, save_index=0, eps_z=eps_z, eps=eps)

            if model['name'] == 'Value':
                p_action = td_data['values']
            elif model['name'] == 'DA':
                p_action = td_data['rpes']
            assert len(p_action) == ntrials

            final_lick_probs = p_action[-n_final_trials:].mean()
            result = {'model': model['name'], 'mode': mode, 'seed': seed, 'final_lick_prob': final_lick_probs, 'lick_probs': p_action}
            results.append(result)

#%% plot Morrens 2020 results for blockwise perturbations

do_save = True

mode_names = {'CS+': 'CS+', 'CS–': 'CS–', 'CS+ (cue stim)': 'CS+ Excitation', 'CS– (cue stim)': 'CS– Excitation'}
for model in models:
    fig = plt.figure(figsize=(2,3))
    xlbls = []
    xi = 0
    for i, mode in enumerate(pert_params.keys()):
        if '*' in mode:
            continue
        xlbls.append(mode_names[mode])
        ys = np.hstack([result['final_lick_prob'] for result in results if result['model'] == model['name'] and result['mode'] == mode])

        xs = np.ones_like(ys) * (xi) + 0.1 * (np.random.rand(len(ys)) - 0.5)
        mu = ys.mean()
        se = ys.std() / np.sqrt(len(ys))
        plt.plot(xs, ys, '.', color='k', alpha=0.1, zorder=-1)
        plt.plot([xi, xi], [mu - se, mu + se], 'k-', linewidth=1)
        plt.plot(xi, mu, 'o', markersize=10, color='k', markeredgecolor='white')
        xi += 1
    
    plt.xticks(np.arange(len(xlbls)), xlbls, rotation=45, ha='right')
    plt.ylabel('Anticipatory lick rate (a.u.)')
    plt.xlim([-0.5, len(xlbls)-0.5])
    plt.tight_layout()
    handle_fig(fig, f"DA_perts_Morrens_summary_{model['name']}", do_save=do_save)
