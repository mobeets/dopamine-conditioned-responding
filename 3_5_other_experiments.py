#%% imports

import numpy as np
from numpy.random import default_rng
from data_loaders import QianBurrell2024, Eshel2016, Matsumoto2016, Kim2020, Amo2024
from data_helpers import get_feature_from_trials, get_data_per_session, zscore_if_possible, build_design_matrix, shuffle_trials
from figure_helpers import handle_fig, plot_correlations

#%% load sessions

D2 = Eshel2016()
D3 = Matsumoto2016()
D4 = Kim2020()
D5a = Amo2024(key='CC_DA')
D5b = Amo2024(key='CC_GCaMP')
D1 = QianBurrell2024()

#%% compare average anticipatory licking on low vs high DA trials

do_save = True
percentiles = (25,75)

conds = [
    {'D': D2, 'trial_type': 'A', 'title': D2.exp_name},
    {'D': D3, 'trial_type': 'A', 'title': D3.exp_name},
    {'D': D4, 'trial_type': 'B', 'title': D4.exp_name},
    {'D': D5b, 'trial_type': 'A', 'title': D5b.exp_name},
    {'D': D5a, 'trial_type': 'A', 'title': D5a.exp_name},
    {'D': D1, 'trial_type': 'A', 'title': D1.exp_name},
    ]
showFirstLick = False
is_qian_burrell = lambda D: 'Qian & Burrell et al., 2025' in D.exp_name

Pts = []
Pts_both = []
names = []
for i, cond in enumerate(conds):
    D = cond['D']
    if not hasattr(D, 'Sessions') or len(D.Sessions) == 0:
        continue
    trial_type = cond['trial_type']
    trial_types = D.trial_types[trial_type][:1]
    da_window = D.da_bin_window
    lick_window = D.lick_bin_windows[trial_type]

    lick_to_hz = 1000 / D.bin_size
    day_indices = None if not is_qian_burrell(D) else (3,4)
    use_mean_spikes = False if ('Qian' in D.exp_name or 'Amo' in D.exp_name) else True

    ymus = []
    ymus_both = []
    for session in D.Sessions:
        if day_indices is not None and session[0]['day_index'] not in day_indices:
            continue
        X, y = get_data_per_session(session, da_window, lick_window, trial_types=trial_types, zscore_spikes=False, use_mean_spikes=use_mean_spikes, subtract_baseline_spikes=False)

        if showFirstLick:
            y = get_feature_from_trials(session, 'time_of_first_lick', None, trial_types, 'nanmean')
            ix = ~(np.isnan(y) | np.isnan(X))
            X = X[ix]; y = y[ix]
            if len(y) == 0:
                continue

        prcs = np.percentile(X, percentiles)
        ixLow = X <= prcs[0]
        ixHigh = X >= prcs[1]
        ymus.append(np.nanmean(y[ixHigh])-np.nanmean(y[ixLow]))
        ymus_both.append((np.nanmean(y[ixLow]), np.nanmean(y[ixHigh])))

    yc = lick_to_hz * np.hstack(ymus)
    Pts.append(yc)
    Pts_both.append(lick_to_hz * np.vstack(ymus_both))
    names.append(cond['title'])

fig = plot_correlations(Pts_both, ylbl="$\Delta$ Anticipatory lick rate (Hz)", xticklabels=names, ystar=4.5, zeroline=True)
handle_fig(fig, 'delta_licking_all_exps', do_save=do_save)

#%% get correlations between licking and CS DA history in empirical data

do_save = True
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
names = []
for i, cond in enumerate(conds):
    D = cond['D']
    if not hasattr(D, 'Sessions') or len(D.Sessions) == 0:
        continue
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
    names.append(cond['title'])

fig = plot_correlations([rhos[:,-1] for rhos in Rhos], ylbl="Corr. between Licks(t) and CS DA(t)", xticklabels=names, zeroline=True)
handle_fig(fig, f"regression_weights_all_experiments", do_save=do_save)
