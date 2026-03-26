#%% imports

import numpy as np
from data_loaders import QianBurrell2024, Eshel2016, Matsumoto2016, HyungGoo2020, Amo2024
from regression_model import get_feature_from_trials, get_data_per_session
from figure_helpers import handle_fig

#%% load sessions

D1 = QianBurrell2024()
D2 = Eshel2016(key='Variable-Expectation')
D3 = Matsumoto2016()
D4 = HyungGoo2020()
D5a = Amo2024(key='CC_DA')
D5b = Amo2024(key='CC_GCaMP')

#%% compare average anticipatory licking on low vs high DA trials

do_save = False
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
