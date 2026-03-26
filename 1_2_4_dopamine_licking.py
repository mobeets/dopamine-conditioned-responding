#%% imports

import numpy as np
from data_loaders import QianBurrell2024
from figure_helpers import plt, plot_trial_average_per_group, plot_trial_average_per_group_rel_to_first_lick, get_time_series_over_sessions, plot_time_series_over_trials, plot_time_series_over_trials_per_animal, plot_average_outcome_per_session, plot_correlations, plot_example_trial, handle_fig
from scipy.stats import ranksums

#%% load sessions

D = QianBurrell2024()

#%% avg dopamine and anticipatory lick rates in Cond, Deg, and CuedRew
# last day, Odor A

do_save = False
figs = plot_average_outcome_per_session(D, colors=['k','k','k'])
handle_fig(figs[0], 'qb_avg_DA', do_save=do_save)
handle_fig(figs[1], 'qb_avg_lick', do_save=do_save)
handle_fig(figs[2], 'qb_value', do_save=do_save)

#%% show example trials (DA and licking)

# (10, 1.23, 13)
# (-1, 4.24, 25)

do_save = False
trials = D.Sessions[23]
trials = [trial for trial in trials if trial['trial_type']=='go']
ctrials = np.vstack([(i, trial['CS_DA_peak'], trial['n_anticipatory_licks']) for i,trial in enumerate(trials) if trial['trial_type']=='go'])
inds = ctrials[np.argsort(ctrials[:,1]),0].astype(int)

fig = plot_example_trial(trials[inds[-1]], D.odor_onset, D.reward_onset)
handle_fig(fig, 'qb_example_trial_1', do_save=do_save)
fig = plot_example_trial(trials[inds[10]], D.odor_onset, D.reward_onset)
handle_fig(fig, 'qb_example_trial_2', do_save=do_save)

#%% show scatter of DA at CS vs anticipatory licks, for example session

do_save = False
high_da_color = 'darkred'
low_da_color = '#ff9999' # light red
trial_types = ('go', 'go_omit')
percentiles = (25,75)
day_index = 4
skipLastThirdOfTrials = False

session_inds = [i for i,trials in enumerate(D.Sessions) if trials[0]['day_index'] == day_index]
trials = D.Sessions[session_inds[4]]
print('Mouse', trials[0]['mouse'])

trials = [trial for trial in trials if trial['trial_type'] in trial_types]
ctrials = np.vstack([(i, trial['CS_DA_peak'], trial['n_anticipatory_licks']) for i,trial in enumerate(trials)])

if skipLastThirdOfTrials:
    print("Excluding last third of trials")
    n_half_trials = len(ctrials) // 3
    ctrials = ctrials[:-n_half_trials]

da_low = np.percentile(ctrials[:,1], percentiles[0])
da_high = np.percentile(ctrials[:,1], percentiles[1])
lick_low = np.percentile(ctrials[:,2], percentiles[0])
lick_high = np.percentile(ctrials[:,2], percentiles[1])
high_da_trials = ctrials[ctrials[:,1] > da_high,0].astype(int)
low_da_trials = ctrials[ctrials[:,1] < da_low,0].astype(int)
other_trials = ctrials[(ctrials[:,1] >= da_low) & (ctrials[:,1] <= da_high),0].astype(int)
high_trials = ctrials[(ctrials[:,1] > da_high) & (ctrials[:,2] > lick_high),0].astype(int)
low_trials = ctrials[(ctrials[:,1] < da_low) & (ctrials[:,2] < lick_low),0].astype(int)
low_trial = low_trials[-1]
high_trial = high_trials[-1]

stat, p_value = ranksums(ctrials[low_da_trials,2], ctrials[high_da_trials,2], alternative='less')
print(f"Anticipatory lick count (high DA > low DA): Wilcoxon rank-sum test: statistic={stat:0.3f}, p-value={p_value}")

fig = plt.figure(figsize=(3,3))
plt.scatter(ctrials[low_trial,1], ctrials[low_trial,2], 50, color=low_da_color, facecolors='none')
plt.scatter(ctrials[high_trial,1], ctrials[high_trial,2], 50, color=high_da_color, facecolors='none')
plt.vlines([da_low, da_high], ymin=0, ymax=20, color='gray', linestyle='--', linewidth=1)
plt.scatter(ctrials[other_trials,1], ctrials[other_trials,2], 10, color='black')
plt.scatter(ctrials[high_da_trials,1], ctrials[high_da_trials,2], 10, color=high_da_color)
plt.scatter(ctrials[low_da_trials,1], ctrials[low_da_trials,2], 10, color=low_da_color)
# plt.hlines([lick_low, lick_high], xmin=-1, xmax=22, color='gray', linestyle='--', linewidth=1)
plt.xlabel('DA peak at CS onset (z-score)')
plt.ylabel('# Anticipatory licks')
# plt.xticks([0,10,20])
# plt.yticks([0,5,10,15])
handle_fig(fig, 'qb_scatter_da_vs_licks', do_save=do_save)

fig = plot_example_trial(trials[low_trial], D.odor_onset, D.reward_onset, ymin=-2.5, ymax=14)
handle_fig(fig, 'qb_example_trial_1', do_save=do_save)
fig = plot_example_trial(trials[high_trial], D.odor_onset, D.reward_onset, ymin=-2.5, ymax=14)
handle_fig(fig, 'qb_example_trial_2', do_save=do_save)

#%% avg DA and lick rate during trial, split by DA quartile
# last day of Cond, Odor A

do_save = False
trial_types = ('go', 'go_omit')
day_indices = (4,)
group_key = 'CS_DA_peak'
percentiles = (25,75)
ignoreFirstHalfOfTrialsEachSession = False
ignoreLastHalfOfTrialsEachSession = False

fig, vals = plot_trial_average_per_group(D.Sessions, group_key=group_key, percentiles=percentiles, ykey='licks', trial_types=trial_types, day_indices=day_indices, odor_onset=D.odor_onset, reward_onset=D.reward_onset, ignoreFirstHalfOfTrialsEachSession=ignoreFirstHalfOfTrialsEachSession, ignoreLastHalfOfTrialsEachSession=ignoreLastHalfOfTrialsEachSession)
handle_fig(fig, 'qb_licks_psth_split', do_save=do_save)

x1,v1 = vals['low']
x2,v2 = vals['high']
v1s = v1[:,(x1 > 0) & (x1 < D.reward_onset-D.odor_onset)].sum(axis=1)
v2s = v2[:,(x2 > 0) & (x2 < D.reward_onset-D.odor_onset)].sum(axis=1)
stat, p_value = ranksums(v1s[~np.isnan(v1s)], v2s[~np.isnan(v2s)], alternative='less')
print(f"Anticipatory lick rate (high DA > low DA): Wilcoxon rank-sum test: statistic={stat:0.3f}, p-value={p_value}")

fig, vals = plot_trial_average_per_group(D.Sessions, group_key=group_key, percentiles=percentiles, ykey='licks', trial_types=trial_types, day_indices=day_indices, showFirstLickOnly=True, odor_onset=D.odor_onset, reward_onset=D.reward_onset, ignoreFirstHalfOfTrialsEachSession=ignoreFirstHalfOfTrialsEachSession, ignoreLastHalfOfTrialsEachSession=ignoreLastHalfOfTrialsEachSession)
handle_fig(fig, 'qb_firstlicks_psth_split', do_save=do_save)

stat, p_value = ranksums(vals['low'][1][:,0], vals['high'][1][:,0], alternative='greater')
print(f"Time of first anticipatory lick (low DA < high DA): Wilcoxon rank-sum test: statistic={stat:0.3f}, p-value={p_value}")

#%% lick rate, aligned to first anticipatory lick
# last day of Cond, Odor A

do_save = False
trial_types = ('go', 'go_omit')
day_indices = (4,)
group_key = 'CS_DA_peak'
percentiles = (25,75)
# percentiles = (50,50)

fig, vals = plot_trial_average_per_group_rel_to_first_lick(D.Sessions, group_key=group_key, percentiles=percentiles, trial_types=trial_types, day_indices=day_indices, odor_onset=D.odor_onset, reward_onset=D.reward_onset)
handle_fig(fig, 'qb_lickrate_psth_split', do_save=do_save)

x1,v1 = vals['low']
x2,v2 = vals['high']
v1s = np.nanmean(v1[:,x1 > 0], axis=1)
v2s = np.nanmean(v2[:,x2 > 0], axis=1)
stat, p_value = ranksums(v1s[~np.isnan(v1s)], v2s[~np.isnan(v2s)], alternative='less')
print(f"Anticipatory lick rate (high DA > low DA): Wilcoxon rank-sum test: statistic={stat:0.3f}, p-value={p_value}")

# fig = plot_trial_average_per_group_rel_to_first_lick(D.Sessions, trial_types=trial_types, day_indices=day_indices, odor_onset=D.odor_onset, reward_onset=D.reward_onset)
# handle_fig(fig, 'qb_lickrate_psth_mean')

#%% anticip. lick rate and DA at CS onset, on last day

do_save = False
trial_types = ('go', 'go_omit')
day_indices = [4]
da_window = D.da_bin_window
lick_window = D.lick_bin_windows['A']

all_vals, ylbls = get_time_series_over_sessions(D.Sessions, da_window, lick_window, D.mouse_names, day_indices=day_indices, trial_types=trial_types, subtract_baseline_spikes=False, use_mean_spikes=True)

rhos = []
for mouse_index in np.unique(all_vals[:,0].astype(int)):
    cvals = all_vals[all_vals[:,0] == mouse_index]
    rho = np.corrcoef(cvals[:,-2:].T)[0,1]
    rhos.append(rho)

# plot correlations
print(f"mean={np.mean(rhos):0.3f}, se={np.std(rhos)/np.sqrt(len(rhos)):0.3f}, sd={np.std(rhos):0.3f}, n={len(rhos)}")
fig = plot_correlations([np.hstack(rhos)], ylim=(0,1), figsize=(1.25,2.55))
handle_fig(fig, 'qb_correlations_lastday', do_save=do_save)

#%% anticip. lick rate and DA at CS onset, over days/trials
# Cond, Odor A

do_save = False
trial_types = ('go', 'go_omit')
day_indices = np.arange(5)
da_window = D.da_bin_window
lick_window = D.lick_bin_windows['A']

all_vals, ylbls = get_time_series_over_sessions(D.Sessions, da_window, lick_window, D.mouse_names, day_indices=day_indices, trial_types=trial_types, subtract_baseline_spikes=False, use_mean_spikes=True)

# grand average
fig = plot_time_series_over_trials(all_vals, ylbls)
handle_fig(fig, 'qb_lickrate_and_da_over_trials', do_save=do_save)

# example session
mouse_index = 0
fig = plot_time_series_over_trials(all_vals[all_vals[:,0] == mouse_index], ylbls)
handle_fig(fig, 'qb_lickrate_and_da_over_trials_example_v0', do_save=do_save)

# now get data with gaps between days, for clarity when showing individual sessions
gap_vals, ylbls = get_time_series_over_sessions(D.Sessions, da_window, lick_window, D.mouse_names, day_indices=day_indices, trial_types=trial_types, subtract_baseline_spikes=False, use_mean_spikes=True, day_offset=10)

# example sessions
example_mouse_indices = [0,5]
for mid in example_mouse_indices:
    fig = plot_time_series_over_trials(gap_vals[gap_vals[:,0] == mid], ylbls, experiment_name=f'Mouse {D.mouse_names[mid].split("_")[1]}')
    handle_fig(fig, f'qb_lickrate_and_da_over_trials_example_{mid}', do_save=do_save)

# all sessions
fig, rhos = plot_time_series_over_trials_per_animal(gap_vals, ylbls, mouse_names=D.mouse_names)
handle_fig(fig, 'qb_lickrate_and_da_over_trials_peranimal', do_save=do_save)

# plot correlations
print(f"mean={np.mean(rhos)}, se={np.std(rhos)/np.sqrt(len(rhos))}, sd={np.std(rhos)}, n={len(rhos)}")
fig = plot_correlations([np.hstack(rhos)], ylim=(0,1), figsize=(1.25,2.55))
handle_fig(fig, 'qb_correlations', do_save=do_save)
