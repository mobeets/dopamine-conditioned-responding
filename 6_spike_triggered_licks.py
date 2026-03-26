#%% imports

import numpy as np
import scipy.stats
from data_loaders import QianBurrell2024
from figure_helpers import plt, handle_fig
from regression_model import zscore_if_possible

#%% load sessions

D = QianBurrell2024()

#%% function helpers

def plot_licks_and_da(xs, licks_around_peak, da_around_peak, clr_licks='k', clr_da=(0.6,0.2,0.2), title='', ylims=None):
    """
    Plot licks and DA signal around DA peak.
    xs: time points relative to DA peak (in ms)
    licks_around_peak: 2D array of shape (n_trials, len(xs)) with licks data
    da_around_peak: 2D array of shape (n_trials, len(xs)) with DA signal data
    title: title for the plot
    """
    fig = plt.figure(figsize=(3, 3))
    # Licks (left y-axis)
    mus_licks = np.nanmean(licks_around_peak, axis=0)
    ses_licks = np.nanstd(licks_around_peak, axis=0) / np.sqrt(np.sum(~np.isnan(licks_around_peak), axis=0))

    ax1 = plt.gca()
    ax1.plot(xs, mus_licks, '.-', zorder=1, color=clr_licks, alpha=0.9, markersize=4, label='Licks')
    ax1.fill_between(xs, mus_licks - ses_licks, mus_licks + ses_licks, color=clr_licks, linewidth=0, alpha=0.2, zorder=-1)
    ax1.set_xlabel('Time rel. to DA peak (ms)')
    ax1.set_ylabel('Lick rate (Hz)', color=clr_licks)
    if ylims is not None and ylims[0] is not None:
        ax1.set_ylim(ylims[0])
    ax1.tick_params(axis='y', labelcolor=clr_licks)

    # DA (right y-axis)
    if da_around_peak is None:
        ax2 = None
    else:
        ax2 = ax1.twinx()
        mus_da = np.nanmean(da_around_peak, axis=0)
        ses_da = np.nanstd(da_around_peak, axis=0) / np.sqrt(np.sum(~np.isnan(da_around_peak), axis=0))
        ax2.plot(xs, mus_da, '.-', zorder=0, color=clr_da, alpha=0.9, markersize=4, label='DA')
        ax2.fill_between(xs, mus_da - ses_da, mus_da + ses_da, color=clr_da, linewidth=0, alpha=0.2, zorder=-1)
        ax2.set_ylabel('Dopamine activity (z-score)', color=clr_da, rotation=270, va='bottom')
        ax2.tick_params(axis='y', labelcolor=clr_da)
        ax2.spines['right'].set_visible(True)
        ax2.spines['right'].set_color(clr_da)     # match DA line color
        ax2.spines['right'].set_linewidth(1)   # adjust as needed
        if ylims is not None and ylims[1] is not None:
            ax2.set_ylim(ylims[1])

    # Vertical line
    ax1.axvline(0, color='k', linestyle='--', label='DA peak', zorder=-2, alpha=0.5)
    if ax2 is not None:
        ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    return fig, [ax1, ax2]

def get_spike_triggered_licks(D, mouse_name=None, trial_types=None, trial_period=None, day_indices=None, t_pre=30, t_post=30, da_threshold=3.0, da_threshold_max=None, da_refractory=10, bin_size=20, da_stats=None, trigger_on_licks_not_da=False, trials_only=False):
    """
    mouse_name is the name of the mouse (or list of mice) to analyze, or None for all mice
    trial_types is a list of trial types to include in the analysis
    trial_period is 'ITI', 'ISI', or None (for pre-CS, or post-CS-and-pre-US)
    day_indices is a list of day indices to include in the analysis
    t_pre is time steps before DA transient to look at licks
    t_post is time steps after DA transient to look at licks
    da_threshold is z-score threshold to identify DA transients
    da_refractory is time steps of refractory period after DA transient
    bin_size is time in ms of bin size for DA and licking rate (used to scale x-axis)
    """
    if mouse_name is None:
        sessions = D.Sessions
        title = 'All mice'
    elif type(mouse_name) is str:
        sessions = [session for session in D.Sessions if session[0]['mouse'] == mouse_name]
        title = f"Mouse {mouse_name}"
    else:
        sessions = [session for session in D.Sessions if session[0]['mouse'] in mouse_name]
        title = f"Mice {', '.join(mouse_name)}"
    if len(sessions) == 0:
        return None, None, None, None

    trial_type_name = ','.join(trial_types) if trial_types is not None else 'all'
    trial_period_name = trial_period if trial_period is not None else 'all'
    title += f" ({trial_period_name} time steps, {trial_type_name} trials)"

    all_da = []
    all_licks = []
    all_times = []
    tts = []
    trials = []
    for session in sessions:
        for t, trial in enumerate(session):
            if trial_types is not None and trial['trial_type'] not in trial_types:
                continue
            if day_indices is not None and trial['day_index'] not in day_indices:
                continue
            tts.append(trial['trial_type'])

            da = trial['DA']
            licks = trial['licks']
            if len(da) == 0 or len(licks) == 0:
                continue

            # include pre-CS, or post-CS-and-pre-US
            if trial_period is None:
                ix = np.ones_like(trial['times'], dtype=bool)
            elif trial_period == 'ITI':
                ix = (trial['times'] < 2.0)
            elif trial_period == 'ISI':
                ix = ((trial['times'] > 3.01) & (trial['times'] < 4.01))
            else:
                raise ValueError(f"Unknown trial period: {trial_period}")
            da = da[ix]
            licks = licks[ix]

            all_da.append(da)
            all_licks.append(licks)
            all_times.append(trial['times'][ix])
            trial_data = {'da': da, 'licks': licks, 'times': trial['times'][ix], 'trial_type': trial['trial_type'], 'day_index': trial['day_index'], 'mouse': trial['mouse']}
            trials.append(trial_data)

            # pad with nan so we don't go across trials
            nans = np.ones(2*da.shape[0])
            all_da.append(np.nan * nans)
            all_licks.append(np.nan * nans)
            all_times.append(np.nan * nans)

    if trials_only:
        return trials
    if len(all_da) == 0:
        return None, None, None, None
    
    all_da = np.hstack(all_da)
    all_licks = np.hstack(all_licks)
    all_times = np.hstack(all_times)
    if da_stats is not None:
        # use provided DA stats
        da_mean = da_stats['da_mean']
        da_std = da_stats['da_std']
        lick_mean = da_stats['lick_mean']
        lick_std = da_stats['lick_std']
    else:
        da_mean = np.nanmean(all_da)
        da_std = np.nanstd(all_da)
        lick_mean = np.nanmean(all_licks)
        lick_std = np.nanstd(all_licks)

    zscored_da = zscore_if_possible(all_da, np.array([da_mean]), np.array([da_std]))
    all_da = zscored_da.copy()  # keep the z-scored version for plotting
    ixpeak = zscored_da > da_threshold
    if da_threshold_max is not None:
        ixpeak &= (zscored_da <= da_threshold_max)
    peak_times = np.where(ixpeak)[0]
    if trigger_on_licks_not_da:
        zscored_licks = zscore_if_possible(all_licks, np.array([lick_mean]), np.array([lick_std]))
        ixpeak = zscored_licks > da_threshold
        if da_threshold_max is not None:
            ixpeak &= (zscored_licks <= da_threshold_max)
        peak_times = np.where(ixpeak)[0]
        # smooth_licks = np.convolve(all_licks, np.ones(5), mode='valid')
        # peak_times = np.where(smooth_licks >= 2)[0]

    # remove peaks too close together
    peak_times = peak_times[np.diff(np.hstack(([-np.inf], peak_times))) > da_refractory]

    licks_around_peak = []
    da_around_peak = []
    for peak in peak_times:
        # get licks around the peak
        start = max(0, peak - t_pre)
        end = min(len(all_licks), peak + t_post)

        c_licks = all_licks[start:end]
        c_da = all_da[start:end]
        if len(c_licks) < t_pre + t_post:
            continue
        licks_around_peak.append(c_licks)
        da_around_peak.append(c_da)
    
    if len(licks_around_peak) == 0:
        return None, None, None, None

    licks_around_peak = np.vstack(licks_around_peak)
    da_around_peak = np.vstack(da_around_peak)
    xs = bin_size * np.arange(-t_pre, t_post)
    return xs, licks_around_peak, da_around_peak, {'da_mean': da_mean, 'da_std': da_std, 'all_da': all_da, 'all_licks': all_licks, 'lick_mean': lick_mean, 'lick_std': lick_std}

def dprime_vector(Z1, Z2, ignore_nan=True):
    """
    Compute per-feature d' between Z1 and Z2.
    
    Parameters
    ----------
    Z1 : ndarray of shape (T1, N) or (T1,)
        First set of samples (T1 trials, N features).
    Z2 : ndarray of shape (T2, N) or (T2,)
        Second set of samples (T2 trials, N features).
    
    Returns
    -------
    dprimes : ndarray of shape (N,)
        d' value for each feature.
    """
    if len(Z1.shape) == 1:
        Z1 = Z1[:, None]
    if len(Z2.shape) == 1:
        Z2 = Z2[:, None]
    if not ignore_nan:
        raise Exception('not implemented')

    m1 = np.nanmean(Z1, axis=0)
    m2 = np.nanmean(Z2, axis=0)
    v1 = np.nanvar(Z1, axis=0, ddof=1)  # unbiased variance
    v2 = np.nanvar(Z2, axis=0, ddof=1)

    # pooled std
    s = np.sqrt(0.5 * (v1 + v2))
    
    # avoid division by zero
    s[s == 0] = np.nan  # set zero std to NaN to avoid division by zero
    dprimes = (m1 - m2) / s
    dprimes[s == 0] = np.nan
    
    return dprimes

def dprime_licks(D, mouse_name=None, trial_types=None, trial_period=None, day=None, t_for_dprime=200, da_stats=None, da_threshold=3.0, trigger_on_licks_not_da=False):
    xs, licks_around_peak, da_around_peak, da_stats_c = get_spike_triggered_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day_indices=[day], da_stats=da_stats, da_threshold=da_threshold, trigger_on_licks_not_da=trigger_on_licks_not_da)
    if xs is None:
        return np.nan, None, None, {'da_mean': np.nan, 'da_std': np.nan, 'lick_mean': np.nan, 'lick_std': np.nan}

    if trigger_on_licks_not_da:
        vs = da_around_peak
        reverse = True
    else:
        vs = licks_around_peak
        reverse = False
    ix_pre = (xs >= -t_for_dprime) & (xs < 0)
    ix_post = (xs >= 0) & (xs <= t_for_dprime)
    vs_pre = np.nansum(vs[:,ix_pre], axis=1)
    vs_post = np.nansum(vs[:,ix_post], axis=1)
    dp = dprime_vector(vs_post, vs_pre)[0]
    da_stats_c['licks_pre'] = vs[:,ix_pre]
    da_stats_c['licks_post'] = vs[:,ix_post]
    if reverse:
        # when triggering on licks, we expect DA to be higher prior than post
        dp = -dp
    return dp, vs_pre, vs_post, da_stats_c

#%% plot example trial of licking and DA during ITI

do_save = False

trial_types = None; trial_period = 'ITI'
clr_da = 'red'
mouse_names = np.unique([session[0]['mouse'] for session in D.Sessions])
mouse_name = mouse_names[7]
day_indices = [0]
ylims = None
trial_index = 50

xs, licks_around_peak, da_around_peak, cur_stats = get_spike_triggered_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day_indices=day_indices, da_stats=None, trigger_on_licks_not_da=False)
bin_size = np.median(np.diff(xs)).astype(int)

fig = plt.figure(figsize=(3,3))
plt.plot(xs, da_around_peak[trial_index], 'k-')
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
ymax = 7.5
for x,y in zip(xs, licks_around_peak[trial_index]):
    if y > 0:
        plt.plot([x, x], [ymax-0.5, ymax], 'k-', alpha=1.0, linewidth=1)

plt.xlabel('Time rel. to DA peak (ms)')
plt.ylabel('Dopamine activity (z-score)')
handle_fig(fig, f"spike_triggered_licks_example_trial", do_save=do_save)

#%% spike-triggered average of licking relative to DA transients

do_save = False
trial_types = None; trial_period = 'ITI'
block_name = 'conditioning'
day_indices = np.arange(5)
clr_da = 'black'
ylims = [[1.5,7.5],[-0.8,4.5]]

xs, licks_around_peak, da_around_peak, cur_stats = get_spike_triggered_licks(D, mouse_name=None, trial_types=trial_types, trial_period=trial_period, day_indices=day_indices, da_stats=None, trigger_on_licks_not_da=False)
bin_size = np.median(np.diff(xs)).astype(int)
fig, axs = plot_licks_and_da(xs, (1000/bin_size)*licks_around_peak, da_around_peak, clr_da='k', clr_licks='darkred', ylims=ylims)
handle_fig(fig, f"spike_triggered_licks_{block_name}", do_save=do_save)

#%% spike-triggered average of licking relative to DA transients

do_save = False

trial_types = None; trial_period = 'ITI'
day_indices = np.arange(5)
da_thresholds = np.arange(-1,6)

cmap = plt.get_cmap('Reds', len(da_thresholds))
fig = plt.figure()
for i, da_threshold in enumerate(da_thresholds):
    xs, licks_around_peak, da_around_peak, _ = get_spike_triggered_licks(D, mouse_name=None, trial_types=trial_types, trial_period=trial_period, day_indices=day_indices, da_stats=None, da_threshold=da_threshold, da_threshold_max=da_threshold+1, trigger_on_licks_not_da=False)

    bin_size = np.median(np.diff(xs)).astype(int)
    licks_around_peak = (1000/bin_size) * licks_around_peak  # convert to Hz
    mus = np.nanmean(licks_around_peak, axis=0)
    ses = np.nanstd(licks_around_peak, axis=0) / np.sqrt(np.sum(~np.isnan(licks_around_peak), axis=0))
    h = plt.plot(xs, mus, color=cmap(i), label=f"{da_threshold:.1f}–{da_threshold+1:.1f}", alpha=0.9, linewidth=1, markersize=4)
    plt.fill_between(xs, mus - ses, mus + ses, color=h[0].get_color(), linewidth=0, alpha=0.05, zorder=-1)
plt.xlabel('Time rel. to DA peak (ms)')
plt.ylabel('Lick rate (Hz)')
plt.legend(fontsize=6, title='DA peak range:', title_fontsize=6)
plt.ylim([1.0,6.5])

handle_fig(fig, f"spike_triggered_licks_dose", do_save=do_save)

#%% plot raster of licks

do_save = False
t_for_dprime = 250
trial_types = None; trial_period = 'ITI'
day = 0
mouse_name = mouse_names[7]

_, _, _, all_stats = get_spike_triggered_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day_indices=np.arange(10))
dp, licks_pre, licks_post, da_stats_c = dprime_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day=day, da_stats=all_stats, t_for_dprime=t_for_dprime)

stats, pval = scipy.stats.wilcoxon(licks_pre, licks_post)
print(f"Mouse {mouse_name}, day {day}, d' = {dp:.2f}")
print(f"Wilcoxon signed-rank test: statistic={stats:.3f}, p-value={pval:.3f}")
licks = np.hstack([da_stats_c['licks_pre'], da_stats_c['licks_post']])
licks[licks == 0] = np.nan # to make transparent

fig = plt.figure(figsize=(3, 3))
plt.imshow(-licks, cmap='gray', aspect='auto', interpolation='nearest')
xs = np.arange(0, t_for_dprime, bin_size)
times = np.hstack([-xs[::-1], xs[1:]])
plt.xticks(ticks=np.arange(0, licks.shape[1])[2::5]-0.5, labels=times[2::5], rotation=45)
plt.axvline(np.where(times == 0)[0][0]-0.5, ymin=0, ymax=1, color='gray', linestyle='-', linewidth=1, zorder=-1)
plt.xlabel('Time rel. to DA peak (ms)')
plt.ylabel('Trials')
plt.tight_layout()
handle_fig(fig, f"spike_triggered_licks_hist", do_save=do_save)

#%% plot d-prime of licks pre- and post-DA peak

do_save = False
days = np.arange(10)
trial_types = None; trial_period = 'ITI'
trigger_on_licks_not_da = False
t_for_dprime = 250
mouse_names = np.unique([session[0]['mouse'] for session in D.Sessions])

for group_name in ['degradation', 'cued reward']:
    all_dprimes = []
    all_licks_pre = []
    all_licks_post = []
    for mouse_name in mouse_names:
        if group_name == 'degradation' and '_0' not in mouse_name:
            continue
        elif group_name == 'cued reward' and '_C' not in mouse_name:
            continue
        elif group_name == 'all':
            pass

        # get da_mean for this mouse
        _, _, _, da_stats_mouse = get_spike_triggered_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day_indices=days, trigger_on_licks_not_da=trigger_on_licks_not_da)
        # da_stats_mouse = None

        dps = []
        for day in days:
            dp, licks_pre, licks_post, da_stats = dprime_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day=day, t_for_dprime=t_for_dprime, da_stats=da_stats_mouse,  trigger_on_licks_not_da=trigger_on_licks_not_da)
            all_licks_pre.append(licks_pre)
            all_licks_post.append(licks_post)
            dps.append(dp)
        dps = np.array(dps)
        all_dprimes.append(dps)

    print(f"{group_name} mice, days {days[0]}-{days[-1]}")
    all_dprimes = np.vstack(all_dprimes)
    all_dps = all_dprimes.flatten(); all_dps = all_dps[~np.isnan(all_dps)]
    stats, pval = scipy.stats.wilcoxon(all_dps)
    print(f"Wilcoxon ranksum test: statistic={stats:.3f}, p-value={pval:.3f}")

    fig = plt.figure()
    plt.plot(days+1, all_dprimes.T, '.-', color='k', alpha=0.5, linewidth=1, markersize=4)

    mus = np.nanmean(all_dprimes, axis=0)
    ses = np.nanstd(all_dprimes, axis=0) / np.sqrt(np.sum(~np.isnan(all_dprimes), axis=0))
    plt.plot(days+1, mus, 'k-', linewidth=2)
    plt.fill_between(days+1, mus - ses, mus + ses, color='k', alpha=0.2, zorder=-1, linewidth=0)

    plt.axhline(0, color='k', linestyle='--', alpha=0.5, zorder=-1)
    plt.xlabel('Day')
    if trigger_on_licks_not_da:
        plt.ylabel("d' of DA pre-lick peak vs post-DA peak")
    else:
        plt.ylabel("d' of licks after vs. before DA peak")
    plt.ylim([-1.11,1.11])
    handle_fig(fig, f"spike_triggered_licks_dprimes_{group_name}", do_save=do_save)

# %%
