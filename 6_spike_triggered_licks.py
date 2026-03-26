#%% imports

from figure_helpers import cond_clrs
from regression_model import cv_scores
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import statsmodels.api as sm
import scipy.stats

#%%

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

#%%

def get_spike_triggered_licks(D1, mouse_name=None, trial_types=None, trial_period=None, day_indices=None, t_pre=30, t_post=30, da_threshold=3.0, da_threshold_max=None, da_refractory=10, bin_size=20, da_stats=None, trigger_on_licks_not_da=False, trials_only=False):
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
        sessions = D1.Sessions
        title = 'All mice'
    elif type(mouse_name) is str:
        sessions = [session for session in D1.Sessions if session[0]['mouse'] == mouse_name]
        title = f"Mouse {mouse_name}"
    else:
        sessions = [session for session in D1.Sessions if session[0]['mouse'] in mouse_name]
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

#%%

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

def dprime_licks(D1, mouse_name=None, trial_types=None, trial_period=None, day=None, t_for_dprime=200, da_stats=None, da_threshold=3.0, trigger_on_licks_not_da=False):
    xs, licks_around_peak, da_around_peak, da_stats_c = get_spike_triggered_licks(D1, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day_indices=[day], da_stats=da_stats, da_threshold=da_threshold, trigger_on_licks_not_da=trigger_on_licks_not_da)
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

#%%

def get_dprime_cs_licks(D, mouse_name=None, trial_types=None, day_indices=None):
    """
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
        return None

    dps = []
    tts = []
    for session in sessions:
        xs = []
        ys = []
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
            
            t_start_da = D.odor_onset
            t_offset_da = 1.0
            t_start_licks = D.odor_onset
            t_offset_licks = 2.5

            # t_start_da = 0.0
            # t_offset_da = 1.0
            # t_start_licks = 0.0
            # t_offset_licks = 2.0

            ix_da = (trial['times'] >= t_start_da) & (trial['times'] <= t_start_da + t_offset_da)
            da = da[ix_da].mean()
            ix_licks = (trial['times'] >= t_start_licks) & (trial['times'] <= t_start_licks + t_offset_licks)
            licks = licks[ix_licks].mean()
            xs.append(da)
            ys.append(licks)
        if len(xs) == 0 or len(ys) == 0:
            continue
        xs = np.hstack(xs)
        x_thresh = np.nanmedian(xs)
        ys = np.hstack(ys)
        dp = dprime_vector(ys[xs >= x_thresh], ys[xs < x_thresh])[0]
        dps.append(dp)
    if len(dps) == 0:
        return None
    dps = np.array(dps)
    return dps

D = D1
days = np.arange(10)
trial_types = ('go', 'go_omit')
# trial_types = ('c_reward', 'c_omit')
# trial_types = None

mouse_names = np.unique([session[0]['mouse'] for session in D.Sessions])
clrs = ['#0076BA', '#FF7F0E']  # blue, orange
for i, nm in enumerate(['_0', '_C']):
    if nm != 'all':
        mouse_name = [mouse for mouse in mouse_names if nm in mouse]
    else:
        mouse_name = None
    print(nm, len(mouse_name))
    clr = clrs[i]
    mus = []
    ses = []
    for day in days:
        dps = get_dprime_cs_licks(D, mouse_name=mouse_name, trial_types=trial_types, day_indices=[day])
        if dps is None or len(dps) == 0:
            mus.append(np.nan)
            ses.append(np.nan)
            continue
        mus.append(np.mean(dps))
        ses.append(np.std(dps) / np.sqrt(len(dps)))  # standard error of the mean
    mus = np.array(mus)
    ses = np.array(ses)
    plt.plot(days, mus, 'o-', color=clr, label=nm, markersize=4, linewidth=1, zorder=1)
    plt.fill_between(days, mus - ses, mus + ses, color=clr, alpha=0.2, zorder=-1)
plt.xlabel('day')
plt.ylabel("d' of licks (high CS DA vs low CS DA)")

#%% plot example trial of licking and DA during ITI

do_save = False
D = D1

trial_types = None; trial_period = 'ITI'

clr_da = 'red'
mouse_names = np.unique([session[0]['mouse'] for session in D1.Sessions])
mouse_name = mouse_names[7]
day_indices = [0]

ylims = None # [[1.5,7.5],[-0.8,4.5]]

xs, licks_around_peak, da_around_peak, cur_stats = get_spike_triggered_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day_indices=day_indices, da_stats=None, trigger_on_licks_not_da=False)
bin_size = np.median(np.diff(xs)).astype(int)

trial_index = 50

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
D = D1

# trial_types = ('background',); trial_period = None
trial_types = None; trial_period = 'ITI'

block_name = 'conditioning'
# block_name = 'degradation'
# block_name = 'example'

clr_da = 'black'
mouse_names = np.unique([session[0]['mouse'] for session in D1.Sessions])
mouse_name = None
if block_name == 'conditioning':
    day_indices = np.arange(5)
    # clr_da = cond_clrs[0]
elif block_name == 'degradation':
    day_indices = np.arange(5,10)
    # clr_da = cond_clrs[1]
elif block_name == 'example':
    day_indices = [0]
    # clr_da = cond_clrs[0]
    mouse_name = mouse_names[7]
else:
    day_indices = None

trigger_on_licks_not_da = False
# ylims = [[-2,25],[-1,5]]
# ylims = [[0,10],[-1,5]]
ylims = [[1.5,7.5],[-0.8,4.5]]
if block_name == 'example':
    ylims = [[0, 13], None]
# ylims = None

# _, _, _, all_stats = get_spike_triggered_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day_indices=np.arange(10), trigger_on_licks_not_da=trigger_on_licks_not_da)
all_stats = None

xs, licks_around_peak, da_around_peak, cur_stats = get_spike_triggered_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day_indices=day_indices, da_stats=all_stats, trigger_on_licks_not_da=trigger_on_licks_not_da)
bin_size = np.median(np.diff(xs)).astype(int)
if block_name == 'example':
    da_around_peak = None
fig, axs = plot_licks_and_da(xs, (1000/bin_size)*licks_around_peak, da_around_peak, clr_da='k', clr_licks='darkred', ylims=ylims)
handle_fig(fig, f"spike_triggered_licks_{block_name}", do_save=do_save)

#%% spike-triggered average of licking relative to DA transients

do_save = False
D = D1

# trial_types = ('background',); trial_period = None
trial_types = None; trial_period = 'ITI'

day_indices = np.arange(4)

mouse_names = np.unique([session[0]['mouse'] for session in D1.Sessions])
# mouse_name = mouse_names[5]
mouse_name = None
# mouse_name = [mouse for mouse in mouse_names if '_C' in mouse]

da_thresholds = np.arange(-1,6)
cmap = plt.get_cmap('Reds', len(da_thresholds))
fig = plt.figure()
for i, da_threshold in enumerate(da_thresholds):
    xs, licks_around_peak, da_around_peak, _ = get_spike_triggered_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day_indices=day_indices, da_stats=None, da_threshold=da_threshold, da_threshold_max=da_threshold+1, trigger_on_licks_not_da=False)

    bin_size = np.median(np.diff(xs)).astype(int)
    licks_around_peak = (1000/bin_size) * licks_around_peak  # convert to Hz
    mus = np.nanmean(licks_around_peak, axis=0)
    ses = np.nanstd(licks_around_peak, axis=0) / np.sqrt(np.sum(~np.isnan(licks_around_peak), axis=0))
    h = plt.plot(xs, mus, color=cmap(i), label=f"{da_threshold:.1f}–{da_threshold+1:.1f}", alpha=0.9, linewidth=1, markersize=4)
    plt.fill_between(xs, mus - ses, mus + ses, color=h[0].get_color(), linewidth=0, alpha=0.05, zorder=-1)
plt.xlabel('Time rel. to DA peak (ms)')
plt.ylabel('Lick rate (Hz)')
plt.legend(fontsize=6, title='DA peak range:', title_fontsize=6)
plt.ylim([1.5,7.5])

handle_fig(fig, f"spike_triggered_licks_dose", do_save=do_save)

#%% plot raster of licks

do_save = False
D = D1
t_for_dprime = 250

# trial_types = ('background',); trial_period = None
trial_types = None; trial_period = 'ITI'
day = 0
mouse_name = mouse_names[7]

_, _, _, all_stats = get_spike_triggered_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day_indices=np.arange(10))
# all_stats = None

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
D = D1

group_name = 'degradation'
group_name = 'cued reward'
group_name = 'all'

# trial_types = ('background',); trial_period = None
trial_types = None; trial_period = 'ITI'
# trial_types = ('go', 'go_omit'); trial_period = 'ISI'

title = ''
if trial_types is not None:
    title += '/'.join(trial_types) + ' trials'
if trial_period is not None:
    if title:
        title += ', '
    title += f"during {trial_period}"

trigger_on_licks_not_da = False
t_for_dprime = 250
days = np.arange(5)
mouse_names = np.unique([session[0]['mouse'] for session in D1.Sessions])
# mouse_names = [mouse_names[7]]
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
    # da_stats_mouse = {'da_mean': 0.0, 'da_std': 1.0}
    # da_stats_mouse = None

    dps = []
    for day in days:
        dp, licks_pre, licks_post, da_stats = dprime_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day=day, t_for_dprime=t_for_dprime, da_stats=da_stats_mouse,  trigger_on_licks_not_da=trigger_on_licks_not_da)
        # if show_da_mean:
        #     dp = da_stats['da_mean']
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
# plt.title(title)
plt.ylim([-1.11,1.11])
# if not show_da_mean:
#     plt.ylim([-1.01,2.01])
# else:
#     plt.ylabel('DA mean (z-score)')
handle_fig(fig, f"spike_triggered_licks_dprimes_{group_name}", do_save=do_save)

#%% show logistic with different slopes and intercepts

# plot logistic function
xs = np.linspace(-4, 4, 100)
# params = [(0.1, 0), (0.5, 0), (0.7, 0), (0.7, 1.0), (1.0, 0), (2.0, 0)]
params = [(0.5, 0), (2, 0), (2, 1)]

plt.figure()
for beta,intercept in params:
    f = lambda x: 1 / (1 + np.exp(-beta * (x - intercept)))
    plt.plot(xs, f(xs), '-', alpha=0.5, markersize=1, label=f'$\\beta$={beta}, $x_0$={intercept}')
plt.xlabel('DA (z-score)')
plt.ylabel('P(lick)')
plt.legend(fontsize=8)
plt.show()

#%% show logistic with different betas

# plot logistic function
xs = np.linspace(-5, 5, 100)
betas = [0.1, 0.2, 0.5, 0.7, 1.0, 2.0]

plt.figure()
for beta in betas:
    f = lambda x: 1 / (1 + np.exp(-beta * x))
    plt.plot(xs, f(xs), '-', alpha=0.5, markersize=1, label=f'$\\beta$={beta}')
plt.xlabel('DA (z-score)')
plt.ylabel('P(lick)')
plt.title('Logistic function')
plt.legend(fontsize=8)
plt.show()

ntrials = 20000
x = 3 * np.random.randn(ntrials)
# x = np.random.standard_t(df=3, size=ntrials)

da_thresh = 3.0

plt.figure()
for beta in betas:
    f = lambda x: 1 / (1 + np.exp(-beta * x))
    y = f(x) + np.random.randn(ntrials) * 0.5  # add some noise
    ix = x > da_thresh
    dp = dprime_vector(y[ix], y[~ix])[0]
    plt.plot(beta, dp, '.')
plt.xlabel('$\\beta$')
plt.ylabel("d' of P(lick) post-DA vs pre-DA peak")

#%% show d' of logistic with different intercepts

intercepts = np.linspace(-2, 2, 11)
beta = 0.1

xs = np.linspace(-5, 5, 100)
plt.figure()
for intercept in intercepts:
    f = lambda x: 1 / (1 + np.exp(-beta * (x - intercept)))
    plt.plot(xs, f(xs), '-', alpha=0.5, markersize=1, label=f'intercept={intercept:.2f}')
plt.xlabel('DA (z-score)')
plt.ylabel('P(lick)')
plt.title('Logistic function')
plt.legend(fontsize=8)
plt.show()

ntrials = 20000
x = 3 * np.random.randn(ntrials)
plt.figure()
for intercept in intercepts:
    f = lambda x: 1 / (1 + np.exp(-beta * (x - intercept)))
    y = f(x) + np.random.randn(ntrials) * 0.5  # add some noise
    ix = x > da_thresh
    dp = dprime_vector(y[ix], y[~ix])[0]
    plt.plot(intercept, dp, '.')
plt.xlabel('intercept')
plt.ylabel("d' of P(lick) post-DA vs pre-DA peak")

#%% show d' of logistic with different sigmas

ntrials = 20000
x = 3 * np.random.randn(ntrials)
sigmas = np.linspace(0.1, 2.0, 21)
beta = 0.5
plt.figure()
for sigma in sigmas:
    f = lambda x: 1 / (1 + np.exp(-beta * x))
    y = f(x) + np.random.randn(ntrials) * sigma  # add some noise
    ix = x > da_thresh
    dp = dprime_vector(y[ix], y[~ix])[0]
    plt.plot(sigma, dp, '.')
plt.xlabel('$\sigma$')
plt.ylabel("d' of P(lick) post-DA vs pre-DA peak")

#%%

def logistic_X_to_y(Xc, y, t_offset=0, n_window=1, N_min=None, do_sum=False):
    """
    """
    if t_offset > 0:
        Xc = Xc[:-t_offset]
        y = y[t_offset:]
    elif t_offset < 0:
        Xc = Xc[-t_offset:]
        y = y[:t_offset]

    # make X which contains history of X from last n_window time steps
    X = np.zeros((len(Xc) - n_window + 1, n_window))
    for t in range(n_window-1, len(Xc)):
        X[t-n_window+1,:] = Xc[(t-n_window+1):(t+1)]
    if do_sum:
        X = X.sum(axis=1, keepdims=True)
    y = y[(n_window-1):]  # adjust y to match X

    ix = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
    X = X[ix]; y = y[ix]
    X = zscore_if_possible(X)
    data = {'X': X, 'y': y}

    kf = KFold(n_splits=10, shuffle=True, random_state=555)
    lls = []
    for train_idx, test_idx in kf.split(data['X']):
        X_train, X_test = data['X'][train_idx], data['X'][test_idx]
        y_train, y_test = data['y'][train_idx], data['y'][test_idx]
        model = LogisticRegression(max_iter=1000, random_state=42)
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue

        if N_min is not None and len(X_train) > N_min:
            ix = np.random.choice(len(X_train), N_min, replace=False)
            X_train = X_train[ix]
            y_train = y_train[ix]
        model.fit(X_train, y_train)
        LL = -log_loss(y_test, model.predict_proba(X_test)[:,1])
        lls.append(LL)
    # score = np.nanmean(lls)
    model.fit(data['X'], data['y'])
    # coef = model.coef_[0][0]
    # intercept = model.intercept_[0]
    return model, lls

#%% fit logistic regression to predict licks from DA, and compare to d'

D = D1#_raw

# trial_types = ('background',); trial_period = None
trial_types = None; trial_period = 'ITI'

# day_indices = [0,4,9]
day_indices = np.arange(10)
# day_indices = [0]

mouse_names = np.unique([session[0]['mouse'] for session in D1.Sessions])
mouse_name = mouse_names[7]

# color map for days
cmap = plt.get_cmap('viridis', len(day_indices))

# get stats to use for all sessions
_, _, _, all_stats = get_spike_triggered_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day_indices=day_indices)

t_offset = 0 # offset between DA and licks, in time steps
n_window = 1 # window size, in time steps, for counting licks

params = []
plt.figure()
for day in day_indices:
    dp, _, _, stats = dprime_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day=day, t_for_dprime=200, da_stats=all_stats)
    if 'all_da' not in stats or 'all_licks' not in stats:
        continue

    Xc = stats['all_da']
    y = stats['all_licks']

    model, lls = logistic_X_to_y(Xc, y, t_offset=t_offset, n_window=n_window, N_min=None, do_sum=False)
    score = np.nanmean(lls)
    coef = model.coef_[0][0]
    intercept = model.intercept_[0]
    
    params.append((dp, coef, intercept, score))

    if day not in [0,4,9]:
        continue

    # plot logistic model using model.coef_ and model.intercept_
    f = lambda h: 1 / (1 + np.exp(-coef * (h - intercept)))
    h = np.linspace(-3, 3, 100)
    plt.plot(h, f(h), '-', alpha=0.5, label=f'Day {day}', color=cmap(day))
    plt.ylim([0,1])
plt.xlabel('DA (z-score)')
plt.ylabel('P(lick)')
plt.legend(fontsize=6)
plt.show()

params = np.vstack(params)
cparams = params.copy()
# cparams = cparams[:,:-1]
param_names = ["d'", '$\\beta$', 'intercept', 'LL']
print(np.nanmean(cparams, axis=0))

plt.figure(), plt.plot(cparams[:,-1]), plt.ylim([-2, 0]), plt.show()

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
for i in range(cparams.shape[1]):
    plt.plot(cparams[:, i], '.-', alpha=0.5, label=param_names[i])
plt.legend(fontsize=6)
plt.xlabel('day index')
plt.subplot(1, 2, 2)
cparams = cparams.copy()
# replace nan with column mean, to get correlation matrix
for i in range(cparams.shape[1]):
    cparams[np.isnan(cparams[:, i]), i] = np.nanmean(cparams[:,i])   
plt.imshow(np.corrcoef(cparams.T), cmap='RdBu')
plt.xticks(range(len(param_names)), param_names, rotation=45)
plt.yticks(range(len(param_names)), param_names, rotation=45)
plt.clim([-1,1])
plt.colorbar(fraction=0.03)
plt.tight_layout()
plt.show()

#%% fit logistic regression for all mice

N_min = None
D = D1#_raw

# trial_types = ('background',); trial_period = None
trial_types = None; trial_period = 'ITI'

# day_indices = [0,4,9]
day_indices = np.arange(10)
# day_indices = [0]

mouse_names = np.unique([session[0]['mouse'] for session in D.Sessions])
cmap = plt.get_cmap('viridis', len(day_indices))

t_offset = 1
n_window = 1 # window size, in time steps, for counting licks

plt.figure(figsize=(9, 9))
all_params = []
for i, mouse_name in enumerate(mouse_names):

    # get stats to use for all sessions
    _, _, _, all_stats = get_spike_triggered_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day_indices=day_indices)

    plt.subplot(4,4,i+1)
    params = []
    for day in day_indices:
        dp, _, _, stats = dprime_licks(D, mouse_name=mouse_name, trial_types=trial_types, trial_period=trial_period, day=day, t_for_dprime=200, da_stats=all_stats)
        if 'all_da' not in stats or 'all_licks' not in stats:
            continue

        Xc = stats['all_da']
        y = stats['all_licks']

        model, lls = logistic_X_to_y(Xc, y, t_offset=t_offset, n_window=n_window, N_min=None)
        score = np.nanmean(lls)
        coef = model.coef_[0][0]
        intercept = model.intercept_[0]
        params.append((day, dp, coef, intercept, score))
        # todo: if n_window > 1, coef is an array

        if day not in [0,4,8]:
            continue

        # plot logistic model using model.coef_ and model.intercept_
        f = lambda h: 1 / (1 + np.exp(-coef * (h - intercept)))
        h = np.linspace(-3, 3, 100)
        plt.plot(h, f(h), '-', alpha=0.5, label=f'Day {day}', color=cmap(day))
    
    params = np.vstack(params)
    all_params.append(params)
    LL = np.median(params[:,-1])
    plt.xlabel('DA (z-score)')
    plt.ylabel('P(lick)')
    plt.title(f"{mouse_name}, {LL=:0.1f}")
    if i == 0:
        plt.legend(fontsize=6)
    params = np.vstack(params)
plt.tight_layout()

#%% plot avg logistic regression parameters per mouse group

pnm = "d'"; pi = 1
pnm = '$\\beta$'; pi = 2
pnm = 'intercept'; pi = 3
# pnm = 'LL'; pi = 4

x1 = []
x2 = []
for i, param in enumerate(all_params):
    p = param#[[0,4,9]]
    v = p[:,pi]
    if '_C' in mouse_names[i]:
        x2.append(v)
    else:
        x1.append(v)
x1 = np.vstack(x1)
x2 = np.vstack(x2)
x1[x1 > 1e4] = np.nan
x2[x2 > 1e4] = np.nan

lbls = ['Degradation', 'Cued Reward']
for i,x in enumerate([x1,x2]):
    mus = np.nanmean(x, axis=0)
    xs = np.arange(len(mus))
    ses = np.nanstd(x, axis=0) / np.sqrt(np.sum(~np.isnan(x), axis=0))
    lbs = mus - ses; ubs = mus + ses

    mus = np.nanmedian(x, axis=0)
    lbs = np.nanpercentile(x, 25, axis=0); ubs = np.nanpercentile(x, 75, axis=0)

    h = plt.plot(xs, mus, label=lbls[i])
    plt.fill_between(xs, lbs, ubs, color=h[0].get_color(), alpha=0.2, zorder=-1, linewidth=0)

plt.legend(fontsize=8)
plt.ylabel(pnm)
plt.xlabel('day index')

#%% play around with t_offset and n_window for LL of the fits

mouse_name = 'FgDA_09'
day = 8
t_offset = 0
n_window = 1
# N_min = 1500; nreps = 5
N_min = None; nreps = 1

xs, licks_around_peak, da_around_peak, cur_stats = get_spike_triggered_licks(D, mouse_name=mouse_name, trial_types=None, trial_period='ITI', day_indices=[day])
Xc = cur_stats['all_da'].copy()
yc = cur_stats['all_licks'].copy()
bin_size = D.bin_size

# default color cycle
clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']

for j, n_window in enumerate([1,2,3]):
    clr = clrs[j % len(clrs)]
    for t_offset in np.arange(-20,30):
        X = Xc.copy()
        y = yc.copy()
        if t_offset > 0:
            X = X[:-t_offset]
            y = y[t_offset:]
        elif t_offset < 0:
            X = X[-t_offset:]
            y = y[:t_offset]
        if n_window > 1:
            # count mean licks in a window
            y = np.convolve(y, np.ones(n_window), mode='valid')
            X = X[:len(y)]

        ix = ~np.isnan(X) & ~np.isnan(y)
        X = X[ix]
        y = y[ix]
        X = zscore_if_possible(X)#[:,None]

        # resample N points
        if N_min is not None and len(X) < N_min:
            continue
        Xa = X.copy()
        ya = y.copy()

        rep_scores = []
        for _ in range(nreps):
            if N_min is not None and len(Xa) > N_min:
                ix = np.random.choice(len(Xa), N_min, replace=False)
                X = Xa[ix].copy()
                y = ya[ix].copy()
            else:
                X = Xa.copy()
                y = ya.copy()

            # Fit binomial regression
            X_ = sm.add_constant(X)
            data = {'X': X_, 'y': y}
            kf = KFold(n_splits=10, shuffle=True, random_state=555)
            scores = []
            tr_scores = []
            for train_idx, test_idx in kf.split(data['X']):
                X_train, X_test = data['X'][train_idx], data['X'][test_idx]
                y_train, y_test = data['y'][train_idx], data['y'][test_idx]
                
                model = sm.GLM(y_train / n_window, X_train, family=sm.families.Binomial(), freq_weights=n_window*np.ones_like(y_train))
                result = model.fit()
                # Compute log-likelihood on test data (total, then average)
                p_test = result.predict(X_test)
                ll_test = np.sum(y_test * np.log(p_test) +  (n_window - y_test) * np.log(1 - p_test))
                mean_ll_test = ll_test / len(y_test)
                scores.append(mean_ll_test)
                tr_scores.append(result.llf / result.nobs)  # also store the log-likelihood of the fitted model
            
            if len(scores) == 0:
                continue
            score = np.mean(scores)
            coef = result.params[1]
            intercept = result.params[0]
            rep_scores.append(score)

        # print(t_offset, len(X_train))
        score = np.mean(rep_scores) if rep_scores else np.nan
        plt.plot(t_offset * bin_size, score, '.', color=clr)
        # plt.plot(n_window, score, 'o')
plt.xlabel('offset (ms)')
plt.ylabel('test log-likelihood')

#%%

N_min = None # 3000
mouse_name = 'FgDA_09'
day_indices = [0,1,2,3,4]
# day_indices = [0]
t_offsets = np.arange(-5,5)
n_windows = [1,5,8,10]#3,5,10,15,20]

LLs = {}
xs, licks_around_peak, da_around_peak, cur_stats = get_spike_triggered_licks(D, mouse_name=mouse_name, trial_types=None, trial_period='ITI', day_indices=day_indices)

for j, n_window in enumerate(n_windows):
    lls = []
    for t_offset in t_offsets:
        Xc = cur_stats['all_da'].copy()
        y = cur_stats['all_licks'].copy()

        if t_offset > 0:
            Xc = Xc[:-t_offset]
            y = y[t_offset:]
        elif t_offset < 0:
            Xc = Xc[-t_offset:]
            y = y[:t_offset]

        # make X which contains history of X from last n_window time steps
        X = np.zeros((len(Xc) - n_window + 1, n_window))
        for t in range(n_window-1, len(Xc)):
            X[t-n_window+1,:] = Xc[(t-n_window+1):(t+1)]
        y = y[(n_window-1):]  # adjust y to match X

        ix = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
        X = X[ix]; y = y[ix]
        X = zscore_if_possible(X)

        data = {'X': X, 'y': y}
        kf = KFold(n_splits=10, shuffle=True, random_state=555)
        scores = []
        for train_idx, test_idx in kf.split(data['X']):
            X_train, X_test = data['X'][train_idx], data['X'][test_idx]
            y_train, y_test = data['y'][train_idx], data['y'][test_idx]
            model = LogisticRegression(max_iter=1000, random_state=42)

            if N_min is not None and len(X_train) > N_min:
                ix = np.random.choice(len(X_train), N_min, replace=False)
                X_train = X_train[ix]
                y_train = y_train[ix]
            model.fit(X_train, y_train)
            LL = -log_loss(y_test, model.predict_proba(X_test)[:,1])
            scores.append(LL)
        score = np.mean(scores)
        lls.append(score)
    if n_window not in LLs:
        LLs[n_window] = []
    LLs[n_window].append(lls)

clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']
for j, (n_window, lls) in enumerate(LLs.items()):
    lls = np.vstack(lls)
    mus = lls.mean(axis=0)
    ses = lls.std(axis=0) / np.sqrt(np.sum(~np.isnan(lls), axis=0))
    plt.plot(t_offsets, mus, '.-', color=clrs[j % len(clrs)])
    plt.fill_between(t_offsets, mus - ses, mus + ses, color=clrs[j % len(clrs)], alpha=0.2, zorder=-1, linewidth=0)
plt.xlabel('offset (ms)')
plt.ylabel('test log-likelihood')

#%% licks on unpred water

# trialtype = 'go'; offset = 0
# trialtype = 'c_reward'; offset = 5
trialtype = 'unpred_water'; offset = 5

df = D.dfl[D.dfl.trialtype == trialtype].copy()
# dfa = df[(df.mouse == 'FgDA_01')]
dfa = df

days = np.array([0,1,2,3,4])+offset
cmap = plt.get_cmap('viridis', len(days))
for i, day in enumerate(days):
    dfc = dfa[(dfa.day >= day) & (dfa.day <= day+0)]
    trs = dfc.trialnumber.unique()
    bins = np.arange(0,20,0.25)
    vs_pre = []
    vs_post = []
    first_lick_times = []
    for tr, dfcc in dfc.groupby('trialnumber'):
        clicks = dfcc.lick.values
        if ~np.any(clicks >= 5.5):
            continue
        t_first_lick = np.where(clicks >= 5.5)[0][0]
        first_lick_times.append(clicks[t_first_lick] - 5.5)
        
        c_clicks = clicks[:(t_first_lick+1)]
        counts = np.histogram(c_clicks, bins=bins)[0]
        vs_pre.append(counts)

        c_clicks = clicks[(t_first_lick+1):]
        counts = np.histogram(c_clicks, bins=bins)[0]
        vs_post.append(counts)

    # plt.plot(bins[:-1] - 5.5, np.nanmean(np.vstack(vs_pre), axis=0), '-', linewidth=1, alpha=0.5, color=cmap(i))
    # plt.plot(bins[:-1] - 5.5, np.nanmean(np.vstack(vs_post), axis=0), '-', linewidth=1, alpha=0.5, color=cmap(i), label=f'Day {day}')
    # plt.xlabel('time rel. to first lick (s)')
    # plt.ylabel('lick rate (Hz)')
    # continue

    bins = np.arange(0, 1, 0.05)
    counts = np.histogram(first_lick_times, bins=bins)
    plt.plot(counts[1][:-1], 100*counts[0]/counts[0].sum(), '-', color=cmap(i), label=f'Day {day}', linewidth=1, alpha=0.5)
    plt.xlabel('time of first consummative lick (s)')
    plt.ylabel('percent of trials')

plt.legend(fontsize=6)
plt.title(trialtype)

# %%
