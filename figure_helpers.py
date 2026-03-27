#%% imports

import os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold
from data_helpers import zscore_if_possible, get_data_per_session
import scipy.stats

import matplotlib as mpl
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = (3,3)

DA_color = 'black'

def handle_fig(fig, filename, do_save=True, overwrite=True, ext='pdf', savedir='figures'):
    if not do_save:
        return
    outfile = os.path.join(savedir, filename + '.' + ext)
    if os.path.exists(outfile):
        extra = 'Overwriting.' if overwrite else 'Will not overwrite.'
        print(f"WARNING: {outfile=} already exists. {extra}")
        if not overwrite:
            return
    fig.savefig(outfile, bbox_inches='tight')

#%% for showing bulk averages per condition

cond_clrs = {0: '#4E9E7C', 1: '#E88355', 2: '#7571AE', 3: 'gray'}
def plot_average_outcome_per_session(D, colors=cond_clrs):
    if 'Qian' not in D.exp_name:
        raise Exception('Invalid experiment name.')
    
    trial_types = ('go', 'go_omit')
    da_window = D.da_bin_window
    lick_window = D.lick_bin_windows['A']
    subtract_baseline_spikes = False
    use_mean_spikes = False

    session_types = [{'name': 'Conditioning', 'day_indices': (3,4,), 'mouse_match': '', 'color': colors[0]}, 
        {'name': 'Degradation', 'day_indices': (8,9,), 'mouse_match': '_0', 'color': colors[1]},
        {'name': 'Cued Reward', 'day_indices': (8,9,), 'mouse_match': '_C', 'color': colors[2]}]

    pts = []
    lick_to_hz = 1000 / D.bin_size
    for session_type in session_types:
        cpts = []
        for i, session in enumerate(D.Sessions):
            if session_type['mouse_match'] not in session[0]['mouse']:
                continue
            if session[0]['day_index'] not in session_type['day_indices']:
                continue
            # get DA and licking per trial
            X, Y = get_data_per_session(session, da_window, lick_window, trial_types=trial_types, zscore_spikes=False, use_mean_spikes=use_mean_spikes, subtract_baseline_spikes=subtract_baseline_spikes)
            cpts.append((X.mean(), lick_to_hz * Y.mean()))
        cpts = np.vstack(cpts)
        pts.append(cpts)

    xs = np.arange(len(session_types))
    lbls = [session_type['name'] for session_type in session_types]

    figs = []
    for di, ylbl in [(0, 'CS Dopamine activity (z-score)'), (1, 'Anticipatory lick rate (Hz)'), (2, 'Objective CS value (a.u.)')]:
        fig = plt.figure(figsize=(1.5,3))
        for i,x in enumerate(xs):
            if di < 2:
                cpts = pts[i][:,di]
                mu = np.mean(cpts); se = np.std(cpts) / np.sqrt(len(cpts))
            else:
                # Empirical value
                mu = 1; se = 0
            clr = session_types[i]['color']
            h = plt.plot(x, mu, 'o', color=clr, markersize=10, markeredgecolor='k')
            # plt.plot(x*np.ones(len(cpts)), cpts, '.', color=h[0].get_color(), zorder=-2)
            plt.plot([x,x], [mu-se, mu+se], 'k-', zorder=-1)
        plt.xticks(ticks=xs, labels=lbls, rotation=45, ha='right')
        plt.ylabel(ylbl)
        plt.xlim([-0.5, 2.5])
        if di == 2:
            plt.ylim([0, 1.2])
            plt.yticks([0, 1])
        figs.append(fig)
    return figs

def plot_example_trial(trial, odor_onset=None, reward_onset=None, ymin=-1.5, ymax=12):
    xs = trial['times']
    ys = trial['licks']
    ix = (ys > 0)

    if odor_onset is not None:
        xs = xs - odor_onset
        if reward_onset is not None:
            reward_onset = reward_onset - odor_onset
        odor_onset = 0
    
    fig = plt.figure()
    plt.plot(xs, trial['DA'], color='k')
    rng = np.random.default_rng(567)
    # plt.plot(xs[ix], (ymax-1)*ys[ix] + 0.0*rng.normal(size=sum(ix)), 'k.', alpha=0.8, markersize=3)
    # plot small vertical lines for licks
    for x in xs[ix]:
        plt.plot([x, x], [ymax-1, ymax], 'k-', alpha=1.0, linewidth=1)

    yl = [ymin, ymax]
    if odor_onset is not None:
        plt.plot(odor_onset*np.ones(2), yl, 'k--', alpha=0.5, linewidth=1, zorder=-1)
    if reward_onset is not None:
        plt.plot(reward_onset*np.ones(2), yl, 'k--', alpha=0.5, linewidth=1, zorder=-1)
    plt.tight_layout()
    plt.yticks(np.arange(0,ymax+1,5))
    plt.ylim([yl[0], ymax])
    plt.xlabel('Time rel. to CS onset (s)')
    plt.ylabel('Dopamine activity (z-score)')
    print('CS_DA_peak={:0.3f}, n_anticipatory_licks={}'.format(trial['CS_DA_peak'], trial['n_anticipatory_licks']))
    return fig

#%% for splitting/plotting trials by being below/above some per-session threshold

def get_criteria_per_session(all_data, group_key, percentiles, trial_types, ignoreFirstHalfOfTrialsEachSession=False, ignoreLastHalfOfTrialsEachSession=False):

    all_crits = []
    for i, data in enumerate(all_data):
        if len(data) == 0:
            all_crits.append(None)
            continue

        mid_trial_index = np.median([d['trial_index'] for d in data])
        def trial_keeper(item):
            keep = True
            if item['trial_type'] not in trial_types:
                keep = False
            elif ignoreFirstHalfOfTrialsEachSession and item['trial_index'] < mid_trial_index:
                keep = False
            elif ignoreLastHalfOfTrialsEachSession and item['trial_index'] > mid_trial_index:
                keep = False
            return keep
        cdata = [item for item in data if trial_keeper(item)]
        if len(cdata) == 0:
            all_crits.append(None)
            continue
        
        stats = np.array([item[group_key] for item in cdata if item[group_key] is not None and ~np.isnan(item[group_key])])
        crits = np.percentile(stats, percentiles)
        # print(len(stats), len(cdata), (stats <= crits[0]).sum(), (stats > crits[1]).sum())
        
        groups = []
        for item in data:
            if not trial_keeper(item) or item[group_key] is None or np.isnan(item[group_key]):
                cur_group = 0
            elif item[group_key] <= crits[0]:
                cur_group = 1
                vkey = 'low'
            elif item[group_key] > crits[1]:
                cur_group = 3
                vkey = 'high'
            else:
                cur_group = 2
                vkey = 'mid'
            # exemplar = exemplars[vkey]
            groups.append(cur_group)
        
        all_crits.append({'criteria': crits, 'groups': np.hstack(groups)})
    return all_crits

def plot_trial_average_per_group(all_data, group_key='CS_DA_peak', ykey=None, percentiles=None, groups_to_show=None, trial_types=None, day_indices=None, showFirstLickOnly=False, ignoreFirstHalfOfTrialsEachSession=False, ignoreLastHalfOfTrialsEachSession=False, odor_onset=None, reward_onset=None, bin_size=50, show_legend=False, ylims=None, title=None, fig=None):
    if percentiles is None and groups_to_show is None:
        # show median
        percentiles = (0, 100)
        groups_to_show = ['mid'] # show median
    elif groups_to_show is None:
        groups_to_show = ['low', 'high']
    elif percentiles is None:
        percentiles = (25, 75) # will show lower and upper quartile

    if trial_types is None:
        trial_types = ['go']
    if odor_onset is None:
        raise Exception('Must provide odor_onset in order to find anticipatory licks.')
    lick_to_hz = 1000 / bin_size

    exemplars = {'low': {}, 'mid': {}, 'high': {}}
    all_crits = get_criteria_per_session(all_data, group_key, percentiles, trial_types, ignoreFirstHalfOfTrialsEachSession=ignoreFirstHalfOfTrialsEachSession, ignoreLastHalfOfTrialsEachSession=ignoreLastHalfOfTrialsEachSession)
    assert len(all_crits) == len(all_data)
    for data, crits in zip(all_data, all_crits):
        if day_indices is not None and data[0].get('day_index', 0) not in day_indices:
            continue

        criteria = crits['criteria']
        groups = crits['groups']
        group_key_names = ['ignore', 'low', 'mid', 'high']

        for group, item in zip(groups, data):
            if group == 0:
                continue
            vkey = group_key_names[group]
            exemplar = exemplars[vkey]

            if 'licks' not in exemplar:
                exemplar['licks'] = []
                exemplar['DA'] = []
                exemplar['times'] = item['times']
                exemplar['first_lick'] = []
                exemplar['first_lick_time'] = []
            
            exemplar['licks'].append(item['licks'])
            exemplar['DA'].append(item['DA'])

            licks_first_only = 0*item['licks']
            isi_licks = np.where(item['licks'][item['times'] > odor_onset] > 0)[0]
            if len(isi_licks) > 0:
                isi_offset = np.where(item['times'] > odor_onset)[0][0]
                licks_first_only[isi_licks[0] + isi_offset] = 1
                exemplar['first_lick_time'].append(item['times'][isi_licks[0]])
            else:
                # prevent these trials from being counted in average
                licks_first_only = np.nan*licks_first_only
                exemplar['first_lick_time'].append(np.nan)
            exemplar['first_lick'].append(licks_first_only)
            exemplars[vkey] = exemplar

    showDA = (ykey is None) or (ykey == 'DA')
    showLicks = (ykey is None) or (ykey == 'licks')
    clrs = {'low': '#ff9999', 'mid': 'black', 'high': 'darkred'}
    
    lbls = {'low': 'Low CS DA', 'mid': 'DA in median', 'high': 'High CS DA'}
    if fig is None:
        if ykey is None:
            fig = plt.figure(figsize=(3,6))
        else:
            if showFirstLickOnly:
                fig = plt.figure(figsize=(1.5,3))
            else:
                fig = plt.figure(figsize=(3,3))
    vals = {}
    for c,name in enumerate(groups_to_show):
        exemplar = exemplars[name]
        if len(exemplar) == 0:
            continue

        if showDA:
            if ykey is None:
                plt.subplot(2,1,1)
            vs = np.vstack(exemplar['DA'])
            mu = np.nanmean(vs, axis=0)
            se = np.nanstd(vs, axis=0) / np.sqrt((~np.isnan(vs.sum(axis=1))).sum())
            plt.plot(exemplar['times'] - odor_onset, mu, '-', color=clrs[name], label=lbls[name])
            plt.fill_between(exemplar['times'] - odor_onset, mu-se, mu+se, color=clrs[name], alpha=0.2, zorder=-1, linewidth=0)

            plt.ylabel('Dopamine activity (a.u.)')
            if showFirstLickOnly:
                if odor_onset is not None and reward_onset is not None:
                    plt.xlim([odor_onset - odor_onset, reward_onset - odor_onset])

        if showLicks:
            if ykey is None:
                plt.subplot(2,1,2)
            vs = np.vstack(exemplar['licks'])
            if showFirstLickOnly:
                # vs = np.vstack(exemplar['first_lick'])
                vs = np.vstack(exemplar['first_lick_time'])
            else:
                vs = lick_to_hz * vs
            
            mu = np.nanmean(vs, axis=0)
            # if showFirstLickOnly:
            #     mu = mu / mu.sum() # ensure this is a proportion
            se = np.nanstd(vs, axis=0) / np.sqrt((~np.isnan(vs.sum(axis=1))).sum())
            if showFirstLickOnly:
                rng = np.random.default_rng(567)
                xs = c + rng.normal(scale=0.05, size=len(vs))
                plt.plot(xs, vs, '.', color=clrs[name], alpha=0.1, markersize=1, zorder=-2)
                plt.plot(c, mu, 'o', color=clrs[name])
                plt.plot([c,c], [mu-se, mu+se], '-', color=clrs[name], alpha=0.5, zorder=-1, linewidth=2)
            else:
                xs = exemplar['times'] - odor_onset
                plt.plot(xs, mu, '-', color=clrs[name])
                plt.fill_between(xs, mu-se, mu+se, color=clrs[name], alpha=0.2, zorder=-1, linewidth=0)
            vals[name] = (xs, vs)

            if showFirstLickOnly:
                plt.xlim([-0.5, len(groups_to_show)-0.5])
                plt.ylabel('Time of first anticipatory lick (s)')
                # if odor_onset is not None and reward_onset is not None:
                #     plt.xlim([odor_onset - odor_onset, reward_onset - odor_onset])
                # plt.ylabel('Proportion of trials')
            else:
                plt.ylabel('Lick rate (Hz)')
                plt.ylim([0,1])
        
        if showFirstLickOnly:
            pass # plt.xlabel('Time of first anticipatory lick (s)')
        else:
            plt.xlabel('Time rel. to CS onset (s)')
        if show_legend:
            plt.legend(fontsize=8)

    for i in ([1,2] if ykey is None else [1]):
        if ykey is None:
            plt.subplot(2,1,i)
        if i == 1 and title is not None:
            plt.title(title)
        if not showFirstLickOnly:
            plt.axis('tight')
        if ylims is None:
            yl = plt.ylim()
        else:
            yl = ylims[i-1]
            if yl:
                plt.ylim(yl)
            else:
                yl = plt.ylim()
        if showFirstLickOnly:
            plt.xticks(np.arange(len(groups_to_show)), labels=[lbls[name] for name in groups_to_show], rotation=45, ha='right')
        else:
            if reward_onset is not None:
                plt.plot((reward_onset - odor_onset)*np.ones(2), yl, 'k--', alpha=0.5, linewidth=1, zorder=-1)
            if odor_onset is not None:
                plt.plot((odor_onset - odor_onset)*np.ones(2), yl, 'k--', alpha=0.5, linewidth=1, zorder=-1)
    plt.tight_layout()
    return fig, vals

def plot_trial_average_per_group_rel_to_first_lick(all_data, group_key='CS_DA_peak', percentiles=None, groups_to_show=None, trial_types=None, day_indices=None, showFirstLickOnly=False, ignoreFirstHalfOfTrialsEachSession=False, ignoreLastHalfOfTrialsEachSession=False, t_pre=40, odor_onset=None, reward_onset=None, bin_size=50):

    if percentiles is None and groups_to_show is None:
        # show median
        percentiles = (0, 100)
        groups_to_show = ['mid'] # show median
    elif groups_to_show is None:
        groups_to_show = ['low', 'high']
    elif percentiles is None:
        percentiles = (25, 75) # will show lower and upper quartile

    if trial_types is None:
        trial_types = ['go']
    if odor_onset is None or reward_onset is None:
        raise Exception('Must provide odor_onset and reward_onset to find first anticipatory lick')
    lick_to_hz = 1000 / bin_size

    all_crits = get_criteria_per_session(all_data, group_key, percentiles, trial_types, ignoreFirstHalfOfTrialsEachSession=ignoreFirstHalfOfTrialsEachSession, ignoreLastHalfOfTrialsEachSession=ignoreLastHalfOfTrialsEachSession)

    yss = []
    yss_crit = {}
    assert len(all_crits) == len(all_data)
    for i, (crits, data) in enumerate(zip(all_crits, all_data)):
        if len(data) == 0 or crits is None:
            continue

        mid_trial_index = np.median([d['trial_index'] for d in data])
        groups = crits['groups']
        group_key_names = ['ignore', 'low', 'mid', 'high']
        for group, item in zip(groups, data):
            if group == 0:
                continue
            lkey = group_key_names[group]

            if lkey not in yss_crit:
                yss_crit[lkey] = []

            ls = item['licks']
            ixc = (ls > 0) & (item['times'] > odor_onset+0.0) & (item['times'] <= reward_onset)
            if ixc.sum() == 0:
                continue
            t0 = np.where(item['times'] > odor_onset)[0][0]
            t = np.where(ixc)[0][0]
            t_end = np.where(item['times'] > reward_onset)[0][0]
            
            ys = lick_to_hz * item['licks']
            ysc = ys[(t-t_pre):t_end]
            ysc = np.hstack([ysc, np.nan*np.ones(len(ys)-len(ysc))])
            yss.append(ysc)
            yss_crit[lkey].append(ysc)

    fig = plt.figure()
    clrs = {'low': '#ff9999', 'mid': 'black', 'high': 'darkred'}
    vals = {}
    for name, yss in yss_crit.items():
        if name not in groups_to_show:
            continue
        yss = np.vstack(yss)
        if yss.shape[0] == 0:
            continue
        mus = np.nanmean(yss, axis=0)
        ns = yss.shape[0] - np.isnan(yss).sum(axis=0)
        sds = np.nanstd(yss, axis=0)
        ses = sds / np.sqrt(ns)
        # xs = item['times']
        td = item['times'][1] - item['times'][0]
        xs = td*(np.arange(len(mus)) - t_pre)
        ix = ns > 0.2*max(ns)
        mus[xs == 0] = np.nan
        vals[name] = (xs, yss)
        plt.plot(xs[ix], mus[ix], '-', color=clrs[name])
        plt.fill_between(xs[ix], mus[ix]-ses[ix], mus[ix]+ses[ix], color=clrs[name], alpha=0.2, zorder=-1, label='name', linewidth=0)

    plt.xlabel('Time rel. to first anticipatory lick (s)')
    plt.ylabel('Lick rate (Hz)')
    return fig, vals

#%% for plotting data across trials over days/sessions

def moving_average(data_set, K):
    weights = np.ones(K) / K
    return np.convolve(data_set, weights, mode='valid')

def get_time_series_over_sessions(Sessions, da_window, lick_window, mouse_names, day_indices=None, trial_types=None, trial_type_key='trial_type', subtract_baseline_spikes=False, use_mean_spikes=False, bin_size=50, day_offset=0):
    trial_counts = {}
    lick_to_hz = 1000 / bin_size

    all_vals = []
    for mouse_index, mouse_name in enumerate(mouse_names):
        for i, session in enumerate(Sessions):
            if mouse_name != session[0]['mouse']:
                continue
            day_index = session[0]['day_index']
            if day_indices is not None and day_index not in day_indices:
                continue
            # get DA and licking per trial
            X, Y = get_data_per_session(session, da_window, lick_window, trial_types=trial_types, zscore_spikes=False, use_mean_spikes=use_mean_spikes, subtract_baseline_spikes=subtract_baseline_spikes, trial_type_key=trial_type_key)
            
            trial_counts[(mouse_index, day_index)] = len(X)
            trial_offset = 0
            for d in range(day_index):
                ckey = (mouse_index, d)
                trial_offset += trial_counts.get(ckey, 0)
                trial_offset += day_offset # add buffer trials between each day

            vals = [(mouse_index, day_index, trial_offset+i, y * lick_to_hz, x) for i, (x,y) in enumerate(zip(X,Y))]
            all_vals.extend(vals)
    all_vals = np.vstack(all_vals)
    return all_vals, ('Anticipatory lick rate (Hz)', 'CS Dopamine activity (z-score)')

def plot_time_series_over_trials(all_vals, ylbls, clr=DA_color, clr2='darkred', experiment_name='Example experiment'):
    fig = plt.figure()
    ax1 = plt.gca()
    ax1.tick_params(axis='y', labelcolor=clr2)
    ax1.set_ylabel(ylbls[0], color=clr2)
    ax2 = ax1.twinx()
    ax2.set_ylabel(ylbls[1], color=DA_color, rotation=270, va='bottom')
    ax2.tick_params(axis='y', labelcolor=DA_color)
    ax1.set_xlabel('# Trials')

    xs = all_vals[:,2] # trial offset
    xsa = np.arange(xs.min(), xs.max()+1)
    pts1 = []
    pts2 = []
    for x in xsa:
        vs = all_vals[xs == x,-2] # lick rate per trial
        if len(vs) > 0:
            mu = np.nanmean(vs)
            ns = (~np.isnan(vs)).sum()
            se = np.nanstd(vs) / np.sqrt(ns)
            pts1.append((x, mu, se, ns))
        else:
            pts1.append((x, np.nan, 0, np.nan))

        vs = all_vals[xs == x,-1] # DA per trial
        if len(vs) > 0:
            mu = np.nanmean(vs)
            ns = (~np.isnan(vs)).sum()
            se = np.nanstd(vs) / np.sqrt(ns)
            pts2.append((x, mu, se, ns))
        else:
            pts2.append((x, np.nan, 0, np.nan))

    pts1 = np.vstack(pts1)
    pts2 = np.vstack(pts2)
    xs1 = pts1[:,0]
    xs2 = pts2[:,0]

    ix1_nan = np.isnan(pts1[:,-1])
    ix2_nan = np.isnan(pts2[:,-1])
    ix1 = (pts1[:,-1] >= 0.5*pts1[~ix1_nan,-1].max()) | ix1_nan
    ix2 = (pts2[:,-1] >= 0.5*pts2[~ix2_nan,-1].max()) | ix2_nan

    ax1.plot(xs1[ix1], pts1[ix1,1], '-', color=clr2, alpha=0.8, linewidth=1)
    ax2.plot(xs2[ix2], pts2[ix2,1], '-',  color=clr, alpha=0.8, linewidth=1)

    ax1.fill_between(xs1[ix1], pts1[ix1,1]-pts1[ix1,2], pts1[ix1,1]+pts1[ix1,2], color=clr2, alpha=0.2, zorder=-1, linewidth=0)
    ax2.fill_between(xs2[ix2], pts2[ix2,1]-pts2[ix2,2], pts2[ix2,1]+pts2[ix2,2], color=clr, alpha=0.2, zorder=-1, linewidth=0)
    ixrho = ix1 & ix2 & ~np.isnan(pts1[:,1]) & ~np.isnan(pts2[:,1])
    rho_avg = np.corrcoef(pts1[ixrho,1], pts2[ixrho,1])[0,1]
    
    nexps = len(np.unique(all_vals[:,0]))
    prefix = 'Total' if nexps > 1 else 'Example experiment'
    print(f"{prefix} Pearson's correlation: {rho_avg:0.3f}")
    if nexps == 1:
        plt.title(f'{experiment_name} ($\\rho = {rho_avg:0.3f}$)')
    else:
        plt.title(f'All experiments ($\\rho = {rho_avg:0.3f}$)')
    return fig

def plot_time_series_over_trials_per_animal(all_vals, ylbls, mouse_names=None, kSmooth=0, clr=DA_color, clr2='darkred'):

    is_binary = len(np.unique(all_vals[:,-2])) == 2
    kSmooth = kSmooth if is_binary else 0
    show_scatter = False

    ms = all_vals[:,0]
    msa = np.unique(ms)
    if len(msa) > 1:
        ncols = np.ceil(np.sqrt(len(msa))).astype(int)
        nrows = np.ceil(len(msa) / ncols).astype(int)
        fig = plt.figure(figsize=(2.75*ncols,2.25*nrows))
    else:
        fig = plt.figure()

    rhos = []
    all_mus = []
    ntrials_per_bin = 1
    for m in msa:
        xs = all_vals[:,2]
        xsa = np.arange(0, max(np.unique(xs)), ntrials_per_bin)
        pts1 = []
        pts2 = []

        c = int(m)+1
        if c >= 4:
            c += 1 # we want the 4th panel blank
        if c >= 8:
            c += 1 # we also want the 8th panel blank
        plt.subplot(ncols, nrows, c)
        ax1 = plt.gca()
        ax1.tick_params(axis='y', labelcolor=clr2)
        ax2 = ax1.twinx()
        if True: #int(m) == 0:
            ax1.set_ylabel(ylbls[0], color=clr2)
            ax2.set_ylabel(ylbls[1], color=clr, rotation=270, va='bottom')
            ax1.set_xlabel('# Trials')
        else:
            pass
        # xticks = np.array([0, 5*int(max(xs)/10), 10*int(max(xs)/10)])
        xticks = np.array([0, 150, 300])
        plt.xticks(ticks=xticks, labels=ntrials_per_bin*xticks)
        ax2.tick_params(axis='y', labelcolor=clr)

        for i, x in enumerate(xsa):
            if i == len(xsa)-1:
                continue
            ix = (xs >= xsa[i]) & (xs < xsa[i+1]) & (ms == m)
            vs = all_vals[ix,-2]
            if len(vs) > 0:
                mu = np.nanmean(vs)
                ns = (~np.isnan(vs)).sum()
                se = np.nanstd(vs) / np.sqrt(ns)
                pts1.append((x, mu, se, ns))
            else:
                pts1.append((x, np.nan, 0, np.nan))

            vs = all_vals[ix,-1]
            if len(vs) > 0:
                mu = np.nanmean(vs)
                ns = (~np.isnan(vs)).sum()
                se = np.nanstd(vs) / np.sqrt(ns)
                pts2.append((x, mu, se, ns))
            else:
                pts2.append((x, np.nan, 0, np.nan))

        pts1 = np.vstack(pts1)
        pts2 = np.vstack(pts2)
        xs1 = pts1[:,0]
        xs2 = pts2[:,0]
        ix1_nan = np.isnan(pts1[:,-1])
        ix2_nan = np.isnan(pts2[:,-1])

        ix1 = (pts1[:,-1] >= 0.5*pts1[~ix1_nan,-1].max()) | ix1_nan
        ix2 = (pts2[:,-1] >= 0.5*pts2[~ix2_nan,-1].max()) | ix2_nan
        if kSmooth > 0:
            # todo: may not be fixed once we add in nans
            ysc = np.ma.array(pts1[:,1], mask=~ix1)
            ysc1 = moving_average(ysc, kSmooth)
            xsc1 = np.arange(len(ysc1))
            ysc = np.ma.array(pts2[:,1], mask=~ix2)
            ysc2 = moving_average(ysc, kSmooth)
            xsc2 = np.arange(len(ysc2))
        else:
            xsc1 = xs1[ix1]
            ysc1 = pts1[ix1,1]
            xsc2 = xs2[ix2]
            ysc2 = pts2[ix2,1]
        
        if show_scatter:
            ax1.plot(ysc2, ysc1, '.', color=clr2, alpha=0.8)
        else:
            ax1.plot(xsc1, ysc1, '-', color=clr2, alpha=0.8, linewidth=1)
            ax2.plot(xsc2, ysc2, '-', color=clr, alpha=0.8, linewidth=1)
        ixrho = ix1 & ix2 & ~ix1_nan & ~ix2_nan
        rho = np.corrcoef(pts1[ixrho,1], pts2[ixrho,1])[0,1]
        all_mus.append((pts1[ixrho,1], pts2[ixrho,1]))
        rhos.append(rho)

        ttl = mouse_names[int(m)].replace('FgDA_', 'Mouse ')
        plt.title(f'{ttl} ($\\rho = {rho:0.3f}$)')

    plt.tight_layout()
    rhos = np.vstack(rhos)[:,0]

    print(f"Pearson's correlation: {np.percentile(rhos, [25, 50, 75]).round(3)}")
    return fig, rhos

def plot_correlations(Rhos, ylim=None, ylbl="Correlation between CS DA and licking", xticklabels=None, ystar=None, zeroline=False, paired=True, figsize=None):
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:   
        fig = plt.figure(figsize=(3.5 if len(Rhos) > 1 else 1.25, 3 if not xticklabels else 4))
    rng = np.random.default_rng(66)
    for i, rhos_both in enumerate(Rhos):
        if len(rhos_both.shape) == 2:
            rhos = rhos_both[:,1] - rhos_both[:,0]
        else:
            rhos = rhos_both[:,None]
        mu = np.mean(rhos)
        se = np.std(rhos) / np.sqrt(len(rhos))
        xs = (i+1)*np.ones(len(rhos)) + 0.08*rng.normal(size=len(rhos))
        plt.plot(xs, rhos, 'ko', markersize=3, alpha=0.25, markeredgecolor='none', zorder=-3)
        plt.plot((i+1)*np.ones(2), [mu-se, mu+se], 'k-', linewidth=1, zorder=-2)
        plt.plot(i+1, mu, 'ko')
        if len(rhos_both.shape) == 2:
            if paired:
                # find p-value of non-parametric paired t-test
                test = scipy.stats.wilcoxon(rhos_both[:,0], rhos_both[:,1])
            else:
                # find p-value of non-parametric unpaired t-test
                # test = scipy.stats.mannwhitneyu(rhos_both[:,0], rhos_both[:,1])
                test = scipy.stats.ranksums(rhos_both[:,0], rhos_both[:,1])
                print(test)
            pval = test.pvalue
            n_stars = int(pval < 0.05) + int(pval < 0.01) + int(pval < 0.001)
            # is_sig = mu-2*se > 0
            if ystar is not None and n_stars > 0:
                star_str = '*' * n_stars
                plt.text(i+1, ystar, star_str, fontsize=12, ha='center', va='bottom')
                # plt.plot(i+1, ystar, 'k*', markersize=5)
    if xticklabels is None:
        plt.xticks([])
    else:
        ticks = 1+np.arange(len(xticklabels))
        plt.xticks(ticks=ticks, labels=xticklabels, rotation=45, ha='right')
    
    if len(Rhos) == 1:
        # remove bottom axis
        plt.gca().spines['bottom'].set_visible(False)
        plt.xlim([0.75, 1.25])
    else:
        plt.xlim([0.5, len(Rhos)+0.5])
    if zeroline:
        plt.plot(plt.xlim(), [0, 0], 'k-', linewidth=1, zorder=-5, alpha=0.5)
    if ylim is not None:
        plt.ylim(ylim)
    plt.ylabel(ylbl)
    plt.tight_layout()
    return fig
