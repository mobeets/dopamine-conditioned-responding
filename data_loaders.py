#%% imports

import glob
import os.path
import numpy as np
from scipy.io import loadmat
import h5py
import pandas as pd

# set base data directory to home user folder
DATADIR = os.path.expanduser('~/code/vta-data')

#%% add DA at CS onset

def add_DA_at_CS_onset(D, da_window=None, use_mean_spikes=True, key_name='CS_DA_peak', baseline_center=False):
    da_window = D.da_bin_window if da_window is None else da_window
    f = np.mean if use_mean_spikes else np.max
    for trials in D.Sessions:
        for trial in trials:
            trial[key_name] = f(trial['DA'][da_window[0]:(da_window[1]+1)])
            if baseline_center:
                trial[key_name] = trial[key_name] - np.mean(trial['DA'][:da_window[0]])

def add_first_anticipatory_lick_time(D, key_name='time_of_first_lick'):
    for trials in D.Sessions:
        for trial in trials:
            licks = trial['licks']
            times = trial['times']
            if D.reward_onset is not None:
                reward_onset = D.reward_onset
            else:
                trial_type = int(trial['trial_type'])
                odor = [x for x,ys in D.trial_types.items() if trial_type in ys and x != 'all'][0]
                reward_onset = D.reward_times[odor]
            ixAnticip = (times > D.odor_onset) & (times < reward_onset)
            anticipatory_licks = licks[ixAnticip]
            if anticipatory_licks.sum() < 1:
                time_of_first_lick = np.nan
            else:
                t_first_lick = np.where(anticipatory_licks > 0)[0][0]
                time_of_first_lick = times[ixAnticip][t_first_lick]
            trial[key_name] = time_of_first_lick

#%% QianBurrell2024

class QianBurrell2024:
    """
    VS (lateral nucleus accumbens) - fluorometry w/ calcium sensor
    """
    __paper_url__ = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10871210/'
    __data_url__ = 'https://datadryad.org/dataset/doi:10.5061/dryad.mkkwh71f6'

    def __init__(self, datadir=DATADIR):
        self.exp_name = 'Qian & Burrell et al., 2025'
        # n.b. this is data from just one fiber, but there were more in other parts of VS
        self.datadir = os.path.join(datadir, 'QianBurrell2025')
        self.dfl = pd.read_parquet(os.path.join(self.datadir, 'data/lick_data_for_jay.parquet'))
        self.dfd = pd.read_parquet(os.path.join(self.datadir, 'data/photo_data_for_jay.parquet'))
        
        self.odor_onset = 2.01
        self.sampling_rate = 20
        self.bin_size = int(1000 / self.sampling_rate)
        self.reward_onset = self.odor_onset + 3.5
        self.mouse_names = self.dfl.mouse.unique()
        self.trial_types = {'all': ['go', 'go_omit', 'no_go'], 'A': ['go', 'go_omit'], 'C': ['c_reward', 'c_omit'], 'D': ['no_go']}

        self.odor_onset_bin = int(self.odor_onset * 1000 / self.bin_size)
        lick_bin_onset = self.odor_onset_bin
        lick_bin_offset = int(self.reward_onset * 1000 / self.bin_size)
        self.lick_bin_windows = {cue: (lick_bin_onset, lick_bin_offset) for cue in self.trial_types}
        
        self.da_window_length = 0.5 # secs
        self.da_bin_length = int(1000*self.da_window_length/self.bin_size)
        self.da_bin_window = (self.odor_onset_bin, self.odor_onset_bin + self.da_bin_length)
        self.da_bin_starts = np.arange(20, 80)
        
        self.Sessions = self.load_all_data(self.dfl, self.dfd, self.odor_onset, self.reward_onset, self.sampling_rate)
    
    def load_all_data(self, dfl, dfd, odor_onset, reward_onset, sampling_rate, mouse_names=None, day_indices=None, trial_types=None):
        if mouse_names is None:
            mouse_names = np.unique(dfl.mouse)
        if day_indices is None:
            day_indices = dfl.day.unique()

        all_data = []
        for i, mouse_name in enumerate(mouse_names):
            print(f'Processing {mouse_name=} ({i+1} of {len(mouse_names)})')
            for day_index in day_indices:
                df = dfl
                ix = (df.mouse == mouse_name) & (df.day == day_index)
                if trial_types is not None:
                    ix = ix & df.trialtype.isin(trial_types)
                dflc = df[ix]

                df = dfd
                ix = (df.mouse == mouse_name) & (df.day == day_index)
                if trial_types is not None:
                    ix = ix & df.trialtype.isin(trial_types)
                dfdc = df[ix]
                DA_max = dfdc.data.max()

                if len(dflc) == 0 or len(dfdc) == 0:
                    if len(dfdc) + len(dflc) > 0:
                        print(f"ERROR: we have missing DA or licking data for session from {mouse_name=}, {day_index=}")
                    continue

                # get avg number of licks during pre-odor period for this mouse-session
                dfb = dfl[(dfl.mouse == mouse_name) & (dfl.day == day_index)]
                # method 1: threshold is lick > mean
                baseline_lick_count = (dfb.lick.values < odor_onset).sum()
                baseline_lick_rate_thresh = baseline_lick_count / (odor_onset * len(dfb.trialnumber.unique()))
                
                # method 2: threshold is lick > mean + SE
                baseline_lick_rates = np.array([(dfcc.lick.values < odor_onset).sum() for k,dfcc in dfb.groupby('trialnumber')]) / odor_onset
                mu = np.mean(baseline_lick_rates)
                se = np.std(baseline_lick_rates) / np.sqrt(len(baseline_lick_rates))
                baseline_lick_rate_thresh = mu + se

                min_licks_if_anticipatory = baseline_lick_rate_thresh * (reward_onset - odor_onset)

                data = []
                trs = np.unique(df[ix].trialnumber)
                for i, tr in enumerate(trs):
                    dfc = dfdc[(dfdc.trialnumber == tr)]
                    times = dfc.time.values / sampling_rate
                    DA = dfc.data.values

                    dfc = dflc[(dflc.trialnumber == tr)]
                    activity = dfc.lick.values
                    activity = activity[activity >= 0]
                    if len(activity) < 1:
                        continue

                    lick_time_bins = np.round(sampling_rate*activity).astype(int)
                    licks = np.zeros(len(times))
                    licks[lick_time_bins[lick_time_bins < len(times)]] = 1

                    # idea 0: DA during ISI
                    US_DA = DA[(times >= reward_onset)].max()

                    # idea 1: show average licks when DA at CS is in top vs lower percentile
                    CS_DA_peak = DA[(times > odor_onset) & (times <= odor_onset + 1)].max()

                    # idea 1b: show average licks when DA early in ISI is in top vs lower percentile
                    ISI_DA_peak = DA[(times > odor_onset+1) & (times <= odor_onset+2)].max()

                    # idea 2: show DA when anticipatory licks are in top vs lower percentile
                    anticip_dur = ((times > odor_onset) & (times <= reward_onset)).sum()
                    n_anticipatory_licks = licks[(times > odor_onset) & (times <= reward_onset)].sum()

                    # pre-odor licks
                    baseline_dur = (times < odor_onset).sum()
                    n_baseline_licks = licks[(times < odor_onset)].sum()

                    # Find baseline spike rate (per bin)
                    DA_bsr = DA[(times >= odor_onset-1) & (times <= odor_onset)].mean()

                    # idea 3: show DA when first anticipatory lick is in early vs later percentile
                    ixAnticip = (activity > odor_onset) & (activity <= reward_onset)
                    if ixAnticip.sum() <= 1:
                        time_of_first_lick = np.nan
                        time_of_second_lick = np.nan
                    else:
                        time_of_first_lick = activity[ixAnticip][0] - odor_onset
                        time_of_second_lick = activity[ixAnticip][1] - odor_onset

                    # did show an anticipatory lick
                    did_anticipatory_lick = int(ixAnticip.sum() > min_licks_if_anticipatory)
                    if not did_anticipatory_lick:
                        time_of_first_lick = np.nan
                        time_of_second_lick = np.nan

                    trialtype = dfc.trialtype.unique()
                    assert len(trialtype) == 1

                    data.append({'US_DA': US_DA, 'CS_DA_peak': CS_DA_peak, 'ISI_DA_peak': ISI_DA_peak, 'n_anticipatory_licks': n_anticipatory_licks, 'n_baseline_licks': n_baseline_licks, 'delta_lick_rate': n_anticipatory_licks/anticip_dur - n_baseline_licks/baseline_dur, 'time_of_first_lick': time_of_first_lick, 'time_of_second_lick': time_of_second_lick, 'baseline_DA_rate': DA_bsr,
                    'did_anticipatory_lick': did_anticipatory_lick, 'licks': licks, 'DA': DA, 'times': times, 'day_index': day_index, 'trial_index': i, 'mouse': mouse_name, 'trial_type': trialtype[0], 'min_licks_if_anticipatory': min_licks_if_anticipatory, 'cell_name': 0})

                all_data.append(data)
        return all_data

#%% helper functions

def mat_to_dict(mat_obj):
    """
    Recursively converts mat-object loaded using scipy.io.loadmat to a dictionary.
    """
    if isinstance(mat_obj, dict):
        # Recursively apply the function for each dictionary value
        return {key: mat_to_dict(value) for key, value in mat_obj.items()}
    elif isinstance(mat_obj, np.ndarray):
        # For arrays, check if it is an object array or structured array
        if mat_obj.dtype.names:  # Structured array (like MATLAB struct)
            return {name: mat_to_dict(mat_obj[name]) for name in mat_obj.dtype.names}
        elif mat_obj.size == 1:
            return mat_to_dict(mat_obj.item())  # Unpack arrays of size 1
        else:
            return [mat_to_dict(element) for element in mat_obj]
    else:
        return mat_obj

#%% Eshel2016

class Eshel2016:
    """
    lateral VTA - ephys
    """
    __paper_url__ = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC4767554/'
    __data_url__ = None

    def __init__(self, key='Variable-Expectation', datadir=DATADIR):
        """
        Variable-Expectation (Trial type, Meaning):
        1           90% expectation, received reward
        2           50% expectation, received reward
        3           10% expectation, received reward
        4           free reward (only some mice have)
        5           10% expectation, no reward
        6           50% expectation, no reward
        7           90% expectation, no reward 
        8           predicted airpuff
        9           free airpuff
        10          airpuff omission

        note: Variable-Reward has no licks (must be corrupt data)
        """
        self.key = key
        self.exp_name = 'Eshel et al., 2016'
        self.datadir = os.path.join(datadir, 'Eshel2016')
        self.sessionfiles = glob.glob(os.path.join(self.datadir, self.key, '*.mat'))
        self.bin_size = 50
        self.t_pre = 1000
        self.t_post = 4000
        self.odor_onset = 0.0
        self.times = np.arange(-self.t_pre, self.t_post+1, self.bin_size)[:-1] / 1000
        if self.key == 'Variable-Expectation':
            self.reward_onset = 2.0
            self.trial_types = {'A': [1,7], 'B': [2,6], 'C': [3,5], 'D': [8,10], 'all': [1,7,2,6,3,5,8,10]}
        elif 'Variable-Reward' in self.key:
            self.reward_onset = 1.5
            self.trial_types = {'A': [8], 'B': [9], 'C': [10], 'D': [11], 'E': [12], 'F': [13], 'G': [14], 'H': [15], 'all': [8,9,10,11,12,13,14,15]}
        
        self.odor_onset_bin = np.where(self.times >= self.odor_onset)[0][0]
        lick_bin_onset = self.odor_onset_bin
        lick_bin_offset = np.where(self.times >= self.reward_onset)[0][0]
        self.lick_bin_windows = {cue: (lick_bin_onset, lick_bin_offset) for cue in self.trial_types}

        self.da_window_length = 0.5 # secs
        self.da_bin_length = int(1000*self.da_window_length/self.bin_size)
        self.da_bin_window = (self.odor_onset_bin, self.odor_onset_bin + self.da_bin_length)
        self.da_bin_window_pre = (self.odor_onset_bin - self.da_bin_length, self.odor_onset_bin)
        self.da_bin_starts = np.arange(0, 60)

        Sessions = []
        mouse_names = []
        for i, sessionfile in enumerate(self.sessionfiles):
            meta_info = os.path.split(sessionfile)[1].replace('.mat','').split('_') # H_2012-05-08_18-50-20_TT6_1_formatted
            mouse_name = meta_info[0]
            session_name = meta_info[1]
            cell_name = meta_info[3] + '_' + meta_info[4]
            data = mat_to_dict(loadmat(sessionfile))
            trials = self.extract_trial_data(data, bin_size=self.bin_size, t_pre=self.t_pre, t_post=self.t_post, mouse_name=mouse_name, session_name=session_name, cell_name=cell_name)
            Sessions.append(trials)
            mouse_names.append(mouse_name)
        self.Sessions = Sessions
        self.mouse_names = sorted(list(set(mouse_names)))
        add_DA_at_CS_onset(self)
        add_first_anticipatory_lick_time(self)

    def extract_trial_data(self, d, bin_size, t_pre, t_post, mouse_name=None, session_name=None, cell_name=None):
        """
        Extracts spike and lick times for each trial, discretized into time bins.
        
        Parameters:
            d (dict): The dictionary containing 'events' and 'responses'.
            bin_size (float): The bin size (in seconds).
            t_pre (float): Time before odor onset to include (in seconds).
            t_post (float): Time after odor onset to include (in seconds).
            
        Returns:
            results (list): A list of dictionaries containing 'licks' and 'spikes' for each trial.
        """
        events = d['events']
        responses = d['responses']
        
        odor_on = events['odorOn'] # N x 1
        trial_types = events['trialType'] # N x 1

        spikes = np.array(responses['spike'])
        licks = np.array(responses['lick'])
        times = np.arange(-t_pre, t_post+1, bin_size)[:-1] / 1000

        results = []

        # Loop over each trial
        for i in range(len(odor_on)):
            if np.isnan(odor_on[i]):
                continue

            trial_result = {'licks': [], 'DA': [], 'times': times, 'trial_type': trial_types[i], 'baseline_DA_rate': None, 'trial_index': i, 'mouse': mouse_name, 'day_index': session_name, 'cell_name': cell_name}

            # Define the time window for this trial
            trial_start_time = odor_on[i] - t_pre
            trial_end_time = odor_on[i] + t_post
            
            # Discretize time into bins from trial_start_time to trial_end_time
            time_bins = np.arange(trial_start_time, trial_end_time+1, bin_size)

            # Find baseline spike rate per bin
            bsr = ((spikes >= odor_on[i]-1000) & (spikes <= odor_on[i])).sum() # Hz
            trial_result['baseline_DA_rate'] = bsr * bin_size / 1000

            # Extract spikes for this trial
            ixc = (spikes >= trial_start_time) & (spikes < trial_end_time)
            spikes_trial = spikes[ixc] # spikes on this trial
            spike_times_hist, _ = np.histogram(spikes_trial, time_bins)
            trial_result['DA'] = spike_times_hist

            # Extract licks for this trial
            ixc = (licks >= trial_start_time) & (licks < trial_end_time)
            licks_trial = licks[ixc] # licks on this trial
            lick_times_hist, _ = np.histogram(licks_trial, time_bins)
            trial_result['licks'] = lick_times_hist

            # Add the trial result to the overall list
            results.append(trial_result)

        # z-score
        rates = np.array([result['baseline_DA_rate'] for result in results])
        mu = np.mean(rates)
        sd = np.std(rates)
        for result in results:
            result['DA'] = (result['DA'] - mu) / sd
        return results

#%% Matsumoto2016

class Matsumoto2016:
    """
    lateral VTA - ephys
    """
    __paper_url__ = 'https://elifesciences.org/articles/17328'
    __data_url__ = 'https://doi.org/10.5061/dryad.8cz8w9gxj'

    def __init__(self, key='_HighR*', datadir=DATADIR):
        """

        n.b. assuming we have 7 trial types where:
            '1:CS_water_US_water', 90%
            '2:CS_water_US_nothing', 10%
            '3:CS_nothing_US_nothing',
            '4:CS_airpuff_US_airpuff',
            '5:CS_airpuff_US_nothing',
            '6:US_freewater',
            '7:US_freeairpuff'
        """
        self.exp_name = 'Matsumoto et al., 2016'
        self.datadir = os.path.join(datadir, 'Matsumoto2016')
        self.key = key # HighR, HighR2, LowR, Mixed
        self.sessionfiles = glob.glob(os.path.join(self.datadir, '*' + self.key + '.mat'))
        self.bin_size = 50
        self.t_pre = 1000
        self.t_post = 4000
        self.odor_onset = 0.0
        self.reward_onset = 2.0
        self.times = np.arange(-self.t_pre, self.t_post+1, self.bin_size)[:-1] / 1000
        self.trial_types = {'all': [1,2,3,4,5], 'A': [1,2], 'B': [3], 'C': [4,5]}

        self.odor_onset_bin = np.where(self.times >= self.odor_onset)[0][0]
        lick_bin_onset = self.odor_onset_bin
        lick_bin_offset = np.where(self.times >= self.reward_onset)[0][0]
        self.lick_bin_windows = {cue: (lick_bin_onset, lick_bin_offset) for cue in self.trial_types}

        self.da_window_length = 0.5 # secs
        self.da_bin_length = int(1000*self.da_window_length/self.bin_size)
        self.da_bin_window = (self.odor_onset_bin, self.odor_onset_bin + self.da_bin_length)
        self.da_bin_window_pre = (self.odor_onset_bin - self.da_bin_length, self.odor_onset_bin)
        self.da_bin_starts = np.arange(0, 60)

        Sessions = []
        mouse_names = []
        for i, sessionfile in enumerate(self.sessionfiles):
            meta_info = os.path.split(sessionfile)[1].replace('.mat', '').split('_') # hr053_2015-09-18_09-46-27_TT4_1_HighR
            mouse_name = meta_info[0]
            session_name = meta_info[1]
            cell_name = meta_info[3] + '_' + meta_info[4]
            data = mat_to_dict(loadmat(sessionfile))
            trials = self.extract_trial_data(data, bin_size=self.bin_size, t_pre=self.t_pre, t_post=self.t_post, mouse_name=mouse_name, session_name=session_name, cell_name=cell_name)
            Sessions.append(trials)
            mouse_names.append(mouse_name)
        self.Sessions = Sessions
        self.mouse_names = list(set(mouse_names))
        add_DA_at_CS_onset(self)
        add_first_anticipatory_lick_time(self)

    def extract_trial_data(self, d, bin_size, t_pre, t_post, mouse_name=None, session_name=None, cell_name=None):
        """
        Parameters:
            d (dict): The data containing trials of licks and spikes
            bin_size (float): The bin size (in seconds).
            t_pre (float): Time before odor onset to include (in seconds).
            t_post (float): Time after odor onset to include (in seconds).
        """
        trial_types = d['events']['trialType']
        spikes = np.array(d['responses']['spike'])
        licks = np.array(d['responses']['lick'])
        times = np.arange(-t_pre, t_post+1, bin_size)[:-1] / 1000

        odor_on = np.array(d['events']['odorOn'])
        
        results = []
        for i in range(len(odor_on)):
            if np.isnan(odor_on[i]):
                continue
            trial_result = {'licks': [], 'DA': [], 'times': times, 'baseline_DA_rate': None, 'trial_type': trial_types[i], 'trial_index': i, 'mouse': mouse_name, 'day_index': session_name, 'cell_name': cell_name}

            # Define the time window for this trial
            trial_start_time = odor_on[i] - t_pre
            trial_end_time = odor_on[i] + t_post
            
            # Discretize time into bins from trial_start_time to trial_end_time
            time_bins = np.arange(trial_start_time, trial_end_time+1, bin_size)

            # Find baseline spike rate (per bin)
            bsr = ((spikes >= odor_on[i]-1000) & (spikes <= odor_on[i])).sum() # Hz
            trial_result['baseline_DA_rate'] = bsr * bin_size / 1000

            # Extract spikes for this trial
            ixc = (spikes >= trial_start_time) & (spikes < trial_end_time)
            spikes_trial = spikes[ixc] # spikes on this trial
            spike_times_hist, _ = np.histogram(spikes_trial, time_bins)
            trial_result['DA'] = spike_times_hist

            # Extract licks for this trial
            ixc = (licks >= trial_start_time) & (licks < trial_end_time)
            licks_trial = licks[ixc] # licks on this trial
            lick_times_hist, _ = np.histogram(licks_trial, time_bins)
            trial_result['licks'] = lick_times_hist

            # Add the trial result to the overall list
            results.append(trial_result)
        
        # z-score
        rates = np.array([result['baseline_DA_rate'] for result in results])
        mu = np.mean(rates)
        sd = np.std(rates)
        for result in results:
            result['DA'] = (result['DA'] - mu) / sd

        return results

#%% Kim2020

class Kim2020:
    """
    VTA - ephys, 100% rewarded
    """
    __paper_url__ = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7736562/'
    __data_url__ = None

    def __init__(self, datadir=DATADIR):
        self.exp_name = 'Kim et al., 2020'
        self.datadir = os.path.join(datadir, 'Kim2020')
        self.f = h5py.File(os.path.join(self.datadir, 'DA_SU_ODOR_NEW.h5'))

        self.t_pre = 1500
        self.t_post = 5000
        self.bin_size = 50
        self.trial_types = {'all': [1,2,3,4], 'A': [1], 'B': [2], 'C': [3], 'D': [4]}
        self.reward_times = {'A': 0.6, 'B': 1.5, 'C': 3.75, 'D': 9.375} # secs
        self.times = np.arange(-self.t_pre, self.t_post+1, self.bin_size)[:-1] / 1000
        self.odor_onset = 0
        self.reward_onset = None

        # identify lick window which is the period right before reward onset
        self.odor_onset_bin = np.where(self.times >= self.odor_onset)[0][0]
        lick_bin_onset = self.odor_onset_bin
        self.lick_bin_windows = {}
        for cue, reward_time in self.reward_times.items():
            if reward_time <= self.times.max():
                lick_bin_offset = np.where(self.times >= reward_time)[0][0]
            else:
                lick_bin_offset = len(self.times)-1
            self.lick_bin_windows[cue] = (lick_bin_onset, lick_bin_offset)

        self.da_window_length = 0.5 # secs
        self.da_bin_length = int(1000*self.da_window_length/self.bin_size)
        self.da_bin_window = (self.odor_onset_bin, self.odor_onset_bin + self.da_bin_length)
        self.da_bin_window_pre = (self.odor_onset_bin - self.da_bin_length, self.odor_onset_bin)
        self.da_bin_starts = np.arange(0, 60)

        Sessions = []
        mouse_names = []
        lick_lookup = {}
        last_mouse_name = 0
        for cell_index in self.f.keys():
            
            # use hash of licks to identify sessions
            # n.b. we treat each session as a new mouse
            licks = np.array(self.f[cell_index]['lick'])[0,:]
            lick_hash = (licks.min(), licks.max())
            if lick_hash in lick_lookup:
                mouse_name = lick_lookup[lick_hash]
            else:
                last_mouse_name += 1
                mouse_name = last_mouse_name
                lick_lookup[lick_hash] = mouse_name
            mouse_names.append(mouse_name)
            
            trials = self.extract_trial_data(self.f[cell_index], bin_size=self.bin_size, t_pre=self.t_pre, t_post=self.t_post, mouse_name=mouse_name, day_index=0, cell_name=cell_index)
            Sessions.append(trials)
        self.Sessions = Sessions
        self.mouse_names = list(set(mouse_names))
        add_DA_at_CS_onset(self)
        add_first_anticipatory_lick_time(self)

    def extract_trial_data(self, d, bin_size, t_pre, t_post, mouse_name=None, day_index=None, cell_name=None):
        """
        d['grp_idx'][0,:] - condition index
        event[0,:] - trial start time
        event[1,:] - trial start cue
        event[2,:] - reward time
        event[4,:] - unknown, but ranges 2 - 5.5s after reward time
        event[8,:] - same as above, plus 1 or 2 ms

        Parameters:
            d (hdf5 group): The data containing trials of licks and spikes
            bin_size (float): The bin size (in seconds).
            t_pre (float): Time before odor onset to include (in seconds).
            t_post (float): Time after odor onset to include (in seconds).
        """
        trial_types = d['grp_idx'][0,:]
        spikes = np.array(d['spike'])
        licks = np.array(d['lick'])
        times = np.arange(-t_pre, t_post+1, bin_size)[:-1] / 1000

        event = np.array(d['event'])
        assert np.isnan(event[[2,5,6,7]]).all()
        # E = event[[0,1,3,4,8],:]
        # assert E.shape[0] == len(trial_types)
        odor_on = event[1,:]
        # trial_on = event[0,:]
        # rew_on = event[3,:]
        
        results = []
        for i in range(event.shape[1]):
            trial_result = {'licks': [], 'DA': [], 'times': times, 'baseline_DA_rate': None, 'trial_type': trial_types[i], 'trial_index': i, 'mouse': mouse_name, 'day_index': day_index, 'cell_name': cell_name}

            # Define the time window for this trial
            trial_start_time = odor_on[i] - t_pre
            trial_end_time = odor_on[i] + t_post
            
            # Discretize time into bins from trial_start_time to trial_end_time
            time_bins = np.arange(trial_start_time, trial_end_time+1, bin_size)

            # Find baseline spike rate (per bin)
            bsr = ((spikes >= odor_on[i]-1000) & (spikes <= odor_on[i])).sum() # Hz
            trial_result['baseline_DA_rate'] = bsr * bin_size / 1000

            # Extract spikes for this trial
            ixc = (spikes >= trial_start_time) & (spikes < trial_end_time)
            spikes_trial = spikes[ixc] # spikes on this trial
            spike_times_hist, _ = np.histogram(spikes_trial, time_bins)
            trial_result['DA'] = spike_times_hist

            # Extract licks for this trial
            ixc = (licks >= trial_start_time) & (licks < trial_end_time)
            licks_trial = licks[ixc] # licks on this trial
            lick_times_hist, _ = np.histogram(licks_trial, time_bins)
            trial_result['licks'] = lick_times_hist

            # Add the trial result to the overall list
            results.append(trial_result)
        
        # z-score
        rates = np.array([result['baseline_DA_rate'] for result in results])
        mu = np.mean(rates)
        sd = np.std(rates)
        for result in results:
            result['DA'] = (result['DA'] - mu) / sd
        
        return results

#%% Amo2024

class Amo2024:
    """
    CC_Glu: (VTA, fluorometry w/ glutamate sensor)
    CC_GCaMP: (VTA, fluorometry w/ calcium sensor)
    CC_DA: (VS, fluorometry w/ dopamine sensor)
    """
    __paper_url__ = 'https://doi.org/10.1016/j.neuron.2023.12.019'
    __data_url__ = 'https://datadryad.org/stash/dataset/doi:10.5061/dryad.ncjsxkt25'

    def __init__(self, key='CC_DA', datadir=DATADIR):
        """
        1: Free reward trial (reward at 1001)
        9: 80% reward predicting cue followed by reward (cue at 1001 and reward at 4001)
        10: 80% reward predicting cue followed by reward omission (cue at 1001)
        11: 40% reward predicting cue followed by reward (cue at 1001 and reward at 4001)
        12: 40% reward predicting cue followed by reward omission (cue at 1001)
        14: 0% reward (nothing) predicting cue (cue at 1001)
        """
        self.exp_name = f"Amo et al., 2024 ({key.replace('CC_', '')})"
        self.datadir = os.path.join(datadir, 'Amo2024')
        self.key = key

        self.sessionfiles = glob.glob(os.path.join(self.datadir, self.key + '_*.mat'))
        self.bin_size = 50
        self.t_pre = 1000
        self.t_post = 6000
        self.odor_onset = 0
        self.reward_onset = 3
        self.times = np.arange(-self.t_pre, self.t_post+1, self.bin_size)[:-1] / 1000
        self.trial_types = {'A': [9,10], 'B': [11,12], 'nogo': [14], 'all': [9,10,11,12,14]}

        self.odor_onset_bin = np.where(self.times >= self.odor_onset)[0][0]
        lick_bin_onset = self.odor_onset_bin
        lick_bin_offset = np.where(self.times >= self.reward_onset)[0][0]
        self.lick_bin_windows = {cue: (lick_bin_onset, lick_bin_offset) for cue in self.trial_types}

        self.da_window_length = 0.5 # secs
        self.da_bin_length = int(1000*self.da_window_length/self.bin_size)
        self.da_bin_window = (self.odor_onset_bin, self.odor_onset_bin + self.da_bin_length)
        self.da_bin_window_pre = (self.odor_onset_bin - self.da_bin_length, self.odor_onset_bin)
        self.da_bin_starts = np.arange(0, 60)

        Sessions = []
        mouse_names = []
        for i, sessionfile in enumerate(self.sessionfiles):
            data = loadmat(sessionfile)

            # get meta data about session from its filename
            meta_info = sessionfile.replace('.mat', '').split('_') # e.g., 'CC_DA_524_session1'
            assert(len(meta_info) == 4)
            mouse_name = meta_info[2]
            mouse_names.append(mouse_name)
            day_index = int(meta_info[3].replace('session', ''))
            cell_name = 0
            
            # load all trials in session
            trials = self.extract_trial_data(data, bin_size=self.bin_size, t_pre=self.t_pre, t_post=self.t_post, mouse_name=mouse_name, day_index=day_index, cell_name=cell_name)
            Sessions.append(trials)
        
        self.mouse_names = list(set(mouse_names))
        self.Sessions = Sessions
        add_DA_at_CS_onset(self, baseline_center=False, use_mean_spikes=False)
        add_first_anticipatory_lick_time(self)
    
    def smooth_data(self, x, bins):
        y = np.zeros(len(bins)-1)
        for i in range(len(bins) - 1):
            y[i] = np.mean(x[bins[i]:bins[i+1]])
        return y

    def extract_trial_data(self, d, bin_size, t_pre, t_post, mouse_name=None, day_index=None, cell_name=None, baseline_center=True):
        """
        Parameters:
            d (dict): The data containing trials of licks and spikes
            bin_size (float): The bin size (in seconds).
            t_pre (float): Time before odor onset to include (in seconds).
            t_post (float): Time after odor onset to include (in seconds).
        """
        trial_types = d['TrialType'][:,0]
        spikes = np.array(d['dFoverFGCaMP2'])
        licks = np.array(d['lickmatrix'])
        times = np.arange(-t_pre, t_post+1, bin_size)[:-1] / 1000
        odor_on = 1001

        if len(trial_types) != spikes.shape[0]:
            # n.b. ryu said this is what to do
            trial_types = trial_types[:spikes.shape[0]]
        
        results = []
        for i in range(spikes.shape[1]): # for each trial
            trial_result = {'licks': [], 'DA': [], 'baseline_DA_rate': None, 'times': times, 'trial_type': trial_types[i], 'mouse': mouse_name, 'day_index': day_index, 'trial_index': i, 'cell_name': cell_name}

            # Define the time window for this trial
            trial_start_time = odor_on - t_pre
            trial_end_time = odor_on + t_post
            
            # Discretize time into bins from trial_start_time to trial_end_time
            time_bins = np.arange(trial_start_time, trial_end_time+1, bin_size)

            trial_result['DA'] = self.smooth_data(spikes[:,i].T, time_bins)

            # Find baseline spike rate (per bin)
            ixt = (times >= -1) & (times < 0)
            trial_result['baseline_DA_rate'] = trial_result['DA'][ixt].mean()
            
            if baseline_center:
                trial_result['DA'] = trial_result['DA'] - trial_result['baseline_DA_rate']

            # Extract licks for this trial
            licks_trial = np.where(licks[:,i])[0] # licks on this trial
            ixc = (licks_trial >= trial_start_time) & (licks_trial < trial_end_time)
            clicks_trial = licks_trial[ixc]
            lick_times_hist, _ = np.histogram(clicks_trial, time_bins)
            trial_result['licks'] = lick_times_hist

            # Add the trial result to the overall list
            results.append(trial_result)
        return results
