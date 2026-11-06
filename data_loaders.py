#%% imports

import glob
import os.path
import numpy as np
from scipy.io import loadmat
import h5py
import pandas as pd

# set base data directory to home user folder
DATADIR = os.path.expanduser('~/Documents/Work/Data/vta-data')

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
            if 'reward_onset' in trial:
                reward_onset = trial['reward_onset']
            elif D.reward_onset is not None:
                reward_onset = D.reward_onset
            else:
                trial_type = int(trial['trial_type'])
                odor = [x for x,ys in D.trial_types.items() if trial_type in ys and x != 'all'][0]
                reward_onset = D.reward_times[odor]
            ixAnticip = (times > D.odor_onset) & (times < reward_onset)
            anticipatory_licks = licks[ixAnticip]
            if not np.any(anticipatory_licks > 0):
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
    __zip_url__ = 'https://drive.google.com/file/d/1fxXSRjgUev4AKNvFyAHuY10xNJLXoxcK/view?usp=drive_link'

    def __init__(self, mouse_names=None, datadir=DATADIR):
        self.exp_name = 'Qian & Burrell et al., 2025'
        # n.b. this is data from just one fiber, but there were more in other parts of VS
        self.datadir = os.path.join(datadir, 'QianBurrell2025')
        if not os.path.exists(self.datadir):
            print(f"QianBurrell2025: Data directory {self.datadir} not found. Please download from {self.__zip_url__} and place the uncompressed folder named 'QianBurrell2025' in {DATADIR=}.")
            return None
        if not os.path.exists(os.path.join(self.datadir, 'lick_data.parquet')) or not os.path.exists(os.path.join(self.datadir, 'photo_data.parquet')):
            print(f"QianBurrell2025: Data files should be accessible at {self.datadir}/lick_data.parquet and {self.datadir}/photo_data.parquet. Please download from {self.__zip_url__} and place the uncompressed folder named 'QianBurrell2025' in {DATADIR=}.")
            return None
        self.dfl = pd.read_parquet(os.path.join(self.datadir, 'lick_data.parquet'))
        self.dfd = pd.read_parquet(os.path.join(self.datadir, 'photo_data.parquet'))
        
        self.odor_onset = 2.01
        self.sampling_rate = 20
        self.bin_size = int(1000 / self.sampling_rate)
        self.reward_onset = self.odor_onset + 3.5
        if mouse_names is not None:
            self.mouse_names = mouse_names
        else:
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
        
        self.Sessions = self.load_all_data(self.dfl, self.dfd, self.odor_onset, self.reward_onset, self.sampling_rate, self.mouse_names)
    
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
        if not self.sessionfiles:
            return None
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
        if not self.sessionfiles:
            return None
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

#%% Starkweather2017

class Starkweather2017:
    """
    VTA - ephys
    """
    __paper_url__ = 'https://www.nature.com/articles/nn.4520'
    __data_url__ = 'https://crcns.org/data-sets/bst/vta-1/about-vta-1'

    def __init__(self, key='Task1', datadir=DATADIR, skip_missing_licks=False,
                 combine_neurons=False, year='2017', condition=None,
                 skip_unknown_trial_types=True, skip_task2b=True):
        self.year = str(year)
        self.exp_name = f'Starkweather et al., {self.year}'
        if self.year not in ('2017', '2018'):
            raise ValueError("year must be '2017' or '2018'")
        valid_tasks = ('Task1', 'Task2', 'Task2b', None) if self.year == '2017' else ('Task1', 'Task2', None)
        if key not in valid_tasks:
            raise ValueError(f'key must be one of {valid_tasks} for {self.year}')
        if condition not in (None, 'Saline', 'SalB'):
            raise ValueError('condition must be Saline, SalB, or None')
        if self.year == '2017' and condition is not None:
            raise ValueError('condition is available only for 2018')
        self.condition = condition
        self.key = key
        self.skip_missing_licks = skip_missing_licks
        self.skip_unknown_trial_types = skip_unknown_trial_types
        self.skip_task2b = skip_task2b
        self.combine_neurons = combine_neurons
        self.datadir = os.path.join(datadir, 'Starkweather', self.year)
        
        self.bin_size = 50 # ms
        self.t_pre = 1000 # ms
        self.t_post = 4000 # ms
        self.times = np.arange(-self.t_pre, self.t_post, self.bin_size) / 1000
        self.odor_onset = 0.0
        self.reward_onset = None
        self.reward_times = {'A': 2.0, 'B': 2.0 if key == 'Task2b' else 1.2,
                             'C': 2.8, 'D': 2.0}
        self.trial_types = {'all': [1,2,3,4], 'A': [1], 'B': [2],
                            'C': [3], 'D': [4]}
        if self.year == '2018':
            self.trial_types['all'].append(0)
            self.trial_types['unknown'] = [0]
        self.odor_onset_bin = np.where(self.times >= self.odor_onset)[0][0]
        self.lick_bin_windows = {
            cue: (self.odor_onset_bin, np.where(self.times >= onset)[0][0])
            for cue, onset in self.reward_times.items()}
        self.lick_bin_windows['all'] = (
            self.odor_onset_bin, np.where(self.times >= 2.0)[0][0])
        if self.year == '2018':
            self.lick_bin_windows['unknown'] = self.lick_bin_windows['all']
        self.da_window_length = 0.5
        self.da_bin_length = int(1000*self.da_window_length/self.bin_size)
        self.da_bin_window = (self.odor_onset_bin,
                              self.odor_onset_bin + self.da_bin_length)
        self.da_bin_window_pre = (self.odor_onset_bin - self.da_bin_length,
                                  self.odor_onset_bin)
        self.da_bin_starts = np.arange(0, 60)

        neurons = pd.read_parquet(os.path.join(self.datadir, 'neurons.parquet'))
        trials = pd.read_parquet(os.path.join(self.datadir, 'trials.parquet'))
        spikes = pd.read_parquet(os.path.join(self.datadir, 'spikes.parquet'))
        if self.year == '2018':
            # Adapt CRCNS identifiers and source codes to the 2017 schema.
            neurons = neurons.rename(columns={'crcns_file': 'neuron_id'})
            trials = trials.rename(columns={'crcns_file': 'neuron_id'})
            spikes = spikes.rename(columns={'crcns_file': 'neuron_id'})
            odor_by_code = {**{code: 'A' for code in range(1, 10)},
                            10: 'B', 11: 'C', 13: 'A', 14: 'B',
                            15: 'C', 16: 'D'}
            trials['odor'] = trials.trial_type.map(odor_by_code)
            lick_sources = neurons.set_index('neuron_id').lick_source
            trials['lick_aligned'] = (
                trials.neuron_id.map(lick_sources).eq('csc64') &
                trials.licks_rel_ms.notna())
            if condition is not None:
                neurons = neurons[neurons.condition == condition]
        if key is not None:
            neurons = neurons[neurons.task == key]
        elif self.skip_task2b:
            neurons = neurons[neurons.task != 'Task2b']
        trial_groups = trials.groupby('neuron_id', sort=False)
        spike_groups = {cell: group.spike_ms.to_numpy()
                        for cell, group in spikes.groupby('neuron_id', sort=False)}
        self.Sessions = []
        for neuron in neurons.itertuples(index=False):
            data = trial_groups.get_group(neuron.neuron_id).sort_values('trial')
            session = self.extract_trial_data(
                data, spike_groups.get(neuron.neuron_id, np.array([])), neuron)
            if self.skip_unknown_trial_types:
                session = [trial for trial in session
                           if pd.notna(trial['original_trial_type'])
                           and trial['trial_type'] != 0]
            if self.skip_missing_licks:
                session = [trial for trial in session if trial['lick_aligned']]
            if session:
                self.Sessions.append(session)
        if self.combine_neurons:
            self.Sessions = self._combine_neuron_sessions(self.Sessions)
        self.mouse_names = sorted({session[0]['mouse'] for session in self.Sessions})
        add_DA_at_CS_onset(self)
        add_first_anticipatory_lick_time(self)

    def _combine_neuron_sessions(self, sessions):
        """
        align trials by original index and average normalized neural data
        """
        recordings = {}
        for session in sessions:
            key = (session[0]['mouse'], session[0]['day_index'])
            recordings.setdefault(key, []).append(session)
        combined_sessions = []
        for (mouse, day_index), neuron_sessions in recordings.items():
            neuron_ids = tuple(session[0]['cell_name'] for session in neuron_sessions)
            trials_by_neuron = [{trial['trial_index']: trial for trial in session}
                                for session in neuron_sessions]
            indices = sorted(trials_by_neuron[0])
            if any(sorted(trials) != indices for trials in trials_by_neuron):
                raise ValueError(f'Cannot combine {mouse}, {day_index}: trial indices differ')
            combined = []
            for index in indices:
                trials = [lookup[index] for lookup in trials_by_neuron]
                first = trials[0]
                for trial in trials[1:]:
                    fields = ('trial_type', 'task', 'reward_onset', 'rewarded',
                              'lick_aligned')
                    if self.year == '2018':
                        fields += ('condition',)
                    if (any(trial[field] != first[field] for field in fields) or
                            not np.array_equal(trial['licks'], first['licks'],
                                               equal_nan=True)):
                        raise ValueError(
                            f'Cannot combine {mouse}, {day_index}, trial {index}: '
                            'task events or licking differ between neurons')
                trial = first.copy()
                trial['DA_per_neuron'] = np.column_stack([t['DA'] for t in trials])
                trial['DA'] = trial['DA_per_neuron'].mean(axis=1)
                trial['baseline_DA_rate_per_neuron'] = np.array(
                    [t['baseline_DA_rate'] for t in trials])
                trial['baseline_DA_rate'] = trial['baseline_DA_rate_per_neuron'].mean()
                trial['neuron_ids'] = neuron_ids
                trial['cell_name'] = f'{mouse}_{day_index}'
                combined.append(trial)
            combined_sessions.append(combined)
        return combined_sessions

    def extract_trial_data(self, trials, spikes, neuron):
        """
        bin neuralynx-clock events (ms), then normalize per neuron
        """
        edges = np.arange(-self.t_pre, self.t_post+1, self.bin_size)
        odor_codes = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        results = []
        for trial in trials.itertuples(index=False):
            if pd.isna(trial.odor_on_ms): # free reward: no CS to align to
                continue
            relative_spikes = spikes - trial.odor_on_ms
            in_window = ((relative_spikes >= -self.t_pre) &
                         (relative_spikes < self.t_post))
            counts, _ = np.histogram(relative_spikes[in_window], edges)
            baseline = ((relative_spikes >= -1000) &
                        (relative_spikes <= 0)).sum() * self.bin_size / 1000
            lick_aligned = bool(trial.lick_aligned) and trial.licks_rel_ms is not None
            if lick_aligned:
                lick_times = np.asarray(trial.licks_rel_ms, dtype=float)
                in_window = (lick_times >= -self.t_pre) & (lick_times < self.t_post)
                licks, _ = np.histogram(lick_times[in_window], edges)
            else:
                licks = np.full(len(self.times), np.nan)
            expected_delay = self.reward_times.get(trial.odor, np.nan)
            if trial.odor == 'B':
                expected_delay = 2.0 if neuron.task == 'Task2b' else 1.2
            rewarded = pd.notna(trial.reward_on_ms)
            reward_onset = ((trial.reward_on_ms - trial.odor_on_ms) / 1000
                            if rewarded else expected_delay)
            results.append({
                'licks': licks, 'DA': counts, 'times': self.times,
                'baseline_DA_rate': baseline,
                'trial_type': odor_codes.get(trial.odor, 0),
                'trial_index': trial.trial, 'mouse': neuron.mouse,
                'day_index': neuron.nlx_stamp, 'cell_name': neuron.neuron_id,
                'original_trial_type': trial.trial_type, 'task': neuron.task,
                'reward_onset': reward_onset, 'rewarded': rewarded,
                'lick_aligned': lick_aligned})
            if self.year == '2018':
                results[-1]['condition'] = neuron.condition
        if results:
            rates = np.array([trial['baseline_DA_rate'] for trial in results])
            mu, sd = rates.mean(), rates.std()
            # A constant baseline cannot define a z-score; keep DA finite.
            for trial in results:
                trial['DA'] = (trial['DA'] - mu) / (sd if sd > 0 else 1.0)
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
        filename = os.path.join(self.datadir, 'DA_SU_ODOR_NEW.h5')
        if not os.path.exists(filename):
            return None
        self.f = h5py.File(filename)

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
        if not self.sessionfiles:
            return None
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
