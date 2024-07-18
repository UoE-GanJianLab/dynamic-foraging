from lib.calculation import get_relative_spike_times
from os.path import join as pjoin
from os import listdir
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import tqdm

data_root = pjoin('data', 'relative_values')

session_names = []
cell_names = []
background_firing_pearson_r = []
background_firing_p_values = []
response_firing_pearson_r = []
response_firing_p_values = []

for session in tqdm.tqdm(glob(pjoin(data_root, '*.npy'))):
    relative_values = np.load(session)
    session_name = session.split('/')[-1].split('.')[0]

    # load behaviour data
    session_behaviour_path = pjoin('data', 'behaviour_data', session_name + '.csv')
    session_behaviour = pd.read_csv(session_behaviour_path)
    # remove nan trials
    session_behaviour = session_behaviour[~session_behaviour['trial_response_side'].isna()]
    cue_times = session_behaviour['cue_time'].values

    # load the spike times of the session
    session_spike_time_path = pjoin('data', 'spike_times', 'sessions', session_name)

    for cell in glob(pjoin(session_spike_time_path, '*_*')):
        cell_name = cell.split('/')[-1].split('.')[0]
        cell_spike_times = np.load(cell)
        background_firing = get_relative_spike_times(cell_spike_times, cue_times, window_left=-1, window_right=-.5)
        response_firing = get_relative_spike_times(cell_spike_times, cue_times, window_left=0, window_right=1.5)
        # calculate the firing rate
        background_firing_rate = [len(x)/.5 for x in background_firing]
        response_firing_rate = [len(x)/1.5 for x in response_firing]

        # calculate the pearson correlation and p value with the relative values
        background_firing_pearson_r.append(pearsonr(background_firing_rate, relative_values)[0])
        background_firing_p_values.append(pearsonr(background_firing_rate, relative_values)[1])
        response_firing_pearson_r.append(pearsonr(response_firing_rate, relative_values)[0])
        response_firing_p_values.append(pearsonr(response_firing_rate, relative_values)[1])

        session_names.append(session_name)
        cell_names.append(cell_name)

# save the results as a csv file
result = pd.DataFrame({'session_name': session_names,
                        'cell_name': cell_names,
                        'background_firing_pearson_r': background_firing_pearson_r,
                        'background_firing_p_values': background_firing_p_values,
                        'response_firing_pearson_r': response_firing_pearson_r,
                        'response_firing_p_values': response_firing_p_values})
result.to_csv(pjoin('data', 'relative_value_correlation.csv'), index=False)


