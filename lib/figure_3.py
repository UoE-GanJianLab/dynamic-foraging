from os.path import join as pjoin, isdir
from os import listdir
from glob import glob

import numpy as np
import pandas as pd
import tqdm

from lib.calculation import moving_window_mean, get_relative_spike_times

def figure_3_panel_bc():
    spike_time_dir = pjoin('data', 'spike_times')
    relative_spike_time_dir = pjoin('data', 'relative_spike_time_trials')
    behaviour_root = pjoin('data', 'behaviour_data')

    # iterate through the sessions
    for session_name in tqdm.tqdm(listdir(behaviour_root)):
        if isdir(pjoin(behaviour_root, session_name)):
            continue

        # loead the behaviour data
        behaviour_path = pjoin(behaviour_root, session_name)
        behaviour_data = pd.read_csv(behaviour_path)
        cue_times = behaviour_data['cue_time']

        # get the index of nan trials in behaviour data as signal only trials
        signal = behaviour_data['trial_response_side'].isna()
        signal_idx = np.where(signal)[0]

        # get the index of unrewarded non_nan trials as movement trials
        mvt = behaviour_data['trial_response_side'].notna()
        mvt = mvt & (behaviour_data['trial_reward'] == 0) 
        mvt_idx = np.where(mvt)[0]

        # get the index of rewarded non_nan trials as reward trials
        reward = behaviour_data['trial_response_side'].notna()
        reward = reward & (behaviour_data['trial_reward'] == 1)
        reward_idx = np.where(reward)[0]


        # load the pfc spike times
        for pfc_times in glob(pjoin(spike_time_dir, session_name, 'pfc_*')):
            pfc_times = np.load(pfc_times)


