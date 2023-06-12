import numpy as np
from scipy.signal import correlate

from os.path import join as pjoin
from os import listdir
from typing import List, Tuple, Dict
from glob import glob

import pandas as pd
import numpy as np

from lib.conversion import one_to_zero_cell, zero_to_one_cell

behaviour_root = pjoin("data", "behaviour_data")

# calculate the mean of a centered moving window, if window is not complete, 
# use the available data without padding
def moving_window_mean(data: np.ndarray, window_size=5) -> np.ndarray:
    data_len = data.size
    output = np.zeros(data_len)

    for i in range(data_len):
        if i < window_size // 2:
            output[i] = np.mean(data[:i+window_size//2+1])
        elif i > data_len - window_size // 2 - 1:
            output[i] = np.mean(data[i-window_size//2:])
        else:
            output[i] = np.mean(data[i-window_size//2:i+window_size//2+1])

    return output

# calculate the mean of a moving window whose right end is the current index
# if window is not complete, use the available data without padding 
def moving_window_mean_prior(data: np.ndarray, window_size=5) -> np.ndarray:
    data_len = data.size
    output = np.zeros(data_len)

    for i in range(data_len):
        if i < window_size:
            output[i] = np.mean(data[:i+1])
        else:
            output[i] = np.mean(data[i-window_size:i+1])

    return output

# calculate the firing rate of a neuron in a given window wrt cue time
def get_firing_rate_window(cue_times: np.ndarray, spike_times:np.ndarray, window_left: float, window_right: float) -> np.ndarray:
    spike_ptr = 0

    firing_rates = []

    for cue in cue_times:
        cur_count = 0

        window_left_cur = cue + window_left
        window_right_cur = cue + window_right

        # move the pointers into window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_left_cur:
            spike_ptr += 1
        
        # count the amount of spikes in window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_right_cur:
            spike_ptr += 1
            cur_count += 1
        
        
        firing_rates.append(cur_count / (window_right - window_left))       

    return firing_rates


def get_relative_spike_times(spike_times: np.ndarray, cue_times: np.ndarray, window_left: float, window_right: float) -> np.ndarray:
    spike_ptr = 0

    relative_spike_times = []

    for cue in cue_times:
        relative_spike_times_trial = []
        window_left_cur = cue + window_left
        window_right_cur = cue + window_right

        # move the pointers into window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_left_cur:
            spike_ptr += 1
        
        # count the amount of spikes in window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_right_cur:
            relative_spike_times_trial.append(spike_times[spike_ptr] - cue)
            spike_ptr += 1
        
        relative_spike_times.append(relative_spike_times_trial)
        
    return relative_spike_times


# calculate the normalized cross correlation (Wei's standard) of two signals
def get_normalized_cross_correlation(pfc_trial_times: np.ndarray, str_trial_times: np.ndarray) -> np.ndarray:
    # create constant signal with the mean of the times
    pfc_trial_times_const = np.ones(len(pfc_trial_times)) * np.mean(pfc_trial_times)
    str_trial_times_const = np.ones(len(str_trial_times)) * np.mean(str_trial_times)

    # cross correlate the relative time signals
    cross_cor = correlate(str_trial_times, pfc_trial_times, mode='same')
    cross_cor_const = correlate(str_trial_times_const, pfc_trial_times_const, mode='same') 

    # calculate normalized cross correlation
    normalized_cross_corr = np.divide(cross_cor - cross_cor_const, cross_cor_const, out=np.zeros_like(cross_cor_const), where=cross_cor_const!=0)

    return normalized_cross_corr

# calculate the performance of all sessions using cross-correlation metric
# returns a dictionary of session name to performance, and the average performance
def get_session_performances() -> Tuple[Dict[str, float], float]:
    results = {}
    performances = []
    # calculates the xcorr between rightP and proportion of rightward resonse
    for behaviour_path in glob(pjoin(behaviour_root, '*.csv')):
        session_name = behaviour_path.split('/')[-1].split('.')[0]
        session_data = pd.read_csv(behaviour_path)
        right_prob = session_data['rightP']
        left_prob = session_data['leftP']
        right_proportion = np.convolve((session_data['trial_response_side']==1).to_numpy(), np.ones(20)/20, mode='same')
        left_proportion = np.convolve((session_data['trial_response_side']==-1).to_numpy(), np.ones(20)/20, mode='same')
        # normalization
        right_prob = np.array(right_prob) / np.linalg.norm(right_prob)
        left_prob = np.array(left_prob) / np.linalg.norm(left_prob)
        right_proportion = right_proportion / np.linalg.norm(right_proportion)
        left_proportion = left_proportion / np.linalg.norm(left_proportion)

        xcorr_right = np.correlate(right_prob, right_proportion)
        xcorr_left = np.correlate(left_prob, left_proportion)
        corrs = [xcorr_left, xcorr_right]
        corrs_averaged = np.mean(corrs, axis=0)
        performance = np.max(corrs_averaged)
        performances.append(performance)
        results[session_name] = performance
    
    return results, np.mean(performances)


# relative positions to cue_time
ITI_LEFT = -1
ITI_RIGHT = 0
RESPONSE_LEFT = 0
RESPONSE_RIGHT = 1.5


# get the firing rate 
def get_response_bg_firing(cue_times, spike_times):
    ptr = 0
    response_mag = []
    bg_firing = []

    for cue in cue_times:
        iti_count = 0
        response_count = 0

        iti_left = cue + ITI_LEFT
        iti_right = cue + ITI_RIGHT
        response_left = cue + RESPONSE_LEFT
        response_right = cue + RESPONSE_RIGHT

        # move the pointers into iti window
        while ptr < len(spike_times) and spike_times[ptr] < iti_left:
            ptr += 1

        # count the amount of spikes in iti
        while ptr < len(spike_times) and spike_times[ptr] < iti_right:
            ptr += 1
            iti_count += 1

        # move the pointer to response time window
        while ptr < len(spike_times) and spike_times[ptr] < response_left:
            ptr += 1

        # count the amount of spikes in response time
        while ptr < len(spike_times) and spike_times[ptr] < response_right:
            ptr += 1
            response_count += 1
        
        bg_firing.append(iti_count / (ITI_RIGHT - ITI_LEFT))
        response_mag.append(abs(response_count / (RESPONSE_RIGHT - RESPONSE_LEFT) - iti_count / (ITI_RIGHT - ITI_LEFT)))

    return response_mag, bg_firing