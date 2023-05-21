import numpy as np
from scipy.signal import correlate

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

def moving_window_mean_prior(data: np.ndarray, window_size=5) -> np.ndarray:
    data_len = data.size
    output = np.zeros(data_len)

    for i in range(data_len):
        if i < window_size:
            output[i] = np.mean(data[:i+1])
        else:
            output[i] = np.mean(data[i-window_size:i+1])

    return output

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