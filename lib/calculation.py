import numpy as np
from scipy.signal import correlate # type: ignore

from os.path import join as pjoin
from os import listdir
from typing import List, Tuple, Dict
from glob import glob

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.stats import circmean # type: ignore
from astropy.stats import rayleightest
from lib.conversion import one_to_zero_cell, zero_to_one_cell

behaviour_root = pjoin("data", "behaviour_data")

# relative positions to cue_time
ITI_LEFT = -1
ITI_RIGHT = -0.5
RESPONSE_LEFT = 0
RESPONSE_RIGHT = 1.5


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


def _check_arg(x, xname):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('%s must be one-dimensional.' % xname)
    return x

# crosscorrelation with maxlag
def crosscorrelation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.
    """
    x = _check_arg(x, 'x')
    y = _check_arg(y, 'y')
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)

# calculate the firing rate of a neuron in a given window wrt cue time
def get_spikes_in_window(cue_times: np.ndarray, spike_times:np.ndarray, window_left: float, window_right: float) -> np.ndarray:
    spike_ptr = 0

    in_window_spikes = []

    for cue in cue_times:
        window_left_cur = cue + window_left
        window_right_cur = cue + window_right

        # backtrack in case of overlapping cue time windows
        while spike_ptr > 0 and spike_times[spike_ptr-1] > window_left_cur:
            spike_ptr -= 1

        # move the pointers into window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_left_cur:
            spike_ptr += 1
        
        # count the amount of spikes in window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_right_cur:
            in_window_spikes.append(spike_times[spike_ptr])
            spike_ptr += 1

    in_window_spikes = np.array(in_window_spikes)
        
    return in_window_spikes


def get_spikes_outside_window(cue_times: np.ndarray, spike_times:np.ndarray, window_left: float, window_right: float) -> np.ndarray:
    spike_ptr = 0

    outside_window_spikes = []

    for cue in cue_times:

        window_left_cur = cue + window_left
        window_right_cur = cue + window_right

        # move the pointers into window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_left_cur:
            outside_window_spikes.append(spike_times[spike_ptr])
            spike_ptr += 1

        # count the amount of spikes in window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_right_cur:
            spike_ptr += 1

        
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_right_cur + 1:
            outside_window_spikes.append(spike_times[spike_ptr])
            spike_ptr += 1

    outside_window_spikes = np.array(outside_window_spikes)
        
    return outside_window_spikes


# calculate the firing rate of a neuron in a given window wrt cue time
def get_firing_rate_window(cue_times: np.ndarray, spike_times:np.ndarray, window_left: float, window_right: float) -> List[float]:
    spike_ptr = 0

    firing_rates = []

    for cue in cue_times:
        cur_count = 0

        window_left_cur = cue + window_left
        window_right_cur = cue + window_right

        # backtrack in case of overlapping cue time windows
        while spike_ptr > 0 and spike_times[spike_ptr-1] > window_left_cur:
            spike_ptr -= 1

        # move the pointers into window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_left_cur:
            spike_ptr += 1
        
        # count the amount of spikes in window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_right_cur:
            spike_ptr += 1
            cur_count += 1
        
        
        firing_rates.append(cur_count / (window_right - window_left))       

    return firing_rates


def get_relative_spike_times(spike_times: np.ndarray, cue_times: np.ndarray, window_left: float, window_right: float) -> List[List[float]]:
    spike_ptr = 0

    relative_spike_times = []

    for cue in cue_times:
        relative_spike_times_trial = []
        window_left_cur = cue + window_left
        window_right_cur = cue + window_right

        # backtrack in case of overlapping cue time windows
        while spike_ptr > 0 and spike_times[spike_ptr-1] > window_left_cur:
            spike_ptr -= 1

        # move the pointers into window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_left_cur:
            spike_ptr += 1
        
        # count the amount of spikes in window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_right_cur:
            relative_spike_times_trial.append(spike_times[spike_ptr] - cue)
            spike_ptr += 1
        
        relative_spike_times.append(relative_spike_times_trial)
        
    return relative_spike_times

def get_relative_firing_rate_binned(spike_times: np.ndarray, cue_times: np.ndarray, window_left: float, window_right: float, bin_size: int) -> List[List[float]]:
    relative_spike_times = get_relative_spike_times_flat(spike_times, cue_times, window_left, window_right)
    relative_spike_times_binned = bin_array(relative_spike_times, window_left, window_right, bin_size)
    relative_firing_rate = relative_spike_times_binned / (len(cue_times)*bin_size)
    return relative_firing_rate
    

def get_relative_spike_times_flat(spike_times: np.ndarray, cue_times: np.ndarray, window_left: float, window_right: float) -> List[List[float]]:
    spike_ptr = 0

    relative_spike_times = []

    for cue in cue_times:
        window_left_cur = cue + window_left
        window_right_cur = cue + window_right

        # backtrack in case of overlapping cue time windows
        while spike_ptr > 0 and spike_times[spike_ptr-1] > window_left_cur:
            spike_ptr -= 1

        # move the pointers into window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_left_cur:
            spike_ptr += 1
        
        # count the amount of spikes in window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_right_cur:
            relative_spike_times.append(spike_times[spike_ptr] - cue)
            spike_ptr += 1
                
    return relative_spike_times


def get_spike_times_in_window(spike_times: np.ndarray, cue_times: np.ndarray, window_left: float, window_right: float) -> List[List[float]]:
    spike_ptr = 0

    relative_spike_times = []

    for cue in cue_times:
        window_left_cur = cue + window_left
        window_right_cur = cue + window_right

        cur_spikes = []

        # backtrack in case of overlapping cue time windows
        while spike_ptr > 0 and spike_times[spike_ptr-1] > window_left_cur:
            spike_ptr -= 1

        # move the pointers into window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_left_cur:
            spike_ptr += 1
        
        # count the amount of spikes in window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_right_cur:
            cur_spikes.append(spike_times[spike_ptr])
            spike_ptr += 1

        relative_spike_times.append(cur_spikes)
                
    return relative_spike_times


def get_relative_spike_times_brute_force(spike_times: np.ndarray, cue_times: np.ndarray, window_left: float, window_right: float) -> List[List[float]]:
    # use O(n^2) algorithm to calculate relative spike times
    relative_spike_times = []

    for cue in cue_times:

        for spike in spike_times:
            if spike > cue + window_left and spike < cue + window_right:
                relative_spike_times.append(spike - cue)
        

    return relative_spike_times


# calculate the normalized cross correlation (Wei's standard) of two signals
def get_normalized_cross_correlation(pfc_trial_times: np.ndarray, str_trial_times: np.ndarray, maxlag: int = 0) -> np.ndarray:
    # create constant signal with the mean of the times
    pfc_trial_times_const = np.ones(len(pfc_trial_times)) * np.mean(pfc_trial_times)
    str_trial_times_const = np.ones(len(str_trial_times)) * np.mean(str_trial_times)

    # cross correlate the relative time signals
    cross_cor = crosscorrelation(str_trial_times, pfc_trial_times, maxlag)
    cross_cor_const = crosscorrelation(str_trial_times_const, pfc_trial_times_const, maxlag) 

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
        right_prob = np.array(session_data['rightP'])
        left_prob = np.array(session_data['leftP'])
        right_proportion = moving_window_mean((session_data['trial_response_side']==1).to_numpy(), 20)
        left_proportion = moving_window_mean((session_data['trial_response_side']==-1).to_numpy(), 20)

        # subtract the mean of each signal
        # right_prob = right_prob - np.mean(right_prob)
        # left_prob = left_prob - np.mean(left_prob)
        # right_proportion = right_proportion - np.mean(right_proportion)
        # left_proportion = left_proportion - np.mean(left_proportion)        

        xcorr_right = crosscorrelation(right_prob, right_proportion, maxlag=50)
        xcorr_left = crosscorrelation(left_prob, left_proportion, maxlag=50)

        # normalize the result to simulate the effect of matlab coeff mode
        xcorr_right = xcorr_right / np.sqrt(np.sum(right_prob**2) * np.sum(right_proportion**2))
        xcorr_left = xcorr_left / np.sqrt(np.sum(left_prob**2) * np.sum(left_proportion**2))

        corrs = [xcorr_left, xcorr_right]
        corrs_averaged = np.mean(corrs, axis=0)
        performance = np.max(corrs_averaged)
        performances.append(performance)
        results[session_name] = performance
    
    return results, np.mean(performances, dtype=float)



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

    response_mag = np.array(response_mag)
    bg_firing = np.array(bg_firing)

    return response_mag, bg_firing

def wrap_to_pi(angle):
    """
    Wrap an angle to the range [-pi, pi].

    Parameters
    ----------
    angle : float or array_like
        The angle(s) to wrap, in radians.

    Returns
    -------
    wrapped_angle : float or ndarray
        The angle(s) wrapped to the range [-pi, pi].

    """
    wrapped_angle = np.mod(angle + np.pi, 2*np.pi) - np.pi
    return wrapped_angle


def circ_mtest(angles, mu):
    # Calculate the circular mean
    mean_angle = circmean(angles, low=-np.pi, high=np.pi)

    # Perform the one-sample circular mean test
    p_value = rayleightest(angles - mean_angle)

    return p_value


# the probe is considered to have drifted if the neurons have 0 firing in 10 consecutive trials
def check_probe_drift(firing_rates: np.ndarray) -> bool:
    # check if there is 0 firing in 10 consecutive trials
    for i in range(len(firing_rates) - 10):
        if np.all(firing_rates[i:i+10] == 0):
            return True

    return False

def get_mean_and_sem(signals):
    return np.mean(signals, axis=0), np.std(signals, axis=0) / np.sqrt(len(signals))

def bin_array(data, window_left, window_right, bin_size):
    discretized_data = np.digitize(data, bins=np.arange(window_left, window_right+bin_size, bin_size), right=True)
    print(discretized_data)
    discretized_data = [np.sum(discretized_data==i) for i in range(1, int((window_right-window_left)/bin_size)+1)]
    return np.array(discretized_data)