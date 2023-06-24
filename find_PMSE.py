# type: ignore
from os.path import join as pjoin, isdir, basename, isfile
from os import listdir, mkdir, rmdir
from glob import glob
from shutil import rmtree
import re
import csv
import sys
import multiprocessing as mp
from functools import partial
import time

import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.signal import find_peaks # type: ignore
from lib.calculation import get_spikes_in_window, get_spikes_outside_window, get_relative_spike_times_flat

# relative time window before pfc time
LEFT = -0.025
# relative time window after pfc time
RIGHT = 0.025
FREQ=2000
PMSE_WINDOW = [0.0005, 0.007]
std_multiplier = 2.5

ITI_LEFT = -1
ITI_RIGHT = -0.5
RESPONSE_LEFT = 0
RESPONSE_RIGHT = 1.5


def jitter(str_spikes):
    return np.add(str_spikes, np.random.uniform(low = -0.005, high=0.005, size=len(str_spikes)))

# returns the mean of the jittered performance as well as 3 std
def get_mean(pfc_spikes, str_spikes):
    jittered_array = []

    for i in range(500):
        jittered_str = jitter(str_spikes)
        relative_times = get_relative_spike_times_flat(spike_times= jittered_str, cue_times= pfc_spikes, window_left=LEFT, window_right=RIGHT)
        # relative_times = get_relative_times(pfc_spikes, jittered_str)
        bins = np.histogram(relative_times, bins=np.arange(start=LEFT, stop=RIGHT + 1/(2*FREQ), step=1/FREQ))[0]

        jittered_array.append(bins)
    
    mean_array = np.mean(a=jittered_array, axis=0)
    std_array = np.add(np.std(a=jittered_array, axis=0)*std_multiplier, mean_array)

    return mean_array, std_array

def FWHM(peak, bins):
    left, right = 0, 0
    # half maximum
    counts = bins[peak]
    HM = counts / 2
    while peak - left > 0 and bins[peak - left] >= HM:
        left += 1
        counts += bins[left]
        
    while peak + right < len(bins) and bins[peak + right] >= HM:
        right += 1
        counts += bins[right]

    return left, right, counts


# Define the function to process a single session
def process_session(s, behaviour_path, strs, pfcs):
    session_all = []
    str_all = []
    pfc_all = []
    peak_all = []
    peak_width_all = []
    counts_in_peak_all = []

    if not isdir(pjoin('data', 'PMSE', s)):
        mkdir(pjoin('data', 'PMSE', s))

    behaviour_data = pd.read_csv(behaviour_path)
    cue_time = np.array(behaviour_data['cue_time'])

    for st in strs:
        str_name = basename(st).split('.')[0]
        str_data = np.load(st)

        # str_data = get_spikes_in_window(cue_times=cue_time, spike_times=str_data, window_left=ITI_LEFT, window_right=ITI_RIGHT)
        str_data = get_spikes_outside_window(cue_times=cue_time, spike_times=str_data, window_left=RESPONSE_LEFT, window_right=RESPONSE_RIGHT)

        for pfc in pfcs:
            pfc_name = basename(pfc).split('.')[0]
            pfc_data = np.load(pfc)

            # pfc_data = get_spikes_in_window(cue_times=cue_time, spike_times=pfc_data, window_left=ITI_LEFT, window_right=ITI_RIGHT)
            pfc_data = get_spikes_outside_window(cue_times=cue_time, spike_times=pfc_data, window_left=RESPONSE_LEFT, window_right=RESPONSE_RIGHT)

            relative_times = get_relative_spike_times_flat(spike_times=str_data, cue_times=pfc_data, window_left=LEFT, window_right=RIGHT)

            if isfile(pjoin('data', 'PMSE', s, f"{str_name}_{pfc_name}.npy")):
                data = np.load(pjoin('data', 'PMSE', s, f"{str_name}_{pfc_name}.npy"))
                mean, std, bins = data
            else:
                bins = np.histogram(relative_times, bins=np.arange(start=LEFT, stop=RIGHT + 1/(2*FREQ), step=1/FREQ))[0]
                # times of the left edge of the bin
                mean, std  = get_mean(pfc_data, str_data)
                np.save(pjoin('data', 'PMSE', s, f"{str_name}_{pfc_name}.npy"), arr=[mean, std, bins])

            # percentage over the average needs to be greater than 30%
            higher_than_mean = np.greater(bins, mean)
            higher_percentage = np.sum(higher_than_mean) / len(bins)

            if higher_percentage <= 0.3:
                continue

            # Method 4
            left_ind = int((PMSE_WINDOW[0] - LEFT) * FREQ)
            right_ind = int((PMSE_WINDOW[1] - LEFT) * FREQ)
            
            real_peaks = []
            bins_in_window = bins[left_ind: right_ind]
            heights = std[left_ind: right_ind]
            peaks, properties = find_peaks(bins_in_window, height=heights)
            if len(peaks) > 0:
                for peak in peaks:
                    if bins[peak + left_ind] > mean[peak + left_ind] + 10:
                        # check full width at half maximum
                        left, right, counts = FWHM(peak + left_ind, bins)
                        if (right + left + 1) * (1/FREQ) <= 0.003:
                            real_peaks.append(peak + left_ind)
                            session_all.append(s)
                            str_all.append(str(str_name))
                            pfc_all.append(str(pfc_name))
                            peak_all.append(peak+left_ind)
                            peak_width_all.append((left + right + 1) * (1/FREQ))
                            counts_in_peak_all.append(counts)
                            
                        
            if len(real_peaks) > 0:
                fig, ax = plt.subplots()
                sns.histplot(x=relative_times, bins=np.arange(start=LEFT, stop=RIGHT + 1/(2*FREQ), step=1/FREQ), ax=ax) # type: ignore
                sns.lineplot(x=np.arange(start=LEFT+1/(2* FREQ), stop=RIGHT, step=1/FREQ), y=bins, ax=ax)
                sns.lineplot(x=np.arange(start=LEFT+1/(2* FREQ), stop=RIGHT, step=1/FREQ), y=mean, ax=ax)
                sns.lineplot(x=np.arange(start=LEFT+1/(2* FREQ), stop=RIGHT, step=1/FREQ), y=std, ax=ax)
                for peak in real_peaks:
                    ax.plot(LEFT + 1/(2*FREQ) + peak * (1/FREQ), bins[peak], 'ro')

                ax.axvline(x=PMSE_WINDOW[0])
                ax.axvline(x=PMSE_WINDOW[1])

                fig.savefig(pjoin('data', 'PMSE', 'qualified', f"{s}_{str_name}_{pfc_name}.png"), dpi=400)
                plt.close(fig)

    return (session_all, str_all, pfc_all, peak_all, peak_width_all, counts_in_peak_all)

# Define the function to process a group of sessions
def process_session_group(sessions):
    session_all = []
    str_all = []
    pfc_all = []
    peak_all = []
    peak_width_all = []
    counts_in_peak_all = []

    for s in sessions:
        behaviour_path = pjoin('data', 'behaviour_data', s+'.csv')
        session_path = pjoin('data', 'spike_times', 'sessions', s)
        strs = glob(pjoin(session_path, 'str_*'))
        pfcs = glob(pjoin(session_path, 'pfc_*'))

        results = process_session(s, behaviour_path, strs, pfcs)
        session_all.extend(results[0])
        str_all.extend(results[1])
        pfc_all.extend(results[2])
        peak_all.extend(results[3])
        peak_width_all.extend(results[4])
        counts_in_peak_all.extend(results[5])

    return (session_all, str_all, pfc_all, peak_all, peak_width_all, counts_in_peak_all)

# Define the main function
def find_PMSE_parallel(reset=False):
    sessions = listdir(pjoin('data', 'spike_times', 'sessions'))

    # remove all content of the qualified folder
    if not isdir(pjoin('data', 'PMSE', 'qualified')):
        # rmtree(pjoin('data', 'PMSE', 'qualified'))
        mkdir(pjoin('data', 'PMSE', 'qualified'))
    else:
        rmtree(pjoin('data', 'PMSE', 'qualified'))
        mkdir(pjoin('data', 'PMSE', 'qualified'))

    if reset:
        rmtree(pjoin('data', 'PMSE'))
        mkdir(pjoin('data', 'PMSE'))
        mkdir(pjoin('data', 'PMSE', 'qualified'))

    # Split the sessions into 3 groups and run them using multiple processes
    num_processes = 4
    session_groups = [sessions[i::num_processes] for i in range(num_processes)]
    with mp.Pool(processes=num_processes) as pool:
        results = []
        start_time = time.time()
        for result in tqdm(pool.imap_unordered(process_session_group, session_groups), total=len(sessions)):
            results.append(result)
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time / len(results) * (len(sessions) - len(results))
            tqdm.write(f'Remaining time: {remaining_time:.2f} seconds')

    # Combine the results from all processes
    session_all = []
    str_all = []
    pfc_all = []
    peak_all = []
    peak_width_all = []
    counts_in_peak_all = []
    for result in results:
        session_all.extend(result[0])
        str_all.extend(result[1])
        pfc_all.extend(result[2])
        peak_all.extend(result[3])
        peak_width_all.extend(result[4])
        counts_in_peak_all.extend(result[5])

    # Write the results to a CSV file
    results_df = pd.DataFrame({'session': session_all, 'str': str_all, 'pfc': pfc_all, 'peak': peak_all, 'peak_width': peak_width_all, 'counts_in_peak': counts_in_peak_all})
    results_df.to_csv(pjoin('data', 'mono_pairs.csv'), index=False)

find_PMSE_parallel(reset=False)