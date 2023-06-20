from os.path import join as pjoin, isdir, basename, isfile
from os import listdir, mkdir, rmdir
from glob import glob
from shutil import rmtree

import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.signal import find_peaks
from lib.calculation import get_spikes_in_window, get_relative_spike_times_flat

# relative time window before pfc time
LEFT = -0.025
# relative time window after pfc time
RIGHT = 0.025
FREQ=2000
PMSE_WINDOW = [0.0005, 0.008]
std_multiplier = 2.5

ITI_LEFT = -1
ITI_RIGHT = -0.5

# def in_bin(pfc, st):
#     return st >= pfc + LEFT and st <= pfc + RIGHT

# def ordered(l):
#     return all(l[i] <= l[i+1] for i in range(len(l) - 1))


# # get the str spike times relative to the pfc spike times
# def get_relative_times(pfc, str):
#     results = []
#     p_ptr = 0
#     s_ptr = 0

#     while p_ptr < len(pfc) and s_ptr < len(str):
#         if pfc[p_ptr] + LEFT > str[s_ptr]:
#             s_ptr += 1
#         elif in_bin(pfc[p_ptr], str[s_ptr]):
#             results.append(str[s_ptr] - pfc[p_ptr])
#             s_ptr += 1
#             if s_ptr == len(str):
#                 p_ptr += 1
#                 # if s_ptr has overshot for the current pfc
#                 while p_ptr < len(pfc) and s_ptr - 1 >= 0 and in_bin(pfc[p_ptr], str[s_ptr-1]):
#                     s_ptr -= 1
#         else:
#             p_ptr += 1
#             # if s_ptr has overshot for the current pfc
#             while p_ptr < len(pfc) and s_ptr - 1 >= 0 and in_bin(pfc[p_ptr], str[s_ptr-1]):
#                 s_ptr -= 1

#     return results


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


def find_PMSE(reset=False):
    sessions = listdir(pjoin('data', 'spike_times', 'sessions'))
    fig, ax = plt.subplots()

    session_all = []
    str_all = []
    pfc_all = []
    peak_all = []
    peak_width_all = []
    counts_in_peak_all = []

    if reset:
        rmtree(pjoin('data', 'PMSE'))
        mkdir(pjoin('data', 'PMSE'))
        mkdir(pjoin('data', 'PMSE', 'qualified'))

    for s in sessions:
        behaviour_path = pjoin('data', 'behaviour_data', s+'.csv')
        behaviour_data = pd.read_csv(behaviour_path)
        cue_time = behaviour_data['cue_time']

        session_path = pjoin('data', 'spike_times', 'sessions', s)
        strs = glob(pjoin(session_path, 'str_*'))
        pfcs = glob(pjoin(session_path, 'pfc_*'))
        pbar = tqdm(total=len(strs)*len(pfcs))
        if not isdir(pjoin('data', 'PMSE', s)):
            mkdir(pjoin('data', 'PMSE', s))

        for st in strs:
            str_name = basename(st).split('.')[0]
            str_data = np.load(st)

            str_data = get_spikes_in_window(cue_times=cue_time, spike_times=str_data, window_left=ITI_LEFT, window_right=ITI_RIGHT)

            for pfc in pfcs:
                ax.clear()
                pfc_name = basename(pfc).split('.')[0]
                pfc_data = np.load(pfc)

                pfc_data = get_spikes_in_window(cue_times=cue_time, spike_times=pfc_data, window_left=ITI_LEFT, window_right=ITI_RIGHT)

                relative_times = get_relative_spike_times_flat(spike_times=str_data, cue_times=pfc_data, window_left=LEFT, window_right=RIGHT)
                # relative_times = get_relative_times(pfc_data, str_data)

                # # find the items that are different between the two methods
                # diff = np.setdiff1d(relative_times, relative_times_org)
                # if len(diff) > 0:
                #     print(f"diff: {diff}")

                if isfile(pjoin('data', 'PMSE', s, f"{str_name}_{pfc_name}.npy")):
                    data = np.load(pjoin('data', 'PMSE', s, f"{str_name}_{pfc_name}.npy"))
                    mean, std, bins = data
                else:
                    bins = np.histogram(relative_times, bins=np.arange(start=LEFT, stop=RIGHT + 1/(2*FREQ), step=1/FREQ))[0]
                    # times of the left edge of the bin
                    mean, std  = get_mean(pfc_data, str_data)
                    np.save(pjoin('data', 'PMSE', s, f"{str_name}_{pfc_name}.npy"), arr=[mean, std, bins])
                pbar.update(1)

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
                        if bins[peak + left_ind] > mean[peak + left_ind] + 7:
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
                    sns.histplot(x=relative_times, bins=np.arange(start=LEFT, stop=RIGHT + 1/(2*FREQ), step=1/FREQ), ax=ax)
                    sns.lineplot(x=np.arange(start=LEFT+1/(2* FREQ), stop=RIGHT, step=1/FREQ), y=bins, ax=ax)
                    sns.lineplot(x=np.arange(start=LEFT+1/(2* FREQ), stop=RIGHT, step=1/FREQ), y=mean, ax=ax)
                    sns.lineplot(x=np.arange(start=LEFT+1/(2* FREQ), stop=RIGHT, step=1/FREQ), y=std, ax=ax)
                    for peak in real_peaks:
                        ax.plot(LEFT + 1/(2*FREQ) + peak * (1/FREQ), bins[peak], 'ro')

                ax.axvline(x=PMSE_WINDOW[0])
                ax.axvline(x=PMSE_WINDOW[1])

                if len(real_peaks) > 0:
                    fig.savefig(pjoin('data', 'PMSE', 'qualified', f"{s}_{str_name}_{pfc_name}.png"), dpi=400)

    results = pd.DataFrame({'session': session_all, 'str': str_all, 'pfc': pfc_all, 'peak': peak_all, 'peak_width': peak_width_all, 'counts_in_peak': counts_in_peak_all})
    results.to_csv(pjoin('data', f'mono_pairs.csv'))

find_PMSE(reset=True)