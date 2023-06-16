from os.path import join as pjoin, isdir, basename, isfile
from os import listdir, mkdir
import numpy as np
from glob import glob
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.signal import find_peaks

from lib.calculation import get_spikes_in_window, get_relative_spike_times

# relative time window before pfc time
LEFT = -0.01
# relative time window after pfc time
RIGHT = 0.02
FREQ=2000
PMSE_WINDOW = [0.0005, 0.007]
std_multiplier = 3

ITI_LEFT = -1
ITI_RIGHT = -0.5


def jitter(str_spikes):
    return np.add(str_spikes, np.random.uniform(low = -0.005, high=0.005, size=len(str_spikes)))

# returns the mean of the jittered performance as well as 3 std
def get_mean(pfc_spikes, str_spikes):
    jittered_array = []

    for i in range(500):
        jittered_str = jitter(str_spikes)
        # remove spikes not in intertrial interval
        relative_times = get_relative_spike_times(jittered_str, pfc_spikes, window_left=LEFT, window_right=RIGHT)
        bins = np.histogram(relative_times, bins=np.arange(start=LEFT, stop=RIGHT + 1/(2*FREQ), step=1/FREQ))
        jittered_array.append(bins[0])
    
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
    sessions = listdir(pjoin('data', 'spike_times'))
    fig, ax = plt.subplots()

    session_all = []
    str_all = []
    pfc_all = []
    peak_all = []
    peak_width_all = []
    counts_in_peak_all = []

    for s in sessions:
        # session_df = []
        # str_df = []
        # pfc_df = []
        # peak_df = []
        # peak_width_df = []
        # counts_in_peak_df = []
        behaviour_path = pjoin('data', 'behaviour_data', s)
        behaviour_data = np.load(pjoin(behaviour_path, 'behaviour_data.npy'))
        cue_time = behaviour_data['cue_time']

        session_path = pjoin('data', 'spike_times', s)
        strs = glob(pjoin(session_path, 'str_*'))
        pfcs = glob(pjoin(session_path, 'pfc_*'))
        pbar = tqdm(total=len(strs)*len(pfcs))
        if not isdir(pjoin('data', 'PMSE', s)):
            mkdir(pjoin('data', 'PMSE', s))

        if not isdir(pjoin('data', 'PMSE', s, 'qualified')):
            mkdir(pjoin('data', 'PMSE', s, 'qualified'))

        for st in strs:
            str_name = basename(st).split('.')[0]
            str_data = np.load(st)

            str_data = get_spikes_in_window(cue_time, str_data, ITI_LEFT, ITI_RIGHT)

            for pfc in pfcs:
                ax.clear()
                pfc_name = basename(pfc).split('.')[0]
                pfc_data = np.load(pfc)

                pfc_data = get_spikes_in_window(cue_time, pfc_data, ITI_LEFT, ITI_RIGHT)

                relative_times = get_relative_spike_times(spike_times=str_data, cue_times=pfc_data, window_left=LEFT, window_right=RIGHT)

                if isfile(pjoin('data', 'PMSE', s, f"{str_name}_{pfc_name}.npy")) and not reset:
                    data = np.load(pjoin('data', 'PMSE', s, f"{str_name}_{pfc_name}.npy"))
                    mean, std, bins = data
                else:
                    bins = np.histogram(relative_times, bins=np.arange(start=LEFT, stop=RIGHT+0.0001, step=1/FREQ))
                    # times of the left edge of the bin
                    left_times = bins[1][:-1]
                    bins = bins[0]
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