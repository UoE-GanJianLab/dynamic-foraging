# type: ignore
from os.path import join as pjoin
from os import listdir, mkdir
from os.path import basename
from typing import List, Tuple
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from scipy.signal import butter, filtfilt, hilbert, detrend # type: ignore
from lib.calculation import circ_mtest
from scipy.stats import circmean # type: ignore

import tqdm

from lib.file_utils import get_dms_pfc_paths_mono, get_dms_pfc_paths_all
from lib.calculation import get_response_bg_firing

# relative positions to cue_time
ITI_LEFT = -1
ITI_RIGHT = -0.5
RESPONSE_LEFT = 0
RESPONSE_RIGHT = 1.5

spike_data_root = pjoin('data', 'spike_times', 'sessions')
behaviour_root = pjoin('data', 'behaviour_data')
relative_value_root = pjoin('data', 'prpd')

def fig_5_panel_b(pfc_mag, str_mag):
    session_length = len(pfc_mag)
    # green is striatum, black is PFC, left is striatum, right is pfc
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))
    axes_2 = axes[0].twinx()
    sns.lineplot(x=np.arange(session_length, dtype=int), y=str_mag, ax=axes[0], color='green')
    sns.lineplot(x=np.arange(session_length, dtype=int), y=pfc_mag, ax=axes_2, color='black')
    axes[0].tick_params(axis='y', colors='green')

    # low_pass filter
    b, a = butter(N=4, Wn=10/session_length, btype='low', output='ba')
    filtered_pfc = filter_signal(pfc_mag, b, a)
    filtered_str = filter_signal(str_mag, b, a)

    # plot filtered signal
    sns.lineplot(x=np.arange(session_length, dtype=int), y=filtered_str, ax=axes[1], color='green')
    sns.lineplot(x=np.arange(session_length, dtype=int), y=filtered_pfc, ax=axes[1], color='black')

    # hilbert transform
    phase_pfc = hilbert_transform(filtered_pfc)
    phase_str = hilbert_transform(filtered_str)
    sns.lineplot(x=np.arange(session_length, dtype=int), y=phase_str, ax=axes[2], color='green')
    sns.lineplot(x=np.arange(session_length, dtype=int), y=phase_pfc, ax=axes[2], color='black')

    plt.show()
    return fig


def fig_5_panel_c(phase_diffs: List[float], phase_diffs_bg: List[float], bin_size: int, zero_ymin: bool = True) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    hist, edge = np.histogram(phase_diffs, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    dist = y_max - y_min
    axes[0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bg, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[1].set_ylim(y_min, y_max)
    sns.histplot(phase_diffs, ax=axes[0], bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size), color='black', kde=True) # type: ignore
    sns.histplot(phase_diffs_bg, ax=axes[1], bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size), color='black', kde=True) # type: ignore

    # set y label
    axes[0].set_ylabel('Number of Cell Pairs')
    axes[1].set_ylabel('Number of Cell Pairs')

    # set x label
    axes[0].set_xlabel('Phase Difference (radians)')
    axes[1].set_xlabel('Phase Difference (radians)')

    # Set the x-axis tick labels to pi
    set_xticks_and_labels_pi(axes[0])
    set_xticks_and_labels_pi(axes[1])

    # remove the top and right spines
    remove_top_and_right_spines(axes[0])
    remove_top_and_right_spines(axes[1])

    return fig

# def get_figure_5_panel_d(mono: bool = False, bin_size: int, zero_ymin: bool = True):


def fig_5_panel_d(phase_diffs: List[float], phase_diffs_bg: List[float], phase_diffs_bad: List[float], phase_diffs_bg_bad: List[float], bin_size: int, zero_ymin: bool = True) -> Figure:
    mid = int(bin_size / 2)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    hist, edge = np.histogram(phase_diffs, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[0][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bg, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[0][1].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bad, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[1][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bg_bad, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[1][1].set_ylim(y_min, y_max)


    sns.histplot(phase_diffs, ax=axes[0][0], bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size), color='blue', kde=True) # type: ignore
    sns.histplot(phase_diffs_bg, ax=axes[0][1], bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size), color='blue', kde=True) # type: ignore
    sns.histplot(phase_diffs_bad, ax=axes[1][0], bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size), color='red', kde=True) # type: ignore
    sns.histplot(phase_diffs_bg_bad, ax=axes[1][1], bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size), color='red', kde=True) # type: ignore

    # set y label
    axes[0][1].set_ylabel('Number of Cell Pairs')
    axes[0][0].set_ylabel('Number of Cell Pairs')

    # set x label
    axes[0][0].set_xlabel('Phase Difference (radians)')
    axes[0][1].set_xlabel('Phase Difference (radians)')
    axes[1][0].set_xlabel('Phase Difference (radians)')
    axes[1][1].set_xlabel('Phase Difference (radians)')

    # Set the x-axis tick labels to pi
    set_xticks_and_labels_pi(axes[0][0])
    set_xticks_and_labels_pi(axes[0][1])
    set_xticks_and_labels_pi(axes[1][0])
    set_xticks_and_labels_pi(axes[1][1])

    # remove top and right spines
    remove_top_and_right_spines(axes[0][0])
    remove_top_and_right_spines(axes[0][1])
    remove_top_and_right_spines(axes[1][0])
    remove_top_and_right_spines(axes[1][1])

    return fig

def get_figure_5_panel_e(mono: bool=False, reset: bool=False, no_nan: bool=False) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    bin_size = 36
    mid = int(bin_size / 2)
    zero_ymin = False

    phase_diff_bg_pfc = []
    phase_diff_bg_dms = []
    phase_diff_response_pfc = []
    phase_diff_response_dms = []

    if mono:
        str_pfc_paths = get_dms_pfc_paths_mono()
        
        for mono_pair in tqdm.tqdm(str_pfc_paths.iterrows()):
            session_path = mono_pair[1]['session_path']
            pfc_path = mono_pair[1]['pfc_path']
            dms_path = mono_pair[1]['dms_path']

            session_name = basename(session_path).split('.')[0]

            pfc_times = np.load(pfc_path)
            dms_times = np.load(dms_path)

            behaviour_data = pd.read_csv(session_path)
            # remove the nan trials
            if no_nan:
                behaviour_data = behaviour_data[~behaviour_data['trial_reward'].isna()]
            cue_times = np.array(behaviour_data['cue_time'].values)
            
            relative_value_path = pjoin(relative_value_root, session_name + '.npy')
            relative_values = np.load(relative_value_path)

            pfc_mag, pfc_bg = get_response_bg_firing(cue_times=cue_times, spike_times=pfc_times)
            dms_mag, dms_bg = get_response_bg_firing(cue_times=cue_times, spike_times=dms_times)

            # calculate the phase difference wrt relative value
            phase_diff_mag = phase_diff(pfc_mag, relative_values)
            phase_diff_bg = phase_diff(pfc_bg, relative_values)

            phase_diff_response_pfc.append(phase_diff_mag)
            phase_diff_bg_pfc.append(phase_diff_bg)

            phase_diff_mag = phase_diff(dms_mag, relative_values)
            phase_diff_bg = phase_diff(dms_bg, relative_values)

            phase_diff_response_dms.append(phase_diff_mag)
            phase_diff_bg_dms.append(phase_diff_bg)
    else:
        for session_name in tqdm.tqdm(listdir(spike_data_root)):
            session_path = pjoin(spike_data_root, session_name)
            relative_value_path = pjoin(relative_value_root, session_name + '.npy')
            relative_values = np.load(relative_value_path)
            behaviour_path = pjoin(behaviour_root, session_name + '.csv')
            behaviour_data = pd.read_csv(behaviour_path)
            # remove all the nan trials
            if no_nan:
                behaviour_data = behaviour_data[~behaviour_data['trial_reward'].isna()]
            cue_times = np.array(behaviour_data['cue_time'].values)

            # load the pfc cells
            for pfc_path in glob(pjoin(session_path, 'pfc_*.npy')):
                pfc_times = np.load(pfc_path)
                pfc_name = basename(pfc_path).split('.')[0]
                pfc_mag, pfc_bg = get_response_bg_firing(cue_times=cue_times, spike_times=pfc_times)

                # calculate the phase difference wrt relative value
                phase_diff_mag = phase_diff(pfc_mag, relative_values)
                phase_diff_bg = phase_diff(pfc_bg, relative_values)

                phase_diff_response_pfc.append(phase_diff_mag)
                phase_diff_bg_pfc.append(phase_diff_bg)
            
            # load the dms cells
            for dms_path in glob(pjoin(session_path, 'dms_*.npy')):
                dms_times = np.load(dms_path)
                dms_name = basename(dms_path).split('.')[0]
                dms_mag, dms_bg = get_response_bg_firing(cue_times=cue_times, spike_times=dms_times)

                # calculate the phase difference wrt relative value
                phase_diff_mag = phase_diff(dms_mag, relative_values)
                phase_diff_bg = phase_diff(dms_bg, relative_values)

                phase_diff_response_dms.append(phase_diff_mag)
                phase_diff_bg_dms.append(phase_diff_bg)

    hist, edge = np.histogram(phase_diff_response_pfc, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[0][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diff_bg_pfc, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[0][1].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diff_response_dms, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[1][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diff_bg_dms, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[1][1].set_ylim(y_min, y_max)

    sns.histplot(phase_diff_response_pfc, ax=axes[0][0], bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size), color='black', kde=False) # type: ignore
    sns.histplot(phase_diff_bg_pfc, ax=axes[0][1], bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size), color='black', kde=False) # type: ignore
    sns.histplot(phase_diff_response_dms, ax=axes[1][0], bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size), color='black', kde=False) # type: ignore
    sns.histplot(phase_diff_bg_dms, ax=axes[1][1], bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size), color='black', kde=False) # type: ignore

    # calculate the circular mean for each group
    mean = circmean(phase_diff_response_pfc, low=-np.pi, high=np.pi)
    p_value = circ_mtest(phase_diff_response_pfc, mean)
    print(f'PFC response: {mean} {p_value}')
    mean = circmean(phase_diff_bg_pfc, low=-np.pi, high=np.pi)
    p_value = circ_mtest(phase_diff_bg_pfc, mean)
    print(f'PFC bg: {mean} {p_value}')
    mean = circmean(phase_diff_response_dms, low=-np.pi, high=np.pi)
    p_value = circ_mtest(phase_diff_response_dms, mean)
    print(f'DMS response: {mean} {p_value}')
    mean = circmean(phase_diff_bg_dms,  low=-np.pi, high=np.pi)
    p_value = circ_mtest(phase_diff_bg_dms, mean)
    print(f'DMS bg: {mean} {p_value}')


    # set y label
    axes[0][1].set_ylabel('Number of PFC Pairs')
    axes[0][0].set_ylabel('Number of DMS Pairs')

    # set x label
    axes[0][0].set_xlabel('Phase Difference (radians)')
    axes[0][1].set_xlabel('Phase Difference (radians)')
    axes[1][0].set_xlabel('Phase Difference (radians)')
    axes[1][1].set_xlabel('Phase Difference (radians)')

    # Set the x-axis tick labels to pi
    set_xticks_and_labels_pi(axes[0][0])
    set_xticks_and_labels_pi(axes[0][1])
    set_xticks_and_labels_pi(axes[1][0])
    set_xticks_and_labels_pi(axes[1][1])

    # remove top and right spines
    remove_top_and_right_spines(axes[0][0])
    remove_top_and_right_spines(axes[0][1])
    remove_top_and_right_spines(axes[1][0])
    remove_top_and_right_spines(axes[1][1])



def set_xticks_and_labels_pi(ax: plt.Axes):
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])

def remove_top_and_right_spines(ax: plt.Axes):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def phase_diff_pfc_str(pfc_mag, str_mag, pfc_bg, str_bg) -> Tuple[float, float]:
    session_length = len(pfc_mag)
    # green is striatum, black is PFC, left is striatum, right is pfc
    # low_pass filter
    b, a = butter(N=4, Wn=10/session_length, btype='low', output='ba')
    filtered_pfc = filter_signal(pfc_mag, b, a)
    filtered_str = filter_signal(str_mag, b, a)
    phase_pfc, phase_str = hilbert_transform(filtered_pfc), hilbert_transform(filtered_str)

    filtered_pfc_bg = filter_signal(pfc_bg, b, a)
    filtered_str_bg = filter_signal(str_bg, b, a)
    phase_pfc_bg, phase_str_bg = hilbert_transform(filtered_pfc_bg), hilbert_transform(filtered_str_bg)

    phase_diff = circmean(phase_pfc - phase_str, high=np.pi, low=-np.pi)
    phase_diff_bg = circmean(phase_pfc_bg - phase_str_bg, high=np.pi, low=-np.pi)
    # phase_diff = circmean(phase_pfc - phase_str)
    # phase_diff_bg = circmean(phase_pfc_bg - phase_str_bg)

    return phase_diff, phase_diff_bg

def phase_diff(sig1, sig2) -> float:
    length = len(sig1)
    # low_pass filter
    b, a = butter(N=4, Wn=10/length, btype='low', output='ba')
    sig1 = filter_signal(sig1, b, a)
    sig2 = filter_signal(sig2, b, a)
    phase1 = hilbert_transform(sig1)
    phase2 = hilbert_transform(sig2)
    phase_diff = circmean(phase1 - phase2, low=-np.pi, high=np.pi)
    return phase_diff

# low pass filter
def filter_signal(signal, b, a) -> np.ndarray:
    filtered_signal = filtfilt(b=b, a=a, x=signal)
    filtered_signal = detrend(filtered_signal, type='constant')
    return filtered_signal

# hilbert transform
def hilbert_transform(signal) -> np.ndarray:
    hilbert_signal = hilbert(signal)
    phase = np.angle(hilbert_signal)
    return phase