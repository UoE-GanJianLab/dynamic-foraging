from os.path import join as pjoin, isdir
from os import listdir, mkdir
from os.path import basename
from typing import List, Tuple, Dict
from glob import glob
from shutil import rmtree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from scipy.signal import butter, filtfilt, hilbert, detrend # type: ignore
from lib.calculation import circ_mtest
from scipy.stats import circmean, ttest_1samp # type: ignore]
from scipy.io import savemat # type: ignore
from pingouin import circ_corrcc
from tqdm import tqdm
# suppress warning
import warnings
warnings.filterwarnings("ignore")

from lib.file_utils import get_dms_pfc_paths_mono, get_dms_pfc_paths_all
from lib.calculation import get_response_bg_firing, get_session_performances, moving_window_mean_prior, check_probe_drift


# relative positions to cue_time
ITI_LEFT = -1
ITI_RIGHT = -0.5
RESPONSE_LEFT = 0
RESPONSE_RIGHT = 1.5

spike_data_root = pjoin('data', 'spike_times', 'sessions')
behaviour_root = pjoin('data', 'behaviour_data')
relative_value_root = pjoin('data', 'prpd')

figure_5_data_root = pjoin('figure_data', 'figure_6')
if not isdir(figure_5_data_root):
    mkdir(figure_5_data_root)
figure_5_panel_b_data_root_prpd = pjoin(figure_5_data_root, 'panel_b_prpd')
if not isdir(figure_5_panel_b_data_root_prpd):
    mkdir(figure_5_panel_b_data_root_prpd)
figure_5_panel_b_data_root_relative_value = pjoin(figure_5_data_root, 'panel_b_relative_value')
if not isdir(figure_5_panel_b_data_root_relative_value):
    mkdir(figure_5_panel_b_data_root_relative_value)
figure_5_panel_b_figure_path_prpd = pjoin('figures', 'all_figures', 'figure_6', 'panel_b_prpd')
if not isdir(figure_5_panel_b_figure_path_prpd):
    mkdir(figure_5_panel_b_figure_path_prpd)
figure_5_panel_b_figure_path_relative_value = pjoin('figures', 'all_figures', 'figure_6', 'panel_b_relative_value')
if not isdir(figure_5_panel_b_figure_path_relative_value):
    mkdir(figure_5_panel_b_figure_path_relative_value)

def get_fig_5_panel_b(mono: bool = False, nonan: bool = False):
    if mono:
        mono_pairs = get_dms_pfc_paths_mono()

        for ind, row in mono_pairs.iterrows():
            behaviour_data = pd.read_csv(row['session_path'])
            if nonan:
                behaviour_data = behaviour_data[behaviour_data['trial_reward'].notna()]
            pfc_times = np.load(row['pfc_path'])
            dms_times = np.load(row['dms_path'])

            session_name = basename(row['session_path']).split('.')[0]
            relative_value = np.load(pjoin(relative_value_root, session_name + '.npy'))
            if relative_value_root == pjoin('data', 'relative_values'):
                # smoothen the relative value
                relative_value = moving_window_mean_prior(relative_value, 10)

            pfc_name = basename(row['pfc_path']).split('.')[0]
            dms_name = basename(row['dms_path']).split('.')[0]
            fig_name = '_'.join([session_name, pfc_name, dms_name]) + '.png'
            fig_path = pjoin('figures', 'all_figures', 'figure_5', 'panel_b', fig_name)

            cue_times = behaviour_data['cue_time'].tolist()
            pfc_mag, pfc_bg = get_response_bg_firing(cue_times=cue_times, spike_times=pfc_times)
            dms_mag, dms_bg = get_response_bg_firing(cue_times=cue_times, spike_times=dms_times)

            draw_fig_5_panel_b(session_name, pfc_name, dms_name,pfc_mag, dms_mag, relative_value)
    else:
        for session_name in listdir(spike_data_root):
            behaviour = pjoin(behaviour_root, session_name + '.csv')
            behaviour_data = pd.read_csv(behaviour)
            cue_times = behaviour_data['cue_time'].tolist()
            relative_value = np.load(pjoin(relative_value_root, session_name + '.npy'))
            if relative_value_root == pjoin('data', 'relative_values'):
                # smoothen the relative value
                relative_value = moving_window_mean_prior(relative_value, 10)

            # count the number of pfc and str pairs and use it for progress bar
            pfc_count = len(glob(pjoin(spike_data_root, session_name, 'pfc_*')))
            dms_count = len(glob(pjoin(spike_data_root, session_name, 'dms_*')))
            total_count = pfc_count * dms_count
            progress_bar = tqdm(total=total_count, desc=session_name)

            for pfc in glob(pjoin(spike_data_root, session_name, 'pfc_*')):
                pfc_times = np.load(pfc)
                pfc_mag, pfc_bg = get_response_bg_firing(cue_times=cue_times, spike_times=pfc_times)
                if check_probe_drift(pfc_mag):
                    progress_bar.update(dms_count)
                    continue
                for dms in glob(pjoin(spike_data_root, session_name, 'dms_*')):
                    dms_times = np.load(dms)
                    dms_mag, dms_bg = get_response_bg_firing(cue_times=cue_times, spike_times=dms_times) 
                    if check_probe_drift(dms_mag):
                        progress_bar.update(1)
                        continue
                    pfc_name = basename(pfc).split('.')[0]
                    dms_name = basename(dms).split('.')[0]
                    
                    fig = draw_fig_5_panel_b(session_name, pfc_name, dms_name, pfc_mag, dms_mag, relative_value) 
                    fig_name = '_'.join([session_name, pfc_name, dms_name]) + '.png'
                    if relative_value_root == pjoin('data', 'relative_values'):
                        fig_path = pjoin(figure_5_panel_b_figure_path_relative_value, fig_name)
                    else:
                        fig_path = pjoin(figure_5_panel_b_figure_path_prpd, fig_name)

                    fig.savefig(fig_path, dpi=300)
                    plt.close(fig)
                    progress_bar.update(1)


# TODO add relative value signal
def draw_fig_5_panel_b(session_name, pfc_name, dms_name, pfc_mag, dms_mag, relative_values = []):
    session_length = len(pfc_mag)
    # green is striatum, black is PFC, left is striatum, right is pfc
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))
    # get the z score of pfc, dms and relative value
    pfc_mag = (pfc_mag - np.mean(pfc_mag)) / np.std(pfc_mag)
    dms_mag = (dms_mag - np.mean(dms_mag)) / np.std(dms_mag)
    relative_values = (relative_values - np.mean(relative_values)) / np.std(relative_values)

    sns.lineplot(x=np.arange(session_length, dtype=int), y=dms_mag, ax=axes[0], color='green')
    sns.lineplot(x=np.arange(session_length, dtype=int), y=pfc_mag, ax=axes[0], color='black')
    sns.lineplot(x=np.arange(session_length, dtype=int), y=relative_values, ax=axes[0], color='red')

    # low_pass filter
    b, a = butter(N=4, Wn=10/session_length, btype='low', output='ba')
    filtered_pfc = filter_signal(pfc_mag, b, a)
    filtered_dms = filter_signal(dms_mag, b, a)
    filtered_relative_values = filter_signal(relative_values, b, a)

    # plot filtered signal
    sns.lineplot(x=np.arange(session_length, dtype=int), y=filtered_dms, ax=axes[1], color='green')
    sns.lineplot(x=np.arange(session_length, dtype=int), y=filtered_pfc, ax=axes[1], color='black')
    sns.lineplot(x=np.arange(session_length, dtype=int), y=filtered_relative_values, ax=axes[1], color='red')

    # hilbert transform
    phase_pfc = hilbert_transform(filtered_pfc)
    phase_dms = hilbert_transform(filtered_dms)
    phase_relative_values = hilbert_transform(filtered_relative_values)
    sns.lineplot(x=np.arange(session_length, dtype=int), y=phase_dms, ax=axes[2], color='green')
    sns.lineplot(x=np.arange(session_length, dtype=int), y=phase_pfc, ax=axes[2], color='black')
    sns.lineplot(x=np.arange(session_length, dtype=int), y=phase_relative_values, ax=axes[2], color='red')

    data_file_name = '_'.join([session_name, pfc_name, dms_name]) + '.csv'

    if relative_value_root == pjoin('data', 'relative_values'):
        # store the data in a dataframe
        figure_6_panel_b_data = pd.DataFrame({'trial_index': np.arange(len(relative_values), dtype=int)+1, 'pfc_mag_standardized': pfc_mag, 'dms_mag_standardized': dms_mag, 'relative_value_standardized': relative_values, 'pfc_mag_filtered': filtered_pfc, 'dms_mag_filtered': filtered_dms, 'relative_value_filtered': filtered_relative_values, 'pfc_phase': phase_pfc, 'dms_phase': phase_dms, 'relative_value_phase': phase_relative_values})
        figure_6_panel_b_data.to_csv(pjoin(figure_5_panel_b_data_root_relative_value, data_file_name), index=False)
    else:
        # store the data in a dataframe
        figure_6_panel_b_data = pd.DataFrame({'trial_index': np.arange(len(relative_values), dtype=int)+1, 'pfc_mag_standardized': pfc_mag, 'dms_mag_standardized': dms_mag, 'prpd_standardized': relative_values, 'pfc_mag_filtered': filtered_pfc, 'dms_mag_filtered': filtered_dms, 'prpd_filtered': filtered_relative_values, 'pfc_phase': phase_pfc, 'dms_phase': phase_dms, 'prpd_phase': phase_relative_values})
        figure_6_panel_b_data.to_csv(pjoin(figure_5_panel_b_data_root_prpd, data_file_name), index=False)
    
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

def get_figure_5_panel_d(mono: bool = False, bin_size: int=36, zero_ymin: bool = True):
    good_sessions = []
    # iti correlated
    phase_diffs = []
    phase_diffs_bg = []
    phase_diffs_bad = []
    phase_diffs_bg_bad = []

    phase_diffs_session_mean = []
    phase_diffs_session_mean_bg = []
    phase_diffs_session_mean_bad = []
    phase_diffs_session_mean_bg_bad = []

    performances, cutoff = get_session_performances()

    if mono:
        session_phase_diffs_good: Dict[str, List] = {}
        session_phase_diffs_good_bg: Dict[str, List] = {}
        session_phase_diffs_bad: Dict[str, List] = {}
        session_phase_diffs_bad_bg: Dict[str, List] = {}

        mono_pairs = get_dms_pfc_paths_mono()

        for ind, row in mono_pairs.iterrows():
            behaviour_data = pd.read_csv(row['session_path'])
            pfc_times = np.load(row['pfc_path'])
            str_times = np.load(row['dms_path'])

            session_name = basename(row['session_path']).split('.')[0]

            # create an entry for the session in the dictionary if it doesn't exist
            if session_name not in session_phase_diffs_good.keys():
                session_phase_diffs_good[session_name] = []
                session_phase_diffs_bad[session_name] = []
                session_phase_diffs_good_bg[session_name] = []
                session_phase_diffs_bad_bg[session_name] = []

            cue_times = behaviour_data['cue_time'].tolist()
            pfc_mag, pfc_bg = get_response_bg_firing(cue_times=cue_times, spike_times=pfc_times)
            dms_mag, dms_bg = get_response_bg_firing(cue_times=cue_times, spike_times=str_times)

            phase_d, phase_d_bg = phase_diff_pfc_dms(pfc_mag=pfc_mag, pfc_bg=pfc_bg, dms_mag=dms_mag, dms_bg=dms_bg)
            if performances[session_name] > cutoff:
                phase_diffs.append(phase_d)
                phase_diffs_bg.append(phase_d_bg)

                session_phase_diffs_good[session_name].append(phase_d)
                session_phase_diffs_good_bg[session_name].append(phase_d_bg)
            else:
                phase_diffs_bad.append(phase_d)
                phase_diffs_bg_bad.append(phase_d_bg)

                session_phase_diffs_bad[session_name].append(phase_d)
                session_phase_diffs_bad_bg[session_name].append(phase_d_bg)
            
        # calculate the number of good bad pairs for each session
        for session_name in session_phase_diffs_good.keys():
            if session_phase_diffs_good[session_name]:
                phase_diffs_session_mean.append(circmean(session_phase_diffs_good[session_name], low=-np.pi, high=np.pi))
                phase_diffs_session_mean_bg.append(circmean(session_phase_diffs_good_bg[session_name], low=-np.pi, high=np.pi))
            if session_phase_diffs_bad[session_name]:
                phase_diffs_session_mean_bad.append(circmean(session_phase_diffs_bad[session_name], low=-np.pi, high=np.pi))
                phase_diffs_session_mean_bg_bad.append(circmean(session_phase_diffs_bad_bg[session_name], low=-np.pi, high=np.pi))
    else:
        for session_name in tqdm(listdir(spike_data_root)):
            behaviour = pjoin(behaviour_root, session_name + '.csv')
            behaviour_data = pd.read_csv(behaviour)
            cue_times = behaviour_data['cue_time'].tolist()

            cur_good = []
            cur_good_bg = []
            cur_bad = []
            cur_bad_bg = []

            good = performances[session_name] > cutoff

            for pfc in glob(pjoin(spike_data_root, session_name, 'pfc_*')):
                pfc_times = np.load(pfc)
                pfc_mag, pfc_bg = get_response_bg_firing(cue_times=cue_times, spike_times=pfc_times)
                for dms in glob(pjoin(spike_data_root, session_name, 'dms_*')):
                    str_times = np.load(dms)
                    str_mag, str_bg = get_response_bg_firing(cue_times=cue_times, spike_times=str_times)            
                    phase_d, phase_d_bg = phase_diff_pfc_dms(pfc_mag=pfc_mag, pfc_bg=pfc_bg, dms_mag=str_mag, dms_bg=str_bg)
                    if good:
                        phase_diffs.append(phase_d)
                        phase_diffs_bg.append(phase_d_bg)

                        cur_good.append(phase_d)
                        cur_good_bg.append(phase_d_bg)
                    else:
                        phase_diffs_bad.append(phase_d)
                        phase_diffs_bg_bad.append(phase_d_bg)

                        cur_bad.append(phase_d)
                        cur_bad_bg.append(phase_d_bg)

            if good:
                good_sessions.append(session_name)
                phase_diffs_session_mean.append(circmean(cur_good, low=-np.pi, high=np.pi))
                phase_diffs_session_mean_bg.append(circmean(cur_good_bg, low=-np.pi, high=np.pi))
            else:
                phase_diffs_session_mean_bad.append(circmean(cur_bad, low=-np.pi, high=np.pi))
                phase_diffs_session_mean_bg_bad.append(circmean(cur_bad_bg, low=-np.pi, high=np.pi))

    if mono:    
        savemat('circular_data_panel_d_mono.mat', {'array_1': phase_diffs_session_mean, 'array_2': phase_diffs_session_mean_bg, 'array_3': phase_diffs_session_mean_bad, 'array_4': phase_diffs_session_mean_bg_bad})   
    else:
        savemat('circular_data_panel_d_all.mat', {'array_1': phase_diffs_session_mean, 'array_2': phase_diffs_session_mean_bg, 'array_3': phase_diffs_session_mean_bad, 'array_4': phase_diffs_session_mean_bg_bad})

    # print out the good sessions to terminal with each entry being a separate line
    for good_session in good_sessions:
        print(good_session)

    fig = draw_fig_5_panel_d(phase_diffs=phase_diffs, phase_diffs_bg=phase_diffs_bg, phase_diffs_bad=phase_diffs_bad, phase_diffs_bg_bad=phase_diffs_bg_bad, bin_size=36, zero_ymin=zero_ymin)

def draw_fig_5_panel_d(phase_diffs: List[float], phase_diffs_bg: List[float], phase_diffs_bad: List[float], phase_diffs_bg_bad: List[float], bin_size: int, zero_ymin: bool = True) -> Figure:
    mid = int(bin_size / 2)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    hist, edge = np.histogram(phase_diffs, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    good_response_count = hist
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[0][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bg, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    good_bg_count = hist
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[0][1].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bad, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    bad_response_count = hist
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[1][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bg_bad, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    bad_bg_count = hist
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

    # store the histogram data in the following format
    # |bin_center|good_response_count|good_bg_count|bad_response_count|bad_bg_count|
    bin_centers = np.arange(-np.pi, np.pi, 2 * np.pi / bin_size) + np.pi / bin_size
    panel_d_data = pd.DataFrame({'bin_center': bin_centers, 'good_response_count': good_response_count, 'good_bg_count': good_bg_count, 'bad_response_count': bad_response_count, 'bad_bg_count': bad_bg_count})
    panel_d_data.to_csv(pjoin(figure_5_data_root, 'pnel_c_data.csv'), index=False)

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

def get_figure_5_panel_e(mono: bool=False, reset: bool=False, no_nan: bool=False, zero_ymin: bool=False, bin_size:int =36) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    bin_size = 36
    mid = int(bin_size / 2)

    phase_diff_bg_pfc = []
    phase_diff_bg_dms = []
    phase_diff_response_pfc = []
    phase_diff_response_dms = []

    pfc_response_sig_count = 0
    pfc_bg_sig_count = 0

    dms_response_sig_count = 0
    dms_bg_sig_count = 0

    pfc_count = 0
    dms_count = 0

    if mono:
        dms_pfc_paths = get_dms_pfc_paths_mono()

        pfc_count = len(dms_pfc_paths)
        dms_count = pfc_count

        session_phase_diffs_pfc: Dict[str, List] = {}
        session_phase_diffs_pfc_bg: Dict[str, List] = {}
        session_phase_diffs_dms: Dict[str, List] = {}
        session_phase_diffs_dms_bg: Dict[str, List] = {}
        
        for mono_pair in tqdm(dms_pfc_paths.iterrows()):
            session_path = mono_pair[1]['session_path']
            pfc_path = mono_pair[1]['pfc_path']
            dms_path = mono_pair[1]['dms_path']

            session_name = basename(session_path).split('.')[0]

            # create an entry for the session in the dictionary if it doesn't exist
            if session_name not in session_phase_diffs_pfc.keys():
                session_phase_diffs_pfc[session_name] = []
                session_phase_diffs_pfc_bg[session_name] = []
                session_phase_diffs_dms[session_name] = []
                session_phase_diffs_dms_bg[session_name] = []

            pfc_times = np.load(pfc_path)
            dms_times = np.load(dms_path)

            behaviour_data = pd.read_csv(session_path)
            # remove the nan trials
            if no_nan:
                behaviour_data = behaviour_data[~behaviour_data['trial_reward'].isna()]
            cue_times = np.array(behaviour_data['cue_time'].values)
            
            relative_value_path = pjoin(relative_value_root, session_name + '.npy')
            relative_values = np.load(relative_value_path)
            if no_nan:
                # smoothen relative values
                relative_values = moving_window_mean_prior(relative_values, 10)

            phase_relative_values = get_phase(relative_values)

            pfc_mag, pfc_bg = get_response_bg_firing(cue_times=cue_times, spike_times=pfc_times)
            dms_mag, dms_bg = get_response_bg_firing(cue_times=cue_times, spike_times=dms_times)

            phase_pfc_mag = get_phase(pfc_mag)
            if circ_corrcc(phase_pfc_mag, phase_relative_values)[1] < 0.05:
                pfc_response_sig_count += 1
                # calculate the phase difference wrt relative value
                phase_diff_mag = phase_diff(relative_values, pfc_mag)
                phase_diff_response_pfc.append(phase_diff_mag)

            phase_pfc_bg = get_phase(pfc_bg)
            if circ_corrcc(phase_pfc_bg, phase_relative_values)[1] < 0.05:
                pfc_bg_sig_count += 1
                phase_diff_bg = phase_diff(relative_values, pfc_bg)
                phase_diff_bg_pfc.append(phase_diff_bg)

            phase_dms_mag = get_phase(dms_mag)
            if circ_corrcc(phase_dms_mag, phase_relative_values)[1] < 0.05:
                dms_response_sig_count += 1
                phase_diff_mag = phase_diff(relative_values, dms_mag)
                phase_diff_response_dms.append(phase_diff_mag)
            
            phase_dms_bg = get_phase(dms_bg)
            if circ_corrcc(phase_dms_bg, phase_relative_values)[1] < 0.05:
                dms_bg_sig_count += 1
                phase_diff_bg = phase_diff(relative_values, dms_bg)
                phase_diff_bg_dms.append(phase_diff_bg)
    else:
        # initialize a set of strong correlated cells
        strong_correlated_cells_pfc = []
        strong_correlated_cells_pfc_bg = []
        strong_correlated_cells_dms = []
        strong_correlated_cells_dms_bg = []

        session_names = []
        session_names_bg = []

        for session_name in tqdm(listdir(spike_data_root)):
            session_path = pjoin(spike_data_root, session_name)
            relative_value_path = pjoin(relative_value_root, session_name + '.npy')
            relative_values = np.load(relative_value_path)
            if no_nan:
                # smoothen relative values
                relative_values = moving_window_mean_prior(relative_values, 10)
            # get the z score of the relative values
            relative_values = (relative_values - np.mean(relative_values)) / np.std(relative_values)
            phase_relative_values = get_phase(relative_values)
            behaviour_path = pjoin(behaviour_root, session_name + '.csv')
            behaviour_data = pd.read_csv(behaviour_path)
            # remove all the nan trials
            if no_nan:
                behaviour_data = behaviour_data[~behaviour_data['trial_reward'].isna()]
            cue_times = np.array(behaviour_data['cue_time'].values)

            session_strong_correlated_cells_pfc = []
            session_strong_correlated_cells_pfc_bg = []
            session_strong_correlated_cells_dms = []
            session_strong_correlated_cells_dms_bg = []

            # load the pfc cells
            for pfc_path in glob(pjoin(session_path, 'pfc_*.npy')):
                pfc_count += 1
                pfc_times = np.load(pfc_path)
                pfc_name = basename(pfc_path).split('.')[0]
                pfc_mag, pfc_bg = get_response_bg_firing(cue_times=cue_times, spike_times=pfc_times)
                
                if np.std(pfc_mag) != 0:
                    pfc_mag = (pfc_mag - np.mean(pfc_mag)) / np.std(pfc_mag)
                    pfc_mag_phase = get_phase(pfc_mag) 
                    if circ_corrcc(pfc_mag_phase, phase_relative_values)[1] < 0.05:
                        session_strong_correlated_cells_pfc.append(pfc_name)
                        pfc_response_sig_count += 1
                        phase_diff_mag = phase_diff(relative_values, pfc_mag)
                        phase_diff_response_pfc.append(phase_diff_mag)

                if np.std(pfc_bg) != 0:
                    pfc_bg = (pfc_bg - np.mean(pfc_bg)) / np.std(pfc_bg)
                    pfc_bg_phase = get_phase(pfc_bg)
                    if circ_corrcc(pfc_bg_phase, phase_relative_values)[1] < 0.05:
                        session_strong_correlated_cells_pfc_bg.append(pfc_name)
                        pfc_bg_sig_count += 1
                        phase_diff_bg = phase_diff(relative_values, pfc_bg)
                        phase_diff_bg_pfc.append(phase_diff_bg)
                
            # load the dms cells
            for dms_path in glob(pjoin(session_path, 'dms_*.npy')):
                dms_count += 1
                dms_times = np.load(dms_path)
                dms_name = basename(dms_path).split('.')[0]
                dms_mag, dms_bg = get_response_bg_firing(cue_times=cue_times, spike_times=dms_times)
                if np.std(dms_mag) != 0:
                    session_strong_correlated_cells_dms.append(dms_name)
                    # get the z score of the firing rates
                    dms_mag = (dms_mag - np.mean(dms_mag)) / np.std(dms_mag)
                    dms_mag_phase = get_phase(dms_mag)
                    if circ_corrcc(dms_mag_phase, phase_relative_values)[1] < 0.05:
                        dms_response_sig_count += 1
                        phase_diff_mag = phase_diff(relative_values, dms_mag)
                        phase_diff_response_dms.append(phase_diff_mag)
                if np.std(dms_bg) != 0:
                    dms_bg = (dms_bg - np.mean(dms_bg)) / np.std(dms_bg)
                    dms_bg_phase = get_phase(dms_bg)
                    if circ_corrcc(dms_bg_phase, phase_relative_values)[1] < 0.05:
                        session_strong_correlated_cells_dms_bg.append(dms_name)
                        dms_bg_sig_count += 1
                        phase_diff_bg = phase_diff(relative_values, dms_bg)
                        phase_diff_bg_dms.append(phase_diff_bg)

            # remove the duplicates in the session list
            session_strong_correlated_cells_pfc = list(set(session_strong_correlated_cells_pfc))
            session_strong_correlated_cells_pfc_bg = list(set(session_strong_correlated_cells_pfc_bg))
            session_strong_correlated_cells_dms = list(set(session_strong_correlated_cells_dms))
            session_strong_correlated_cells_dms_bg = list(set(session_strong_correlated_cells_dms_bg))

            for pfc in session_strong_correlated_cells_pfc:
                for dms in session_strong_correlated_cells_dms:
                    strong_correlated_cells_pfc.append(pfc)
                    strong_correlated_cells_dms.append(dms)
                    session_names.append(session_name)
            
            for pfc in session_strong_correlated_cells_pfc_bg:
                for dms in session_strong_correlated_cells_dms_bg:
                    strong_correlated_cells_pfc_bg.append(pfc)
                    strong_correlated_cells_dms_bg.append(dms)
                    session_names_bg.append(session_name)

    

    hist, edge = np.histogram(phase_diff_response_pfc, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    pfc_response_count = hist
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[0][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diff_bg_pfc, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    pfc_bg_count = hist
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[0][1].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diff_response_dms, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    dms_response_count = hist
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[1][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diff_bg_dms, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    dms_bg_count = hist
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

    bin_centers = np.arange(-np.pi, np.pi, 2 * np.pi / bin_size) + np.pi / bin_size
    panel_e_data = pd.DataFrame({'bin_center': bin_centers, 'pfc_response_count': pfc_response_count, 'pfc_bg_count': pfc_bg_count, 'dms_response_count': dms_response_count, 'dms_bg_count': dms_bg_count})
    panel_e_data.to_csv(pjoin(figure_5_data_root, 'panel_d_data.csv'), index=False)

    # calculate the circular mean for each group
    mean = circmean(phase_diff_response_pfc, low=-np.pi, high=np.pi)
    print(f'PFC response: {mean}')
    mean = circmean(phase_diff_bg_pfc, low=-np.pi, high=np.pi)
    print(f'PFC bg: {mean}')
    mean = circmean(phase_diff_response_dms, low=-np.pi, high=np.pi)
    print(f'DMS response: {mean}')
    mean = circmean(phase_diff_bg_dms,  low=-np.pi, high=np.pi)
    print(f'DMS bg: {mean}')

    if mono:
        savemat('circular_data_panel_e_mono.mat', {'array_1': phase_diff_response_pfc, 'array_2': phase_diff_bg_pfc, 'array_3': phase_diff_response_dms, 'array_4': phase_diff_bg_dms})
    else:
        savemat('circular_data_panel_e_all.mat', {'array_1': phase_diff_response_pfc, 'array_2': phase_diff_bg_pfc, 'array_3': phase_diff_response_dms, 'array_4': phase_diff_bg_dms})

    print(f'PFC response: {pfc_response_sig_count} / {pfc_count}')
    print(f'PFC bg: {pfc_bg_sig_count} / {pfc_count}')
    print(f'DMS response: {dms_response_sig_count} / {dms_count}')
    print(f'DMS bg: {dms_bg_sig_count} / {dms_count}')

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
    

    # save the strong correlated pairs as two csv files
    strong_correlated_data = {'pfc': strong_correlated_cells_pfc, 'dms': strong_correlated_cells_dms, 'session': session_names}
    strong_correlated_data = pd.DataFrame(strong_correlated_data)
    strong_correlated_data.to_csv(pjoin('data', 'strong_circular_correlated_cells_pairs.csv'), index=False)

    strong_correlated_data = {'pfc': strong_correlated_cells_pfc_bg, 'dms': strong_correlated_cells_dms_bg, 'session': session_names_bg}
    strong_correlated_data = pd.DataFrame(strong_correlated_data)
    strong_correlated_data.to_csv(pjoin('data', 'strong_circular_correlated_cells_pairs_bg.csv'), index=False)

    return fig

def get_figure_5_panel_extra(mono: bool=False, reset: bool=False, no_nan: bool=False, zero_ymin: bool=False, bin_size:int =36) -> Figure:
    # divide the trials for each session into plateau and transitioning trials
    phase_diff_plateau, phase_diff_transition, phase_diff_plateau_bg, phase_diff_transition_bg = [], [], [], []

    if mono:
        dms_pfc_paths = get_dms_pfc_paths_mono()
        for mono_pair in tqdm(dms_pfc_paths.iterrows()):
            session_path: str = mono_pair[1]['session_path']
            pfc_path = mono_pair[1]['pfc_path']
            dms_path = mono_pair[1]['dms_path']

            session_name = basename(session_path).split('.')[0]

            # load the session data
            session_data = pd.read_csv(pjoin(behaviour_root, session_name + '.csv'))
            cue_times = session_data['cue_time'].tolist()
            leftP = session_data['leftP']

            pfc_mag, pfc_bg = get_response_bg_firing(cue_times=cue_times, spike_times=np.load(pfc_path))
            dms_mag, dms_bg = get_response_bg_firing(cue_times=cue_times, spike_times=np.load(dms_path))

            switches = find_switch(leftP)

            plateau_trial_indices = []
            transition_trial_indices = []

            # if there are less than 20 trials before or after the switch
            # skip the switch
            for switch in switches:
                if switch < 20 or switch > len(leftP) - 20:
                    continue
                plateau_trial_indices += (range(switch-20, switch))
                transition_trial_indices += (range(switch, switch+20))

            # get the phase difference for the plateau and transition trials
            phase_diff_session, phase_diff_session_bg = phase_diff_pfc_dms_array(pfc_mag=pfc_mag, pfc_bg=pfc_bg, dms_mag=dms_mag, dms_bg=dms_bg)

            phase_diff_plateau.append(circmean(phase_diff_session[plateau_trial_indices], low=-np.pi, high=np.pi))
            phase_diff_transition.append(circmean(phase_diff_session[transition_trial_indices], low=-np.pi, high=np.pi))
            phase_diff_plateau_bg.append(circmean(phase_diff_session_bg[plateau_trial_indices], low=-np.pi, high=np.pi))
            phase_diff_transition_bg.append(circmean(phase_diff_session_bg[transition_trial_indices], low=-np.pi, high=np.pi))
    else:
        dms_pfc_paths = get_dms_pfc_paths_all()

        for session in tqdm(dms_pfc_paths):
            session_name = session[0]
            cue_times = session[1]
            trial_reward = session[2]
            all_pairs = session[3]

            session_data = pd.read_csv(pjoin(behaviour_root, session_name + '.csv'))
            leftP = session_data['leftP']
            swithes = find_switch(leftP)

            plateau_trial_indices = []
            transition_trial_indices = []

            # if there are less than 20 trials before or after the switch
            # skip the switch
            for switch in swithes:
                if switch < 20 or switch > len(leftP) - 20:
                    continue
                plateau_trial_indices += (range(switch-20, switch))
                transition_trial_indices += (range(switch, switch+20))

            for cell_pair in all_pairs:
                pfc_path = cell_pair[1]
                dms_path = cell_pair[0]

                pfc_time = np.load(pfc_path)
                dms_time = np.load(dms_path)

                pfc_mag, pfc_bg = get_response_bg_firing(cue_times=cue_times, spike_times=pfc_time)
                dms_mag, dms_bg = get_response_bg_firing(cue_times=cue_times, spike_times=dms_time)

                phase_diff_session, phase_diff_session_bg = phase_diff_pfc_dms_array(pfc_mag=pfc_mag, pfc_bg=pfc_bg, dms_mag=dms_mag, dms_bg=dms_bg)

                phase_diff_plateau.append(circmean(phase_diff_session[plateau_trial_indices], low=-np.pi, high=np.pi))
                phase_diff_transition.append(circmean(phase_diff_session[transition_trial_indices], low=-np.pi, high=np.pi))
                phase_diff_plateau_bg.append(circmean(phase_diff_session_bg[plateau_trial_indices], low=-np.pi, high=np.pi))
                phase_diff_transition_bg.append(circmean(phase_diff_session_bg[transition_trial_indices], low=-np.pi, high=np.pi))

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    bin_size = 36
    mid = int(bin_size / 2)

    hist, edge = np.histogram(phase_diff_plateau, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[0][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diff_plateau_bg, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[0][1].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diff_transition, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[1][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diff_transition_bg, bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size))
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[1][1].set_ylim(y_min, y_max)

    sns.histplot(phase_diff_plateau, ax=axes[0][0], bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size), color='blue', kde=True) # type: ignore
    sns.histplot(phase_diff_plateau_bg, ax=axes[0][1], bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size), color='blue', kde=True) # type: ignore
    sns.histplot(phase_diff_transition, ax=axes[1][0], bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size), color='red', kde=True) # type: ignore
    sns.histplot(phase_diff_transition_bg, ax=axes[1][1], bins=np.arange(-np.pi, np.pi+2 * np.pi / bin_size, 2 * np.pi / bin_size), color='red', kde=True) # type: ignore

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


def set_xticks_and_labels_pi(ax: plt.Axes):
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])


def remove_top_and_right_spines(ax: plt.Axes):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def phase_diff_pfc_dms(pfc_mag, dms_mag, pfc_bg, dms_bg) -> Tuple[float, float]:
    session_length = len(pfc_mag)
    # green is striatum, black is PFC, left is striatum, right is pfc
    # low_pass filter
    b, a = butter(N=4, Wn=10/session_length, btype='low', output='ba')
    filtered_pfc = filter_signal(pfc_mag, b, a)
    filtered_dms = filter_signal(dms_mag, b, a)
    phase_pfc, phase_dms = hilbert_transform(filtered_pfc), hilbert_transform(filtered_dms)

    filtered_pfc_bg = filter_signal(pfc_bg, b, a)
    filtered_dms_bg = filter_signal(dms_bg, b, a)
    phase_pfc_bg, phase_dms_bg = hilbert_transform(filtered_pfc_bg), hilbert_transform(filtered_dms_bg)

    phase_diff = circmean(phase_pfc - phase_dms, high=np.pi, low=-np.pi)
    phase_diff_bg = circmean(phase_pfc_bg - phase_dms_bg, high=np.pi, low=-np.pi)

    return phase_diff, phase_diff_bg


def phase_diff_pfc_dms_array(pfc_mag, dms_mag, pfc_bg, dms_bg) -> Tuple[np.ndarray, np.ndarray]:
    session_length = len(pfc_mag)
    # green is striatum, black is PFC, left is striatum, right is pfc
    # low_pass filter
    b, a = butter(N=4, Wn=10/session_length, btype='low', output='ba')
    filtered_pfc = filter_signal(pfc_mag, b, a)
    filtered_dms = filter_signal(dms_mag, b, a)
    phase_pfc, phase_dms = hilbert_transform(filtered_pfc), hilbert_transform(filtered_dms)

    filtered_pfc_bg = filter_signal(pfc_bg, b, a)
    filtered_dms_bg = filter_signal(dms_bg, b, a)
    phase_pfc_bg, phase_dms_bg = hilbert_transform(filtered_pfc_bg), hilbert_transform(filtered_dms_bg)

    phase_diff = phase_pfc - phase_dms
    phase_diff_bg = phase_pfc_bg - phase_dms_bg

    return np.array(phase_diff), np.array(phase_diff_bg)


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

def get_phase(signal) -> np.ndarray:
    length = len(signal)
    b, a = butter(N=4, Wn=10/length, btype='low', output='ba')
    signal = filter_signal(signal, b, a)
    hilbert_signal = hilbert(signal)
    phase = np.angle(hilbert_signal)
    return phase

# return the indices of the trials where the high reward side switches
def find_switch(leftP: np.ndarray) -> List[int]:
    switch_indices = []
    for i in range(len(leftP)-1):
        if leftP[i] != leftP[i+1]:
            switch_indices.append(i)
    return switch_indices