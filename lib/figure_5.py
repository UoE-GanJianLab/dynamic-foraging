from os.path import join as pjoin
from os import listdir, mkdir
from os.path import basename
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, hilbert, detrend
from scipy.stats import circmean


import sys
sys.path.append('lib')

from lib.file_utils import get_strong_corr

# relative positions to cue_time
ITI_LEFT = -1
ITI_RIGHT = 0
RESPONSE_LEFT = 0
RESPONSE_RIGHT = 1.5

def fig_5_panel_b(pfc_mag, str_mag) -> plt.figure:
    session_length = len(pfc_mag)
    # green is striatum, black is PFC, left is striatum, right is pfc
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))
    axes_2 = axes[0].twinx()
    sns.lineplot(x=np.arange(session_length, dtype=int), y=str_mag, ax=axes[0], color='green')
    sns.lineplot(x=np.arange(session_length, dtype=int), y=pfc_mag, ax=axes_2, color='black')
    axes[0].tick_params(axis='y', colors='green')

    # low_pass filter
    b, a = butter(N=4, Wn=10/session_length, btype='low', output='ba')
    filtered_pfc = filtfilt(b=b, a=a, x=pfc_mag)
    filtered_str = filtfilt(b=b, a=a, x=str_mag)
    # only subtracts the mean
    filtered_pfc = detrend(filtered_pfc, type='constant')
    filtered_str = detrend(filtered_str, type='constant')
    sns.lineplot(x=np.arange(session_length, dtype=int), y=filtered_str, ax=axes[1], color='green')
    sns.lineplot(x=np.arange(session_length, dtype=int), y=filtered_pfc, ax=axes[1], color='black')

    # hilbert transform
    hilbert_pfc = hilbert(filtered_pfc)
    hilbert_str = hilbert(filtered_str)
    phase_pfc = np.angle(hilbert_pfc)
    phase_str = np.angle(hilbert_str)
    sns.lineplot(x=np.arange(session_length, dtype=int), y=phase_str, ax=axes[2], color='green')
    sns.lineplot(x=np.arange(session_length, dtype=int), y=phase_pfc, ax=axes[2], color='black')

    plt.show()
    return fig


def fig_5_panel_c(phase_diffs: List[float], phase_diffs_bg: List[float], bin_size: int) -> plt.figure:
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    hist, edge = np.histogram(phase_diffs, bins=bin_size)
    y_min = 0
    y_max = np.max(hist)
    dist = y_max - y_min
    y_min = y_min - dist * 0.1
    y_max = y_max + dist * 0.1
    axes[0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bg, bins=bin_size)
    y_min = 0
    y_max = np.max(hist)
    dist = y_max - y_min
    y_min = y_min - dist * 0.1
    y_max = y_max + dist * 0.1
    axes[1].set_ylim(y_min, y_max)
    sns.histplot(phase_diffs, ax=axes[0], bins=bin_size, color='black', kde=True)
    sns.histplot(phase_diffs_bg, ax=axes[1], bins=bin_size, color='black', kde=True)
    return fig

def fig_5_panel_d(phase_diffs: List[float], phase_diffs_bg: List[float], phase_diffs_bad: List[float], phase_diffs_bg_bad: List[float], bin_size: int) -> plt.figure:
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    hist, edge = np.histogram(phase_diffs, bins=bin_size)
    y_min = 0
    y_max = np.max(hist)
    dist = y_max - y_min
    y_min = y_min - dist * 0.1
    y_max = y_max + dist * 0.1
    axes[0][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bg, bins=bin_size)
    y_min = 0
    y_max = np.max(hist)
    dist = y_max - y_min
    y_min = y_min - dist * 0.1
    y_max = y_max + dist * 0.1
    axes[0][1].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bad, bins=bin_size)
    y_min = 0
    y_max = np.max(hist)
    dist = y_max - y_min
    y_min = y_min - dist * 0.1
    y_max = y_max + dist * 0.1
    axes[1][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bg_bad, bins=bin_size)
    y_min = 0
    y_max = np.max(hist)
    dist = y_max - y_min
    y_min = y_min - dist * 0.1
    y_max = y_max + dist * 0.1
    axes[1][1].set_ylim(y_min, y_max)
    sns.histplot(phase_diffs, ax=axes[0][0], bins=bin_size, color='blue', kde=True)
    sns.histplot(phase_diffs_bg, ax=axes[0][1], bins=bin_size, color='blue', kde=True)
    sns.histplot(phase_diffs_bad, ax=axes[1][0], bins=bin_size, color='red', kde=True)
    sns.histplot(phase_diffs_bg_bad, ax=axes[1][1], bins=bin_size, color='red', kde=True)
    return fig

def phase_diff(pfc_mag, str_mag, pfc_bg, str_bg) -> Tuple[float, float]:
    session_length = len(pfc_mag)
    # green is striatum, black is PFC, left is striatum, right is pfc
    # low_pass filter
    b, a = butter(N=4, Wn=10/session_length, btype='low', output='ba')
    filtered_pfc = filtfilt(b=b, a=a, x=pfc_mag)
    filtered_str = filtfilt(b=b, a=a, x=str_mag)
    filtered_pfc = detrend(filtered_pfc, type='constant')
    filtered_str = detrend(filtered_str, type='constant')
    # hilbert transform
    hilbert_pfc = hilbert(filtered_pfc)
    hilbert_str = hilbert(filtered_str)
    phase_pfc = np.angle(hilbert_pfc)
    phase_str = np.angle(hilbert_str)

    filtered_pfc_bg = filtfilt(b=b, a=a, x=pfc_bg)
    filtered_str_bg = filtfilt(b=b, a=a, x=str_bg)
    filtered_pfc_bg = detrend(filtered_pfc_bg, type='constant')
    filtered_str_bg = detrend(filtered_str_bg, type='constant')
    # hilbert transform
    hilbert_pfc_bg = hilbert(filtered_pfc_bg)
    hilbert_str_bg = hilbert(filtered_str_bg)
    phase_pfc_bg = np.angle(hilbert_pfc_bg)
    phase_str_bg = np.angle(hilbert_str_bg)

    phase_diff = circmean(phase_pfc - phase_str, high=np.pi, low=-np.pi)
    phase_diff_bg = circmean(phase_pfc_bg - phase_str_bg, high=np.pi, low=-np.pi)

    return phase_diff, phase_diff_bg