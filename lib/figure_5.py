from os.path import join as pjoin
from os import listdir, mkdir
from os.path import basename
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from scipy.signal import butter, filtfilt, hilbert, detrend # type: ignore
from scipy.stats import circmean # type: ignore



import sys
sys.path.append('lib')

from lib.file_utils import get_str_pfc_strong_corr_mono

# relative positions to cue_time
ITI_LEFT = -1
ITI_RIGHT = 0
RESPONSE_LEFT = 0
RESPONSE_RIGHT = 1.5

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


def fig_5_panel_c(phase_diffs: List[float], phase_diffs_bg: List[float], bin_size: int, zero_ymin: bool = True) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    hist, edge = np.histogram(phase_diffs, bins=bin_size)
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist)
    dist = y_max - y_min
    y_min = y_min - dist * 0.1
    y_max = y_max + dist * 0.1
    axes[0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bg, bins=bin_size)
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist)
    dist = y_max - y_min
    y_min = y_min - dist * 0.1
    y_max = y_max + dist * 0.1
    axes[1].set_ylim(y_min, y_max)
    sns.histplot(phase_diffs, ax=axes[0], bins=bin_size, color='black', kde=True) # type: ignore
    sns.histplot(phase_diffs_bg, ax=axes[1], bins=bin_size, color='black', kde=True) # type: ignore

    # set y label
    axes[0].set_ylabel('Number of Cell Pairs')
    axes[1].set_ylabel('Number of Cell Pairs')

    # set x label
    axes[0].set_xlabel('Phase Difference (radians)')
    axes[1].set_xlabel('Phase Difference (radians)')

    # Set the x-axis tick labels to pi
    set_xticks_and_labels_pi(axes[0])
    set_xticks_and_labels_pi(axes[1])

    return fig


def fig_5_panel_d(phase_diffs: List[float], phase_diffs_bg: List[float], phase_diffs_bad: List[float], phase_diffs_bg_bad: List[float], bin_size: int, zero_ymin: bool = True) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    hist, edge = np.histogram(phase_diffs, bins=bin_size)
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist)
    dist = y_max - y_min
    y_min = y_min - dist * 0.1
    y_max = y_max + dist * 0.1
    axes[0][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bg, bins=bin_size)
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist)
    dist = y_max - y_min
    y_min = y_min - dist * 0.1
    y_max = y_max + dist * 0.1
    axes[0][1].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bad, bins=bin_size)
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist)
    dist = y_max - y_min
    y_min = y_min - dist * 0.1
    y_max = y_max + dist * 0.1
    axes[1][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bg_bad, bins=bin_size)
    if zero_ymin:
        y_min = 0
    else:
        # 10% lower than the lowest value
        y_min = np.min(hist) * 0.95
    y_max = np.max(hist)
    dist = y_max - y_min
    y_min = y_min - dist * 0.1
    y_max = y_max + dist * 0.1
    axes[1][1].set_ylim(y_min, y_max)
    sns.histplot(phase_diffs, ax=axes[0][0], bins=bin_size, color='blue', kde=True) # type: ignore
    sns.histplot(phase_diffs_bg, ax=axes[0][1], bins=bin_size, color='blue', kde=True) # type: ignore
    sns.histplot(phase_diffs_bad, ax=axes[1][0], bins=bin_size, color='red', kde=True) # type: ignore
    sns.histplot(phase_diffs_bg_bad, ax=axes[1][1], bins=bin_size, color='red', kde=True) # type: ignore

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

    return fig

def fig_5_panel_e(phase_diffs_pdrp_pfc: List[float], phase_diff_pdrp_pfc_bg: List[float], phase_diff_pdrp_str: List[float], phase_diff_pdrp_str_bg: List[float], bin_size: int) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    hist, edge = np.histogram(phase_diffs_pdrp_pfc, bins=bin_size)
    y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[0][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diff_pdrp_pfc_bg, bins=bin_size)
    y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[0][1].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diff_pdrp_str, bins=bin_size)
    y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[1][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diff_pdrp_str_bg, bins=bin_size)
    y_min = np.min(hist) * 0.95
    y_max = np.max(hist) * 1.05
    axes[1][1].set_ylim(y_min, y_max)
    sns.histplot(phase_diffs_pdrp_pfc, ax=axes[0][0], bins=bin_size, color='blue', kde=True) # type: ignore
    sns.histplot(phase_diff_pdrp_pfc_bg, ax=axes[0][1], bins=bin_size, color='blue', kde=True) # type: ignore
    sns.histplot(phase_diff_pdrp_str, ax=axes[1][0], bins=bin_size, color='red', kde=True) # type: ignore
    sns.histplot(phase_diff_pdrp_str_bg, ax=axes[1][1], bins=bin_size, color='red', kde=True) # type: ignore

    # set y label
    axes[0][1].set_ylabel('Number of Cells')
    axes[0][0].set_ylabel('Number of Cells')

    # set x label
    axes[0][0].set_xlabel('Phase Difference (radians)')
    axes[0][1].set_xlabel('Phase Difference (radians)')
    axes[1][0].set_xlabel('Phase Difference (radians)')

    # Set the x-axis tick labels to pi
    set_xticks_and_labels_pi(axes[0][0])
    set_xticks_and_labels_pi(axes[0][1])
    set_xticks_and_labels_pi(axes[1][0])
    set_xticks_and_labels_pi(axes[1][1])

    return fig


def set_xticks_and_labels_pi(ax):
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])


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

    return phase_diff, phase_diff_bg

def phase_diff(sig1, sig2) -> float:
    length = len(sig1)
    # low_pass filter
    b, a = butter(N=4, Wn=10/length, btype='low', output='ba')
    sig1 = filter_signal(sig1, b, a)
    sig2 = filter_signal(sig2, b, a)
    phase1 = hilbert_transform(sig1)
    phase2 = hilbert_transform(sig2)
    phase_diff = circmean(phase1 - phase2, high=np.pi, low=-np.pi)
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