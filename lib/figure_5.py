from os.path import join as pjoin
from os import listdir, mkdir
from os.path import basename
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, hilbert, detrend

import sys
sys.path.append('lib')

from lib.extraction import get_strong_corr

# relative positions to cue_time
ITI_LEFT = -1
ITI_RIGHT = 0
RESPONSE_LEFT = 0
RESPONSE_RIGHT = 1.5

def get_response_mag_bg_firing(cue_times, pfc_times, str_times):
    pfc_ptr = 0
    str_ptr = 0

    pfc_mag = []
    str_mag = []

    pfc_bg_firing = []
    str_bg_firing = []

    for cue in cue_times:
        iti_pfc_count = 0
        response_pfc_count = 0
        iti_str_count = 0
        response_str_count = 0

        iti_left = cue + ITI_LEFT
        iti_right = cue + ITI_RIGHT
        response_left = cue + RESPONSE_LEFT
        response_right = cue + RESPONSE_RIGHT

        # move the pointers into iti window
        while pfc_ptr < len(pfc_times) and pfc_times[pfc_ptr] < iti_left:
            pfc_ptr += 1
        
        while str_ptr < len(str_times) and str_times[str_ptr] < iti_left:
            str_ptr += 1

        # count the amount of spikes in iti
        while pfc_ptr < len(pfc_times) and pfc_times[pfc_ptr] < iti_right:
            pfc_ptr += 1
            iti_pfc_count += 1
        
        while str_ptr < len(str_times) and str_times[str_ptr] < iti_right:
            str_ptr += 1
            iti_str_count += 1

        # move the pointer to response time window
        while pfc_ptr < len(pfc_times) and pfc_times[pfc_ptr] < response_left:
            pfc_ptr += 1
        
        while str_ptr < len(str_times) and str_times[str_ptr] < response_left:
            str_ptr += 1

        # count the amount of spikes in response time
        while pfc_ptr < len(pfc_times) and pfc_times[pfc_ptr] < response_right:
            pfc_ptr += 1
            response_pfc_count += 1
        
        while str_ptr < len(str_times) and str_times[str_ptr] < response_right:
            str_ptr += 1
            response_str_count += 1
        
        pfc_bg_firing.append(iti_pfc_count / (ITI_RIGHT - ITI_LEFT))
        str_bg_firing.append(iti_str_count / (ITI_RIGHT - ITI_LEFT))
        
        pfc_mag.append(abs(response_pfc_count / (RESPONSE_RIGHT - RESPONSE_LEFT) - iti_pfc_count / (ITI_RIGHT - ITI_LEFT)))
        str_mag.append(abs(response_str_count / (RESPONSE_RIGHT - RESPONSE_LEFT) - iti_str_count / (ITI_RIGHT - ITI_LEFT)))

    return pfc_mag, str_mag, pfc_bg_firing, str_bg_firing

def fig_5_panel_d(phase_diffs: List[float], phase_diffs_bg: List[float], phase_diffs_bad: List[float], phase_diffs_bg_bad: List[float], bin_size: int) -> plt.figure:
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    hist, edge = np.histogram(phase_diffs, bins=bin_size)
    y_min = np.min(hist)
    y_max = np.max(hist)
    dist = y_max - y_min
    y_min = y_min - dist * 0.1
    y_max = y_max + dist * 0.1
    axes[0][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bg, bins=bin_size)
    y_min = np.min(hist)
    y_max = np.max(hist)
    dist = y_max - y_min
    y_min = y_min - dist * 0.1
    y_max = y_max + dist * 0.1
    axes[0][1].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bad, bins=bin_size)
    y_min = np.min(hist)
    y_max = np.max(hist)
    dist = y_max - y_min
    y_min = y_min - dist * 0.1
    y_max = y_max + dist * 0.1
    axes[1][0].set_ylim(y_min, y_max)
    hist, edge = np.histogram(phase_diffs_bg_bad, bins=bin_size)
    y_min = np.min(hist)
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