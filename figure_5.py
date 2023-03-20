from os.path import join as pjoin
from os import listdir, mkdir
from os.path import basename

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, hilbert, detrend

import sys
sys.path.append('lib')

from extraction import get_strong_corr

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