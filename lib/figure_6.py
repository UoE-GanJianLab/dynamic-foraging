import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.signal import correlate

from lib.file_utils import get_sessions_names
from lib.calculation import moving_window_mean, get_firing_rate_window

# using firing during intertrial interval (ITI) window -1 to -0.5ms
def figure_6_panel_c(pfc_times: np.ndarray, str_times: np.ndarray, cue_times: np.ndarray, pfc_name: str, str_name: str, rewarded: np.ndarray):
    # calculate the cross correlation of the two signals during a centered 20-trial long window
    pfc_firing_rates = get_firing_rate_window(cue_times, pfc_times, -1, -0.5)
    str_firing_rates = get_firing_rate_window(cue_times, str_times, -1, -0.5)

    # smoothen
    pfc_firing_rates = moving_window_mean(np.array(pfc_firing_rates), 20)
    str_firing_rates = moving_window_mean(np.array(str_firing_rates), 20)

    # normalize firing rates to [0, 1]
    pfc_firing_rates = (pfc_firing_rates - np.min(pfc_firing_rates)) / (np.max(pfc_firing_rates) - np.min(pfc_firing_rates))
    str_firing_rates = (str_firing_rates - np.min(str_firing_rates)) / (np.max(str_firing_rates) - np.min(str_firing_rates))

    # calculate the cross correlation
    corr = correlate(pfc_firing_rates, str_firing_rates, mode='same')
    


# using firing during 1-3ms after cue
def figure_6_panel_e(pfc_times: np.ndarray, str_times: np.ndarray, cue_times: np.ndarray, pfc_name: str, str_name: str, rewarded: np.ndarray):
    # calculate the cross correlation of the two signals during a centered 20-trial long window
    pfc_firing_rates = get_firing_rate_window(cue_times, pfc_times, 1, 3)
    str_firing_rates = get_firing_rate_window(cue_times, str_times, 1, 3)