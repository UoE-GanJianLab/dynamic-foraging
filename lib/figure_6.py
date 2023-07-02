import os
from os.path import join as pjoin, isfile, isdir, basename 
from glob import glob
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, ttest_ind, spearmanr # type: ignore
from scipy.signal import correlate # type: ignore

from lib.calculation import moving_window_mean, get_firing_rate_window, moving_window_mean_prior, get_relative_spike_times, get_normalized_cross_correlation, crosscorrelation
from lib.file_utils import get_dms_pfc_paths_all, get_dms_pfc_paths_mono

behaviour_root = pjoin('data', 'behaviour_data')
spike_root = pjoin('data', 'spike_times', 'sessions')

figure_6_data_root = pjoin('data', 'spike_times', 'figure_6')

p_value_threshold = 0.05


# using firing during intertrial interval (ITI) window -1 to -0.5ms
def get_interconnectivity_strength(pfc_times: np.ndarray, dms_times: np.ndarray, cue_times: np.ndarray, reset: bool=False) -> np.ndarray:
    pfc_relative_spike_times = get_relative_spike_times(pfc_times, cue_times, -1, -0.5)
    dms_relative_spike_times = get_relative_spike_times(dms_times, cue_times, -1, -0.5)

    # calculate the cross correlation
    interconnectivity_strength = []
    for i in range(len(cue_times)):
        # get the 10 trials index before the current trial and the 10 trials index after 
        # the current trial if the current trial is within the first 10 trials or last 10 
        # trials, use the incomplete window without padding
        if i < 10:
            indices = np.arange(0, i + 11)
        elif i > len(cue_times) - 11:
            indices = np.arange(i - 10, len(cue_times))
        else:
            indices = np.arange(i - 10, i + 11)

        # empty histogram array
        pfc_trial_times = []
        str_trial_times = []

        for ind in indices:    
            pfc_trial_times += pfc_relative_spike_times[ind]
            str_trial_times += dms_relative_spike_times[ind]

        pfc_trial_times = np.histogram(pfc_trial_times, bins=np.arange(-1, -0.5, 0.01))[0]
        str_trial_times = np.histogram(str_trial_times, bins=np.arange(-1, -0.5, 0.01))[0]


        # if any of the array is empty, append 0
        if len(pfc_trial_times) == 0 or len(str_trial_times) == 0:
            interconnectivity_strength.append(0)
            continue
        normalized_cross_corr = get_normalized_cross_correlation(pfc_trial_times, str_trial_times, 50)
        interconnectivity_strength.append(np.max(np.abs(normalized_cross_corr)))

    interconnectivity_strength = np.array(interconnectivity_strength)
    
    return interconnectivity_strength
    

# reset means whether to recalculate the values disregarding the saved files
def figure_6_poster_panel_ab(session_name: str, pfc_name: str, dms_name: str,pfc_times: np.ndarray, dms_times: np.ndarray, cue_times: np.ndarray, reward_proportion: np.ndarray, reset: bool = False):
    if isfile(pjoin(figure_6_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy')) and not reset:
        # load the interconnectivity strength
        interconnectivity_strength = np.load(pjoin(figure_6_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy')) 
    else:
        # calculate the interconnectivity strength
        interconnectivity_strength = get_interconnectivity_strength(pfc_times, dms_times, cue_times, reset)
        # load the interconnectivity strength
        np.save(pjoin(figure_6_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy'), interconnectivity_strength) 


    # plot reward proportion vs cross correlation in twinx plot
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
    ax1.plot(reward_proportion, color='tab:blue')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Reward proportion', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.plot(interconnectivity_strength, color='tab:red')
    ax2.set_ylabel('Cross correlation', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # calculate pearson r and the p value, set it as figure title
    r, p = pearsonr(reward_proportion, interconnectivity_strength)
    fig.suptitle(f'Pearson r: {r:.2f}, p: {p:.2f}, {pfc_name} vs {dms_name}')

    plt.close()

    # calculate the overall cross correlation
    overall_cross_cor = crosscorrelation(interconnectivity_strength, reward_proportion, maxlag=50)

    return p, r, overall_cross_cor

def get_figure_6_poster_panel_ab(mono: bool = False, reset: bool = False):
    if mono:
        str_pfc_paths = get_dms_pfc_paths_mono()
        
        for mono_pair in str_pfc_paths.iterrows():
            session_path = mono_pair[1]['session_path']
            pfc_path = mono_pair[1]['pfc_path']
            dms_path = mono_pair[1]['dms_path']

            session_name = basename(session_path).split('.')[0]
            pfc_name = basename(pfc_path).split('.')[0]
            dms_name = basename(dms_path).split('.')[0]

            pfc_times = np.load(pfc_path)
            dms_times = np.load(dms_path)

            behaviour_data = pd.read_csv(session_path)
            # ignore the nan trials
            behaviour_data = behaviour_data[~np.isnan(behaviour_data['trial_reward'])]  
            cue_time = np.array(behaviour_data['cue_time'].values)            
            trial_reward = np.array(behaviour_data['trial_reward'].values)
            reward_proportion = moving_window_mean_prior(trial_reward, 20)

            # plot figure 6 poster panel ab
            p, r, _ = figure_6_poster_panel_ab(session_name, pfc_name, dms_name, pfc_times, dms_times, cue_time, reward_proportion, reset=reset)
    else:
        str_pfc_paths = get_dms_pfc_paths_all()

        for session in str_pfc_paths:
            session_name = session[0]
            cue_time = session[1]
            trial_reward = session[2]
            reward_proportion = moving_window_mean_prior(trial_reward, 20)

            for pair in session[3]:
                pfc_path = pair[0]
                dms_path = pair[1]
            
                pfc_name = basename(pfc_path).split('.')[0]
                dms_name = basename(dms_path).split('.')[0]

                pfc_times = np.load(pfc_path)
                dms_times = np.load(dms_path)

                # plot figure 6 poster panel ab
                p, r, _ = figure_6_poster_panel_ab(session_name, pfc_name, dms_name, pfc_times, dms_times, cue_time, reward_proportion, reset=reset)


def figure_6_poster_panel_c(session_name: str, pfc_name: str, dms_name: str, pfc_times: np.ndarray, str_times: np.ndarray, cue_times: np.ndarray, reward_proportion: np.ndarray, reset: bool = False):
    # load the interconnectivity strength if it exists
    if isfile(pjoin(figure_6_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy')) and not reset:
        interconnectivity_strength = np.load(pjoin(figure_6_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy'))
    else:
        # calculate the interconnectivity strength
        interconnectivity_strength = get_interconnectivity_strength(pfc_times, str_times, cue_times)
        # load the interconnectivity strength
        np.save(pjoin(figure_6_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy'), interconnectivity_strength)

    # calculate the overall cross correlation
    overall_cross_cor = crosscorrelation(interconnectivity_strength, reward_proportion, maxlag=50)

    discretized_reward_proportion = np.digitize(reward_proportion, bins=np.arange(0, 1, 0.2))
    discretized_reward_proportion = discretized_reward_proportion * 0.2 - 0.1

    # plot reward proportion vs cross correlation in twinx plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    
    # plot interconnectivity_strength against reward_proportion
    sns.lineplot(x=discretized_reward_proportion, y=interconnectivity_strength, ax=ax, color='tab:blue', err_style='bars')
    # set x axis tick label 
    ax.set_xticks(np.arange(0, 1, 0.2))

    plt.close()

    # calculate pearson r and the p value, set it as figure title
    r, p = pearsonr(reward_proportion, interconnectivity_strength)


    return p, r, overall_cross_cor


def figure_6_poster_panel_d(mono: bool = False, reset: bool = False):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    sig_rs_positive_percentage = []
    sig_rs_negative_percentage = []

    if mono:
        str_pfc_paths = get_dms_pfc_paths_mono()

        total = len(str_pfc_paths)

        session_total = {}
        session_sig_rs_positive = {}
        session_sig_rs_negative = {}
        
        for mono_pair in str_pfc_paths.iterrows():
            session_path = mono_pair[1]['session_path']
            pfc_path = mono_pair[1]['pfc_path']
            dms_path = mono_pair[1]['dms_path']

            session_name = basename(session_path).split('.')[0]
            pfc_name = basename(pfc_path).split('.')[0]
            dms_name = basename(dms_path).split('.')[0]

            if session_name not in session_total:
                session_total[session_name] = 0
                session_sig_rs_positive[session_name] = 0
                session_sig_rs_negative[session_name] = 0
            
            session_total[session_name] += 1

            pfc_times = np.load(pfc_path)
            dms_times = np.load(dms_path)

            behaviour_data = pd.read_csv(session_path)
            # ignore the nan trials
            behaviour_data = behaviour_data[~np.isnan(behaviour_data['trial_reward'])]
            cue_time = np.array(behaviour_data['cue_time'].values)
            trial_reward = np.array(behaviour_data['trial_reward'].values)
            reward_proportion = moving_window_mean_prior(trial_reward, 20)

            # plot figure 6 poster panel ab
            p, r, _ = figure_6_poster_panel_c(session_name, pfc_name, dms_name, pfc_times, dms_times, cue_time, reward_proportion, reset=reset)

            if p < p_value_threshold:
                if r > 0:
                    session_sig_rs_positive[session_name] += 1
                else:
                    session_sig_rs_negative[session_name] += 1
        # calculate the percentage of significant positive and negative r for each session
        for session_name in session_total:
            sig_rs_positive_percentage.append(session_sig_rs_positive[session_name] / session_total[session_name])
            sig_rs_negative_percentage.append(session_sig_rs_negative[session_name] / session_total[session_name])
    else:
        str_pfc_paths = get_dms_pfc_paths_all()

        for session in str_pfc_paths:
            session_sig_rs_positive = 0
            session_sig_rs_negative = 0
            session_name = session[0]
            cue_time = session[1]
            trial_reward = session[2]
            reward_proportion = moving_window_mean_prior(trial_reward, 20)#

            session_total = len(session[3])

            for pair in session[3]:
                pfc_path = pair[0]
                dms_path = pair[1]
            
                pfc_name = basename(pfc_path).split('.')[0]
                dms_name = basename(dms_path).split('.')[0]

                pfc_times = np.load(pfc_path)
                dms_times = np.load(dms_path)

                # plot figure 6 poster panel ab
                p, r, _ = figure_6_poster_panel_c(session_name, pfc_name, dms_name, pfc_times, dms_times, cue_time, reward_proportion, reset=reset)

                if p < p_value_threshold:
                    if r > 0:
                        session_sig_rs_positive += 1
                    else:
                        session_sig_rs_negative += 1

            # calculate the percentage of significant positive and negative r for each session
            sig_rs_positive_percentage.append(session_sig_rs_positive / session_total)
            sig_rs_negative_percentage.append(session_sig_rs_negative / session_total)



    # t test to see if the percentage of positive and negative significant rs are different
    t, p = ttest_ind(sig_rs_positive_percentage, sig_rs_negative_percentage, alternative='less')
    print(f't: {t}, p: {p}')

    # plot the bar plot with the average percentage of positive and negative significant rs
    sns.barplot(x=['+', '-'], y=[np.mean(sig_rs_positive_percentage), np.mean(sig_rs_negative_percentage)], ax=axes)
    axes.set_ylim(0, 1)

def figure_6_poster_panel_e(mono: bool = False, reset: bool = False):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))

    # 

def figure_6_poster_panel_f(mono: bool = False, reset: bool = False):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))

    #