import os
from os import mkdir
from os.path import join as pjoin, isfile, isdir, basename
from glob import glob
from typing import List, Tuple, Dict
import warnings
from multiprocessing import Pool
from functools import partial
from shutil import rmtree


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from scipy.stats import pearsonr, ttest_ind, spearmanr # type: ignore
from scipy.signal import correlate # type: ignore

from lib.calculation import moving_window_mean_prior, get_relative_spike_times, get_spike_times_in_window, get_normalized_cross_correlation, crosscorrelation, check_probe_drift, get_mean_and_sem
from lib.file_utils import get_dms_pfc_paths_all, get_dms_pfc_paths_mono
from lib.figure_utils import remove_top_and_right_spines, plot_with_sem_error_bar

# ignore constant input warning from scipy
warnings.filterwarnings("ignore", category=RuntimeWarning)

behaviour_root = pjoin('data', 'behaviour_data')
spike_root = pjoin('data', 'spike_times', 'sessions')

figure_6_figure_root = pjoin('figures', 'all_figures', 'figure_8')
if not isdir(figure_6_figure_root):
    mkdir(figure_6_figure_root)
panel_abc_figure_root = pjoin(figure_6_figure_root, 'panel_abc')
if not isdir(panel_abc_figure_root):
    mkdir(panel_abc_figure_root)
panel_abc_significant = pjoin(figure_6_figure_root, 'panel_abc', 'significant')
if not isdir(panel_abc_significant):
    mkdir(panel_abc_significant)


figure_8_data_root = pjoin('data', 'spike_times', 'figure_8')
figure_data_root = pjoin('figure_data', 'figure_8')
if not isdir(figure_data_root):
    mkdir(figure_data_root)
# save the data for trial reward and interconnectivity strength
# in panel a and b folder
panel_a_data_root = pjoin(figure_data_root, 'panel_a')
panel_b_data_root = pjoin(figure_data_root, 'panel_b')
panel_c_data_root = pjoin(figure_data_root, 'panel_c')
if not isdir(panel_a_data_root):
    mkdir(panel_a_data_root)
if not isdir(panel_b_data_root):
    mkdir(panel_b_data_root)
if not isdir(panel_c_data_root):
    mkdir(panel_c_data_root)

p_value_threshold = 0.05

WINDOW_LEFT = -1
WINDOW_RIGHT = 0
BIN_SIZE = 0.01

# using firing during intertrial interval (ITI) window -1 to -0.5ms
def get_interconnectivity_strength(pfc_times: np.ndarray, dms_times: np.ndarray, cue_times: np.ndarray, reset: bool=False) -> np.ndarray:
    pfc_spike_times = get_spike_times_in_window(pfc_times, cue_times, WINDOW_LEFT, WINDOW_RIGHT)
    dms_spike_times = get_spike_times_in_window(dms_times, cue_times, WINDOW_LEFT, WINDOW_RIGHT)

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
        dms_trial_times = []

        for ind in indices:    
            pfc_trial_times += pfc_spike_times[ind]
            dms_trial_times += dms_spike_times[ind]
        
        min_time = cue_times[indices[0]] + WINDOW_LEFT
        max_time = cue_times[indices[-1]] + WINDOW_RIGHT

        # if any of the array is empty, append 0
        if len(pfc_trial_times) == 0 or len(dms_trial_times) == 0:
            interconnectivity_strength.append(0)
            continue

        pfc_trial_times = np.histogram(pfc_trial_times, bins=np.arange(min_time, max_time+BIN_SIZE, BIN_SIZE))[0]
        dms_trial_times = np.histogram(dms_trial_times, bins=np.arange(min_time, max_time+BIN_SIZE, BIN_SIZE))[0]
        
        normalized_cross_corr = get_normalized_cross_correlation(pfc_trial_times, dms_trial_times, 20)
        interconnectivity_strength.append(np.max(np.abs(normalized_cross_corr)))

    interconnectivity_strength = np.array(interconnectivity_strength)
    
    return interconnectivity_strength


def figure_8_panel_abc(session_name: str, pfc_name: str, dms_name: str, pfc_times: np.ndarray, dms_times: np.ndarray, cue_times: np.ndarray, reward_proportion: np.ndarray, reset: bool = False, plot: bool = True):
    # load the interconnectivity strength if it exists
    if isfile(pjoin(figure_8_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy')) and not reset:
        interconnectivity_strength = np.load(pjoin(figure_8_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy'))
    else:
        # calculate the interconnectivity strength
        interconnectivity_strength = get_interconnectivity_strength(pfc_times, dms_times, cue_times)
        # load the interconnectivity strength
        np.save(pjoin(figure_8_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy'), interconnectivity_strength)

    discretized_reward_proportion = np.digitize(reward_proportion, bins=np.arange(0, 1, 0.2))
    # get the mean interconnectivity strength for each reward 
    # proportion bin as well as std error
    interconnectivity_strength_binned = []
    interconnectivity_strength_binned_err = []
    for i in range(1, 6):
        interconnectivity_strength_binned.append(np.mean(interconnectivity_strength[discretized_reward_proportion == i]))
        interconnectivity_strength_binned_err.append(np.std(interconnectivity_strength[discretized_reward_proportion == i]) / np.sqrt(np.sum(discretized_reward_proportion == i)))
    discretized_reward_proportion = discretized_reward_proportion * 0.2 - 0.1

    # calculate pearson r and the p value, set it as figure title
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        try:
            dis_r, dis_p = pearsonr(discretized_reward_proportion, interconnectivity_strength)
        except (RuntimeWarning, UserWarning):
            dis_r, dis_p = 0, 1

    # calculate pearson r and the p value, set it as figure title
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        try:
            r, p = pearsonr(reward_proportion, interconnectivity_strength)
        except (RuntimeWarning, UserWarning):
            r, p = 0, 1
    
    if plot and not isfile(pjoin(panel_abc_figure_root, f'{session_name}_{pfc_name}_{dms_name}.png')) and not check_probe_drift(interconnectivity_strength):
        panel_a_data = pd.DataFrame({'trial_index': np.arange(len(interconnectivity_strength), dtype=int)+1, 'interconnectivity_strength': interconnectivity_strength})

        panel_a_data.to_csv(pjoin(panel_a_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.csv'))

        # plot reward proportion vs cross correlation in twinx plot
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        axes[0].plot(reward_proportion, color='tab:blue')
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('Reward proportion', color='tab:blue')
        axes[0].tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = axes[0].twinx()
        ax2.plot(interconnectivity_strength, color='tab:red')
        ax2.set_ylabel('Cross correlation', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # fig.suptitle(f'Pearson r: {r:.2f}, p: {p:.2f}, {pfc_name} vs {dms_name}')
        plt.close()

        panel_c_data = pd.DataFrame({'discretized_reward_proportion': np.arange(0.1, 1, 0.2), 'interconnectivity_strength_mean': interconnectivity_strength_binned, 'interconnectivity_strength_err': interconnectivity_strength_binned_err})
        panel_c_data.to_csv(pjoin(panel_c_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.csv'))
        
        # plot interconnectivity_strength against reward_proportion with error bar
        sns.lineplot(x=np.arange(0.1, 1, 0.2), y=interconnectivity_strength_binned, ax=axes[1])
        axes[1].errorbar(x=np.arange(0.1, 1, 0.2), y=interconnectivity_strength_binned, yerr=interconnectivity_strength_binned_err, fmt='o', color='black')
        # set x axis tick label 
        axes[1].set_xticks(np.arange(0, 1, 0.2))
        axes[1].set_title(f'Pearson r: {dis_r:.2f}, p: {dis_p:.2f}')

        fig.suptitle(f'Pearson r: {r:.2f}, p: {p:.2f}, {pfc_name} vs {dms_name}')

        if p < p_value_threshold:
            fig.savefig(pjoin(panel_abc_significant, f'{session_name}_{pfc_name}_{dms_name}.png'))
        else:
            fig.savefig(pjoin(panel_abc_figure_root, f'{session_name}_{pfc_name}_{dms_name}.png'))

        plt.close()
    
    # calculate the z score for interconnectivity strength and reward proportion
    interconnectivity_strength_z_score = (interconnectivity_strength - np.mean(interconnectivity_strength)) / np.std(interconnectivity_strength)
    reward_proportion_z_score = (reward_proportion - np.mean(reward_proportion)) / np.std(reward_proportion)
    # calculate the overall cross correlation
    overall_cross_cor = crosscorrelation(interconnectivity_strength_z_score, reward_proportion_z_score, maxlag=50)

    return p, r, list(overall_cross_cor)


# average propotion of significantly positively and negatively correlated prior reward
# proportion and interconnectivity strength. Panel d is for all the data, while panel d
# is for the mono pair data
def figure_8_panel_dh(mono: bool = False, reset: bool = False):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    sig_rs_positive_percentage = []
    sig_rs_negative_percentage = []

    if reset:
        rmtree(figure_8_data_root)
        mkdir(figure_8_data_root)

    if mono:
        dms_pfc_paths = get_dms_pfc_paths_mono()

        session_total = {}
        session_sig_rs_positive = {}
        session_sig_rs_negative = {}
        
        for mono_pair in tqdm.tqdm(dms_pfc_paths.iterrows()):
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
            cue_time = np.array(behaviour_data['cue_time'].values)
            trial_reward = np.array(behaviour_data['trial_reward'].values)
            # fill the nan with 0
            trial_reward[np.isnan(trial_reward)] = 0
            reward_proportion = moving_window_mean_prior(trial_reward, 10)

            # plot figure 6 poster panel ab
            p, r, _ = figure_8_panel_abc(session_name, pfc_name, dms_name, pfc_times, dms_times, cue_time, reward_proportion, reset=reset, plot=False)

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
        dms_pfc_paths = get_dms_pfc_paths_all(no_nan=False)

        with Pool() as pool:
            process_session_partial = partial(process_session_panel_d, reset=reset)
            results = list(tqdm.tqdm(pool.imap(process_session_partial, dms_pfc_paths), total=len(dms_pfc_paths)))

        for result in results:
            sig_rs_positive_percentage.append(result[0])
            sig_rs_negative_percentage.append(result[1])

    # t test to see if the percentage of positive and negative significant rs are different
    t, p = ttest_ind(sig_rs_positive_percentage, sig_rs_negative_percentage, alternative='less')
    print(f't: {t}, p: {p}')

    figure_8_panel_d_data = pd.DataFrame({'positive': sig_rs_positive_percentage, 'negative': sig_rs_negative_percentage})
    if mono:
        figure_8_panel_d_data.to_csv(pjoin(figure_data_root, 'figure_8_panel_dh_mono.csv'), index=False)
    else:
        figure_8_panel_d_data.to_csv(pjoin(figure_data_root, 'figure_8_panel_dh.csv'), index=False)

    # plot the bar plot with the average percentage of positive and negative significant rs
    sig_rs_positive_percentage = np.array(sig_rs_positive_percentage)
    sig_rs_negative_percentage = np.array(sig_rs_negative_percentage)

    sns.boxplot(data=[sig_rs_positive_percentage[sig_rs_positive_percentage<1], sig_rs_negative_percentage[sig_rs_negative_percentage<1]], ax=axes)
    axes.set_ylim(0, 1)



# mean cross correlation between interconnectivity strength and reward proportion
# panel e is for all the data, while panel f is for the mono pair data
def figure_8_panel_ei(mono: bool = False, reset: bool = False):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))

    if reset:
        rmtree(figure_8_data_root)
        mkdir(figure_8_data_root)

    if mono:
        dms_pfc_paths = get_dms_pfc_paths_mono()

        overall_crosscors = []
        
        for mono_pair in tqdm.tqdm(dms_pfc_paths.iterrows()):
            session_path = mono_pair[1]['session_path']
            pfc_path = mono_pair[1]['pfc_path']
            dms_path = mono_pair[1]['dms_path']

            session_name = basename(session_path).split('.')[0]
            pfc_name = basename(pfc_path).split('.')[0]
            dms_name = basename(dms_path).split('.')[0]
            
            pfc_times = np.load(pfc_path)
            dms_times = np.load(dms_path)

            behaviour_data = pd.read_csv(session_path)
            cue_time = np.array(behaviour_data['cue_time'].values)
            trial_reward = np.array(behaviour_data['trial_reward'].values)
            # fill the nan with 0
            trial_reward[np.isnan(trial_reward)] = 0
            reward_proportion = moving_window_mean_prior(trial_reward, 10)

            # plot figure 6 poster panel ab
            p, r, overall_crosscor = figure_8_panel_abc(session_name, pfc_name, dms_name, pfc_times, dms_times, cue_time, reward_proportion, reset=reset, plot=False)

            overall_crosscors.append(overall_crosscor)
    else:
        dms_pfc_paths = get_dms_pfc_paths_all(no_nan=False)

        with Pool() as pool:
            process_session_partial = partial(process_session_panel_f, reset=reset)
            overall_crosscors = list(tqdm.tqdm(pool.imap(process_session_partial, dms_pfc_paths), total=len(dms_pfc_paths)))

        # print the length of arrays in overall_crosscors
        overall_crosscors = np.concatenate(overall_crosscors)

    print(overall_crosscors)
    overall_crosscors, overall_crosscors_sem = get_mean_and_sem(overall_crosscors)

    # save the data into a dataframe
    figure_8_panel_ei_data = pd.DataFrame({'trial_lag': np.arange(-50, 51, 1), 'cross_correlation': overall_crosscors, 'cross_correlation_sem': overall_crosscors_sem})
    if mono:
        figure_8_panel_ei_data.to_csv(pjoin(figure_data_root, 'figure_8_panel_ei_data_mono.csv'), index=False)
    else:
        figure_8_panel_ei_data.to_csv(pjoin(figure_data_root, 'figure_8_panel_ei_data.csv'), index=False)

    # plot overall crosscor
    plot_with_sem_error_bar(ax=axes,x=np.arange(-50, 51, 1, dtype=int), mean=overall_crosscors, sem=overall_crosscors_sem, color='black')
    axes.set_xlabel('Trial Lag')


# split into rewarded and non_rewarded trials
def figure_8_panel_fj(mono: bool = False, reset: bool = False):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))

    rewarded_strength = []
    non_rewarded_strength = []

    if reset:
        rmtree(figure_8_data_root)
        mkdir(figure_8_data_root)
    
    if mono:
        dms_pfc_paths = get_dms_pfc_paths_mono()
        
        for mono_pair in tqdm.tqdm(dms_pfc_paths.iterrows()):
            session_path = mono_pair[1]['session_path']
            pfc_path = mono_pair[1]['pfc_path']
            dms_path = mono_pair[1]['dms_path']

            session_name = basename(session_path).split('.')[0]
            pfc_name = basename(pfc_path).split('.')[0]
            dms_name = basename(dms_path).split('.')[0]

            behaviour_data = pd.read_csv(session_path)
            cue_time = np.array(behaviour_data['cue_time'].values)
            trial_reward = np.array(behaviour_data['trial_reward'].values)

            rewarded_trials = np.where(trial_reward == 1)[0]
            non_rewarded_trials = np.where(trial_reward == 0)[0]

            if isfile(pjoin(figure_8_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy')) and not reset:
                interconnectivity_strength = np.load(pjoin(figure_8_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy'))
            else:
                pfc_times = np.load(pfc_path)
                dms_times = np.load(dms_path)
                # calculate the interconnectivity strength
                interconnectivity_strength = get_interconnectivity_strength(pfc_times, dms_times, cue_time)
                # load the interconnectivity strength
                np.save(pjoin(figure_8_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy'), interconnectivity_strength)

            rewarded_strength.extend(interconnectivity_strength[rewarded_trials])
            non_rewarded_strength.extend(interconnectivity_strength[non_rewarded_trials])
    else:
        dms_pfc_paths = get_dms_pfc_paths_all(no_nan=False)

        with Pool() as pool:
            process_session_partial = partial(process_session_panel_e, reset=reset)
            results = list(tqdm.tqdm(pool.imap(process_session_partial, dms_pfc_paths), total=len(dms_pfc_paths)))

        for result in results:
            rewarded_strength += result[0]
            non_rewarded_strength += result[1]

    figure_8_panel_fj_data_rewarded = pd.DataFrame({'rewarded': rewarded_strength})
    figure_8_panel_fj_data_non_rewarded = pd.DataFrame({'non_rewarded': non_rewarded_strength})
    if mono:
        figure_8_panel_fj_data_rewarded.to_csv(pjoin(figure_data_root, 'figure_8_panel_fj_rewarded_mono.csv'), index=False)
        figure_8_panel_fj_data_non_rewarded.to_csv(pjoin(figure_data_root, 'figure_8_panel_fj_non_rewarded_mono.csv'), index=False)
    else:
        figure_8_panel_fj_data_rewarded.to_csv(pjoin(figure_data_root, 'figure_8_panel_fj_rewarded.csv'), index=False)
        figure_8_panel_fj_data_non_rewarded.to_csv(pjoin(figure_data_root, 'figure_8_panel_fj_non_rewarded.csv'), index=False)

    # plot the bar plot with the average percentage of rewarded and non-rewarded strength
    rewarded_strength = np.array(rewarded_strength)
    non_rewarded_strength = np.array(non_rewarded_strength)

    sns.boxplot(data=[rewarded_strength[rewarded_strength<100], non_rewarded_strength[non_rewarded_strength<100]], ax=axes)
    axes.set_ylim(0, 50)
    # t test to see if the percentage of rewarded and non-rewarded strength are different
    t, p = ttest_ind(rewarded_strength, non_rewarded_strength)
    print(f't: {t}, p: {p}')

# similar to panel e, but split into plateau and transition trials instead of 
# rewarded and non-rewarded trials
def figure_8_panel_gk(mono: bool = False, reset: bool = False):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))

    plateau_strength = []
    transition_strength = []

    if reset:
        rmtree(figure_8_data_root)
        mkdir(figure_8_data_root)

    if mono:
        dms_pfc_paths = get_dms_pfc_paths_mono()
        
        for mono_pair in tqdm.tqdm(dms_pfc_paths.iterrows()):
            session_path = mono_pair[1]['session_path']
            pfc_path = mono_pair[1]['pfc_path']
            dms_path = mono_pair[1]['dms_path']

            session_name = basename(session_path).split('.')[0]
            pfc_name = basename(pfc_path).split('.')[0]
            dms_name = basename(dms_path).split('.')[0]

            behaviour_data = pd.read_csv(session_path)
            cue_time = np.array(behaviour_data['cue_time'].values)
            leftP = np.array(behaviour_data['leftP'].values)

            # get the switches
            switches = find_switch(leftP)
            # find the plateau and transition trials
            plateau_trial_indices = []
            transition_trial_indices = []

            # if there are less than 20 trials before or after the switch
            # skip the switch
            for switch in switches:
                if switch < 20 or switch > len(leftP) - 20:
                    continue
                plateau_trial_indices += (range(switch-20, switch))
                transition_trial_indices += (range(switch, switch+20))

            plateau_trial_indices = np.array(plateau_trial_indices)
            transition_trial_indices = np.array(transition_trial_indices)

            if isfile(pjoin(figure_8_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy')) and not reset:
                interconnectivity_strength = np.load(pjoin(figure_8_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy'))
            else:
                pfc_times = np.load(pfc_path)
                dms_times = np.load(dms_path)
                # calculate the interconnectivity strength
                interconnectivity_strength = get_interconnectivity_strength(pfc_times, dms_times, cue_time)
                # load the interconnectivity strength
                np.save(pjoin(figure_8_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy'), interconnectivity_strength)
            
            plateau_strength.extend(interconnectivity_strength[plateau_trial_indices])
            transition_strength.extend(interconnectivity_strength[transition_trial_indices])
    else:
        dms_pfc_paths = get_dms_pfc_paths_all(no_nan=False)

        with Pool() as pool:
            process_session_partial = partial(process_session_panel_e_plateau_transition, reset=reset)
            results = list(tqdm.tqdm(pool.imap(process_session_partial, dms_pfc_paths), total=len(dms_pfc_paths)))

        for result in results:
            plateau_strength += result[0]
            transition_strength += result[1]

    # remove 0s from the arrays
    plateau_strength = np.array(plateau_strength)
    transition_strength = np.array(transition_strength)

    plateau_strength = plateau_strength[plateau_strength != 0]
    transition_strength = transition_strength[transition_strength != 0]

    # # store the data into a dataframe, with a column indicating the type of trial
    # figure_8_panel_gk_data = pd.DataFrame({'plateau': plateau_strength, 'transition': transition_strength})
    # if mono:
    #     figure_8_panel_gk_data.to_csv(pjoin(figure_data_root, 'figure_8_panel_gk_mono.csv'), index=False)
    # else:
    #     figure_8_panel_gk_data.to_csv(pjoin(figure_data_root, 'figure_8_panel_gk.csv'), index=False)

    # plateau_strength = np.array(plateau_strength)
    # transition_strength = np.array(transition_strength)

    # # plot the plateau and transitioning trials as box plots
    # sns.boxplot(data=plateau_strength, x='trial_type', y='strength', ax=axes)
    sns.boxplot(data=[plateau_strength, transition_strength], ax=axes)
    axes.set_ylim(0, 50)
    
    # t test to see if the percentage of rewarded and non-rewarded strength are different
    t, p = ttest_ind(plateau_strength, transition_strength)
    print(f't: {t}, p: {p}')


def process_session_panel_d(session, reset=False):
    session_sig_rs_positive = 0
    session_sig_rs_negative = 0
    session_name = session[0]
    cue_time = session[1]
    trial_reward = session[2]
    reward_proportion = moving_window_mean_prior(trial_reward, 10)

    # save the reward proportion
    panel_b_data = pd.DataFrame({'trial_index': np.arange(len(reward_proportion), dtype=int)+1, 'reward_proportion': reward_proportion})
    panel_b_data.to_csv(pjoin(panel_b_data_root, f'{session_name}_reward_proportion.csv'))

    session_total = len(session[3])

    for pair in session[3]:
        dms_path = pair[0]
        pfc_path = pair[1]

        pfc_name = basename(pfc_path).split('.')[0]
        dms_name = basename(dms_path).split('.')[0]

        pfc_times = np.load(pfc_path)
        dms_times = np.load(dms_path)

        # plot figure 6 poster panel ab
        p, r, _ = figure_8_panel_abc(session_name, pfc_name, dms_name, pfc_times, dms_times, cue_time, reward_proportion, reset=reset, plot=False)

        if p < p_value_threshold:
            if r > 0:
                session_sig_rs_positive += 1
            else:
                session_sig_rs_negative += 1

    # calculate the percentage of significant positive and negative r for each session
    sig_rs_positive_percentage = session_sig_rs_positive / session_total
    sig_rs_negative_percentage = session_sig_rs_negative / session_total


    return (sig_rs_positive_percentage, sig_rs_negative_percentage)


def process_session_panel_e(session, reset=False):
    session_name = session[0]
    cue_time = session[1]
    trial_reward = session[2]

    rewarded_interconnectivity = []
    non_rewarded_interconnectivity = []

    for pair in session[3]:
        dms_path = pair[0]
        pfc_path = pair[1]

        pfc_name = basename(pfc_path).split('.')[0]
        dms_name = basename(dms_path).split('.')[0]

        if isfile(pjoin(figure_8_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy')) and not reset:
            interconnectivity_strength = np.load(pjoin(figure_8_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy'))
        else:
            pfc_times = np.load(pfc_path)
            dms_times = np.load(dms_path)
            # calculate the interconnectivity strength
            interconnectivity_strength = get_interconnectivity_strength(pfc_times, dms_times, cue_time)
            # load the interconnectivity strength
            np.save(pjoin(figure_8_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy'), interconnectivity_strength)
        
        rewarded_interconnectivity.extend(interconnectivity_strength[trial_reward == 1])
        non_rewarded_interconnectivity.extend(interconnectivity_strength[trial_reward == 0])

    return (rewarded_interconnectivity, non_rewarded_interconnectivity)


def process_session_panel_e_plateau_transition(session, reset=False):
    session_name = session[0]
    cue_time = session[1]
    trial_reward = session[2]

    behaviour_data = pd.read_csv(pjoin(behaviour_root, session_name + '.csv'))
    leftP = np.array(behaviour_data['leftP'])

    switches = find_switch(leftP)
    # find the plateau and transition trials
    plateau_trial_indices = []
    transition_trial_indices = []

    # if there are less than 20 trials before or after the switch
    # skip the switch
    for switch in switches:
        if switch < 20 or switch > len(leftP) - 20:
            continue
        plateau_trial_indices += (range(switch-20, switch))
        transition_trial_indices += (range(switch, switch+20))

    plateau_trial_indices = np.array(plateau_trial_indices)
    transition_trial_indices = np.array(transition_trial_indices)

    plateau_interconnectivity = []
    transition_interconnectivity = []

    for pair in session[3]:
        dms_path = pair[0]
        pfc_path = pair[1]

        pfc_name = basename(pfc_path).split('.')[0]
        dms_name = basename(dms_path).split('.')[0]

        if isfile(pjoin(figure_8_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy')) and not reset:
            interconnectivity_strength = np.load(pjoin(figure_8_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy'))
        else:
            pfc_times = np.load(pfc_path)
            dms_times = np.load(dms_path)
            # calculate the interconnectivity strength
            interconnectivity_strength = get_interconnectivity_strength(pfc_times, dms_times, cue_time)
            # load the interconnectivity strength
            np.save(pjoin(figure_8_data_root, f'{session_name}_{pfc_name}_{dms_name}_interconnectivity_strength.npy'), interconnectivity_strength)
        
        plateau_interconnectivity.extend(interconnectivity_strength[plateau_trial_indices])
        transition_interconnectivity.extend(interconnectivity_strength[transition_trial_indices])

    return (plateau_interconnectivity, transition_interconnectivity)

def process_session_panel_f(session, reset=False):
    session_name = session[0]
    cue_time = session[1]
    trial_reward = session[2]
    reward_proportion = moving_window_mean_prior(trial_reward, 10)

    overall_crosscors = []
    session_total = len(session[3])

    for pair in session[3]:
        dms_path = pair[0]
        pfc_path = pair[1]

        pfc_name = basename(pfc_path).split('.')[0]
        dms_name = basename(dms_path).split('.')[0]

        pfc_times = np.load(pfc_path)
        dms_times = np.load(dms_path)

        # plot figure 6 poster panel ab
        p, r, overall_crosscor = figure_8_panel_abc(session_name, pfc_name, dms_name, pfc_times, dms_times, cue_time, reward_proportion, reset=reset, plot=False)

        overall_crosscors.append(overall_crosscor)
    

    return overall_crosscors


# return the indices of the trials where the high reward side switches
def find_switch(leftP: np.ndarray) -> List[int]:
    switch_indices = []
    for i in range(len(leftP)-1):
        if leftP[i] != leftP[i+1]:
            switch_indices.append(i)
    return switch_indices