from os.path import join as pjoin, isdir, isfile, basename
from os import listdir, mkdir
from shutil import rmtree
from os.path import join as pjoin, isdir
from os import listdir
from glob import glob

import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_ind

from lib.calculation import moving_window_mean, get_relative_spike_times, moving_window_mean_prior
from lib.figure_utils import remove_top_and_right_spines

WINDOW_LEFT = -0.5
WINDOW_RIGHT = 1.5
BIN_SIZE = 0.02 # 20ms bins

spike_time_dir = pjoin('data', 'spike_times')
behaviour_root = pjoin('data', 'behaviour_data')

# TODO: prpd strongly correlated either
# TODO: left and right response split

# load the prpd correlation data
prpd_correlation_data = pd.read_csv(pjoin('data', 'prpd_correlation.csv'))

signal_mvt_reward_file = pjoin(spike_time_dir, 'figure_3', 'signal_mvt_reward.npy')
signal_mvt_reward_high_low_file = pjoin(spike_time_dir, 'figure_3', 'signal_mvt_reward_high_low.npy')

wheel_velocity_file_root = pjoin('data', 'behaviour_data', 'wheel_velocity')

figure_3_data_root = pjoin('figure_data', 'figure_3')
if not isdir(figure_3_data_root):
    print('Creating ' + figure_3_data_root)
    mkdir(figure_3_data_root)

figure_4_data_root = pjoin('figure_data', 'figure_4')
if not isdir(figure_4_data_root):
    print('Creating ' + figure_4_data_root)
    mkdir(figure_4_data_root)

def compute_component_prp_spike_times(component_idx, prp_idx, relative_to_pfc, window_left=WINDOW_LEFT, window_right=WINDOW_RIGHT, bin_size=BIN_SIZE):
    component_spike_times = [relative_to_pfc[idx] for idx in component_idx if idx in prp_idx]
    if len(component_spike_times) == 0:
        return np.zeros(int((window_right - window_left) / bin_size))
    component_spike_times = np.concatenate(component_spike_times)
    component_spike_times = np.histogram(component_spike_times, bins=np.arange(window_left, window_right+bin_size, bin_size))[0]
    component_spike_times = np.divide(component_spike_times, bin_size)
    component_spike_times = np.divide(component_spike_times, len([idx for idx in component_idx if idx in prp_idx]))
    return component_spike_times



# poster panel a,b of figure 3
def get_figure_3_panel_ab(reset=False, prpd=False):
    dms_count = 0
    pfc_count = 0

    if not isdir(pjoin('data', 'spike_times', 'figure_3')):
        mkdir(pjoin('data', 'spike_times', 'figure_3'))

    if reset:
        rmtree(pjoin('data', 'spike_times', 'figure_3'))
        mkdir(pjoin('data', 'spike_times', 'figure_3'))

    if not isfile(signal_mvt_reward_file):
        pfc_signal_binned = []
        pfc_mvt_binned = []
        pfc_reward_binned = []

        dms_mvt_binned = []
        dms_reward_binned = []
        dms_signal_binned = []

        # iterate through the sessions
        for session_name in tqdm.tqdm(listdir(behaviour_root)):
            if isdir(pjoin(behaviour_root, session_name)):
                continue

            # get the prpd correlation for this session
            prpd_correlation = prpd_correlation_data[prpd_correlation_data['session'] == session_name.split('.')[0]]

            # loead the behaviour data
            behaviour_path = pjoin(behaviour_root, session_name)
            behaviour_data = pd.read_csv(behaviour_path)
            cue_times = behaviour_data['cue_time']

            # get the index of nan trials in behaviour data as signal only trials
            signal = behaviour_data['trial_response_side'].isna()
            signal_idx = np.where(signal)[0]


            # get the index of unrewarded non_nan trials as movement trials
            mvt = behaviour_data['trial_response_side'].notna()
            mvt = mvt & (behaviour_data['trial_reward'] == 0) 
            mvt_idx = np.where(mvt)[0]
            mvt_left_idx = np.where(mvt & (behaviour_data['trial_response_side'] == -1))[0]
            mvt_right_idx = np.where(mvt & (behaviour_data['trial_response_side'] == 1))[0]

            print('mvt_left_idx', len(mvt_left_idx))
            print('mvt_right_idx', len(mvt_right_idx))

            # get the index of rewarded non_nan trials as reward trials
            reward = behaviour_data['trial_response_side'].notna()
            reward = reward & (behaviour_data['trial_reward'] == 1)
            reward_idx = np.where(reward)[0]
            reward_left_idx = np.where(reward & (behaviour_data['trial_response_side'] == -1))[0]
            reward_right_idx = np.where(reward & (behaviour_data['trial_response_side'] == 1))[0]

            session_name = session_name.split('.')[0]

            # load the pfc spike times
            for pfc_times in glob(pjoin(spike_time_dir, 'sessions', session_name, 'pfc_*')):
                cell_name = basename(pfc_times).split('.')[0]
                if prpd:
                    try:
                        if prpd_correlation[prpd_correlation['cell'] == cell_name].background_firing_p_values.values[0] > 0.05 or prpd_correlation[prpd_correlation['cell'] == cell_name].response_firing_p_values.values[0] > 0.05:
                            continue
                    except:
                        print('Error: ' + cell_name + ' not found in prpd_correlation')
                        continue
                pfc_count += 1
                binned_file = pjoin(spike_time_dir, 'figure_3', f'{session_name}_{cell_name}.npy')

                # load the pfc spike times
                pfc_times = np.load(pfc_times)
                relative_to_pfc = get_relative_spike_times(pfc_times, np.array(cue_times), WINDOW_LEFT, WINDOW_RIGHT)

                # get the spike times of each trial type
                signal_spike_times = [relative_to_pfc[idx] for idx in signal_idx]
                mvt_spike_times = [relative_to_pfc[idx] for idx in mvt_idx]
                reward_spike_times = [relative_to_pfc[idx] for idx in reward_idx]

                mvt_left_spike_times = [relative_to_pfc[idx] for idx in mvt_left_idx]
                mvt_right_spike_times = [relative_to_pfc[idx] for idx in mvt_right_idx]
                reward_left_spike_times = [relative_to_pfc[idx] for idx in reward_left_idx]
                reward_right_spike_times = [relative_to_pfc[idx] for idx in reward_right_idx]

                # concatenate the spike times of each trial type
                signal_spike_times = np.concatenate(signal_spike_times)
                mvt_spike_times = np.concatenate(mvt_spike_times)
                reward_spike_times = np.concatenate(reward_spike_times)

                mvt_left_spike_times = np.concatenate(mvt_left_spike_times)
                mvt_right_spike_times = np.concatenate(mvt_right_spike_times)
                reward_left_spike_times = np.concatenate(reward_left_spike_times)
                reward_right_spike_times = np.concatenate(reward_right_spike_times)


                # bin the spike times using histoplot
                signal_spike_times = np.histogram(signal_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                mvt_spike_times = np.histogram(mvt_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                reward_spike_times = np.histogram(reward_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]

                mvt_left_spike_times = np.histogram(mvt_left_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                mvt_right_spike_times = np.histogram(mvt_right_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                reward_left_spike_times = np.histogram(reward_left_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                reward_right_spike_times = np.histogram(reward_right_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]

                # convert to firing rate
                signal_spike_times = np.divide(signal_spike_times, BIN_SIZE)
                mvt_spike_times = np.divide(mvt_spike_times, BIN_SIZE)
                reward_spike_times = np.divide(reward_spike_times, BIN_SIZE)

                mvt_left_spike_times = np.divide(mvt_left_spike_times, BIN_SIZE)
                mvt_right_spike_times = np.divide(mvt_right_spike_times, BIN_SIZE)
                reward_left_spike_times = np.divide(reward_left_spike_times, BIN_SIZE)
                reward_right_spike_times = np.divide(reward_right_spike_times, BIN_SIZE)

                # normalize by the number of trials
                signal_spike_times = np.divide(signal_spike_times, len(signal_idx))
                mvt_spike_times = np.divide(mvt_spike_times, len(mvt_idx))
                reward_spike_times = np.divide(reward_spike_times, len(reward_idx))

                mvt_left_spike_times = np.divide(mvt_left_spike_times, len(mvt_left_idx))
                mvt_right_spike_times = np.divide(mvt_right_spike_times, len(mvt_right_idx))
                reward_left_spike_times = np.divide(reward_left_spike_times, len(reward_left_idx))
                reward_right_spike_times = np.divide(reward_right_spike_times, len(reward_right_idx))

                if not isfile(binned_file) or reset:
                    total_spike_times = [np.divide(np.histogram(relative_trial_spikes, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0], BIN_SIZE) for relative_trial_spikes in relative_to_pfc]
                    binned_data = [signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times, mvt_left_spike_times, mvt_right_spike_times, reward_left_spike_times, reward_right_spike_times]
                    binned_data_np = np.empty(len(binned_data), dtype=object)
                    binned_data_np[:] = binned_data
                    # save the binned spike times
                    np.save(binned_file, binned_data_np)

                # add the binned spike times to the total
                pfc_signal_binned.append(signal_spike_times)
                pfc_mvt_binned.append(mvt_spike_times)
                pfc_reward_binned.append(reward_spike_times)

            # load the pfc spike times
            for dms_times in glob(pjoin(spike_time_dir, 'sessions', session_name, 'dms_*')):
                cell_name = basename(dms_times).split('.')[0]
                if prpd:
                    try:
                        if prpd_correlation[prpd_correlation['cell'] == cell_name].background_firing_p_values.values[0] > 0.05 or prpd_correlation[prpd_correlation['cell'] == cell_name].response_firing_p_values.values[0] > 0.05:
                            continue
                    except:
                        print('Error: ' + cell_name + ' not found in prpd_correlation')
                        continue
                dms_count += 1
                binned_file = pjoin(spike_time_dir, 'figure_3', f'{session_name}_{cell_name}.npy')

                # load the pfc spike times
                dms_times = np.load(dms_times)
                relative_to_dms = get_relative_spike_times(dms_times, np.array(cue_times), WINDOW_LEFT, WINDOW_RIGHT)

                # get the spike times of each trial type
                signal_spike_times = [relative_to_dms[idx] for idx in signal_idx]
                mvt_spike_times = [relative_to_dms[idx] for idx in mvt_idx]
                reward_spike_times = [relative_to_dms[idx] for idx in reward_idx]

                mvt_left_spike_times = [relative_to_dms[idx] for idx in mvt_left_idx]
                mvt_right_spike_times = [relative_to_dms[idx] for idx in mvt_right_idx]
                reward_left_spike_times = [relative_to_dms[idx] for idx in reward_left_idx]
                reward_right_spike_times = [relative_to_dms[idx] for idx in reward_right_idx]                

                # concatenate the spike times of each trial type
                signal_spike_times = np.concatenate(signal_spike_times)
                mvt_spike_times = np.concatenate(mvt_spike_times)
                reward_spike_times = np.concatenate(reward_spike_times)

                mvt_left_spike_times = np.concatenate(mvt_left_spike_times)
                mvt_right_spike_times = np.concatenate(mvt_right_spike_times)
                reward_left_spike_times = np.concatenate(reward_left_spike_times)
                reward_right_spike_times = np.concatenate(reward_right_spike_times)

                # bin the spike times using histoplot 10ms bins
                signal_spike_times = np.histogram(signal_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                mvt_spike_times = np.histogram(mvt_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                reward_spike_times = np.histogram(reward_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]

                mvt_left_spike_times = np.histogram(mvt_left_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                mvt_right_spike_times = np.histogram(mvt_right_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                reward_left_spike_times = np.histogram(reward_left_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                reward_right_spike_times = np.histogram(reward_right_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]

                # convert to firing rate
                signal_spike_times = np.divide(signal_spike_times, BIN_SIZE)
                mvt_spike_times = np.divide(mvt_spike_times, BIN_SIZE)
                reward_spike_times = np.divide(reward_spike_times, BIN_SIZE)

                mvt_left_spike_times = np.divide(mvt_left_spike_times, BIN_SIZE)
                mvt_right_spike_times = np.divide(mvt_right_spike_times, BIN_SIZE)
                reward_left_spike_times = np.divide(reward_left_spike_times, BIN_SIZE)
                reward_right_spike_times = np.divide(reward_right_spike_times, BIN_SIZE)

                # normalize by the number of trials
                signal_spike_times = np.divide(signal_spike_times, len(signal_idx))
                mvt_spike_times = np.divide(mvt_spike_times, len(mvt_idx))
                reward_spike_times = np.divide(reward_spike_times, len(reward_idx))

                mvt_left_spike_times = np.divide(mvt_left_spike_times, len(mvt_left_idx))
                mvt_right_spike_times = np.divide(mvt_right_spike_times, len(mvt_right_idx))
                reward_left_spike_times = np.divide(reward_left_spike_times, len(reward_left_idx))
                reward_right_spike_times = np.divide(reward_right_spike_times, len(reward_right_idx))

                if not isfile(binned_file) or reset:
                    total_spike_times = [np.divide(np.histogram(relative_trial_spikes, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0], BIN_SIZE) for relative_trial_spikes in relative_to_dms]
                    binned_data = [signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times, mvt_left_spike_times, mvt_right_spike_times, reward_left_spike_times, reward_right_spike_times]
                    binned_data_np = np.empty(len(binned_data), dtype=object)
                    binned_data_np[:] = binned_data
                    # save the binned spike times
                    np.save(binned_file, binned_data_np)

                # add the binned spike times to the total
                dms_signal_binned.append(signal_spike_times)
                dms_mvt_binned.append(mvt_spike_times)
                dms_reward_binned.append(reward_spike_times)

        # take the mean of the binned spike times, recording the standard error
        pfc_signal_binned_mean = np.mean(pfc_signal_binned, axis=0) 
        pfc_signal_binned_err = np.std(pfc_signal_binned, axis=0) / np.sqrt(pfc_count)
        pfc_mvt_binned_mean = np.mean(pfc_mvt_binned, axis=0)
        pfc_mvt_binned_err = np.std(pfc_mvt_binned, axis=0) / np.sqrt(pfc_count)
        pfc_reward_binned_mean = np.mean(pfc_reward_binned, axis=0)
        pfc_reward_binned_err = np.std(pfc_reward_binned, axis=0) / np.sqrt(pfc_count)

        dms_signal_binned_mean = np.mean(dms_signal_binned, axis=0)
        dms_signal_binned_err = np.std(dms_signal_binned, axis=0) / np.sqrt(dms_count)
        dms_mvt_binned_mean = np.mean(dms_mvt_binned, axis=0)
        dms_mvt_binned_err = np.std(dms_mvt_binned, axis=0) / np.sqrt(dms_count)
        dms_reward_binned_mean = np.mean(dms_reward_binned, axis=0)
        dms_reward_binned_err = np.std(dms_reward_binned, axis=0) / np.sqrt(dms_count)

        # save the binned spike times
        np.save(signal_mvt_reward_file, [pfc_signal_binned_mean, pfc_signal_binned_err, pfc_mvt_binned_mean, pfc_mvt_binned_err, pfc_reward_binned_mean, pfc_reward_binned_err, dms_signal_binned_mean, dms_signal_binned_err, dms_mvt_binned_mean, dms_mvt_binned_err, dms_reward_binned_mean, dms_reward_binned_err])
    else:
        # load the binned spike times
        pfc_signal_binned_mean, pfc_signal_binned_err, pfc_mvt_binned_mean, pfc_mvt_binned_err, pfc_reward_binned_mean, pfc_reward_binned_err, dms_signal_binned_mean, dms_signal_binned_err, dms_mvt_binned_mean, dms_mvt_binned_err, dms_reward_binned_mean, dms_reward_binned_err = np.load(signal_mvt_reward_file)

    # store the data for figure 3 panel A(PFC) and B(DMS) as csv files
    figure_3_panel_A_data = pd.DataFrame({'x_values': np.arange(WINDOW_LEFT+BIN_SIZE/2, WINDOW_RIGHT, BIN_SIZE), 'signal_mean': pfc_signal_binned_mean, 'signal_err': pfc_signal_binned_err, 'mvt_mean': pfc_mvt_binned_mean, 'mvt_err': pfc_mvt_binned_err, 'reward_mean': pfc_reward_binned_mean, 'reward_err': pfc_reward_binned_err})
    figure_3_panel_B_data = pd.DataFrame({'x_values': np.arange(WINDOW_LEFT+BIN_SIZE/2, WINDOW_RIGHT, BIN_SIZE), 'signal_mean': dms_signal_binned_mean, 'signal_err': dms_signal_binned_err, 'mvt_mean': dms_mvt_binned_mean, 'mvt_err': dms_mvt_binned_err, 'reward_mean': dms_reward_binned_mean, 'reward_err': dms_reward_binned_err})
    figure_3_panel_A_data['x_values'] = figure_3_panel_A_data['x_values'].round(2)
    figure_3_panel_B_data['x_values'] = figure_3_panel_B_data['x_values'].round(2)
    figure_3_panel_A_data.to_csv(pjoin(figure_3_data_root, 'figure_3_panel_A_data.csv'), index=False)
    figure_3_panel_B_data.to_csv(pjoin(figure_3_data_root, 'figure_3_panel_B_data.csv'), index=False)


    # plot the three binned spike times as line plots with error bars
    fig_pfc, ax_pfc = plt.subplots(1, 1, figsize=(16, 4))
    fig_dms, ax_dms = plt.subplots(1, 1, figsize=(16, 4))

    # set the title
    ax_pfc.set_title('PFC')
    ax_dms.set_title('DMS')

    # plot the pfc trials
    ax_pfc.plot(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), pfc_signal_binned_mean, color='black', label='signal')
    ax_pfc.fill_between(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), pfc_signal_binned_mean - pfc_signal_binned_err, pfc_signal_binned_mean + pfc_signal_binned_err, color='black', alpha=0.3)
    ax_pfc.plot(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), pfc_mvt_binned_mean, color='red', label='signal+mvt')
    ax_pfc.fill_between(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), pfc_mvt_binned_mean - pfc_mvt_binned_err, pfc_mvt_binned_mean + pfc_mvt_binned_err, color='red', alpha=0.3)
    ax_pfc.plot(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), pfc_reward_binned_mean, color='blue', label='signal+mvt+reward')
    ax_pfc.fill_between(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), pfc_reward_binned_mean - pfc_reward_binned_err, pfc_reward_binned_mean + pfc_reward_binned_err, color='blue', alpha=0.3)
    ax_pfc.set_xticks(np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, 0.5))
    ax_pfc.set_xticklabels(np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, 0.5), fontsize=16)
    ax_pfc.set_ylabel('Firing Rate (Hz)', fontsize=16)
    ax_pfc.set_xlabel('Time from Trial Start (s)', fontsize=16)
    ax_pfc.legend(loc='upper right', fontsize=16)

    # plot the dms trials
    ax_dms.plot(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), dms_signal_binned_mean, color='black', label='signal')
    ax_dms.fill_between(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), dms_signal_binned_mean - dms_signal_binned_err, dms_signal_binned_mean + dms_signal_binned_err, color='black', alpha=0.3)
    ax_dms.plot(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), dms_mvt_binned_mean, color='red', label='signal+mvt')
    ax_dms.fill_between(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), dms_mvt_binned_mean - dms_mvt_binned_err, dms_mvt_binned_mean + dms_mvt_binned_err, color='red', alpha=0.3)
    ax_dms.plot(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), dms_reward_binned_mean, color='blue', label='signal+mvt+reward')
    ax_dms.fill_between(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), dms_reward_binned_mean - dms_reward_binned_err, dms_reward_binned_mean + dms_reward_binned_err, color='blue', alpha=0.3)
    ax_dms.set_xticks(np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, 0.5))
    ax_dms.set_xticklabels(np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, 0.5), fontsize=16)
    ax_dms.set_ylabel('Firing Rate (Hz)', fontsize=16)
    ax_dms.set_xlabel('Time from Trial Start (s)', fontsize=16)
    ax_dms.legend(loc='upper right', fontsize=16)


def get_figure_3_panel_cd():
    if not isfile(signal_mvt_reward_file):
        # print error message
        print('Error: ' + signal_mvt_reward_file + ' does not exist. Run figure_3_panel_bc() first.')

    # load the binned spike times
    pfc_signal_binned_mean, pfc_signal_binned_err, pfc_mvt_binned_mean, pfc_mvt_binned_err, pfc_reward_binned_mean, pfc_reward_binned_err, dms_signal_binned_mean, dms_signal_binned_err, dms_mvt_binned_mean, dms_mvt_binned_err, dms_reward_binned_mean, dms_reward_binned_err = np.load(signal_mvt_reward_file)

    pfc_mvt = pfc_mvt_binned_mean - pfc_signal_binned_mean
    pfc_reward = pfc_reward_binned_mean - pfc_mvt_binned_mean

    dms_mvt = dms_mvt_binned_mean - dms_signal_binned_mean
    dms_reward = dms_reward_binned_mean - dms_mvt_binned_mean

    # save the csv files for panel C and D
    figure_3_panel_C_data = pd.DataFrame({'x_values': np.arange(WINDOW_LEFT+BIN_SIZE/2, WINDOW_RIGHT, BIN_SIZE),'signal': pfc_signal_binned_mean, 'mvt': pfc_mvt, 'reward': pfc_reward})
    figure_3_panel_D_data = pd.DataFrame({'x_values': np.arange(WINDOW_LEFT+BIN_SIZE/2, WINDOW_RIGHT, BIN_SIZE), 'signal': dms_signal_binned_mean, 'mvt': dms_mvt, 'reward': dms_reward})
    figure_3_panel_C_data['x_values'] = figure_3_panel_C_data['x_values'].round(2)
    figure_3_panel_D_data['x_values'] = figure_3_panel_D_data['x_values'].round(2)
    figure_3_panel_C_data.to_csv(pjoin(figure_3_data_root, 'figure_3_panel_C_data.csv'), index=False)
    figure_3_panel_D_data.to_csv(pjoin(figure_3_data_root, 'figure_3_panel_D_data.csv'), index=False)

    # plot the signal, mvt, and reward components
    fig_pfc, ax_pfc = plt.subplots(1, 3, figsize=(12, 3.5))
    fig_dms, ax_dms = plt.subplots(1, 3, figsize=(12, 3.5))

    # plot the pfc trials
    ax_pfc[0].plot(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), pfc_signal_binned_mean, color='black', label='signal')
    ax_pfc[1].plot(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), pfc_mvt, color='red', label='mvt')
    ax_pfc[2].plot(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), pfc_reward, color='blue', label='reward')

    # plot the dms trials
    ax_dms[0].plot(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), dms_signal_binned_mean, color='black', label='signal')
    ax_dms[1].plot(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), dms_mvt, color='red', label='mvt')
    ax_dms[2].plot(np.arange(WINDOW_LEFT, WINDOW_RIGHT, BIN_SIZE), dms_reward, color='blue', label='reward')

    fig_pfc.suptitle('PFC', fontsize=16)
    fig_dms.suptitle('DMS', fontsize=16)


def get_figure_3_panel_ef():
    if not isfile(signal_mvt_reward_file):
        # print error message
        print('Error: ' + signal_mvt_reward_file + ' does not exist. Run figure_3_panel_bc() first.')

    # load the binned spike times
    pfc_signal_binned_mean, pfc_signal_binned_err, pfc_mvt_binned_mean, pfc_mvt_binned_err, pfc_reward_binned_mean, pfc_reward_binned_err, dms_signal_binned_mean, dms_signal_binned_err, dms_mvt_binned_mean, dms_mvt_binned_err, dms_reward_binned_mean, dms_reward_binned_err = np.load(signal_mvt_reward_file)

    pfc_mvt = pfc_mvt_binned_mean - pfc_signal_binned_mean
    pfc_reward = pfc_reward_binned_mean - pfc_mvt_binned_mean

    dms_mvt = dms_mvt_binned_mean - dms_signal_binned_mean
    dms_reward = dms_reward_binned_mean - dms_mvt_binned_mean

    # get the z score of the signal, mvt, and reward components
    pfc_signal_binned_mean = (pfc_signal_binned_mean - np.mean(pfc_signal_binned_mean)) / np.std(pfc_signal_binned_mean)
    pfc_mvt = (pfc_mvt - np.mean(pfc_mvt)) / np.std(pfc_mvt)
    pfc_reward = (pfc_reward - np.mean(pfc_reward)) / np.std(pfc_reward)

    dms_signal_binned_mean = (dms_signal_binned_mean - np.mean(dms_signal_binned_mean)) / np.std(dms_signal_binned_mean)
    dms_mvt = (dms_mvt - np.mean(dms_mvt)) / np.std(dms_mvt)
    dms_reward = (dms_reward - np.mean(dms_reward)) / np.std(dms_reward)


    # store the regressors in X
    X_pfc= np.column_stack((pfc_signal_binned_mean, pfc_mvt, pfc_reward))
    X_pfc = np.column_stack((np.ones(len(pfc_signal_binned_mean)), X_pfc))

    X_dms = np.column_stack((dms_signal_binned_mean, dms_mvt, dms_reward))
    X_dms = np.column_stack((np.ones(len(dms_signal_binned_mean)), X_dms))

    pfc_signal_coeffs = []
    pfc_mvt_coeffs = []
    pfc_reward_coeffs = []
    dms_signal_coeffs = []
    dms_mvt_coeffs = []
    dms_reward_coeffs = []

    # load the binned pfc spike times
    for pfc_file in glob(pjoin(spike_time_dir, 'figure_3', f'*pfc*')):
        signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times, *_ = np.load(pfc_file, allow_pickle=True)

        if not np.max(signal_spike_times) == 0:
            signal_spike_times = signal_spike_times / np.max(signal_spike_times)
            pfc_signal_coeffs.append(np.linalg.lstsq(X_pfc, signal_spike_times, rcond=None)[0][1:])
        if not np.max(mvt_spike_times) == 0:
            mvt_spike_times = mvt_spike_times / np.max(mvt_spike_times)
            pfc_mvt_coeffs.append(np.linalg.lstsq(X_pfc, mvt_spike_times, rcond=None)[0][1:])
        if not np.max(reward_spike_times) == 0:
            reward_spike_times = reward_spike_times / np.max(reward_spike_times)
            pfc_reward_coeffs.append(np.linalg.lstsq(X_pfc, reward_spike_times, rcond=None)[0][1:])

    # load the binned dms spike times
    for dms_file in glob(pjoin(spike_time_dir, 'figure_3', f'*dms*')):
        signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times, *_ = np.load(dms_file, allow_pickle=True)        

        if not np.max(signal_spike_times) == 0:
            signal_spike_times = signal_spike_times / np.max(signal_spike_times)
            dms_signal_coeffs.append(np.linalg.lstsq(X_dms, signal_spike_times, rcond=None)[0][1:])
        if not np.max(mvt_spike_times) == 0:
            mvt_spike_times = mvt_spike_times / np.max(mvt_spike_times)
            dms_mvt_coeffs.append(np.linalg.lstsq(X_dms, mvt_spike_times, rcond=None)[0][1:])
        if not np.max(reward_spike_times) == 0:
            reward_spike_times = reward_spike_times / np.max(reward_spike_times)
            dms_reward_coeffs.append(np.linalg.lstsq(X_dms, reward_spike_times, rcond=None)[0][1:])

    # plot the signal, mvt, and reward components
    fig_pfc, ax_pfc = plt.subplots(1, 3, figsize=(16, 5))
    fig_dms, ax_dms = plt.subplots(1, 3, figsize=(16, 5))

    # plot the pfc coeffs for the signal trials as bar plots with error bars
    # with each bar representing a coefficient
    signal_signal_coeffs = np.array(pfc_signal_coeffs)[:, 0]
    signal_signal_coeffs_mean = np.mean(signal_signal_coeffs)
    signal_signal_coeffs_err = np.std(signal_signal_coeffs) / np.sqrt(len(signal_signal_coeffs))
    signal_mvt_coeffs = np.array(pfc_signal_coeffs)[:, 1]
    signal_mvt_coeffs_mean = np.mean(signal_mvt_coeffs)
    signal_mvt_coeffs_err = np.std(signal_mvt_coeffs) / np.sqrt(len(signal_mvt_coeffs))
    signal_reward_coeffs = np.array(pfc_signal_coeffs)[:, 2]
    signal_reward_coeffs_mean = np.mean(signal_reward_coeffs)
    signal_reward_coeffs_err = np.std(signal_reward_coeffs) / np.sqrt(len(signal_reward_coeffs))

    mvt_signal_coeffs = np.array(pfc_mvt_coeffs)[:, 0]
    mvt_signal_coeffs_mean = np.mean(mvt_signal_coeffs)
    mvt_signal_coeffs_err = np.std(mvt_signal_coeffs) / np.sqrt(len(mvt_signal_coeffs))
    mvt_mvt_coeffs = np.array(pfc_mvt_coeffs)[:, 1]
    mvt_mvt_coeffs_mean = np.mean(mvt_mvt_coeffs)
    mvt_mvt_coeffs_err = np.std(mvt_mvt_coeffs) / np.sqrt(len(mvt_mvt_coeffs))
    mvt_reward_coeffs = np.array(pfc_mvt_coeffs)[:, 2]
    mvt_reward_coeffs_mean = np.mean(mvt_reward_coeffs)
    mvt_reward_coeffs_err = np.std(mvt_reward_coeffs) / np.sqrt(len(mvt_reward_coeffs))

    reward_signal_coeffs = np.array(pfc_reward_coeffs)[:, 0]
    reward_signal_coeffs_mean = np.mean(reward_signal_coeffs)
    # standard error of the mean
    reward_signal_coeffs_err = np.std(reward_signal_coeffs) / np.sqrt(len(reward_signal_coeffs))
    reward_mvt_coeffs = np.array(pfc_reward_coeffs)[:, 1]
    reward_mvt_coeffs_mean = np.mean(reward_mvt_coeffs)
    reward_mvt_coeffs_err = np.std(reward_mvt_coeffs) / np.sqrt(len(reward_mvt_coeffs))
    reward_reward_coeffs = np.array(pfc_reward_coeffs)[:, 2]
    reward_reward_coeffs_mean = np.mean(reward_reward_coeffs)
    reward_reward_coeffs_err = np.std(reward_reward_coeffs)  /  np.sqrt(len(reward_reward_coeffs))

    figure_3_panel_E_data = pd.DataFrame({'coefficient_type': ['signal', 'mvt', 'reward'], 'signal_trials': [signal_signal_coeffs_mean, signal_mvt_coeffs_mean, signal_reward_coeffs_mean], 'signal_trials_err': [signal_signal_coeffs_err, signal_mvt_coeffs_err, signal_reward_coeffs_err], 'mvt_trials': [mvt_signal_coeffs_mean, mvt_mvt_coeffs_mean, mvt_reward_coeffs_mean], 'mvt_trials_err': [mvt_signal_coeffs_err, mvt_mvt_coeffs_err, mvt_reward_coeffs_err], 'reward_trials': [reward_signal_coeffs_mean, reward_mvt_coeffs_mean, reward_reward_coeffs_mean], 'reward_trials_err': [reward_signal_coeffs_err, reward_mvt_coeffs_err, reward_reward_coeffs_err]})
    figure_3_panel_E_data.to_csv(pjoin(figure_3_data_root, 'figure_3_panel_E_data.csv'), index=False)

    ax_pfc[0].bar([0, 1, 2], [signal_signal_coeffs_mean, mvt_signal_coeffs_mean, reward_signal_coeffs_mean], yerr=[signal_signal_coeffs_err, mvt_signal_coeffs_err, reward_signal_coeffs_err], color='k')
    ax_pfc[0].set_xticks([0, 1, 2])
    ax_pfc[0].set_xticklabels(['S', 'SM', 'SMR'])
    ax_pfc[0].set_ylabel('signal coeffs')

    print('PFC')

    print('signal coeffs')

    # do the t test for signal vs mvt and signal vs reward and reward vs mvt
    t_stat, p_val = stats.ttest_ind(signal_signal_coeffs, mvt_signal_coeffs)
    print('signal vs mvt t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(signal_signal_coeffs, reward_signal_coeffs)
    print('signal vs reward t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(mvt_signal_coeffs, reward_signal_coeffs)
    print('mvt vs reward t test: t = ' + str(t_stat) + ', p = ' + str(p_val))


    ax_pfc[1].bar([0, 1, 2], [signal_mvt_coeffs_mean, mvt_mvt_coeffs_mean, reward_mvt_coeffs_mean], yerr=[signal_mvt_coeffs_err, mvt_mvt_coeffs_err, reward_mvt_coeffs_err], color='k')
    ax_pfc[1].set_xticks([0, 1, 2])
    ax_pfc[1].set_xticklabels(['S', 'SM', 'SMR'])
    ax_pfc[1].set_ylabel('mvt coeffs')

    print('mvt coeffs')

    # do the t test for signal vs mvt and signal vs reward and reward vs mvt
    t_stat, p_val = stats.ttest_ind(signal_mvt_coeffs, mvt_mvt_coeffs)
    print('signal vs mvt t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(signal_mvt_coeffs, reward_mvt_coeffs)
    print('signal vs reward t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(mvt_mvt_coeffs, reward_mvt_coeffs)
    print('mvt vs reward t test: t = ' + str(t_stat) + ', p = ' + str(p_val))

    ax_pfc[2].bar([0, 1, 2], [signal_reward_coeffs_mean, mvt_reward_coeffs_mean, reward_reward_coeffs_mean], yerr=[signal_reward_coeffs_err, mvt_reward_coeffs_err, reward_reward_coeffs_err], color='k')
    ax_pfc[2].set_xticks([0, 1, 2])
    ax_pfc[2].set_xticklabels(['S', 'SM', 'SMR'])
    ax_pfc[2].set_ylabel('reward coeffs')

    print('reward coeffs')

    # do the t test for signal vs mvt and signal vs reward and reward vs mvt
    t_stat, p_val = stats.ttest_ind(signal_reward_coeffs, mvt_reward_coeffs)
    print('signal vs mvt t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(signal_reward_coeffs, reward_reward_coeffs)
    print('signal vs reward t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(mvt_reward_coeffs, reward_reward_coeffs)
    print('mvt vs reward t test: t = ' + str(t_stat) + ', p = ' + str(p_val))

    fig_pfc.suptitle('PFC')

    # Similarly for the dms coeffs
    signal_signal_coeffs = np.array(dms_signal_coeffs)[:, 0]
    signal_signal_coeffs_mean = np.mean(signal_signal_coeffs)
    signal_signal_coeffs_err = np.std(signal_signal_coeffs) / np.sqrt(len(signal_signal_coeffs))
    signal_mvt_coeffs = np.array(dms_signal_coeffs)[:, 1]
    signal_mvt_coeffs_mean = np.mean(signal_mvt_coeffs)
    signal_mvt_coeffs_err = np.std(signal_mvt_coeffs) / np.sqrt(len(signal_mvt_coeffs))
    signal_reward_coeffs = np.array(dms_signal_coeffs)[:, 2]
    signal_reward_coeffs_mean = np.mean(signal_reward_coeffs)
    signal_reward_coeffs_err = np.std(signal_reward_coeffs) / np.sqrt(len(signal_reward_coeffs))

    mvt_signal_coeffs = np.array(dms_mvt_coeffs)[:, 0]
    mvt_signal_coeffs_mean = np.mean(mvt_signal_coeffs)
    mvt_signal_coeffs_err = np.std(mvt_signal_coeffs) / np.sqrt(len(mvt_signal_coeffs))
    mvt_mvt_coeffs = np.array(dms_mvt_coeffs)[:, 1]
    mvt_mvt_coeffs_mean = np.mean(mvt_mvt_coeffs)
    mvt_mvt_coeffs_err = np.std(mvt_mvt_coeffs) /  np.sqrt(len(mvt_mvt_coeffs))
    mvt_reward_coeffs = np.array(dms_mvt_coeffs)[:, 2]
    mvt_reward_coeffs_mean = np.mean(mvt_reward_coeffs)
    mvt_reward_coeffs_err = np.std(mvt_reward_coeffs) / np.sqrt(len(mvt_reward_coeffs))

    reward_signal_coeffs = np.array(dms_reward_coeffs)[:, 0]
    reward_signal_coeffs_mean = np.mean(reward_signal_coeffs)
    reward_signal_coeffs_err = np.std(reward_signal_coeffs) / np.sqrt(len(reward_signal_coeffs))
    reward_mvt_coeffs = np.array(dms_reward_coeffs)[:, 1]
    reward_mvt_coeffs_mean = np.mean(reward_mvt_coeffs)
    reward_mvt_coeffs_err = np.std(reward_mvt_coeffs) / np.sqrt(len(reward_mvt_coeffs))
    reward_reward_coeffs = np.array(dms_reward_coeffs)[:, 2]
    reward_reward_coeffs_mean = np.mean(reward_reward_coeffs)
    reward_reward_coeffs_err = np.std(reward_reward_coeffs) / np.sqrt(len(reward_reward_coeffs))

    figure_3_panel_F_data = pd.DataFrame({'coefficient_type': ['signal', 'mvt', 'reward'], 'signal_trials': [signal_signal_coeffs_mean, signal_mvt_coeffs_mean, signal_reward_coeffs_mean], 'signal_trials_err': [signal_signal_coeffs_err, signal_mvt_coeffs_err, signal_reward_coeffs_err], 'mvt_trials': [mvt_signal_coeffs_mean, mvt_mvt_coeffs_mean, mvt_reward_coeffs_mean], 'mvt_trials_err': [mvt_signal_coeffs_err, mvt_mvt_coeffs_err, mvt_reward_coeffs_err], 'reward_trials': [reward_signal_coeffs_mean, reward_mvt_coeffs_mean, reward_reward_coeffs_mean], 'reward_trials_err': [reward_signal_coeffs_err, reward_mvt_coeffs_err, reward_reward_coeffs_err]})
    figure_3_panel_F_data.to_csv(pjoin(figure_3_data_root, 'figure_3_panel_F_data.csv'), index=False)

    ax_dms[0].bar([0, 1, 2], [signal_signal_coeffs_mean, mvt_signal_coeffs_mean, reward_signal_coeffs_mean], yerr=[signal_signal_coeffs_err, mvt_signal_coeffs_err, reward_signal_coeffs_err], color='k')
    ax_dms[0].set_xticks([0, 1, 2])
    ax_dms[0].set_xticklabels(['S', 'SM', 'SMR'])
    ax_dms[0].set_ylabel('signal coeffs')

    print('DMS')

    print('signal coeffs')

    # do the t test for signal vs mvt and signal vs reward and reward vs mvt
    t_stat, p_val = stats.ttest_ind(signal_signal_coeffs, mvt_signal_coeffs)
    print('signal vs mvt t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(signal_signal_coeffs, reward_signal_coeffs)
    print('signal vs reward t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(mvt_signal_coeffs, reward_signal_coeffs)
    print('mvt vs reward t test: t = ' + str(t_stat) + ', p = ' + str(p_val))


    ax_dms[1].bar([0, 1, 2], [signal_mvt_coeffs_mean, mvt_mvt_coeffs_mean, reward_mvt_coeffs_mean], yerr=[signal_mvt_coeffs_err, mvt_mvt_coeffs_err, reward_mvt_coeffs_err], color='k')
    ax_dms[1].set_xticks([0, 1, 2])
    ax_dms[1].set_xticklabels(['S', 'SM', 'SMR'])
    ax_dms[1].set_ylabel('mvt coeffs')

    print('mvt coeffs')

    # do the t test for signal vs mvt and signal vs reward and reward vs mvt
    t_stat, p_val = stats.ttest_ind(signal_mvt_coeffs, mvt_mvt_coeffs)
    print('signal vs mvt t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(signal_mvt_coeffs, reward_mvt_coeffs)
    print('signal vs reward t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(mvt_mvt_coeffs, reward_mvt_coeffs)
    print('mvt vs reward t test: t = ' + str(t_stat) + ', p = ' + str(p_val))

    ax_dms[2].bar([0, 1, 2], [signal_reward_coeffs_mean, mvt_reward_coeffs_mean, reward_reward_coeffs_mean], yerr=[signal_reward_coeffs_err, mvt_reward_coeffs_err, reward_reward_coeffs_err], color='k')
    ax_dms[2].set_xticks([0, 1, 2])
    ax_dms[2].set_xticklabels(['S', 'SM', 'SMR'])
    ax_dms[2].set_ylabel('reward coeffs')

    print('reward coeffs')

    # do the t test for signal vs mvt and signal vs reward and reward vs mvt
    t_stat, p_val = stats.ttest_ind(signal_reward_coeffs, mvt_reward_coeffs)
    print('signal vs mvt t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(signal_reward_coeffs, reward_reward_coeffs)
    print('signal vs reward t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(mvt_reward_coeffs, reward_reward_coeffs)
    print('mvt vs reward t test: t = ' + str(t_stat) + ', p = ' + str(p_val))

    fig_dms.suptitle('DMS')

    # make sure the y label does not overlap with figures
    fig_pfc.tight_layout()
    fig_dms.tight_layout()

def get_figure_3_panel_ef_direction_modulation(prpd=False):
    if not isfile(signal_mvt_reward_file):
        # print error message
        print('Error: ' + signal_mvt_reward_file + ' does not exist. Run figure_3_panel_bc() first.')

    # load the binned spike times
    pfc_signal_binned_mean, pfc_signal_binned_err, pfc_mvt_binned_mean, pfc_mvt_binned_err, pfc_reward_binned_mean, pfc_reward_binned_err, dms_signal_binned_mean, dms_signal_binned_err, dms_mvt_binned_mean, dms_mvt_binned_err, dms_reward_binned_mean, dms_reward_binned_err = np.load(signal_mvt_reward_file)

    pfc_mvt = pfc_mvt_binned_mean - pfc_signal_binned_mean
    pfc_reward = pfc_reward_binned_mean - pfc_mvt_binned_mean

    dms_mvt = dms_mvt_binned_mean - dms_signal_binned_mean
    dms_reward = dms_reward_binned_mean - dms_mvt_binned_mean

    # get the z score of the signal, mvt, and reward components
    pfc_signal_binned_mean = (pfc_signal_binned_mean - np.mean(pfc_signal_binned_mean)) / np.std(pfc_signal_binned_mean)
    pfc_mvt = (pfc_mvt - np.mean(pfc_mvt)) / np.std(pfc_mvt)
    pfc_reward = (pfc_reward - np.mean(pfc_reward)) / np.std(pfc_reward)

    dms_signal_binned_mean = (dms_signal_binned_mean - np.mean(dms_signal_binned_mean)) / np.std(dms_signal_binned_mean)
    dms_mvt = (dms_mvt - np.mean(dms_mvt)) / np.std(dms_mvt)
    dms_reward = (dms_reward - np.mean(dms_reward)) / np.std(dms_reward)


    # store the regressors in X
    X_pfc= np.column_stack((pfc_signal_binned_mean, pfc_mvt, pfc_reward))
    X_pfc = np.column_stack((np.ones(len(pfc_signal_binned_mean)), X_pfc))

    X_dms = np.column_stack((dms_signal_binned_mean, dms_mvt, dms_reward))
    X_dms = np.column_stack((np.ones(len(dms_signal_binned_mean)), X_dms))

    pfc_signal_coeffs_left = []
    pfc_signal_coeffs_right = []
    pfc_mvt_coeffs_left = []
    pfc_mvt_coeffs_right = []
    pfc_reward_coeffs_left = []
    pfc_reward_coeffs_right = []

    dms_signal_coeffs_left = []
    dms_signal_coeffs_right = []
    dms_mvt_coeffs_left = []
    dms_mvt_coeffs_right = []
    dms_reward_coeffs_left = []
    dms_reward_coeffs_right = []


    # load the binned pfc spike times
    for pfc_file in glob(pjoin(spike_time_dir, 'figure_3', f'*pfc*')):
        signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times, mvt_spike_times_left, mvt_spike_times_right, reward_spike_times_left, reward_spike_times_right = np.load(pfc_file, allow_pickle=True)

        # if not np.max(signal_spike_times) == 0:
        #     signal_spike_times_left = signal_spike_times_left / np.max(signal_spike_times_left)
        #     signal_spike_times_right = signal_spike_times_right / np.max(signal_spike_times_right)
        #     pfc_signal_coeffs_left.append(np.linalg.lstsq(X_pfc, signal_spike_times_left, rcond=None)[0][1:])
        #     pfc_signal_coeffs_right.append(np.linalg.lstsq(X_pfc, signal_spike_times_right, rcond=None)[0][1:])
        if not np.max(mvt_spike_times) == 0:
            mvt_spike_times_left = mvt_spike_times_left / np.max(mvt_spike_times_left)
            mvt_spike_times_right = mvt_spike_times_right / np.max(mvt_spike_times_right)
            pfc_mvt_coeffs_left.append(np.linalg.lstsq(X_pfc, mvt_spike_times_left, rcond=None)[0][1:])
            pfc_mvt_coeffs_right.append(np.linalg.lstsq(X_pfc, mvt_spike_times_right, rcond=None)[0][1:])

        if not np.max(reward_spike_times) == 0:
            reward_spike_times_left = reward_spike_times_left / np.max(reward_spike_times_left)
            reward_spike_times_right = reward_spike_times_right / np.max(reward_spike_times_right)
            pfc_reward_coeffs_left.append(np.linalg.lstsq(X_pfc, reward_spike_times_left, rcond=None)[0][1:])
            pfc_reward_coeffs_right.append(np.linalg.lstsq(X_pfc, reward_spike_times_right, rcond=None)[0][1:])


    # load the binned dms spike times
    for dms_file in glob(pjoin(spike_time_dir, 'figure_3', f'*dms*')):
        signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times, mvt_spike_times_left, mvt_spike_times_right, reward_spike_times_left, reward_spike_times_right  = np.load(dms_file, allow_pickle=True)        

        # if not np.max(signal_spike_times) == 0:
        #     signal_spike_times_left = signal_spike_times_left / np.max(signal_spike_times_left)
        #     signal_spike_times_right = signal_spike_times_right / np.max(signal_spike_times_right)
        #     dms_signal_coeffs_left.append(np.linalg.lstsq(X_dms, signal_spike_times_left, rcond=None)[0][1:])
        #     dms_signal_coeffs_right.append(np.linalg.lstsq(X_dms, signal_spike_times_right, rcond=None)[0][1:])
        if not np.max(mvt_spike_times) == 0:
            mvt_spike_times_left = mvt_spike_times_left / np.max(mvt_spike_times_left)
            mvt_spike_times_right = mvt_spike_times_right / np.max(mvt_spike_times_right)
            dms_mvt_coeffs_left.append(np.linalg.lstsq(X_dms, mvt_spike_times_left, rcond=None)[0][1:])
            dms_mvt_coeffs_right.append(np.linalg.lstsq(X_dms, mvt_spike_times_right, rcond=None)[0][1:])
        if not np.max(reward_spike_times) == 0:
            reward_spike_times_left = reward_spike_times_left / np.max(reward_spike_times_left)
            reward_spike_times_right = reward_spike_times_right / np.max(reward_spike_times_right)
            dms_reward_coeffs_left.append(np.linalg.lstsq(X_dms, reward_spike_times_left, rcond=None)[0][1:])
            dms_reward_coeffs_right.append(np.linalg.lstsq(X_dms, reward_spike_times_right, rcond=None)[0][1:])

    # plot the signal, mvt, and reward components
    fig_pfc, ax_pfc = plt.subplots(1, 2, figsize=(16, 5))
    fig_dms, ax_dms = plt.subplots(1, 2, figsize=(16, 5))

    # plot the pfc coeffs for the signal trials as bar plots with error bars
    # with each bar representing a coefficient
    # signal_signal_coeffs_left = np.array(pfc_signal_coeffs_left)[:, 0]
    # signal_signal_coeffs_left_mean = np.mean(signal_signal_coeffs_left)
    # signal_signal_coeffs_left_err = np.std(signal_signal_coeffs_left) / np.sqrt(len(signal_signal_coeffs_left))
    # signal_signal_coeffs_right = np.array(pfc_signal_coeffs_right)[:, 0]
    # signal_signal_coeffs_right_mean = np.mean(signal_signal_coeffs_right)
    # signal_signal_coeffs_right_err = np.std(signal_signal_coeffs_right) / np.sqrt(len(signal_signal_coeffs_right))

    signal_mvt_coeffs_left = np.array(pfc_mvt_coeffs_left)[:, 0]
    # ignore nan values
    signal_mvt_coeffs_left = signal_mvt_coeffs_left[~np.isnan(signal_mvt_coeffs_left)]
    signal_mvt_coeffs_left_mean = np.mean(signal_mvt_coeffs_left)
    signal_mvt_coeffs_left_err = np.std(signal_mvt_coeffs_left) / np.sqrt(len(signal_mvt_coeffs_left))

    signal_mvt_coeffs_right = np.array(pfc_mvt_coeffs_right)[:, 0]
    # ignore nan values
    signal_mvt_coeffs_right = signal_mvt_coeffs_right[~np.isnan(signal_mvt_coeffs_right)]
    signal_mvt_coeffs_right_mean = np.mean(signal_mvt_coeffs_right)
    signal_mvt_coeffs_right_err = np.std(signal_mvt_coeffs_right) / np.sqrt(len(signal_mvt_coeffs_right))

    signal_reward_coeffs_left = np.array(pfc_reward_coeffs_left)[:, 0]
    # ignore nan values
    signal_reward_coeffs_left = signal_reward_coeffs_left[~np.isnan(signal_reward_coeffs_left)]
    signal_reward_coeffs_left_mean = np.mean(signal_reward_coeffs_left)
    signal_reward_coeffs_left_err = np.std(signal_reward_coeffs_left) / np.sqrt(len(signal_reward_coeffs_left))

    signal_reward_coeffs_right = np.array(pfc_reward_coeffs_right)[:, 0]
    # ignore nan values
    signal_reward_coeffs_right = signal_reward_coeffs_right[~np.isnan(signal_reward_coeffs_right)]
    signal_reward_coeffs_right_mean = np.mean(signal_reward_coeffs_right)
    signal_reward_coeffs_right_err = np.std(signal_reward_coeffs_right) / np.sqrt(len(signal_reward_coeffs_right))

    # mvt_signal_coeffs_left = np.array(pfc_signal_coeffs_left)[:, 1]
    # mvt_signal_coeffs_left_mean = np.mean(mvt_signal_coeffs_left)
    # mvt_signal_coeffs_left_err = np.std(mvt_signal_coeffs_left) / np.sqrt(len(mvt_signal_coeffs_left))
    # mvt_signal_coeffs_right = np.array(pfc_signal_coeffs_right)[:, 1]
    # mvt_signal_coeffs_right_mean = np.mean(mvt_signal_coeffs_right)
    # mvt_signal_coeffs_right_err = np.std(mvt_signal_coeffs_right) / np.sqrt(len(mvt_signal_coeffs_right))

    mvt_mvt_coeffs_left = np.array(pfc_mvt_coeffs_left)[:, 1]
    # ignore nan values
    mvt_mvt_coeffs_left = mvt_mvt_coeffs_left[~np.isnan(mvt_mvt_coeffs_left)]
    mvt_mvt_coeffs_left_mean = np.mean(mvt_mvt_coeffs_left)
    mvt_mvt_coeffs_left_err = np.std(mvt_mvt_coeffs_left) / np.sqrt(len(mvt_mvt_coeffs_left))

    mvt_mvt_coeffs_right = np.array(pfc_mvt_coeffs_right)[:, 1]
    # ignore nan values
    mvt_mvt_coeffs_right = mvt_mvt_coeffs_right[~np.isnan(mvt_mvt_coeffs_right)]
    mvt_mvt_coeffs_right_mean = np.mean(mvt_mvt_coeffs_right)
    mvt_mvt_coeffs_right_err = np.std(mvt_mvt_coeffs_right) / np.sqrt(len(mvt_mvt_coeffs_right))

    mvt_reward_coeffs_left = np.array(pfc_reward_coeffs_left)[:, 1]
    # ignore nan values
    mvt_reward_coeffs_left = mvt_reward_coeffs_left[~np.isnan(mvt_reward_coeffs_left)]
    mvt_reward_coeffs_left_mean = np.mean(mvt_reward_coeffs_left)
    mvt_reward_coeffs_left_err = np.std(mvt_reward_coeffs_left) / np.sqrt(len(mvt_reward_coeffs_left))

    mvt_reward_coeffs_right = np.array(pfc_reward_coeffs_right)[:, 1]
    # ignore nan values
    mvt_reward_coeffs_right = mvt_reward_coeffs_right[~np.isnan(mvt_reward_coeffs_right)]
    mvt_reward_coeffs_right_mean = np.mean(mvt_reward_coeffs_right)
    mvt_reward_coeffs_right_err = np.std(mvt_reward_coeffs_right) / np.sqrt(len(mvt_reward_coeffs_right))

    # reward_signal_coeffs_left = np.array(pfc_signal_coeffs_left)[:, 2]
    # reward_signal_coeffs_left_mean = np.mean(reward_signal_coeffs_left)
    # reward_signal_coeffs_left_err = np.std(reward_signal_coeffs_left) / np.sqrt(len(reward_signal_coeffs_left))
    # reward_signal_coeffs_right = np.array(pfc_signal_coeffs_right)[:, 2]
    # reward_signal_coeffs_right_mean = np.mean(reward_signal_coeffs_right)
    # reward_signal_coeffs_right_err = np.std(reward_signal_coeffs_right) / np.sqrt(len(reward_signal_coeffs_right))

    reward_mvt_coeffs_left = np.array(pfc_mvt_coeffs_left)[:, 2]
    # ignore nan values
    reward_mvt_coeffs_left = reward_mvt_coeffs_left[~np.isnan(reward_mvt_coeffs_left)]
    reward_mvt_coeffs_left_mean = np.mean(reward_mvt_coeffs_left)
    reward_mvt_coeffs_left_err = np.std(reward_mvt_coeffs_left) / np.sqrt(len(reward_mvt_coeffs_left))

    reward_mvt_coeffs_right = np.array(pfc_mvt_coeffs_right)[:, 2]
    # ignore nan values
    reward_mvt_coeffs_right = reward_mvt_coeffs_right[~np.isnan(reward_mvt_coeffs_right)]
    reward_mvt_coeffs_right_mean = np.mean(reward_mvt_coeffs_right)
    reward_mvt_coeffs_right_err = np.std(reward_mvt_coeffs_right) / np.sqrt(len(reward_mvt_coeffs_right))

    reward_reward_coeffs_left = np.array(pfc_reward_coeffs_left)[:, 2]
    # ignore nan values
    reward_reward_coeffs_left = reward_reward_coeffs_left[~np.isnan(reward_reward_coeffs_left)]
    reward_reward_coeffs_left_mean = np.mean(reward_reward_coeffs_left)
    reward_reward_coeffs_left_err = np.std(reward_reward_coeffs_left) / np.sqrt(len(reward_reward_coeffs_left))

    reward_reward_coeffs_right = np.array(pfc_reward_coeffs_right)[:, 2]
    # ignore nan values
    reward_reward_coeffs_right = reward_reward_coeffs_right[~np.isnan(reward_reward_coeffs_right)]
    reward_reward_coeffs_right_mean = np.mean(reward_reward_coeffs_right)
    reward_reward_coeffs_right_err = np.std(reward_reward_coeffs_right) / np.sqrt(len(reward_reward_coeffs_right))


    # ax_pfc[0].bar([0, 0.2, 1, 1.2, 2, 2.2], [signal_signal_coeffs_left_mean, signal_signal_coeffs_right_mean, mvt_signal_coeffs_left_mean, mvt_signal_coeffs_right_mean, reward_signal_coeffs_left_mean, reward_signal_coeffs_right_mean], yerr=[signal_signal_coeffs_left_err, signal_signal_coeffs_right_err, mvt_signal_coeffs_left_err, mvt_signal_coeffs_right_err, reward_signal_coeffs_left_err, reward_signal_coeffs_right_err], color='k', width=0.1)
    # ax_pfc[0].set_xticks([0, 1, 2])
    # ax_pfc[0].set_xticklabels(['S', 'SM', 'SMR'])
    # ax_pfc[0].set_ylabel('signal coeffs')

    print('PFC')

    # print('signal coeffs')

    # # do the t test for left vs right
    # t_stat, p_val = stats.ttest_ind(signal_signal_coeffs_left, signal_signal_coeffs_right)
    # print('signal left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    # t_stat, p_val = stats.ttest_ind(mvt_signal_coeffs_left, mvt_signal_coeffs_right)
    # print('mvt left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    # t_stat, p_val = stats.ttest_ind(reward_signal_coeffs_left, reward_signal_coeffs_right)
    # print('reward left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))


    ax_pfc[0].bar([0, 0.2, 1, 1.2, 2, 2.2], [signal_mvt_coeffs_left_mean, signal_mvt_coeffs_right_mean, mvt_mvt_coeffs_left_mean, mvt_mvt_coeffs_right_mean, reward_mvt_coeffs_left_mean, reward_mvt_coeffs_right_mean], yerr=[signal_mvt_coeffs_left_err, signal_mvt_coeffs_right_err, mvt_mvt_coeffs_left_err, mvt_mvt_coeffs_right_err, reward_mvt_coeffs_left_err, reward_mvt_coeffs_right_err], color='k', width=0.1)
    ax_pfc[0].set_xticks([0, 1, 2])
    ax_pfc[0].set_xticklabels(['S', 'SM', 'SMR'])
    ax_pfc[0].set_ylabel('mvt coeffs')

    print('mvt coeffs')

    # do the t test for left vs right
    t_stat, p_val = stats.ttest_ind(signal_mvt_coeffs_left, signal_mvt_coeffs_right)
    print('signal left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(mvt_mvt_coeffs_left, mvt_mvt_coeffs_right)
    print('mvt left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(reward_mvt_coeffs_left, reward_mvt_coeffs_right)
    print('reward left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))

    ax_pfc[1].bar([0, 0.2, 1, 1.2, 2, 2.2], [signal_reward_coeffs_left_mean, signal_reward_coeffs_right_mean, mvt_reward_coeffs_left_mean, mvt_reward_coeffs_right_mean, reward_reward_coeffs_left_mean, reward_reward_coeffs_right_mean], yerr=[signal_reward_coeffs_left_err, signal_reward_coeffs_right_err, mvt_reward_coeffs_left_err, mvt_reward_coeffs_right_err, reward_reward_coeffs_left_err, reward_reward_coeffs_right_err], color='k', width=0.1)
    ax_pfc[1].set_xticks([0, 1, 2])
    ax_pfc[1].set_xticklabels(['S', 'SM', 'SMR'])
    ax_pfc[1].set_ylabel('reward coeffs')

    print('reward coeffs')

    # do the t test for left vs right
    t_stat, p_val = stats.ttest_ind(signal_reward_coeffs_left, signal_reward_coeffs_right)
    print('signal left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(mvt_reward_coeffs_left, mvt_reward_coeffs_right)
    print('mvt left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(reward_reward_coeffs_left, reward_reward_coeffs_right)
    print('reward left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))

    fig_pfc.suptitle('PFC')

    # Similarly for the dms coeffs
    # signal_signal_coeffs_left = np.array(dms_signal_coeffs_left)[:, 0]
    # signal_signal_coeffs_left_mean = np.mean(signal_signal_coeffs_left)
    # signal_signal_coeffs_left_err = np.std(signal_signal_coeffs_left) / np.sqrt(len(signal_signal_coeffs_left))
    # signal_signal_coeffs_right = np.array(dms_signal_coeffs_right)[:, 0]
    # signal_signal_coeffs_right_mean = np.mean(signal_signal_coeffs_right)
    # signal_signal_coeffs_right_err = np.std(signal_signal_coeffs_right) / np.sqrt(len(signal_signal_coeffs_right))

    signal_mvt_coeffs_left = np.array(dms_mvt_coeffs_left)[:, 0]
    # ignore nan values
    signal_mvt_coeffs_left = signal_mvt_coeffs_left[~np.isnan(signal_mvt_coeffs_left)]
    signal_mvt_coeffs_left_mean = np.mean(signal_mvt_coeffs_left)
    signal_mvt_coeffs_left_err = np.std(signal_mvt_coeffs_left) / np.sqrt(len(signal_mvt_coeffs_left))

    signal_mvt_coeffs_right = np.array(dms_mvt_coeffs_right)[:, 0]
    # ignore nan values
    signal_mvt_coeffs_right = signal_mvt_coeffs_right[~np.isnan(signal_mvt_coeffs_right)]
    signal_mvt_coeffs_right_mean = np.mean(signal_mvt_coeffs_right)
    signal_mvt_coeffs_right_err = np.std(signal_mvt_coeffs_right) / np.sqrt(len(signal_mvt_coeffs_right))

    signal_reward_coeffs_left = np.array(dms_reward_coeffs_left)[:, 0]
    # ignore nan values
    signal_reward_coeffs_left = signal_reward_coeffs_left[~np.isnan(signal_reward_coeffs_left)]
    signal_reward_coeffs_left_mean = np.mean(signal_reward_coeffs_left)
    signal_reward_coeffs_left_err = np.std(signal_reward_coeffs_left) / np.sqrt(len(signal_reward_coeffs_left))

    signal_reward_coeffs_right = np.array(dms_reward_coeffs_right)[:, 0]
    # ignore nan values
    signal_reward_coeffs_right = signal_reward_coeffs_right[~np.isnan(signal_reward_coeffs_right)]
    signal_reward_coeffs_right_mean = np.mean(signal_reward_coeffs_right)
    signal_reward_coeffs_right_err = np.std(signal_reward_coeffs_right) / np.sqrt(len(signal_reward_coeffs_right))

    # mvt_signal_coeffs_left = np.array(dms_signal_coeffs_left)[:, 1]
    # mvt_signal_coeffs_left_mean = np.mean(mvt_signal_coeffs_left)
    # mvt_signal_coeffs_left_err = np.std(mvt_signal_coeffs_left) / np.sqrt(len(mvt_signal_coeffs_left))
    # mvt_signal_coeffs_right = np.array(dms_signal_coeffs_right)[:, 1]
    # mvt_signal_coeffs_right_mean = np.mean(mvt_signal_coeffs_right)
    # mvt_signal_coeffs_right_err = np.std(mvt_signal_coeffs_right) / np.sqrt(len(mvt_signal_coeffs_right))

    mvt_mvt_coeffs_left = np.array(dms_mvt_coeffs_left)[:, 1]
    # ignore nan values
    mvt_mvt_coeffs_left = mvt_mvt_coeffs_left[~np.isnan(mvt_mvt_coeffs_left)]
    mvt_mvt_coeffs_left_mean = np.mean(mvt_mvt_coeffs_left)
    mvt_mvt_coeffs_left_err = np.std(mvt_mvt_coeffs_left) / np.sqrt(len(mvt_mvt_coeffs_left))

    mvt_mvt_coeffs_right = np.array(dms_mvt_coeffs_right)[:, 1]
    # ignore nan values
    mvt_mvt_coeffs_right = mvt_mvt_coeffs_right[~np.isnan(mvt_mvt_coeffs_right)]
    mvt_mvt_coeffs_right_mean = np.mean(mvt_mvt_coeffs_right)
    mvt_mvt_coeffs_right_err = np.std(mvt_mvt_coeffs_right) / np.sqrt(len(mvt_mvt_coeffs_right))

    mvt_reward_coeffs_left = np.array(dms_reward_coeffs_left)[:, 1]
    # ignore nan values
    mvt_reward_coeffs_left = mvt_reward_coeffs_left[~np.isnan(mvt_reward_coeffs_left)]
    mvt_reward_coeffs_left_mean = np.mean(mvt_reward_coeffs_left)
    mvt_reward_coeffs_left_err = np.std(mvt_reward_coeffs_left) / np.sqrt(len(mvt_reward_coeffs_left))

    mvt_reward_coeffs_right = np.array(dms_reward_coeffs_right)[:, 1]
    # ignore nan values
    mvt_reward_coeffs_right = mvt_reward_coeffs_right[~np.isnan(mvt_reward_coeffs_right)]
    mvt_reward_coeffs_right_mean = np.mean(mvt_reward_coeffs_right)
    mvt_reward_coeffs_right_err = np.std(mvt_reward_coeffs_right) / np.sqrt(len(mvt_reward_coeffs_right))

    # reward_signal_coeffs_left = np.array(dms_signal_coeffs_left)[:, 2]
    # reward_signal_coeffs_left_mean = np.mean(reward_signal_coeffs_left)
    # reward_signal_coeffs_left_err = np.std(reward_signal_coeffs_left) / np.sqrt(len(reward_signal_coeffs_left))
    # reward_signal_coeffs_right = np.array(dms_signal_coeffs_right)[:, 2]
    # reward_signal_coeffs_right_mean = np.mean(reward_signal_coeffs_right)
    # reward_signal_coeffs_right_err = np.std(reward_signal_coeffs_right) / np.sqrt(len(reward_signal_coeffs_right))

    reward_mvt_coeffs_left = np.array(dms_mvt_coeffs_left)[:, 2]
    # ignore nan values
    reward_mvt_coeffs_left = reward_mvt_coeffs_left[~np.isnan(reward_mvt_coeffs_left)]
    reward_mvt_coeffs_left_mean = np.mean(reward_mvt_coeffs_left)
    reward_mvt_coeffs_left_err = np.std(reward_mvt_coeffs_left) / np.sqrt(len(reward_mvt_coeffs_left))

    reward_mvt_coeffs_right = np.array(dms_mvt_coeffs_right)[:, 2]
    # ignore nan values
    reward_mvt_coeffs_right = reward_mvt_coeffs_right[~np.isnan(reward_mvt_coeffs_right)]
    reward_mvt_coeffs_right_mean = np.mean(reward_mvt_coeffs_right)
    reward_mvt_coeffs_right_err = np.std(reward_mvt_coeffs_right) / np.sqrt(len(reward_mvt_coeffs_right))

    reward_reward_coeffs_left = np.array(dms_reward_coeffs_left)[:, 2]
    # ignore nan values
    reward_reward_coeffs_left = reward_reward_coeffs_left[~np.isnan(reward_reward_coeffs_left)]
    reward_reward_coeffs_left_mean = np.mean(reward_reward_coeffs_left)
    reward_reward_coeffs_left_err = np.std(reward_reward_coeffs_left) / np.sqrt(len(reward_reward_coeffs_left))

    reward_reward_coeffs_right = np.array(dms_reward_coeffs_right)[:, 2]
    # ignore nan values
    reward_reward_coeffs_right = reward_reward_coeffs_right[~np.isnan(reward_reward_coeffs_right)]
    reward_reward_coeffs_right_mean = np.mean(reward_reward_coeffs_right)
    reward_reward_coeffs_right_err = np.std(reward_reward_coeffs_right) / np.sqrt(len(reward_reward_coeffs_right))

    # ax_dms[0].bar([0, 0.2, 1, 1.2, 2, 2.2], [signal_signal_coeffs_left_mean, signal_signal_coeffs_right_mean, mvt_signal_coeffs_left_mean, mvt_signal_coeffs_right_mean, reward_signal_coeffs_left_mean, reward_signal_coeffs_right_mean], yerr=[signal_signal_coeffs_left_err, signal_signal_coeffs_right_err, mvt_signal_coeffs_left_err, mvt_signal_coeffs_right_err, reward_signal_coeffs_left_err, reward_signal_coeffs_right_err], color='k', width=0.1)
    # ax_dms[0].set_xticks([0, 1, 2])
    # ax_dms[0].set_xticklabels(['S', 'SM', 'SMR'])
    # ax_dms[0].set_ylabel('signal coeffs')

    print('DMS')

    # print('signal coeffs')

    # # do the t test for left vs right
    # t_stat, p_val = stats.ttest_ind(signal_signal_coeffs_left, signal_signal_coeffs_right)
    # print('signal left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    # t_stat, p_val = stats.ttest_ind(mvt_signal_coeffs_left, mvt_signal_coeffs_right)
    # print('mvt left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    # t_stat, p_val = stats.ttest_ind(reward_signal_coeffs_left, reward_signal_coeffs_right)
    # print('reward left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))

    ax_dms[0].bar([0, 0.2, 1, 1.2, 2, 2.2], [signal_mvt_coeffs_left_mean, signal_mvt_coeffs_right_mean, mvt_mvt_coeffs_left_mean, mvt_mvt_coeffs_right_mean, reward_mvt_coeffs_left_mean, reward_mvt_coeffs_right_mean], yerr=[signal_mvt_coeffs_left_err, signal_mvt_coeffs_right_err, mvt_mvt_coeffs_left_err, mvt_mvt_coeffs_right_err, reward_mvt_coeffs_left_err, reward_mvt_coeffs_right_err], color='k', width=0.1)
    ax_dms[0].set_xticks([0, 1, 2])
    ax_dms[0].set_xticklabels(['S', 'SM', 'SMR'])
    ax_dms[0].set_ylabel('mvt coeffs')

    print('mvt coeffs')

    # do the t test for left vs right
    t_stat, p_val = stats.ttest_ind(signal_mvt_coeffs_left, signal_mvt_coeffs_right)
    print('signal left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(mvt_mvt_coeffs_left, mvt_mvt_coeffs_right)
    print('mvt left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(reward_mvt_coeffs_left, reward_mvt_coeffs_right)
    print('reward left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))

    ax_dms[1].bar([0, 0.2, 1, 1.2, 2, 2.2], [signal_reward_coeffs_left_mean, signal_reward_coeffs_right_mean, mvt_reward_coeffs_left_mean, mvt_reward_coeffs_right_mean, reward_reward_coeffs_left_mean, reward_reward_coeffs_right_mean], yerr=[signal_reward_coeffs_left_err, signal_reward_coeffs_right_err, mvt_reward_coeffs_left_err, mvt_reward_coeffs_right_err, reward_reward_coeffs_left_err, reward_reward_coeffs_right_err], color='k', width=0.1)
    ax_dms[1].set_xticks([0, 1, 2])
    ax_dms[1].set_xticklabels(['S', 'SM', 'SMR'])
    ax_dms[1].set_ylabel('reward coeffs')

    print('reward coeffs')

    # do the t test for left vs right
    t_stat, p_val = stats.ttest_ind(signal_reward_coeffs_left, signal_reward_coeffs_right)
    print('signal left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(mvt_reward_coeffs_left, mvt_reward_coeffs_right)
    print('mvt left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))
    t_stat, p_val = stats.ttest_ind(reward_reward_coeffs_left, reward_reward_coeffs_right)
    print('reward left vs right t test: t = ' + str(t_stat) + ', p = ' + str(p_val))

    fig_dms.suptitle('DMS')

    # make sure the y label does not overlap with figures
    fig_pfc.tight_layout()
    fig_dms.tight_layout()

    # save all the coefficients mean and sem in a csv file
    coeffs_dataframe = pd.DataFrame({'direction': ['left', 'right'],'pfc_signal_mvt_coeff': [signal_mvt_coeffs_left_mean, signal_mvt_coeffs_right_mean], 'pfc_signal_mvt_sem': [signal_mvt_coeffs_left_err, signal_mvt_coeffs_right_err], 'pfc_mvt_mvt_coeff': [mvt_mvt_coeffs_left_mean, mvt_mvt_coeffs_right_mean], 'pfc_mvt_mvt_sem': [mvt_mvt_coeffs_left_err, mvt_mvt_coeffs_right_err], 'pfc_reward_mvt_coeff': [reward_mvt_coeffs_left_mean, reward_mvt_coeffs_right_mean], 'pfc_reward_mvt_sem': [reward_mvt_coeffs_left_err, reward_mvt_coeffs_right_err], 'pfc_signal_reward_coeff': [signal_reward_coeffs_left_mean, signal_reward_coeffs_right_mean], 'pfc_signal_reward_sem': [signal_reward_coeffs_left_err, signal_reward_coeffs_right_err], 'pfc_mvt_reward_coeff': [mvt_reward_coeffs_left_mean, mvt_reward_coeffs_right_mean], 'pfc_mvt_reward_sem': [mvt_reward_coeffs_left_err, mvt_reward_coeffs_right_err], 'pfc_reward_reward_coeff': [reward_reward_coeffs_left_mean, reward_reward_coeffs_right_mean], 'pfc_reward_reward_sem': [reward_reward_coeffs_left_err, reward_reward_coeffs_right_err]}) 
    coeffs_dataframe.to_csv('pfc_coeffs.csv', index=False)

    coeffs_dataframe = pd.DataFrame({'direction': ['left', 'right'],'dms_signal_mvt_coeff': [signal_mvt_coeffs_left_mean, signal_mvt_coeffs_right_mean], 'dms_signal_mvt_sem': [signal_mvt_coeffs_left_err, signal_mvt_coeffs_right_err], 'dms_mvt_mvt_coeff': [mvt_mvt_coeffs_left_mean, mvt_mvt_coeffs_right_mean], 'dms_mvt_mvt_sem': [mvt_mvt_coeffs_left_err, mvt_mvt_coeffs_right_err], 'dms_reward_mvt_coeff': [reward_mvt_coeffs_left_mean, reward_mvt_coeffs_right_mean], 'dms_reward_mvt_sem': [reward_mvt_coeffs_left_err, reward_mvt_coeffs_right_err], 'dms_signal_reward_coeff': [signal_reward_coeffs_left_mean, signal_reward_coeffs_right_mean], 'dms_signal_reward_sem': [signal_reward_coeffs_left_err, signal_reward_coeffs_right_err], 'dms_mvt_reward_coeff': [mvt_reward_coeffs_left_mean, mvt_reward_coeffs_right_mean], 'dms_mvt_reward_sem': [mvt_reward_coeffs_left_err, mvt_reward_coeffs_right_err], 'dms_reward_reward_coeff': [reward_reward_coeffs_left_mean, reward_reward_coeffs_right_mean], 'dms_reward_reward_sem': [reward_reward_coeffs_left_err, reward_reward_coeffs_right_err]})
    coeffs_dataframe.to_csv('dms_coeffs.csv', index=False)


def get_figure_4_panel_ab():
    if not isfile(signal_mvt_reward_file):
        # print error message
        print('Error: ' + signal_mvt_reward_file + ' does not exist. Run figure_3_panel_bc() first.')

    # load the binned spike times
    pfc_signal_binned_mean, pfc_signal_binned_err, pfc_mvt_binned_mean, pfc_mvt_binned_err, pfc_reward_binned_mean, pfc_reward_binned_err, dms_signal_binned_mean, dms_signal_binned_err, dms_mvt_binned_mean, dms_mvt_binned_err, dms_reward_binned_mean, dms_reward_binned_err = np.load(signal_mvt_reward_file)

    pfc_mvt = pfc_mvt_binned_mean - pfc_signal_binned_mean
    pfc_reward = pfc_reward_binned_mean - pfc_mvt_binned_mean

    dms_mvt = dms_mvt_binned_mean - dms_signal_binned_mean
    dms_reward = dms_reward_binned_mean - dms_mvt_binned_mean

    # standardize the signal, mvt, and reward components
    pfc_signal_binned_mean = (pfc_signal_binned_mean - np.mean(pfc_signal_binned_mean)) / np.std(pfc_signal_binned_mean)
    pfc_mvt = (pfc_mvt - np.mean(pfc_mvt)) / np.std(pfc_mvt)
    pfc_reward = (pfc_reward - np.mean(pfc_reward)) / np.std(pfc_reward)

    dms_signal_binned_mean = (dms_signal_binned_mean - np.mean(dms_signal_binned_mean)) / np.std(dms_signal_binned_mean)
    dms_mvt = (dms_mvt - np.mean(dms_mvt)) / np.std(dms_mvt)
    dms_reward = (dms_reward - np.mean(dms_reward)) / np.std(dms_reward)

    # store the regressors in X
    X_pfc= np.column_stack((pfc_signal_binned_mean, pfc_mvt, pfc_reward))
    X_pfc = np.column_stack((np.ones(len(pfc_signal_binned_mean)), X_pfc))

    X_dms = np.column_stack((dms_signal_binned_mean, dms_mvt, dms_reward))
    X_dms = np.column_stack((np.ones(len(dms_signal_binned_mean)), X_dms))

    pfc_coeffs_high = []
    dms_coeffs_high = []

    pfc_coeffs_low = []
    dms_coeffs_low = []

    # load the binned pfc spike times
    for pfc_file in glob(pjoin(spike_time_dir, 'figure_3', f'*pfc*')):
        signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times = np.load(pfc_file, allow_pickle=True)

        total_spike_times = np.array(total_spike_times)

        session_name = basename(pfc_file).split('_')[0]
        high_prp, low_prp = get_high_low_prp_index(session_name, reset=False)

        spike_times_high = np.sum(total_spike_times[high_prp], axis=0)
        spike_times_low = np.sum(total_spike_times[low_prp], axis=0)

        if not np.max(spike_times_high) == 0:
            spike_times_high = spike_times_high / np.max(spike_times_high)
            pfc_coeffs_high.append(np.linalg.lstsq(X_pfc, spike_times_high, rcond=None)[0][1:])
        if not np.max(spike_times_low) == 0:
            spike_times_low = spike_times_low / np.max(spike_times_low)
            pfc_coeffs_low.append(np.linalg.lstsq(X_pfc, spike_times_low, rcond=None)[0][1:])

    # load the binned dms spike times
    for dms_file in glob(pjoin(spike_time_dir, 'figure_3', f'*dms*')):
        signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times = np.load(dms_file, allow_pickle=True)        

        total_spike_times = np.array(total_spike_times)

        session_name = basename(dms_file).split('_')[0]
        high_prp, low_prp = get_high_low_prp_index(session_name, reset=False)

        spike_times_high = np.sum(total_spike_times[high_prp], axis=0)
        spike_times_low = np.sum(total_spike_times[low_prp], axis=0)

        if not np.max(spike_times_high) == 0:
            spike_times_high = spike_times_high / np.max(spike_times_high)
            dms_coeffs_high.append(np.linalg.lstsq(X_dms, spike_times_high, rcond=None)[0][1:])
        if not np.max(spike_times_low) == 0:
            spike_times_low = spike_times_low / np.max(spike_times_low)
            dms_coeffs_low.append(np.linalg.lstsq(X_dms, spike_times_low, rcond=None)[0][1:])

    pfc_coeffs_high = np.array(pfc_coeffs_high)
    pfc_coeffs_low = np.array(pfc_coeffs_low)

    dms_coeffs_high = np.array(dms_coeffs_high)
    dms_coeffs_low = np.array(dms_coeffs_low)

    # plot the signal, mvt, and reward components
    fig_pfc, ax_pfc = plt.subplots(1, 3, figsize=(16, 5))
    fig_dms, ax_dms = plt.subplots(1, 3, figsize=(16, 5))

    fig_pfc.suptitle('PFC')
    fig_dms.suptitle('DMS')

    # for each coefficient, plot the high and low values with std error bars
    pfc_signals_high = pfc_coeffs_high[:, 0]
    pfc_signals_low = pfc_coeffs_low[:, 0]

    pfc_mvt_high = pfc_coeffs_high[:, 1]
    pfc_mvt_low = pfc_coeffs_low[:, 1]

    pfc_reward_high = pfc_coeffs_high[:, 2]
    pfc_reward_low = pfc_coeffs_low[:, 2]

    dms_signals_high = dms_coeffs_high[:, 0]
    dms_signals_low = dms_coeffs_low[:, 0]

    dms_mvt_high = dms_coeffs_high[:, 1]
    dms_mvt_low = dms_coeffs_low[:, 1]

    dms_reward_high = dms_coeffs_high[:, 2]
    dms_reward_low = dms_coeffs_low[:, 2]

    # calculate the standard error for each coefficient
    pfc_signals_high_err = np.std(pfc_signals_high, axis=0) / np.sqrt(len(pfc_signals_high))
    pfc_signals_low_err = np.std(pfc_signals_low, axis=0) / np.sqrt(len(pfc_signals_low))

    pfc_mvt_high_err = np.std(pfc_mvt_high, axis=0) / np.sqrt(len(pfc_mvt_high))
    pfc_mvt_low_err = np.std(pfc_mvt_low, axis=0) / np.sqrt(len(pfc_mvt_low))

    pfc_reward_high_err = np.std(pfc_reward_high, axis=0) / np.sqrt(len(pfc_reward_high))
    pfc_reward_low_err = np.std(pfc_reward_low, axis=0) / np.sqrt(len(pfc_reward_low))

    dms_signals_high_err = np.std(dms_signals_high, axis=0) / np.sqrt(len(dms_signals_high))
    dms_signals_low_err = np.std(dms_signals_low, axis=0) / np.sqrt(len(dms_signals_low))

    dms_mvt_high_err = np.std(dms_mvt_high, axis=0) / np.sqrt(len(dms_mvt_high))
    dms_mvt_low_err = np.std(dms_mvt_low, axis=0) / np.sqrt(len(dms_mvt_low))

    dms_reward_high_err = np.std(dms_reward_high, axis=0) / np.sqrt(len(dms_reward_high))
    dms_reward_low_err = np.std(dms_reward_low, axis=0) / np.sqrt(len(dms_reward_low))

    figure_4_panel_A_data = pd.DataFrame({'trial_type': ['prp $\geq$ 0.5', 'prp < 0.5'], 'signal_coeffs': [np.mean(pfc_signals_high), np.mean(pfc_signals_low)], 'signal_coeffs_err': [pfc_signals_high_err, pfc_signals_low_err], 'mvt_coeffs': [np.mean(pfc_mvt_high), np.mean(pfc_mvt_low)], 'mvt_coeffs_err': [pfc_mvt_high_err, pfc_mvt_low_err], 'reward_coeffs': [np.mean(pfc_reward_high), np.mean(pfc_reward_low)], 'reward_coeffs_err': [pfc_reward_high_err, pfc_reward_low_err]})
    figure_4_panel_A_data.to_csv(pjoin(figure_4_data_root, 'figure_4_panel_A_data.csv'), index=False)
    figure_4_panel_B_data = pd.DataFrame({'trial_type': ['prp $\geq$ 0.5', 'prp < 0.5'], 'signal_coeffs': [np.mean(dms_signals_high), np.mean(dms_signals_low)], 'signal_coeffs_err': [dms_signals_high_err, dms_signals_low_err], 'mvt_coeffs': [np.mean(dms_mvt_high), np.mean(dms_mvt_low)], 'mvt_coeffs_err': [dms_mvt_high_err, dms_mvt_low_err], 'reward_coeffs': [np.mean(dms_reward_high), np.mean(dms_reward_low)], 'reward_coeffs_err': [dms_reward_high_err, dms_reward_low_err]})
    figure_4_panel_B_data.to_csv(pjoin(figure_4_data_root, 'figure_4_panel_B_data.csv'), index=False)

    # plot the high and low coefficients for each component using bar plots, with error bars using std
    ax_pfc[0].bar(['prp$\geq$0.5', 'prp<0.5'], [np.mean(pfc_signals_high), np.mean(pfc_signals_low)], yerr=[pfc_signals_high_err, pfc_signals_low_err])
    ax_pfc[0].set_title('Signal')
    ax_pfc[0].set_ylabel('Coefficient')

    ax_pfc[1].bar(['prp$\geq$0.5', 'prp<0.5'], [np.mean(pfc_mvt_high), np.mean(pfc_mvt_low)], yerr=[pfc_mvt_high_err, pfc_mvt_low_err])
    ax_pfc[1].set_title('Movement')
    ax_pfc[1].set_ylabel('Coefficient')

    ax_pfc[2].bar(['prp$\geq$0.5', 'prp<0.5'], [np.mean(pfc_reward_high), np.mean(pfc_reward_low)], yerr=[pfc_reward_high_err, pfc_reward_low_err])
    ax_pfc[2].set_title('Reward')
    ax_pfc[2].set_ylabel('Coefficient')

    ax_dms[0].bar(['prp$\geq$0.5', 'prp<0.5'], [np.mean(dms_signals_high), np.mean(dms_signals_low)], yerr=[dms_signals_high_err, dms_signals_low_err])
    ax_dms[0].set_title('Signal')
    ax_dms[0].set_ylabel('Coefficient')

    ax_dms[1].bar(['prp$\geq$0.5', 'prp<0.5'], [np.mean(dms_mvt_high), np.mean(dms_mvt_low)], yerr=[dms_mvt_high_err, dms_mvt_low_err])
    ax_dms[1].set_title('Movement')
    ax_dms[1].set_ylabel('Coefficient')

    ax_dms[2].bar(['prp$\geq$0.5', 'prp<0.5'], [np.mean(dms_reward_high), np.mean(dms_reward_low)], yerr=[dms_reward_high_err, dms_reward_low_err])
    ax_dms[2].set_title('Reward')
    ax_dms[2].set_ylabel('Coefficient')

    # remove the top and bottom spines
    for ax in ax_pfc:
        remove_top_and_right_spines(ax)

    for ax in ax_dms:
        remove_top_and_right_spines(ax)

    # do the t test for each pair of high and low coefficients
    # for each component
    pfc_signal_t, pfc_signal_p = ttest_ind(pfc_signals_high, pfc_signals_low)
    pfc_mvt_t, pfc_mvt_p = ttest_ind(pfc_mvt_high, pfc_mvt_low)
    pfc_reward_t, pfc_reward_p = ttest_ind(pfc_reward_high, pfc_reward_low)

    dms_signal_t, dms_signal_p = ttest_ind(dms_signals_high, dms_signals_low)
    dms_mvt_t, dms_mvt_p = ttest_ind(dms_mvt_high, dms_mvt_low)
    dms_reward_t, dms_reward_p = ttest_ind(dms_reward_high, dms_reward_low)

    # print the t test results
    print('PFC signal t test: t = {}, p = {}'.format(pfc_signal_t, pfc_signal_p))
    print('PFC movement t test: t = {}, p = {}'.format(pfc_mvt_t, pfc_mvt_p))
    print('PFC reward t test: t = {}, p = {}'.format(pfc_reward_t, pfc_reward_p))

    print('DMS signal t test: t = {}, p = {}'.format(dms_signal_t, dms_signal_p))
    print('DMS movement t test: t = {}, p = {}'.format(dms_mvt_t, dms_mvt_p))
    print('DMS reward t test: t = {}, p = {}'.format(dms_reward_t, dms_reward_p))

# split by left and right instead of high and low prp
def get_figure_4_panel_ab_lr():
    if not isfile(signal_mvt_reward_file):
        # print error message
        print('Error: ' + signal_mvt_reward_file + ' does not exist. Run figure_3_panel_bc() first.')

    # load the binned spike times
    pfc_signal_binned_mean, pfc_signal_binned_err, pfc_mvt_binned_mean, pfc_mvt_binned_err, pfc_reward_binned_mean, pfc_reward_binned_err, dms_signal_binned_mean, dms_signal_binned_err, dms_mvt_binned_mean, dms_mvt_binned_err, dms_reward_binned_mean, dms_reward_binned_err = np.load(signal_mvt_reward_file)

    pfc_mvt = pfc_mvt_binned_mean - pfc_signal_binned_mean
    pfc_reward = pfc_reward_binned_mean - pfc_mvt_binned_mean

    dms_mvt = dms_mvt_binned_mean - dms_signal_binned_mean
    dms_reward = dms_reward_binned_mean - dms_mvt_binned_mean

    # standardize the signal, mvt, and reward components
    pfc_signal_binned_mean = (pfc_signal_binned_mean - np.mean(pfc_signal_binned_mean)) / np.std(pfc_signal_binned_mean)
    pfc_mvt = (pfc_mvt - np.mean(pfc_mvt)) / np.std(pfc_mvt)
    pfc_reward = (pfc_reward - np.mean(pfc_reward)) / np.std(pfc_reward)

    dms_signal_binned_mean = (dms_signal_binned_mean - np.mean(dms_signal_binned_mean)) / np.std(dms_signal_binned_mean)
    dms_mvt = (dms_mvt - np.mean(dms_mvt)) / np.std(dms_mvt)
    dms_reward = (dms_reward - np.mean(dms_reward)) / np.std(dms_reward)

    # store the regressors in X
    X_pfc= np.column_stack((pfc_signal_binned_mean, pfc_mvt, pfc_reward))
    X_pfc = np.column_stack((np.ones(len(pfc_signal_binned_mean)), X_pfc))

    X_dms = np.column_stack((dms_signal_binned_mean, dms_mvt, dms_reward))
    X_dms = np.column_stack((np.ones(len(dms_signal_binned_mean)), X_dms))

    pfc_coeffs_right = []
    dms_coeffs_right = []

    pfc_coeffs_left = []
    dms_coeffs_left = []

    # load the binned pfc spike times
    for pfc_file in glob(pjoin(spike_time_dir, 'figure_3', f'*pfc*')):
        signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times, _ = np.load(pfc_file, allow_pickle=True)

        total_spike_times = np.array(total_spike_times)

        session_name = basename(pfc_file).split('_')[0]
        high_prp, low_prp = get_lr_index(session_name, reset=False)

        spike_times_high = np.sum(total_spike_times[high_prp], axis=0)
        spike_times_low = np.sum(total_spike_times[low_prp], axis=0)

        if not np.max(spike_times_high) == 0:
            spike_times_high = spike_times_high / np.max(spike_times_high)
            pfc_coeffs_right.append(np.linalg.lstsq(X_pfc, spike_times_high, rcond=None)[0][1:])
        if not np.max(spike_times_low) == 0:
            spike_times_low = spike_times_low / np.max(spike_times_low)
            pfc_coeffs_left.append(np.linalg.lstsq(X_pfc, spike_times_low, rcond=None)[0][1:])

    # load the binned dms spike times
    for dms_file in glob(pjoin(spike_time_dir, 'figure_3', f'*dms*')):
        signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times, _ = np.load(dms_file, allow_pickle=True)        

        total_spike_times = np.array(total_spike_times)

        session_name = basename(dms_file).split('_')[0]
        high_prp, low_prp = get_lr_index(session_name, reset=False)

        spike_times_high = np.sum(total_spike_times[high_prp], axis=0)
        spike_times_low = np.sum(total_spike_times[low_prp], axis=0)

        if not np.max(spike_times_high) == 0:
            spike_times_high = spike_times_high / np.max(spike_times_high)
            dms_coeffs_right.append(np.linalg.lstsq(X_dms, spike_times_high, rcond=None)[0][1:])
        if not np.max(spike_times_low) == 0:
            spike_times_low = spike_times_low / np.max(spike_times_low)
            dms_coeffs_left.append(np.linalg.lstsq(X_dms, spike_times_low, rcond=None)[0][1:])

    pfc_coeffs_right = np.array(pfc_coeffs_right)
    pfc_coeffs_left = np.array(pfc_coeffs_left)

    dms_coeffs_right = np.array(dms_coeffs_right)
    dms_coeffs_left = np.array(dms_coeffs_left)

    # plot the signal, mvt, and reward components
    fig_pfc, ax_pfc = plt.subplots(1, 3, figsize=(16, 5))
    fig_dms, ax_dms = plt.subplots(1, 3, figsize=(16, 5))

    fig_pfc.suptitle('PFC')
    fig_dms.suptitle('DMS')

    print(pfc_coeffs_right)

    # for each coefficient, plot the high and low values with std error bars
    pfc_signals_high = pfc_coeffs_right[:, 0]
    pfc_signals_low = pfc_coeffs_left[:, 0]

    pfc_mvt_high = pfc_coeffs_right[:, 1]
    pfc_mvt_low = pfc_coeffs_left[:, 1]

    pfc_reward_high = pfc_coeffs_right[:, 2]
    pfc_reward_low = pfc_coeffs_left[:, 2]

    dms_signals_high = dms_coeffs_right[:, 0]
    dms_signals_low = dms_coeffs_left[:, 0]

    dms_mvt_high = dms_coeffs_right[:, 1]
    dms_mvt_low = dms_coeffs_left[:, 1]

    dms_reward_high = dms_coeffs_right[:, 2]
    dms_reward_low = dms_coeffs_left[:, 2]

    # calculate the standard error for each coefficient
    pfc_signals_high_err = np.std(pfc_signals_high, axis=0) / np.sqrt(len(pfc_signals_high))
    pfc_signals_low_err = np.std(pfc_signals_low, axis=0) / np.sqrt(len(pfc_signals_low))

    pfc_mvt_high_err = np.std(pfc_mvt_high, axis=0) / np.sqrt(len(pfc_mvt_high))
    pfc_mvt_low_err = np.std(pfc_mvt_low, axis=0) / np.sqrt(len(pfc_mvt_low))

    pfc_reward_high_err = np.std(pfc_reward_high, axis=0) / np.sqrt(len(pfc_reward_high))
    pfc_reward_low_err = np.std(pfc_reward_low, axis=0) / np.sqrt(len(pfc_reward_low))

    dms_signals_high_err = np.std(dms_signals_high, axis=0) / np.sqrt(len(dms_signals_high))
    dms_signals_low_err = np.std(dms_signals_low, axis=0) / np.sqrt(len(dms_signals_low))

    dms_mvt_high_err = np.std(dms_mvt_high, axis=0) / np.sqrt(len(dms_mvt_high))
    dms_mvt_low_err = np.std(dms_mvt_low, axis=0) / np.sqrt(len(dms_mvt_low))

    dms_reward_high_err = np.std(dms_reward_high, axis=0) / np.sqrt(len(dms_reward_high))
    dms_reward_low_err = np.std(dms_reward_low, axis=0) / np.sqrt(len(dms_reward_low))

    # figure_4_panel_A_data = pd.DataFrame({'trial_type': ['prp $\geq$ 0.5', 'prp < 0.5'], 'signal_coeffs': [np.mean(pfc_signals_high), np.mean(pfc_signals_low)], 'signal_coeffs_err': [pfc_signals_high_err, pfc_signals_low_err], 'mvt_coeffs': [np.mean(pfc_mvt_high), np.mean(pfc_mvt_low)], 'mvt_coeffs_err': [pfc_mvt_high_err, pfc_mvt_low_err], 'reward_coeffs': [np.mean(pfc_reward_high), np.mean(pfc_reward_low)], 'reward_coeffs_err': [pfc_reward_high_err, pfc_reward_low_err]})
    # figure_4_panel_A_data.to_csv(pjoin(figure_4_data_root, 'figure_4_panel_A_data.csv'), index=False)
    # figure_4_panel_B_data = pd.DataFrame({'trial_type': ['prp $\geq$ 0.5', 'prp < 0.5'], 'signal_coeffs': [np.mean(dms_signals_high), np.mean(dms_signals_low)], 'signal_coeffs_err': [dms_signals_high_err, dms_signals_low_err], 'mvt_coeffs': [np.mean(dms_mvt_high), np.mean(dms_mvt_low)], 'mvt_coeffs_err': [dms_mvt_high_err, dms_mvt_low_err], 'reward_coeffs': [np.mean(dms_reward_high), np.mean(dms_reward_low)], 'reward_coeffs_err': [dms_reward_high_err, dms_reward_low_err]})
    # figure_4_panel_B_data.to_csv(pjoin(figure_4_data_root, 'figure_4_panel_B_data.csv'), index=False)

    # plot the high and low coefficients for each component using bar plots, with error bars using std
    ax_pfc[0].bar(['left', 'right'], [np.mean(pfc_signals_high), np.mean(pfc_signals_low)], yerr=[pfc_signals_high_err, pfc_signals_low_err])
    ax_pfc[0].set_title('Signal')
    ax_pfc[0].set_ylabel('Coefficient')

    ax_pfc[1].bar(['left', 'right'], [np.mean(pfc_mvt_high), np.mean(pfc_mvt_low)], yerr=[pfc_mvt_high_err, pfc_mvt_low_err])
    ax_pfc[1].set_title('Movement')
    ax_pfc[1].set_ylabel('Coefficient')

    ax_pfc[2].bar(['left', 'right'], [np.mean(pfc_reward_high), np.mean(pfc_reward_low)], yerr=[pfc_reward_high_err, pfc_reward_low_err])
    ax_pfc[2].set_title('Reward')
    ax_pfc[2].set_ylabel('Coefficient')

    ax_dms[0].bar(['left', 'right'], [np.mean(dms_signals_high), np.mean(dms_signals_low)], yerr=[dms_signals_high_err, dms_signals_low_err])
    ax_dms[0].set_title('Signal')
    ax_dms[0].set_ylabel('Coefficient')

    ax_dms[1].bar(['left', 'right'], [np.mean(dms_mvt_high), np.mean(dms_mvt_low)], yerr=[dms_mvt_high_err, dms_mvt_low_err])
    ax_dms[1].set_title('Movement')
    ax_dms[1].set_ylabel('Coefficient')

    ax_dms[2].bar(['left', 'right'], [np.mean(dms_reward_high), np.mean(dms_reward_low)], yerr=[dms_reward_high_err, dms_reward_low_err])
    ax_dms[2].set_title('Reward')
    ax_dms[2].set_ylabel('Coefficient')

    # remove the top and bottom spines
    for ax in ax_pfc:
        remove_top_and_right_spines(ax)

    for ax in ax_dms:
        remove_top_and_right_spines(ax)

    # do the t test for each pair of high and low coefficients
    # for each component
    pfc_signal_t, pfc_signal_p = ttest_ind(pfc_signals_high, pfc_signals_low)
    pfc_mvt_t, pfc_mvt_p = ttest_ind(pfc_mvt_high, pfc_mvt_low)
    pfc_reward_t, pfc_reward_p = ttest_ind(pfc_reward_high, pfc_reward_low)

    dms_signal_t, dms_signal_p = ttest_ind(dms_signals_high, dms_signals_low)
    dms_mvt_t, dms_mvt_p = ttest_ind(dms_mvt_high, dms_mvt_low)
    dms_reward_t, dms_reward_p = ttest_ind(dms_reward_high, dms_reward_low)

    # print the t test results
    print('PFC signal t test: t = {}, p = {}'.format(pfc_signal_t, pfc_signal_p))
    print('PFC movement t test: t = {}, p = {}'.format(pfc_mvt_t, pfc_mvt_p))
    print('PFC reward t test: t = {}, p = {}'.format(pfc_reward_t, pfc_reward_p))

    print('DMS signal t test: t = {}, p = {}'.format(dms_signal_t, dms_signal_p))
    print('DMS movement t test: t = {}, p = {}'.format(dms_mvt_t, dms_mvt_p))
    print('DMS reward t test: t = {}, p = {}'.format(dms_reward_t, dms_reward_p))

def figure_3_panel_bc():
    spike_time_dir = pjoin('data', 'spike_times')
    relative_spike_time_dir = pjoin('data', 'relative_spike_time_trials')
    behaviour_root = pjoin('data', 'behaviour_data')

    # iterate through the sessions
    for session_name in tqdm.tqdm(listdir(behaviour_root)):
        if isdir(pjoin(behaviour_root, session_name)):
            continue

        # loead the behaviour data
        behaviour_path = pjoin(behaviour_root, session_name)
        behaviour_data = pd.read_csv(behaviour_path)
        cue_times = behaviour_data['cue_time']

        # get the index of nan trials in behaviour data as signal only trials
        signal = behaviour_data['trial_response_side'].isna()
        signal_idx = np.where(signal)[0]

        # get the index of unrewarded non_nan trials as movement trials
        mvt = behaviour_data['trial_response_side'].notna()
        mvt = mvt & (behaviour_data['trial_reward'] == 0) 
        mvt_idx = np.where(mvt)[0]

        # get the index of rewarded non_nan trials as reward trials
        reward = behaviour_data['trial_response_side'].notna()
        reward = reward & (behaviour_data['trial_reward'] == 1)
        reward_idx = np.where(reward)[0]

        # load the pfc spike times
        for pfc_times in glob(pjoin(spike_time_dir, session_name, 'pfc_*')):
            pfc_times = np.load(pfc_times)
            relative_to_pfc = get_relative_spike_times(pfc_times, cue_times, -0.5, 1.5)



def get_high_low_prp_index(session_name: str, reset: bool = False):
    if isfile(pjoin(spike_time_dir, 'figure_3', session_name+'_high.npy')) and not reset:
        high_prp = np.load(pjoin(spike_time_dir, 'figure_3', session_name+'_high.npy'))
        low_prp = np.load(pjoin(spike_time_dir, 'figure_3', session_name+'_low.npy'))
    else:
        # load the behaviour data
        behaviour_data = pd.read_csv(pjoin(behaviour_root, session_name+'.csv'))
        trial_reward = np.array(behaviour_data['trial_reward'])
        # fill in the nan values
        trial_reward[np.isnan(trial_reward)] = 0
        prp = moving_window_mean_prior(trial_reward, 10)
        # get the index of prp > 0.5 and prp < 0.5 and reward == 1
        high_prp = np.where((prp >= 0.5) & (trial_reward==1))[0]
        low_prp = np.where((prp < 0.5) & (trial_reward==1))[0]
        np.save(pjoin(spike_time_dir, 'figure_3', session_name+'_high.npy'), high_prp)
        np.save(pjoin(spike_time_dir, 'figure_3', session_name+'_low.npy'), low_prp)
    # truncate the high_prp and low_prp to the same length
    # TODO why is this necessary?
    high_prp = high_prp[:min(len(high_prp), len(low_prp))]
    low_prp = low_prp[:min(len(high_prp), len(low_prp))]
    return high_prp, low_prp


def get_lr_index(session_name: str, reset: bool = False):
    # load the behaviour data
    behaviour_data = pd.read_csv(pjoin(behaviour_root, session_name+'.csv'))
    trial_reward = np.array(behaviour_data['trial_reward'])
    trial_response_side = np.array(behaviour_data['trial_response_side'])
    right = np.where(trial_response_side==1)[0]
    left = np.where(trial_response_side==-1)[0]

    # truncate the left and right to the same length
    # right = right[:min(len(right), len(left))]
    # left = left[:min(len(right), len(left))]
    return right, left