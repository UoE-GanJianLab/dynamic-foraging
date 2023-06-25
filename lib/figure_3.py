# type: ignore
from os.path import join as pjoin, isdir, isfile, basename
from os import listdir, mkdir
from shutil import rmtree
from glob import glob

import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from lib.calculation import moving_window_mean, get_relative_spike_times

WINDOW_LEFT = -0.5
WINDOW_RIGHT = 1.5
BIN_SIZE = 0.02 # 20ms bins

spike_time_dir = pjoin('data', 'spike_times')
behaviour_root = pjoin('data', 'behaviour_data')

signal_mvt_reward_file = pjoin(spike_time_dir, 'figure_3', 'signal_mvt_reward.npy')

# poster panel a,b of figure 3
def figure_3_panel_bc(reset=False):

    str_count = 0
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

            session_name = session_name.split('.')[0]

            pfc_session = pfc_count

            # load the pfc spike times
            for pfc_times in glob(pjoin(spike_time_dir, 'sessions', session_name, 'pfc_*')):
                pfc_count += 1
                # check if the binned file for this cell exists
                cell_name = basename(pfc_times).split('.')[0]
                binned_file = pjoin(spike_time_dir, 'figure_3', f'{session_name}_{cell_name}.npy')

                # load the pfc spike times
                pfc_times = np.load(pfc_times)
                relative_to_pfc = get_relative_spike_times(pfc_times, np.array(cue_times), WINDOW_LEFT, WINDOW_RIGHT)

                # get the spike times of each trial type
                signal_spike_times = [relative_to_pfc[idx] for idx in signal_idx]
                mvt_spike_times = [relative_to_pfc[idx] for idx in mvt_idx]
                reward_spike_times = [relative_to_pfc[idx] for idx in reward_idx]

                # concatenate the spike times of each trial type
                signal_spike_times = np.concatenate(signal_spike_times)
                mvt_spike_times = np.concatenate(mvt_spike_times)
                reward_spike_times = np.concatenate(reward_spike_times)

                # bin the spike times using histoplot 10ms bins
                signal_spike_times = np.histogram(signal_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                mvt_spike_times = np.histogram(mvt_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                reward_spike_times = np.histogram(reward_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                
                
                if not isfile(binned_file):
                    total_spike_times = np.histogram(np.concatenate(relative_to_pfc), bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                    # save the binned spike times
                    np.save(binned_file, [signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times])


                if not isfile(binned_file):
                    total_spike_times = np.histogram(np.concatenate(relative_to_pfc), bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                    # save the binned spike times
                    np.save(binned_file, [signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times])

                # convert to firing rate
                signal_spike_times = np.divide(signal_spike_times, BIN_SIZE)
                mvt_spike_times = np.divide(mvt_spike_times, BIN_SIZE)
                reward_spike_times = np.divide(reward_spike_times, BIN_SIZE)

                # normalize by the number of trials
                signal_spike_times = np.divide(signal_spike_times, len(signal_idx))
                mvt_spike_times = np.divide(mvt_spike_times, len(mvt_idx))
                reward_spike_times = np.divide(reward_spike_times, len(reward_idx))

                if not isfile(binned_file):
                    total_spike_times = np.histogram(np.concatenate(relative_to_pfc), bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                    # save the binned spike times
                    np.save(binned_file, [signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times])
                
                # add the binned spike times to the total
                pfc_signal_binned.append(signal_spike_times)
                pfc_mvt_binned.append(mvt_spike_times)
                pfc_reward_binned.append(reward_spike_times)

            str_session = str_count

            # load the pfc spike times
            for str_times in glob(pjoin(spike_time_dir, 'sessions', session_name, 'str_*')):
                str_count += 1
                # check if the binned file for this cell exists
                cell_name = basename(str_times).split('.')[0]
                binned_file = pjoin(spike_time_dir, 'figure_3', f'{session_name}_{cell_name}.npy')

                # load the pfc spike times
                str_times = np.load(str_times)
                relative_to_str = get_relative_spike_times(str_times, np.array(cue_times), WINDOW_LEFT, WINDOW_RIGHT)

                # get the spike times of each trial type
                signal_spike_times = [relative_to_str[idx] for idx in signal_idx]
                mvt_spike_times = [relative_to_str[idx] for idx in mvt_idx]
                reward_spike_times = [relative_to_str[idx] for idx in reward_idx]

                # concatenate the spike times of each trial type
                signal_spike_times = np.concatenate(signal_spike_times)
                mvt_spike_times = np.concatenate(mvt_spike_times)
                reward_spike_times = np.concatenate(reward_spike_times)

                # bin the spike times using histoplot 10ms bins
                signal_spike_times = np.histogram(signal_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                mvt_spike_times = np.histogram(mvt_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                reward_spike_times = np.histogram(reward_spike_times, bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                
                
                if not isfile(binned_file):
                    total_spike_times = np.histogram(np.concatenate(relative_to_str), bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                    # save the binned spike times
                    np.save(binned_file, [signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times])


                if not isfile(binned_file):
                    total_spike_times = np.histogram(np.concatenate(relative_to_str), bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                    # save the binned spike times
                    np.save(binned_file, [signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times])

                # convert to firing rate
                signal_spike_times = np.divide(signal_spike_times, BIN_SIZE)
                mvt_spike_times = np.divide(mvt_spike_times, BIN_SIZE)
                reward_spike_times = np.divide(reward_spike_times, BIN_SIZE)

                # normalize by the number of trials
                signal_spike_times = np.divide(signal_spike_times, len(signal_idx))
                mvt_spike_times = np.divide(mvt_spike_times, len(mvt_idx))
                reward_spike_times = np.divide(reward_spike_times, len(reward_idx))

                if not isfile(binned_file):
                    total_spike_times = np.histogram(np.concatenate(relative_to_str), bins=np.arange(WINDOW_LEFT, WINDOW_RIGHT+BIN_SIZE, BIN_SIZE))[0]
                    # save the binned spike times
                    np.save(binned_file, [signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times])
                
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
        dms_signal_binned_err = np.std(dms_signal_binned, axis=0) / np.sqrt(str_count)
        dms_mvt_binned_mean = np.mean(dms_mvt_binned, axis=0)
        dms_mvt_binned_err = np.std(dms_mvt_binned, axis=0) / np.sqrt(str_count)
        dms_reward_binned_mean = np.mean(dms_reward_binned, axis=0)
        dms_reward_binned_err = np.std(dms_reward_binned, axis=0) / np.sqrt(str_count)

        # save the binned spike times
        np.save(signal_mvt_reward_file, [pfc_signal_binned_mean, pfc_signal_binned_err, pfc_mvt_binned_mean, pfc_mvt_binned_err, pfc_reward_binned_mean, pfc_reward_binned_err, dms_signal_binned_mean, dms_signal_binned_err, dms_mvt_binned_mean, dms_mvt_binned_err, dms_reward_binned_mean, dms_reward_binned_err])
    else:
        # load the binned spike times
        pfc_signal_binned_mean, pfc_signal_binned_err, pfc_mvt_binned_mean, pfc_mvt_binned_err, pfc_reward_binned_mean, pfc_reward_binned_err, dms_signal_binned_mean, dms_signal_binned_err, dms_mvt_binned_mean, dms_mvt_binned_err, dms_reward_binned_mean, dms_reward_binned_err = np.load(signal_mvt_reward_file)


    # plot the three binned spike times as line plots with error bars
    fig_pfc, ax_pfc = plt.subplots(1, 1, figsize=(12, 5))
    fig_dms, ax_dms = plt.subplots(1, 1, figsize=(12, 5))

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
    

def figure_3_panel_bc_mid():
    if not isfile(signal_mvt_reward_file):
        # print error message
        print('Error: ' + signal_mvt_reward_file + ' does not exist. Run figure_3_panel_bc() first.')
    
    # load the binned spike times
    pfc_signal_binned_mean, pfc_signal_binned_err, pfc_mvt_binned_mean, pfc_mvt_binned_err, pfc_reward_binned_mean, pfc_reward_binned_err, dms_signal_binned_mean, dms_signal_binned_err, dms_mvt_binned_mean, dms_mvt_binned_err, dms_reward_binned_mean, dms_reward_binned_err = np.load(signal_mvt_reward_file)

    pfc_mvt = pfc_mvt_binned_mean - pfc_signal_binned_mean
    pfc_reward = pfc_reward_binned_mean - pfc_mvt_binned_mean

    dms_mvt = dms_mvt_binned_mean - dms_signal_binned_mean
    dms_reward = dms_reward_binned_mean - dms_mvt_binned_mean

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

def figure_3_panel_bc_bottom():
    if not isfile(signal_mvt_reward_file):
        # print error message
        print('Error: ' + signal_mvt_reward_file + ' does not exist. Run figure_3_panel_bc() first.')
    
    # load the binned spike times
    pfc_signal_binned_mean, pfc_signal_binned_err, pfc_mvt_binned_mean, pfc_mvt_binned_err, pfc_reward_binned_mean, pfc_reward_binned_err, dms_signal_binned_mean, dms_signal_binned_err, dms_mvt_binned_mean, dms_mvt_binned_err, dms_reward_binned_mean, dms_reward_binned_err = np.load(signal_mvt_reward_file)

    pfc_mvt = pfc_mvt_binned_mean - pfc_signal_binned_mean
    pfc_reward = pfc_reward_binned_mean - pfc_mvt_binned_mean

    dms_mvt = dms_mvt_binned_mean - dms_signal_binned_mean
    dms_reward = dms_reward_binned_mean - dms_mvt_binned_mean

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
        signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times = np.load(pfc_file, allow_pickle=True)

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
    for dms_file in glob(pjoin(spike_time_dir, 'figure_3', f'*str*')):
        signal_spike_times, mvt_spike_times, reward_spike_times, total_spike_times = np.load(dms_file, allow_pickle=True)        

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
    fig_pfc, ax_pfc = plt.subplots(1, 3, figsize=(12, 3.5))
    fig_dms, ax_dms = plt.subplots(1, 3, figsize=(12, 3.5))

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

    ax_pfc[0].bar([0, 1, 2], [signal_signal_coeffs_mean, mvt_signal_coeffs_mean, reward_signal_coeffs_mean], yerr=[signal_signal_coeffs_err, mvt_signal_coeffs_err, reward_signal_coeffs_err], color='k')
    ax_pfc[0].set_xticks([0, 1, 2])
    ax_pfc[0].set_xticklabels(['S', 'SM', 'SMR'])
    ax_pfc[0].set_ylabel('signal coeffs')

    ax_pfc[1].bar([0, 1, 2], [signal_mvt_coeffs_mean, mvt_mvt_coeffs_mean, reward_mvt_coeffs_mean], yerr=[signal_mvt_coeffs_err, mvt_mvt_coeffs_err, reward_mvt_coeffs_err], color='k')
    ax_pfc[1].set_xticks([0, 1, 2])
    ax_pfc[1].set_xticklabels(['S', 'SM', 'SMR'])
    ax_pfc[1].set_ylabel('mvt coeffs')

    ax_pfc[2].bar([0, 1, 2], [signal_reward_coeffs_mean, mvt_reward_coeffs_mean, reward_reward_coeffs_mean], yerr=[signal_reward_coeffs_err, mvt_reward_coeffs_err, reward_reward_coeffs_err], color='k')
    ax_pfc[2].set_xticks([0, 1, 2])
    ax_pfc[2].set_xticklabels(['S', 'SM', 'SMR'])
    ax_pfc[2].set_ylabel('reward coeffs')

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
    
    ax_dms[0].bar([0, 1, 2], [signal_signal_coeffs_mean, mvt_signal_coeffs_mean, reward_signal_coeffs_mean], yerr=[signal_signal_coeffs_err, mvt_signal_coeffs_err, reward_signal_coeffs_err], color='k')
    ax_dms[0].set_xticks([0, 1, 2])
    ax_dms[0].set_xticklabels(['S', 'SM', 'SMR'])
    ax_dms[0].set_ylabel('signal coeffs')

    ax_dms[1].bar([0, 1, 2], [signal_mvt_coeffs_mean, mvt_mvt_coeffs_mean, reward_mvt_coeffs_mean], yerr=[signal_mvt_coeffs_err, mvt_mvt_coeffs_err, reward_mvt_coeffs_err], color='k')
    ax_dms[1].set_xticks([0, 1, 2])
    ax_dms[1].set_xticklabels(['S', 'SM', 'SMR'])
    ax_dms[1].set_ylabel('mvt coeffs')
    
    ax_dms[2].bar([0, 1, 2], [signal_reward_coeffs_mean, mvt_reward_coeffs_mean, reward_reward_coeffs_mean], yerr=[signal_reward_coeffs_err, mvt_reward_coeffs_err, reward_reward_coeffs_err], color='k')
    ax_dms[2].set_xticks([0, 1, 2])
    ax_dms[2].set_xticklabels(['S', 'SM', 'SMR'])
    ax_dms[2].set_ylabel('reward coeffs')

    fig_dms.suptitle('DMS')


    





    

