from os.path import join as pjoin, basename, isdir, isfile
from os import mkdir
from glob import glob

import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns

from lib.calculation import moving_window_mean, get_relative_firing_rate_binned, get_mean_and_sem
from lib.figure_utils import remove_top_and_right_spines, plot_with_sem_error_bar


behaviour_data_root = pjoin('data', 'behaviour_data')
spike_firing_root = pjoin('data', 'spike_times', 'sessions')
figure_data_root = pjoin('figure_data', 'figure_1')
panel_c_data_root = pjoin(figure_data_root, 'panel_c')
for dir in [figure_data_root, panel_c_data_root]:
    if not isdir(dir):
        mkdir(dir)

wheel_velocity_data_root = pjoin('data', 'behaviour_data', 'wheel_velocity')
WHEEL_VELOCITY_SAMPLING_FEREQUENCY = 200
DOWNSAMPLING_FREQUENCY = 200
wheel_velocity_time_bin = 1 / DOWNSAMPLING_FREQUENCY

WINDOW_LEFT = 0
WINDOW_RIGHT = 3

WINDOW_LEFT_SPIKE = -0.5
WINDOW_RIGHT_SPIKE = 1.5

def get_figure_1_panel_b():
    # for all sessions, load the cue times
    for session in glob(pjoin(behaviour_data_root, '*.csv')):
        session_name = basename(session).split('.')[0]
        behaviour_data = pd.read_csv(session)
        cue_time = behaviour_data['cue_time'].values
        reward_time = behaviour_data['reward_time'].values
        trial_reward = behaviour_data['trial_reward'].values

        time = np.arange(WINDOW_LEFT, WINDOW_RIGHT, 1/DOWNSAMPLING_FREQUENCY)
        
        # AKED0120210715 does not have a wheel velocity file, thus omitted
        if isfile(pjoin(wheel_velocity_data_root, session_name+'.npy')):
            wheel_velocity, _ = np.load(pjoin(wheel_velocity_data_root, session_name+'.npy'))
        else:
            continue

        for ind, cue in enumerate(cue_time):
            # covert the time in seconds to index in wheel velocity array
            cue_left = cue + WINDOW_LEFT
            cue_right = cue + WINDOW_RIGHT

            cue_left_ind = int(cue_left / (1/DOWNSAMPLING_FREQUENCY))
            cue_right_ind = int(cue_right / (1/DOWNSAMPLING_FREQUENCY))

            trial_wheel_velocity = wheel_velocity[cue_left_ind: cue_right_ind]
            
            if trial_reward[ind] == 1:
                trial_reward_time = reward_time[ind]-cue
                
                # calculated the cumulative wheel movement 0-3s relative to the cue time
                cumulative_wheel_velocity = [np.sum(trial_wheel_velocity[:i]) * (1/DOWNSAMPLING_FREQUENCY) for i in np.arange(len(trial_wheel_velocity))]
                # plot the cumulative wheel velocity against wheel velocity
                fig, axes = plt.subplots(1, 1, figsize=(10, 5))
                axes.plot(time, cumulative_wheel_velocity, label='cumulative velocity')
                axes.plot(time, trial_wheel_velocity, label='real time velocity')
                # plt a tick at reward time
                plt.scatter(x=trial_reward_time, y=max(max(cumulative_wheel_velocity), max(trial_wheel_velocity)), marker='|', s=200, label='Reward Time', c='red')

                plt.legend()
                plt.show()
                plt.close()


def get_figure_1_panel_c():
    cue_aligned_firings_pfc = []
    movement_aligned_firings_pfc = []
    reward_aligned_firings_pfc = []

    cue_aligned_firings_dms = []
    movement_aligned_firings_dms = []
    reward_aligned_firings_dms = []

    time = np.arange(WINDOW_LEFT_SPIKE, WINDOW_RIGHT_SPIKE, 0.02)

    # for all sessions, load the cue times
    for session in glob(pjoin(behaviour_data_root, '*.csv')):
        session_name = basename(session).split('.')[0]
        behaviour_data = pd.read_csv(session)
        cue_time = behaviour_data['cue_time'].values
        reward_time = behaviour_data['reward_time'].values
        trial_reward = behaviour_data['trial_reward'].values
        
        # AKED0120210715 does not have a wheel velocity file, thus omitted
        if isfile(pjoin(wheel_velocity_data_root, session_name+'.npy')):
            wheel_velocity, _ = np.load(pjoin(wheel_velocity_data_root, session_name+'.npy'))
        else:
            continue

        movement_onsets = []
        reward_onsets = []

        for ind, cue in enumerate(cue_time):
            cue_left_ind = int(0 / (1/DOWNSAMPLING_FREQUENCY))
            cue_right_ind = int(7 / (1/DOWNSAMPLING_FREQUENCY))

            trial_wheel_velocity = wheel_velocity[cue_left_ind: cue_right_ind]
            if trial_reward[ind] == 1:
                reward_onsets.append(reward_time[ind]-cue)
            elif trial_reward[ind] == 0:
                movement_onset = find_movement_onset(trial_wheel_velocity)
                if movement_onset != -1:
                    movement_onsets.append(movement_onset)

        for pfc_firing in glob(pjoin(spike_firing_root, session_name, 'pfc_*')):
            firing_data = np.load(pfc_firing)

            cue_aligned_firings_pfc.append(get_relative_firing_rate_binned(cue_time, firing_data, WINDOW_LEFT_SPIKE, WINDOW_RIGHT_SPIKE, 0.02))
            movement_aligned_firings_pfc.append(get_relative_firing_rate_binned(movement_onsets, firing_data, WINDOW_LEFT_SPIKE, WINDOW_RIGHT_SPIKE, 0.02))
            reward_aligned_firings_pfc.append(get_relative_firing_rate_binned(reward_onsets, firing_data, WINDOW_LEFT_SPIKE, WINDOW_RIGHT_SPIKE, 0.02))

        for dms_firing in glob(pjoin(spike_firing_root, session_name, 'pfc_*')):
            firing_data = np.load(dms_firing)

            cue_aligned_firings_dms.append(get_relative_firing_rate_binned(cue_time, firing_data, WINDOW_LEFT_SPIKE, WINDOW_RIGHT_SPIKE, 0.02))
            movement_aligned_firings_dms.append(get_relative_firing_rate_binned(movement_onsets, firing_data, WINDOW_LEFT_SPIKE, WINDOW_RIGHT_SPIKE, 0.02))
            reward_aligned_firings_dms.append(get_relative_firing_rate_binned(reward_onsets, firing_data, WINDOW_LEFT_SPIKE, WINDOW_RIGHT_SPIKE, 0.02))

    cue_aligned_firings_pfc_mean, cue_aligned_firings_pfc_sem = get_mean_and_sem(cue_aligned_firings_pfc)
    movement_aligned_firings_pfc_mean, movement_aligned_firings_pfc_sem = get_mean_and_sem(movement_aligned_firings_pfc)
    reward_aligned_firings_pfc_mean, reward_aligned_firings_pfc_sem = get_mean_and_sem(reward_aligned_firings_pfc)

    cue_aligned_firings_dms_mean, cue_aligned_firings_dms_sem = get_mean_and_sem(cue_aligned_firings_dms)
    movement_aligned_firings_dms_mean, movement_aligned_firings_dms_sem = get_mean_and_sem(movement_aligned_firings_dms)
    reward_aligned_firings_dms_mean, reward_aligned_firings_dms_sem = get_mean_and_sem(reward_aligned_firings_dms)


    # plot the firing rates with different alignment
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    plot_with_sem_error_bar(axes[0], x=time, mean=cue_aligned_firings_pfc_mean, sem=cue_aligned_firings_pfc_sem, label='PFC', color='black')
    plot_with_sem_error_bar(axes[0], x=time, mean=cue_aligned_firings_dms_mean, sem=cue_aligned_firings_dms_sem, label='DMS', color='green')
    plot_with_sem_error_bar(axes[1], x=time, mean=movement_aligned_firings_pfc_mean, sem=movement_aligned_firings_pfc_sem, label='PFC', color='black')
    plot_with_sem_error_bar(axes[1], x=time, mean=movement_aligned_firings_dms_mean, sem=movement_aligned_firings_dms_sem, label='DMS', color='green')
    plot_with_sem_error_bar(axes[2], x=time, mean=reward_aligned_firings_pfc_mean, sem=reward_aligned_firings_pfc_sem, label='PFC', color='black')
    plot_with_sem_error_bar(axes[2], x=time, mean=reward_aligned_firings_dms_mean, sem=reward_aligned_firings_dms_sem, label='DMS', color='green')

                
            
def find_movement_onset(wheel_velocity):
    for i in wheel_velocity:
        if abs(i) > 0.2:
            return i * (1/DOWNSAMPLING_FREQUENCY) + 0
    return -1
