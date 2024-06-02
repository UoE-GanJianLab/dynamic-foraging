from os.path import join as pjoin, basename, isdir, isfile
from os import mkdir, listdir
from glob import glob

import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from lib.calculation import moving_window_mean, get_relative_spike_times, get_relative_firing_rate_binned, get_mean_and_sem, get_firing_rate_window
from lib.figure_utils import remove_top_and_right_spines, plot_with_sem_error_bar
from lib.figure_1 import find_switch

# figures not in the current draft

behaviour_data_root = pjoin('data', 'behaviour_data')
spike_firing_root = pjoin('data', 'spike_times', 'sessions')
figure_data_root = pjoin('figure_data', 'figure_1')
panel_c_data_root = pjoin(figure_data_root, 'panel_c')
for dir in [figure_data_root, panel_c_data_root]:
    if not isdir(dir):
        mkdir(dir)

prpd_modulation_data = pjoin('data', 'prpd_correlation.csv')
prpd_modulation_data = pd.read_csv(prpd_modulation_data)

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


# poster panel a,b of figure 3
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

def prpd_firing_rate_vs_advantageous_proportion(type='ALL'):
    advantageous_proportion = []
    normalized_prpd_firing_rate_response_mag = []

    # iterate through the sessions and find the prpd modulated neurons
    for session_name in tqdm.tqdm(listdir(behaviour_data_root)):
        if isdir(pjoin(behaviour_data_root, session_name)):
            continue

        # load the behaviour data
        behaviour_data = pd.read_csv(pjoin(behaviour_data_root, session_name))

        session_name = session_name.split('.')[0]

        # get the switches
        switches = find_switch(behaviour_data['leftP'].values)

        # calculate the percentage of advantageous trials in 20 trials after each switch
        for switch in switches:
            # if there are less than 20 trials after the switch, skip
            if switch + 20 > len(behaviour_data):
                continue


            if behaviour_data.iloc[switch+1]['leftP'] > behaviour_data.iloc[switch+1]['rightP']:
                advantageous_proportion.append(moving_window_mean(behaviour_data['trial_response_side'].values[switch:switch+20] == 1))
            else:
                advantageous_proportion.append(moving_window_mean(behaviour_data['trial_response_side'].values[switch:switch+20] == -1))
            
            if type == 'PFC':
                cells = glob(pjoin(spike_firing_root, session_name, 'pfc_*'))
            elif type == 'DMS':
                cells = glob(pjoin(spike_firing_root, session_name, 'dms_*'))
            else:
                cells = glob(pjoin(spike_firing_root, session_name, 'pfc_*')) + glob(pjoin(spike_firing_root, session_name, 'dms_*'))
            # load the pfc firing data in the sessions
            for cell_firing in cells:
                cell_name = basename(cell_firing).split('.')[0]

                cell_firing = np.load(cell_firing)
                relative_firing_rate = get_firing_rate_window(cue_times=behaviour_data['cue_time'].values, spike_times=cell_firing, window_left=0, window_right=1.5)
                # normalized the firing rate
                if np.max(relative_firing_rate) != 0:
                    relative_firing_rate = relative_firing_rate / np.max(relative_firing_rate)
                else:
                    continue

                # check if the neuron is prpd modulated
                prpd_modulation = prpd_modulation_data[(prpd_modulation_data['session'] == session_name) & (prpd_modulation_data['cell'] == cell_name)]

                # if cell does not have prpd modulation data, skip
                if prpd_modulation.empty:
                    continue

                if prpd_modulation['response_firing_p_values'].values[0] > 0.05:
                    continue

                # get the firing rate in the current switch
                relative_firing_rate = relative_firing_rate[switch:switch+20]

                if prpd_modulation['response_firing_pearson_r'].values[0] < 0:
                    normalized_prpd_firing_rate_response_mag.append(relative_firing_rate)
                else:
                    # sign flip the negatively modulated neurons
                    normalized_prpd_firing_rate_response_mag.append(1-relative_firing_rate)

    advantageous_proportion = np.array(advantageous_proportion)
    normalized_prpd_firing_rate_response_mag = np.array(normalized_prpd_firing_rate_response_mag)

    print(advantageous_proportion.shape)
    print(normalized_prpd_firing_rate_response_mag.shape)

    # get the mean and sem of the firing rate and advantageous proportion
    mean_firing_rate, sem_firing_rate = get_mean_and_sem(normalized_prpd_firing_rate_response_mag)
    mean_advantageous_proportion, sem_advantageous_proportion = get_mean_and_sem(advantageous_proportion)

    # save the data as csv
    data = pd.DataFrame({
        'firing_rate': mean_firing_rate,
        'sem_firing_rate': sem_firing_rate,
        'advantageous_proportion': mean_advantageous_proportion,
        'sem_advantageous_proportion': sem_advantageous_proportion
    })

    if type == 'PFC':
        data.to_csv(pjoin(panel_c_data_root, 'pfc.csv'))
    elif type == 'DMS':
        data.to_csv(pjoin(panel_c_data_root, 'dms.csv'))
    else:
        data.to_csv(pjoin(panel_c_data_root, 'all.csv'))


    # plot the firing rate against the advantageous proportion
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    axes_1 = axes.twinx()
    plot_with_sem_error_bar(axes, x=range(20), mean=mean_firing_rate, sem=sem_firing_rate, label='Firing Rate', color='black')
    plot_with_sem_error_bar(axes_1, x=range(20), mean=mean_advantageous_proportion, sem=sem_advantageous_proportion, label='Advantageous Proportion', color='red')
    # merge the legend
    axes.legend(loc='upper left')
    axes_1.legend(loc='upper right')
    
    plt.show()
