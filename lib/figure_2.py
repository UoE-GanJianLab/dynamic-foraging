from os.path import join as pjoin, basename, isdir, isfile
from os import mkdir
from glob import glob

import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns

from lib.calculation import moving_window_mean
from lib.figure_utils import remove_top_and_right_spines


behaviour_data_root = pjoin('data', 'behaviour_data')
figure_data_root = pjoin('figure_data', 'figure_1')
panel_c_data_root = pjoin(figure_data_root, 'panel_c')
for dir in [figure_data_root, panel_c_data_root]:
    if not isdir(dir):
        mkdir(dir)

wheel_velocity_data_root = pjoin('data', 'behaviour_data', 'wheel_velocity')
WHEEL_VELOCITY_SAMPLING_FEREQUENCY = 200
DOWNSAMPLING_FREQUENCY = 200
wheel_velocity_time_bin = 1 / DOWNSAMPLING_FREQUENCY

WINDOW_LEFT = -0.5
WINDOW_RIGHT = 1.5

# TODO correct figure_data path with poster figure number
def get_figure_2_panel_a():
    wheel_velocities_rewarded = []
    wheel_velocities_unrewarded = []

    relative_reward_time = []
    total_trial = 0

    # TODO downsample the wheel velocity signal

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

        for ind, cue in enumerate(cue_time):
            # covert the time in seconds to index in wheel velocity array
            cue_left = cue + WINDOW_LEFT
            cue_right = cue + WINDOW_RIGHT

            cue_left_ind = int(cue_left / (1/DOWNSAMPLING_FREQUENCY))
            cue_right_ind = int(cue_right / (1/DOWNSAMPLING_FREQUENCY))

            trial_wheel_velocity = wheel_velocity[cue_left_ind: cue_right_ind]
            
            if trial_reward[ind] == 1:
                wheel_velocities_rewarded.append(np.abs(trial_wheel_velocity))
                relative_reward_time.append(reward_time[ind]-cue)
            elif trial_reward[ind] == 0:
                wheel_velocities_unrewarded.append(np.abs(trial_wheel_velocity))

            total_trial += 1

    wheel_velocities_rewarded_sem = np.std(wheel_velocities_rewarded, axis=0) / np.sqrt(len(wheel_velocities_rewarded))
    wheel_velocities_unrewarded_sem = np.std(wheel_velocities_unrewarded, axis=0) / np.sqrt(len(wheel_velocities_unrewarded))
    wheel_velocities_rewarded = np.mean(wheel_velocities_rewarded, axis=0)
    wheel_velocities_unrewarded = np.mean(wheel_velocities_unrewarded, axis=0)
    

    time = np.arange(WINDOW_LEFT, WINDOW_RIGHT, 1/DOWNSAMPLING_FREQUENCY)
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    axes.plot(time, wheel_velocities_rewarded, label='rewarded', color='blue')
    axes.plot(time, wheel_velocities_unrewarded, label='unrewarded', color='red')
    axes.fill_between(x=time, y1=wheel_velocities_rewarded-wheel_velocities_rewarded_sem, y2=wheel_velocities_rewarded+wheel_velocities_rewarded_sem, color='blue', alpha=0.5)
    axes.fill_between(x=time, y1=wheel_velocities_unrewarded-wheel_velocities_unrewarded_sem, y2=wheel_velocities_unrewarded+wheel_velocities_unrewarded_sem, color='red', alpha=0.5)

    reward_axis = axes.twinx()
    reward_axis.set_ylabel('reward probability')

    # bin the reward times 
    reward_time_digitized = np.digitize(relative_reward_time, bins = np.arange(WINDOW_LEFT, WINDOW_RIGHT+1/DOWNSAMPLING_FREQUENCY, 1/DOWNSAMPLING_FREQUENCY), right=True)
    # count the number of entries in each bin
    reward_time_digitized = np.array([np.sum(reward_time_digitized==index) for index in np.arange(1, 2 * DOWNSAMPLING_FREQUENCY+1, dtype=int)])
    print(reward_time_digitized)
    reward_probability = reward_time_digitized / total_trial

    reward_axis.plot(time, reward_probability, color='black', label='reward probability')

    axes.set_xticks([-0.5, 0, 0.5, 1, 1.5])
    remove_top_and_right_spines(axes)
    plt.legend()
    plt.show()
    plt.close()

    # TODO find the panel and figure number and save the data
    figure_data = pd.DataFrame({'time': time, 'reward_probability': reward_probability, 'wheel_velocities_rewarded': wheel_velocities_rewarded, 'wheel_velocities_rewarded_sem': wheel_velocities_rewarded_sem, 'wheel_velocities_unrewarded': wheel_velocities_unrewarded, 'wheel_velocities_unrewarded_sem': wheel_velocities_unrewarded_sem})
    figure_data.to_csv(path_or_buf=pjoin(figure_data_root, 'figure_1_panel_b.csv'))
    

        
def get_figure_2_panel_c():
    for session in glob(pjoin(behaviour_data_root, '*.csv')):
        session_name = basename(session).split('.')[0]
        behaviour_data = pd.read_csv(session)
        leftP = np.array(behaviour_data['leftP'].values)
        rightP = np.array(behaviour_data['rightP'].values)
        trial_reward = np.array(behaviour_data['trial_reward'].values)
        trial_response_side = np.array(behaviour_data['trial_response_side'].values)
        high_reward_side = leftP < 0.5
        high_reward_side = np.array(high_reward_side, dtype=int)
        high_reward_side[high_reward_side==0] = -1
        chosen_high_reward_side = trial_response_side == high_reward_side

        left_response, right_response = trial_response_side == -1, trial_response_side == 1
        left_response_proportion, right_response_proportion = moving_window_mean(left_response, 20), moving_window_mean(right_response, 20)

        high_reward_proportion = moving_window_mean(chosen_high_reward_side, 20)
        reward_proportion = moving_window_mean(trial_reward == 1, 20)

        trial_indices = np.arange(len(leftP)) + 1

        # calculate the perceived reward probability
        prpd, perceived_left, perceived_right = get_prpd(session_name, trial_response_side, trial_reward)

        relative_values = np.load(pjoin('data', 'relative_values', session_name+'.npy'))
        
        figure_1_panel_c_data = pd.DataFrame({'trial_index': trial_indices, 'leftP': leftP, 'rightP': rightP, 'responses': trial_response_side, 'left_response_proportion': left_response_proportion, 'right_response_proportion': right_response_proportion, 'perceived_left': perceived_left, 'perceived_right': perceived_right, 'prpd': prpd, 'relative_values': relative_values, 'chosen_high_reward_side': chosen_high_reward_side, 'high_reward_proportion': high_reward_proportion, 'reward_proportion': reward_proportion})
        figure_1_panel_c_data.to_csv(pjoin(panel_c_data_root, session_name+'.csv'), index=False)


        # plot the results
        fig, axes = plt.subplots(4, 1, figsize=(10, 15))
        # plot set reward probabilities
        axes[0].plot(trial_indices, leftP, label='leftP')
        axes[0].plot(trial_indices, rightP, label='rightP')
        axes[0].set_xlabel('trials')
        axes[0].set_ylabel('reward probability')
        axes[0].legend()
        # plot the proportion of left and right responses
        axes[1].plot(trial_indices, left_response_proportion, label='left response proportion')
        axes[1].plot(trial_indices, right_response_proportion, label='right response proportion')
        axes[1].set_xlabel('trials')
        axes[1].set_ylabel('proportion')
        axes[1].legend()
        # plot perceived reward probabilities and prpd
        axes[2].plot(trial_indices, perceived_left, label='perceived left')
        axes[2].plot(trial_indices, perceived_right, label='perceived right')
        axes[2].plot(trial_indices, prpd, label='prpd')
        axes[2].set_xlabel('trials')
        axes[2].set_ylabel('reward probability')
        axes[2].legend()
        # plot the proportion of high reward side choices and rewarded trials
        axes[3].plot(trial_indices, high_reward_proportion, label='high reward proportion')
        axes[3].plot(trial_indices, reward_proportion, label='reward proportion')
        axes[3].set_xlabel('trials')
        axes[3].set_ylabel('proportion')
        axes[3].legend()
        fig.suptitle(session_name)
        plt.close(fig)


def get_figure_2_panel_d():
    # load the behaviour data from all sessions
    all_sessions = glob(pjoin(behaviour_data_root, '*.csv'))
    sorted_sessions = sorted(all_sessions)

    high_reward_percentage = []
    rewarded_percentage = []

    for session in sorted_sessions:
        session_name = basename(session).split('.')[0]
        behaviour_data = pd.read_csv(session)
        leftP = np.array(behaviour_data['leftP'].values)
        trial_reward = np.array(behaviour_data['trial_reward'].values)
        trial_response_side = np.array(behaviour_data['trial_response_side'].values)

        # find the indices of the trials where the high reward side switches
        switch_indices = find_switch(leftP)

        # remove the switches that do not have 50 trials before and after
        for switch in switch_indices:
            if switch < 50 or switch > len(leftP)-51:
                continue
            else:
                # get the percentage of rewarded trials and choices made to high reward side
                rewarded_percentage.append(trial_reward[switch-50:switch+51]==1)
                high_reward_side = leftP[switch-50:switch+51] < 0.5
                high_reward_side = np.array(high_reward_side, dtype=int)
                high_reward_side[high_reward_side==0] = -1
                high_reward_percentage.append(trial_response_side[switch-50:switch+51]==high_reward_side)
    
    # calculate the mean percentage of rewarded trials and choices made to high reward side
    rewarded_percentage = np.mean(np.array(rewarded_percentage), axis=0)
    high_reward_percentage = np.mean(np.array(high_reward_percentage), axis=0)

    # save the results as a csv files
    figure_1_panel_d_data = pd.DataFrame({'relative_trial_index': np.arange(-50, 51, dtype=int),'rewarded_percentage': rewarded_percentage, 'high_reward_percentage': high_reward_percentage})
    figure_1_panel_d_data.to_csv(pjoin(figure_data_root, 'figure_1_panel_d_data.csv'), index=False)


    # plot the results as two line plots in one figure
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(rewarded_percentage, label='rewarded trials')
    ax.plot(high_reward_percentage, label='high reward side')
    ax.set_xlabel('trials')
    ax.set_ylabel('percentage')
    ax.legend()
    plt.show()


# return the indices of the trials where the high reward side switches
def find_switch(leftP: np.ndarray) -> List[int]:
    switch_indices = []
    for i in range(len(leftP)-1):
        if leftP[i] != leftP[i+1]:
            switch_indices.append(i)
    return switch_indices

def get_prpd(session_name, trial_response_side, trial_reward):
    crainotomy_side = 'L' if session_name[:6] == "AKED01" else 'R' 
    perceived_left = []
    perceived_right = []
    prpd = []
    left_ptr = 0
    right_ptr = 0

    leftward_index = np.where(trial_response_side == -1)[0]
    rightward_index = np.where(trial_response_side == 1)[0]
    for response in trial_response_side:
        leftward_trials = []
        rightward_trials = []

        if left_ptr < 10:
            leftward_trials = leftward_index[:left_ptr]
        else:
            leftward_trials = leftward_index[left_ptr-10:left_ptr]
        if right_ptr < 10:
            rightward_trials = rightward_index[:right_ptr]
        else:
            rightward_trials = rightward_index[right_ptr-10:right_ptr]

        if len(leftward_trials) == 0:
            leftward_mean = 0
        else:
            # calculate the mean reward probability for the last 10 trials
            leftward_mean = np.mean(trial_reward[leftward_trials])
        if len(rightward_trials) == 0:
            rightward_mean = 0
        else:
            rightward_mean = np.mean(trial_reward[rightward_trials])

        perceived_left.append(leftward_mean)
        perceived_right.append(rightward_mean)
        if crainotomy_side == 'L':
            prpd.append(rightward_mean - leftward_mean)
        else:
            prpd.append(leftward_mean - rightward_mean)

        if response == -1:
            left_ptr += 1
        else:
            right_ptr += 1

    return prpd, perceived_left, perceived_right