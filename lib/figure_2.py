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
figure_data_root = pjoin('figure_data', 'figure_2')
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

# wheel velocity plot smoothened by a moving window of 5
def get_figure_2_panel_b():
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
                # smoothen the wheel velocity before plotting
                axes.plot(time, moving_window_mean(trial_wheel_velocity, 5), label='real time velocity')
                # plt a tick at reward time
                plt.scatter(x=trial_reward_time, y=max(max(cumulative_wheel_velocity), max(trial_wheel_velocity)), marker='|', s=200, label='Reward Time', c='red')

                plt.legend()
                plt.show()
                plt.close()
