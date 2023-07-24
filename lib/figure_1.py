from os.path import join as pjoin, basename, isdir
from os import mkdir
from glob import glob

import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt


behaviour_data_path = pjoin('data', 'behaviour_data')
figure_data_root = pjoin('figure_data', 'figure_1')
if not isdir(figure_data_root):
    mkdir(figure_data_root)


def get_figure_1_panel_d():
    # load the behaviour data from all sessions
    all_sessions = glob(pjoin(behaviour_data_path, '*.csv'))
    sorted_sessions = sorted(all_sessions)

    high_reward_percentage = []
    rewarded_percentage = []

    for session in sorted_sessions:
        session_name = basename(session).split('.')[0]
        print(session_name)
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