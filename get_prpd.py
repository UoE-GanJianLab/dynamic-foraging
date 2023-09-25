# calculate the perceived reward probability difference for each session and save it into data/pdrp folder
# it is the difference between the prior 10 trials' reward probability on the left and right side
from os.path import join as pjoin, basename, exists
from os import makedirs
import numpy as np
import pandas as pd
from glob import glob

# load the data from the sessions folder
sessions_folder = "data/behaviour_data"
sessions_data = []
prpd_folder = "data/prpd"
if not exists(prpd_folder):
    makedirs(prpd_folder)
for behaviour_file in glob(pjoin(sessions_folder, "*.csv")):
    craniotomy_side = "L"
    session_data = pd.read_csv(behaviour_file)
    session_name = basename(behaviour_file).split(".")[0]

    if session_name[:6] == "AKED01":
        craniotomy_side = "R"
    # fill the nan with 0
    session_data = session_data.fillna(0)
    trial_response_side = session_data['trial_response_side']
    trial_reward = np.array(session_data['trial_reward'])

    leftward_index = np.where(trial_response_side == -1)[0]
    rightward_index = np.where(trial_response_side == 1)[0]

    left_ptr = 0
    right_ptr = 0

    prpd = []

    for ind, row in session_data.iterrows():
        # get the last 10 leftward trials, and the last 10 rightward trials
        # do not use padding if the trial is less than 10
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
        if craniotomy_side == "R":
            prpd.append(leftward_mean - rightward_mean)
        else:
            prpd.append(rightward_mean - leftward_mean)

        if row['trial_response_side'] == -1:
            left_ptr += 1
        else:
            right_ptr += 1

    # save the  pdrp into the pdrp folder
    np.save(pjoin(prpd_folder, session_name+'.npy'), prpd)
