from os.path import join as pjoin
from os import listdir

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lib.calculation import moving_window_mean

# todo: refactor to take trial_response_side and trial_reward as input
# nan trials are already filled with 0 before feeding into this function
def panel_A(trial_response_side: np.ndarray, trial_reward: np.ndarray, leftP: np.ndarray, name: str):
    nan_trials = np.nonzero(trial_response_side == 0)
    left_rewarded = np.nonzero(trial_response_side == -1 & trial_reward == 1)
    right_rewarded = np.nonzero(trial_response_side == 1 & trial_reward == 1)
    left_unrewarded = np.nonzero(trial_response_side == -1 & trial_reward == 0)
    right_unrewarded = np.nonzero(trial_response_side == 1 & trial_reward == 0)

    # the indices of the trials
    indices = np.arange(trial_response_side.size)

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(name, fontsize=20)

    sns.scatterplot(x=right_rewarded, y=1.1, marker='|', color='royalblue', ax=axes[0], s=100)
    sns.scatterplot(x=left_rewarded, y=-1.1, marker='|', color='royalblue', ax=axes[0], s=100, label='rewarded')
    sns.scatterplot(x=right_unrewarded, y=1.2, marker='|', color='deeppink', ax=axes[0], s=100, label='no_reward')
    sns.scatterplot(x=left_unrewarded, y=-1.2, marker='|', color='deeppink', ax=axes[0], s=100)
    sns.scatterplot(x=nan_trials, y=-1.15, marker='|', color='black', ax=axes[0], s=100, label='nan_trials')
    sns.scatterplot(x=nan_trials, y=1.15, marker='|', color='black', ax=axes[0], s=100)

    responses = moving_window_mean(trial_response_side, 10)
    sns.lineplot(x=np.arange(trial_response_side.size), y=responses, ax=axes[0], label='choices')

    left_choices = np.nonzero(trial_response_side == 1)
    right_choices = np.nonzero(trial_response_side == -1)
    left_c = np.zeros(len(trial_response_side))
    left_c[left_choices] = 1
    left_c = np.convolve(left_c, np.ones(10)/10, mode='same')
    right_c = np.zeros(len(trial_response_side))
    right_c[right_choices] = 1
    right_c = np.convolve(right_c, np.ones(10)/10, mode='same')
    sns.lineplot(x=indices, y=right_c, color='royalblue', label='choose right', ax=axes[1])
    sns.lineplot(x=indices, y=left_c, color='deeppink', label='choose left', ax=axes[1])

    # sns.scatterplot(x=left_choices, y=1.1, marker='o', color='deeppink', ax=axes[1], s=10, label='left_choices')
    # sns.scatterplot(x=nan_trials, y=1.1, marker='o', color='black', ax=axes[1], s=10, label='nan_trials')
    # sns.scatterplot(x=right_choices, y=1.1, marker='o', color='royalblue', ax=axes[1], s=10, label='right_choices')

    # right p + left p = 1
    rightP = 1 - leftP
    # probability is the maximum of rightP and leftP, and negate the leftP if leftP > rightP
    prob = np.maximum(rightP, leftP)
    prob[leftP > rightP] = -leftP[leftP > rightP]
    sns.lineplot(x=indices, y=prob, ax=axes[0], label='reward probability')

    axes[0].legend(bbox_to_anchor=(1.15, 0.7))
    axes[1].legend(bbox_to_anchor=(1.15, 0.7))

    # plt.savefig(pjoin('figure_4', 'panel_A',name + '.tiff'), dpi=300)
