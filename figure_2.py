from os.path import join as pjoin
from os import listdir

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def panel_A(behaviour_path: str, name):
    df = pd.read_csv(behaviour_path)
    # preprocessing data
    # fill nan values with 0s
    indices = range(df.shape[0])

    df = df.fillna(0)
    nan_trials = df[df['trial_response_side'] == 0].index.tolist()
    left_rewarded = df[(df['trial_response_side'] == -1) & (df['trial_reward'] == 1)].index.tolist()
    right_rewarded = df[(df['trial_response_side'] == 1) & (df['trial_reward'] == 1)].index.tolist()
    left_unrewarded = df[(df['trial_response_side'] == -1) & (df['trial_reward'] == 0)].index.tolist()
    right_unrewarded = df[(df['trial_response_side'] == -1) & (df['trial_reward'] == 0)].index.tolist()

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(name, fontsize=20)

    sns.scatterplot(x=right_rewarded, y=1.1, marker='|', color='royalblue', ax=axes[0], s=100)
    sns.scatterplot(x=left_rewarded, y=-1.1, marker='|', color='royalblue', ax=axes[0], s=100, label='rewarded')
    sns.scatterplot(x=right_unrewarded, y=1.2, marker='|', color='deeppink', ax=axes[0], s=100, label='no_reward')
    sns.scatterplot(x=left_unrewarded, y=-1.2, marker='|', color='deeppink', ax=axes[0], s=100)
    sns.scatterplot(x=nan_trials, y=-1.15, marker='|', color='black', ax=axes[0], s=100, label='nan_trials')
    sns.scatterplot(x=nan_trials, y=1.15, marker='|', color='black', ax=axes[0], s=100)

    responses = np.convolve(df['trial_response_side'].to_numpy(), np.ones(10)/10, mode='same')
    sns.lineplot(x=indices, y=responses.reshape(1, -1)[0], ax=axes[0], label='choices')

    left_choices = df[df['trial_response_side'] == 1].index.tolist()
    right_choices = df[df['trial_response_side'] == -1].index.tolist()
    left_c = np.zeros(len(df))
    left_c[left_choices] = 1
    left_c = np.convolve(left_c, np.ones(10)/10, mode='same')
    right_c = np.zeros(len(df))
    right_c[right_choices] = 1
    right_c = np.convolve(right_c, np.ones(10)/10, mode='same')
    sns.lineplot(x=indices, y=right_c, color='royalblue', label='choose right', ax=axes[1])
    sns.lineplot(x=indices, y=left_c, color='deeppink', label='choose left', ax=axes[1])

    # sns.scatterplot(x=left_choices, y=1.1, marker='o', color='deeppink', ax=axes[1], s=10, label='left_choices')
    # sns.scatterplot(x=nan_trials, y=1.1, marker='o', color='black', ax=axes[1], s=10, label='nan_trials')
    # sns.scatterplot(x=right_choices, y=1.1, marker='o', color='royalblue', ax=axes[1], s=10, label='right_choices')

    df['p'] = df[['leftP', 'rightP']].max(axis=1)
    left = df[df['leftP'] > df['rightP']].index.tolist()
    df.iloc[left] = df.iloc[left].mul(-1)
    sns.lineplot(x=indices, y=df['p'], ax=axes[0], label='reward')

    axes[0].legend(bbox_to_anchor=(1.15, 0.7))
    axes[1].legend(bbox_to_anchor=(1.15, 0.7))

    # plt.savefig(pjoin('figure_4', 'panel_A',name + '.tiff'), dpi=300)
