import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind, spearmanr
from scipy.signal import correlate

from lib.calculation import moving_window_mean, get_firing_rate_window, moving_window_mean_prior, get_relative_spike_times, get_normalized_cross_correlation, crosscorrelation

# using firing during intertrial interval (ITI) window -1 to -0.5ms
# reset means whether to recalculate the values disregarding the saved files
def figure_6_panel_c(pfc_times: np.ndarray, str_times: np.ndarray, cue_times: np.ndarray, pfc_name: str, str_name: str, rewarded: np.ndarray, session_name: str, mono: bool = False, reset: bool = False):
    # if the relative spike time file for current session and cell already exists, load it
    if os.path.exists(f'/data/relative_spike_time_trials/{session_name}_{pfc_name}.npy'):
        pfc_relative_spike_times = np.load(f'/data/relative_spike_time_trials/{session_name}_{pfc_name}.npy')
    else:
        pfc_relative_spike_times = get_relative_spike_times(pfc_times, cue_times, -1, -0.5)
        np.save(f'/data/relative_spike_time_trials/{session_name}_{pfc_name}.npy', pfc_relative_spike_times)
    
    if os.path.exists(f'/data/relative_spike_time_trials/{session_name}_{str_name}.npy'):
        str_relative_spike_times = np.load(f'/data/relative_spike_time_trials/{session_name}_{str_name}.npy')
    else:
        str_relative_spike_times = get_relative_spike_times(str_times, cue_times, -1, -0.5)
        np.save(f'/data/relative_spike_time_trials/{session_name}_{str_name}.npy', str_relative_spike_times)

    # calculate the cross correlation
    cross_cors = []
    for i in range(len(cue_times)):
        pfc_trial_times = pfc_relative_spike_times[i]
        str_trial_times = str_relative_spike_times[i]

        # if any of the array is empty, append 0
        if len(pfc_trial_times) == 0 or len(str_trial_times) == 0:
            cross_cors.append(0)
            continue
        # if the binnning file for current trial already exists, load it
        if os.path.exists(f'/data/inter_trial_binned/10ms_{session_name}_{pfc_name}_{i}.npy'):
            pfc_trial_times = np.load(f'/data/cross_correlation/10ms_{session_name}_{pfc_name}_{i}.npy')
        else:
            # binning with bin size of 10ms using histogram
            pfc_trial_times = np.histogram(pfc_trial_times, bins=np.arange(-1, -0.5, 0.01))[0]
            np.save(f'/data/cross_correlation/10ms_{session_name}_{pfc_name}_{i}.npy', pfc_trial_times)
        if os.path.exists(f'/data/inter_trial_binned/10ms_{session_name}_{str_name}_{i}.npy'):
            str_trial_times = np.load(f'/data/cross_correlation/10ms_{session_name}_{str_name}_{i}.npy')
        else:
            str_trial_times = np.histogram(str_trial_times, bins=np.arange(-1, -0.5, 0.01))[0]
            np.save(f'/data/cross_correlation/10ms_{session_name}_{str_name}_{i}.npy', str_trial_times)

        normalized_cross_corr = get_normalized_cross_correlation(pfc_trial_times, str_trial_times, 100)

        # append the absolute maximum value of the cross correlation
        cross_cors.append(np.max(np.abs(normalized_cross_corr)))

    # smoothen the cross correlation
    cross_cors = moving_window_mean(np.array(cross_cors), 20)

    # reward proportion is the proportion of rewarded trials in the previous 10 trials
    reward_proportion = moving_window_mean_prior(rewarded, 10)

    # plot reward proportion vs cross correlation in twinx plot
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
    ax1.plot(reward_proportion, color='tab:blue')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Reward proportion', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.plot(cross_cors, color='tab:red')
    ax2.set_ylabel('Cross correlation', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # calculate pearson r and the p value, set it as figure title
    r, p = pearsonr(reward_proportion, cross_cors)
    fig.suptitle(f'Pearson r: {r:.2f}, p: {p:.2f}, {pfc_name} vs {str_name}')

    # normalize cross correlation to 0-1, avoid divide by 0 error
    if np.max(cross_cors) - np.min(cross_cors) == 0:
        cross_cors = np.zeros(len(cross_cors))
    else:
        cross_cors = (cross_cors - np.min(cross_cors)) / (np.max(cross_cors) - np.min(cross_cors))

    # calculate the overall cross correlation
    overall_cross_cor = crosscorrelation(cross_cors, reward_proportion, maxlag=50)

    # if the figures directory does not exist, create it
    if not mono:
        if not os.path.exists('figures/figure_6/panel_c'):
            os.makedirs('figures/figure_6/panel_c')
        if not os.path.exists('figures/figure_6/significant'):
            os.makedirs('figures/figure_6/significant')
        
        # if not os.path.exists('figures/figure_6/panel_d'):
        #     os.makedirs('figures/figure_6/panel_d')
    else:
        if not os.path.exists('figures/figure_6/panel_c'):
            os.makedirs('figures/figure_6/panel_c')
        if not os.path.exists('figures/figure_6/significant'):
            os.makedirs('figures/figure_6/significant')
        # if not os.path.exists('mono_figures/figure_6/panel_d'):
        #     os.makedirs('mono_figures/figure_6/panel_d')

    # save the figures
    if not mono:
        fig.savefig(f'figures/figure_6/panel_c/6c_{session_name}_{pfc_name}_{str_name}_cross_correlation.png')
        # fig_overall.savefig(f'figures/figure_6/panel_d/6d_{session_name}_{pfc_name}_{str_name}_overall_cross_correlation.png')
    else:
        fig.savefig(f'figures/figure_6/panel_c/6c_{session_name}_{pfc_name}_{str_name}_cross_correlation_mono.png')
        # fig_overall.savefig(f'mono_figures/figure_6/panel_d/6d_{session_name}_{pfc_name}_{str_name}_overall_cross_correlation_mono.png')

    if p < 0.001:
        if not mono:
            # save the figures in significant folder
            fig.savefig(f'figures/figure_6/significant/6c_{session_name}_{pfc_name}_{str_name}_cross_correlation_significant.png')
            # fig_overall.savefig(f'figures/figure_6/significant/6d_{session_name}_{pfc_name}_{str_name}_overall_cross_correlation_significant.png')
        else:
            # save the figures in significant folder
            fig.savefig(f'mono_figures/figure_6/significant/6c_{session_name}_{pfc_name}_{str_name}_cross_correlation_significant_mono.png')
            # fig_overall.savefig(f'mono_figures/figure_6/significant/6d_{session_name}_{pfc_name}_{str_name}_overall_cross_correlation_significant_mono.png')

    # close the figures
    plt.close(fig)
    # plt.close(fig_overall)

    return fig


# using firing during 1-3ms 
# may be removed in the paper
def figure_6_panel_e(pfc_times: np.ndarray, str_times: np.ndarray, cue_times: np.ndarray, pfc_name: str, str_name: str, rewarded: np.ndarray, session_name: str, mono: bool = False):
    # if the relative spike time file for current session and cell already exists, load it
    if os.path.exists(f'/data/relative_spike_time_trials/{session_name}_{pfc_name}.npy'):
        pfc_relative_spike_times = np.load(f'/data/relative_spike_time_trials/{session_name}_{pfc_name}.npy')
    else:
        pfc_relative_spike_times = get_relative_spike_times(pfc_times, cue_times, -1, -0.5)
        np.save(f'/data/relative_spike_time_trials/{session_name}_{pfc_name}.npy', pfc_relative_spike_times)
    
    if os.path.exists(f'/data/relative_spike_time_trials/{session_name}_{str_name}.npy'):
        str_relative_spike_times = np.load(f'/data/relative_spike_time_trials/{session_name}_{str_name}.npy')
    else:
        str_relative_spike_times = get_relative_spike_times(str_times, cue_times, -1, -0.5)
        np.save(f'/data/relative_spike_time_trials/{session_name}_{str_name}.npy', str_relative_spike_times)

    # calculate the cross correlation
    cross_cors = []
    for i in range(len(cue_times)):
        pfc_trial_times = pfc_relative_spike_times[i]
        str_trial_times = str_relative_spike_times[i]

        # if any of the array is empty, append 0
        if len(pfc_trial_times) == 0 or len(str_trial_times) == 0:
            cross_cors.append(0)
            continue
        # if the binnning file for current trial already exists, load it
        if os.path.exists(f'/data/inter_trial_binned/1ms_{session_name}_{pfc_name}_{i}.npy'):
            pfc_trial_times = np.load(f'/data/cross_correlation/1ms_{session_name}_{pfc_name}_{i}.npy')
        else:
            # binning with bin size of 10ms using histogram
            pfc_trial_times = np.histogram(pfc_trial_times, bins=np.arange(-1, -0.5, 0.01))[0]
            np.save(f'/data/cross_correlation/1ms_{session_name}_{pfc_name}_{i}.npy', pfc_trial_times)
        if os.path.exists(f'/data/inter_trial_binned/1ms_{session_name}_{str_name}_{i}.npy'):
            str_trial_times = np.load(f'/data/cross_correlation/1ms_{session_name}_{str_name}_{i}.npy')
        else:
            str_trial_times = np.histogram(str_trial_times, bins=np.arange(-1, -0.5, 0.01))[0]
            np.save(f'/data/cross_correlation/1ms_{session_name}_{str_name}_{i}.npy', str_trial_times)

        normalized_cross_corr = get_normalized_cross_correlation(pfc_trial_times, str_trial_times, 25)

        # append the absolute maximum value of the cross correlation
        cross_cors.append(np.max(np.mean(normalized_cross_corr[26:28])))

    # smoothen the cross correlation
    cross_cors = moving_window_mean(np.array(cross_cors), 20)

    # reward proportion is the proportion of rewarded trials in the previous 10 trials
    reward_proportion = moving_window_mean_prior(rewarded, 10)

    # plot reward proportion vs cross correlation in twinx plot
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
    ax1.plot(reward_proportion, color='tab:blue')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Reward proportion', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.plot(cross_cors, color='tab:red')
    ax2.set_ylabel('Cross correlation', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # calculate pearson r and the p value, set it as figure title
    r, p = pearsonr(reward_proportion, cross_cors)
    fig.suptitle(f'Pearson r: {r:.2f}, p: {p:.2f}, {pfc_name} vs {str_name}')

    # normalize cross correlation to 0-1, avoid divide by 0 error
    if np.max(cross_cors) - np.min(cross_cors) == 0:
        cross_cors = np.zeros(len(cross_cors))
    else:
        cross_cors = (cross_cors - np.min(cross_cors)) / (np.max(cross_cors) - np.min(cross_cors))

    # calculate the overall cross correlation
    overall_cross_cor = crosscorrelation(cross_cors, reward_proportion, maxlag=50)

    # if the figures directory does not exist, create it
    if not mono:
        if not os.path.exists('figures/figure_6/panel_c'):
            os.makedirs('figures/figure_6/panel_c')
        if not os.path.exists('figures/figure_6/significant'):
            os.makedirs('figures/figure_6/significant')
        
        # if not os.path.exists('figures/figure_6/panel_d'):
        #     os.makedirs('figures/figure_6/panel_d')
    else:
        if not os.path.exists('figures/figure_6/panel_c'):
            os.makedirs('figures/figure_6/panel_c')
        if not os.path.exists('figures/figure_6/significant'):
            os.makedirs('figures/figure_6/significant')
        # if not os.path.exists('mono_figures/figure_6/panel_d'):
        #     os.makedirs('mono_figures/figure_6/panel_d')

    # save the figures
    if not mono:
        fig.savefig(f'figures/figure_6/panel_c/6c_{session_name}_{pfc_name}_{str_name}_cross_correlation.png')
        # fig_overall.savefig(f'figures/figure_6/panel_d/6d_{session_name}_{pfc_name}_{str_name}_overall_cross_correlation.png')
    else:
        fig.savefig(f'figures/figure_6/panel_c/6c_{session_name}_{pfc_name}_{str_name}_cross_correlation_mono.png')
        # fig_overall.savefig(f'mono_figures/figure_6/panel_d/6d_{session_name}_{pfc_name}_{str_name}_overall_cross_correlation_mono.png')

    if p < 0.001:
        if not mono:
            # save the figures in significant folder
            fig.savefig(f'figures/figure_6/significant/6c_{session_name}_{pfc_name}_{str_name}_cross_correlation_significant.png')
            # fig_overall.savefig(f'figures/figure_6/significant/6d_{session_name}_{pfc_name}_{str_name}_overall_cross_correlation_significant.png')
        else:
            # save the figures in significant folder
            fig.savefig(f'mono_figures/figure_6/significant/6c_{session_name}_{pfc_name}_{str_name}_cross_correlation_significant_mono.png')
            # fig_overall.savefig(f'mono_figures/figure_6/significant/6d_{session_name}_{pfc_name}_{str_name}_overall_cross_correlation_significant_mono.png')

    # close the figures
    plt.close(fig)
    # plt.close(fig_overall)

    return fig

def figure_6_poster_panel_c(pfc_times: np.ndarray, str_times: np.ndarray, cue_times: np.ndarray, pfc_name: str, str_name: str, rewarded: np.ndarray, session_name: str, mono: bool = False):
    pfc_relative_spike_times = get_relative_spike_times(pfc_times, cue_times, -1, -0.5)
    str_relative_spike_times = get_relative_spike_times(str_times, cue_times, -1, -0.5)

    # calculate the cross correlation
    cross_cors = []
    for i in range(len(cue_times)):
        pfc_trial_times = pfc_relative_spike_times[i]
        str_trial_times = str_relative_spike_times[i]

        # if any of the array is empty, append 0
        if len(pfc_trial_times) == 0 or len(str_trial_times) == 0:
            cross_cors.append(0)
            continue

        # binning with bin size of 10ms using histogram
        pfc_trial_times = np.histogram(pfc_trial_times, bins=np.arange(-1, -0.5, 0.01))[0]
        str_trial_times = np.histogram(str_trial_times, bins=np.arange(-1, -0.5, 0.01))[0]

        normalized_cross_corr = get_normalized_cross_correlation(pfc_trial_times, str_trial_times, 50)

        # append the absolute maximum value of the cross correlation
        cross_cors.append(np.max(np.abs(normalized_cross_corr)))


    # smoothen the cross correlation
    # cross_cors = moving_window_mean(np.array(cross_cors), 20)

    # reward proportion is the proportion of rewarded trials in the previous 10 trials
    reward_proportion = moving_window_mean_prior(rewarded, 10)

    discretized_reward_proportion = np.digitize(reward_proportion, bins=np.arange(0, 1, 0.2))
    discretized_reward_proportion = discretized_reward_proportion * 0.2 - 0.1

    # plot reward proportion vs cross correlation in twinx plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    
    # plot cross_cors against reward_proportion
    sns.lineplot(x=discretized_reward_proportion, y=cross_cors, ax=ax, color='tab:blue', err_style='bars')
    # set x axis tick label 
    ax.set_xticks(np.arange(0, 1, 0.2))

    # calculate pearson r and the p value, set it as figure title
    r, p = pearsonr(reward_proportion, cross_cors)
    # calculate spearman rank correlation and the p value
    sr, sp = spearmanr(reward_proportion, cross_cors)
    fig.suptitle(f'Pearson r: {r:.2f}, p: {p:.2f}, {pfc_name} vs {str_name}')

    # if the figures directory does not exist, create it
    if not mono:
        if not os.path.exists('figures/figure_6/poster_panel_c'):
            os.makedirs('figures/figure_6/poster_panel_c')
        if not os.path.exists('figures/figure_6/significant'):
            os.makedirs('figures/figure_6/significant')
    else:
        if not os.path.exists('figures/figure_6/poster_panel_c'):
            os.makedirs('figures/figure_6/poster_panel_c')
        if not os.path.exists('figures /figure_6/significant'):
            os.makedirs('figures/figure_6/significant')

    # save the figures
    if not mono:
        fig.savefig(f'figures/figure_6/poster_panel_c/poster_6c_{session_name}_{pfc_name}_{str_name}_cross_correlation.png')
    else:
        fig.savefig(f'figures/figure_6/poster_panel_c/poster_6c_{session_name}_{pfc_name}_{str_name}_cross_correlation_mono.png')

    # close the figures
    plt.close(fig)

    return cross_cors, reward_proportion, p, r, sp, sr


def figure_6_poster_panel_d(rs: np.ndarray, ps: np.ndarray, mono=False, spearman=False):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))

    sig_rs_positive_percentages = []
    sig_rs_negative_percentages = []

    sig_size = 0

    for i in range(len(rs)):
        # get the significant rs and ps
        sig_rs = rs[i][ps[i]<0.01]
        sig_size += len(sig_rs)

        sig_rs_positive = sig_rs[sig_rs>0]
        sig_rs_negative = sig_rs[sig_rs<0]
        
        print(f'{i}th session, positive: {len(sig_rs_positive)}, negative: {len(sig_rs_negative)}')

        # calculate the percentage of positive and negative significant rs
        sig_rs_positive_percentage = len(sig_rs_positive)/len(rs[i])
        sig_rs_negative_percentage = len(sig_rs_negative)/len(rs[i])

        # append the percentage to the list
        sig_rs_positive_percentages.append(sig_rs_positive_percentage)
        sig_rs_negative_percentages.append(sig_rs_negative_percentage)

    print(f'num of significant rs: {sig_size}')

    # t test to see if the percentage of positive and negative significant rs are different
    t, p = ttest_ind(sig_rs_positive_percentages, sig_rs_negative_percentages, alternative='less')
    print(f't: {t}, p: {p}')

    # plot the bar plot with the average percentage of positive and negative significant rs
    sns.barplot(x=['+', '-'], y=[np.mean(sig_rs_positive_percentages), np.mean(sig_rs_negative_percentages)], ax=axes)
    axes.set_ylim(0, 1)

    if not mono:
        # if the figures directory does not exist, create it
        if not os.path.exists('figures/figure_6'):
            os.makedirs('figures/figure_6')
        if not spearman:
            # save the figures
            fig.savefig(f'figures/figure_6/poster_6d.png')
        else:
            fig.savefig(f'figures/figure_6/poster_6d_spearman.png')
    else:
        # if the figures directory does not exist, create it
        if not os.path.exists('figures/figure_6'):
            os.makedirs('figures/figure_6')
        # save the figures
        if not spearman:
            fig.savefig(f'figures/figure_6/poster_6d_mono.png')
        else:
            fig.savefig(f'figures/figure_6/poster_6d_spearman_mono.png')

def figure_6_poster_panel_d_mono_window(rs: np.ndarray, ps: np.ndarray, mono=False):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))

    sig_rs_positive_percentages = []
    sig_rs_negative_percentages = []

    sig_size = 0

    for i in range(len(rs)):
        # get the significant rs and ps
        sig_rs = rs[i][ps[i]<0.01]
        sig_size += len(sig_rs)

        sig_rs_positive = sig_rs[sig_rs>0]
        sig_rs_negative = sig_rs[sig_rs<0]
        
        print(f'{i}th session, positive: {len(sig_rs_positive)}, negative: {len(sig_rs_negative)}')

        # calculate the percentage of positive and negative significant rs
        sig_rs_positive_percentage = len(sig_rs_positive)/len(rs[i])
        sig_rs_negative_percentage = len(sig_rs_negative)/len(rs[i])

        # append the percentage to the list
        sig_rs_positive_percentages.append(sig_rs_positive_percentage)
        sig_rs_negative_percentages.append(sig_rs_negative_percentage)

    print(f'num of significant rs: {sig_size}')

    # t test to see if the percentage of positive and negative significant rs are different
    t, p = ttest_ind(sig_rs_positive_percentages, sig_rs_negative_percentages, alternative='less')
    print(f't: {t}, p: {p}')

    # plot the bar plot with the average percentage of positive and negative significant rs
    sns.barplot(x=['+', '-'], y=[np.mean(sig_rs_positive_percentages), np.mean(sig_rs_negative_percentages)], ax=axes)
    axes.set_ylim(0, 1)

    if not mono:
        # if the figures directory does not exist, create it
        if not os.path.exists('figures/figure_6'):
            os.makedirs('figures/figure_6')

        # save the figures
        fig.savefig(f'figures/figure_6/poster_6d_mono_window.png')
    else:
        # save the figures
        fig.savefig(f'figures/figure_6/poster_6d_mono_window.png')