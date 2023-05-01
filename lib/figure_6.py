import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.signal import correlate

from lib.calculation import moving_window_mean, get_firing_rate_window, moving_window_mean_prior, get_relative_spike_times, get_normalized_cross_correlation

# using firing during intertrial interval (ITI) window -1 to -0.5ms
def figure_6_panel_c(pfc_times: np.ndarray, str_times: np.ndarray, cue_times: np.ndarray, pfc_name: str, str_name: str, rewarded: np.ndarray, session_name: str, mono: bool = False):
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


        normalized_cross_corr = get_normalized_cross_correlation(pfc_trial_times, str_trial_times)

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
    overall_cross_cor = correlate(cross_cors, reward_proportion, mode='same')
    center = len(overall_cross_cor) // 2
    if len(overall_cross_cor) > 100:
        overall_cross_cor = overall_cross_cor[center - 50:center + 50]

    # if the figures directory does not exist, create it
    if not mono:
        if not os.path.exists('figures/figure_6/panel_c'):
            os.makedirs('figures/figure_6/panel_c')
        if not os.path.exists('figures/figure_6/significant'):
            os.makedirs('figures/figure_6/significant')
        
        # if not os.path.exists('figures/figure_6/panel_d'):
        #     os.makedirs('figures/figure_6/panel_d')
    else:
        if not os.path.exists('mono_figures/figure_6/panel_c'):
            os.makedirs('mono_figures/figure_6/panel_c')
        if not os.path.exists('mono_figures/figure_6/significant'):
            os.makedirs('mono_figures/figure_6/significant')
        # if not os.path.exists('mono_figures/figure_6/panel_d'):
        #     os.makedirs('mono_figures/figure_6/panel_d')

    # save the figures
    if not mono:
        fig.savefig(f'figures/figure_6/panel_c/6c_{session_name}_{pfc_name}_{str_name}_cross_correlation.png')
        # fig_overall.savefig(f'figures/figure_6/panel_d/6d_{session_name}_{pfc_name}_{str_name}_overall_cross_correlation.png')
    else:
        fig.savefig(f'mono_figures/figure_6/panel_c/6c_{session_name}_{pfc_name}_{str_name}_cross_correlation_mono.png')
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
    


# using firing during 1-3ms after cue
def figure_6_panel_e(pfc_times: np.ndarray, str_times: np.ndarray, cue_times: np.ndarray, pfc_name: str, str_name: str, rewarded: np.ndarray, session_name: str, mono: bool = False):
    # calculate the cross correlation of the two signals during a centered 20-trial long window
    pfc_firing_rates = get_firing_rate_window(cue_times, pfc_times, 1, 3)
    str_firing_rates = get_firing_rate_window(cue_times, str_times, 1, 3)

    # smoothen
    pfc_firing_rates = moving_window_mean(np.array(pfc_firing_rates), 20)
    str_firing_rates = moving_window_mean(np.array(str_firing_rates), 20)

    # normalize firing rates to [0, 1], avoid division by 0
    pfc_firing_rates = (pfc_firing_rates - np.min(pfc_firing_rates)) / (np.max(pfc_firing_rates) - np.min(pfc_firing_rates) + 1e-6)
    str_firing_rates = (str_firing_rates - np.min(str_firing_rates)) / (np.max(str_firing_rates) - np.min(str_firing_rates) + 1e-6)

    # calculate the cross correlation
    cross_cors = []
    for i in range(len(pfc_firing_rates)):
        if i < 10:
            pfc_window = pfc_firing_rates[:i + 10]
            str_window = str_firing_rates[:i + 10]
        elif i > len(pfc_firing_rates) - 10:
            pfc_window = pfc_firing_rates[i - 10:]
            str_window = str_firing_rates[i - 10:]
        else:
            pfc_window = pfc_firing_rates[i - 10:i + 10]
            str_window = str_firing_rates[i - 10:i + 10]
        cross_cors.append(np.abs(correlate(pfc_window, str_window)[0]))

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

    # calculate the overall cross correlation
    overall_cross_cor = correlate(pfc_firing_rates, str_firing_rates, mode='same')
    center = len(overall_cross_cor) // 2
    if len(overall_cross_cor) > 100:
        overall_cross_cor = overall_cross_cor[center - 50:center + 50]
    fig_overall, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.lineplot(x=np.arange(-len(overall_cross_cor) // 2, len(overall_cross_cor) // 2), y=overall_cross_cor, ax=ax)
    ax.set_xlabel('Trial Lag')
    ax.set_ylabel('Cross correlation')

    # save the figures
    if not mono:
        fig.savefig(f'figures/figure_6/panel_e/6e_{session_name}_{pfc_name}_{str_name}_cross_correlation.png')
    else:
        fig.savefig(f'mono_figures/figure_6/panel_e/6e_{session_name}_{pfc_name}_{str_name}_cross_correlation_mono.png')

    if p < 0.001:
        if not mono:
            # save the figures in significant folder
            fig.savefig(f'figures/figure_6/significant/6e_{session_name}_{pfc_name}_{str_name}_cross_correlation_significant.png')
            fig_overall.savefig(f'figures/figure_6/significant/6f_{session_name}_{pfc_name}_{str_name}_overall_cross_correlation_significant.png')
        else:
            # save the figures in significant folder
            fig.savefig(f'mono_figures/figure_6/significant/6e_{session_name}_{pfc_name}_{str_name}_cross_correlation_significant_mono.png')
            fig_overall.savefig(f'mono_figures/figure_6/significant/6f_{session_name}_{pfc_name}_{str_name}_overall_cross_correlation_significant_mono.png')

    # close the figures
    plt.close(fig)
    plt.close(fig_overall)

    return fig, fig_overall