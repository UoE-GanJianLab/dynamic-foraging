from os.path import join as pjoin
from os import listdir, mkdir
from os.path import basename, isfile, isdir
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy
from scipy.stats import ttest_rel

from lib.calculation import get_firing_rate_window

WINDOW_LEFT = 0.5
WINDOW_RIGHT = 1.5
# 1/BIN_WIDTH must be an integer
BIN_WIDTH = 0.02

spike_data_root = pjoin('data', 'spike_times', 'sessions')
behaviour_root = pjoin('data', 'behaviour_data')
relative_value_root = pjoin('data', 'prpd')

figure_5_data_source = pjoin('figure_data', 'figure_5')
figure_5_panel_abcd_pfc_data_top = pjoin(figure_5_data_source, 'figure_5_panel_abcd_pfc_top')
figure_5_panel_abcd_dms_data_top = pjoin(figure_5_data_source, 'figure_5_panel_abcd_dms_top')
figure_5_panel_abcd_pfc_data_bottom = pjoin(figure_5_data_source, 'figure_5_panel_abcd_pfc_bottom')
figure_5_panel_abcd_dms_data_bottom = pjoin(figure_5_data_source, 'figure_5_panel_abcd_dms_bottom')

figure_5_figure_root = pjoin('figures', 'all_figures', 'figure_5')
figure_5_data_root = pjoin('figure_data', 'figure_5')

spike_firing_root = pjoin('data', 'spike_times', 'sessions')
all_mono_pairs = pd.read_csv(pjoin('data', 'mono_pairs.csv'))

ITI_WINDOW_LEFT = -1
ITI_WINDOW_RIGHT = -0.5
RESPONSE_WINDOW_LEFT = 0
RESPONSE_WINDOW_RIGHT = 1.5

figure_5_figure_root = pjoin('figures', 'all_figures', 'figure_5')
figure_5_panel_abcd_figure_root = pjoin(figure_5_figure_root, 'panel_abcd')
figure_5_panel_abcd_pfc_root = pjoin(figure_5_panel_abcd_figure_root, 'pfc')
figure_5_panel_abcd_dms_root = pjoin(figure_5_panel_abcd_figure_root, 'dms')

for dir in [figure_5_figure_root, figure_5_panel_abcd_figure_root, figure_5_panel_abcd_pfc_root, figure_5_panel_abcd_dms_root, figure_5_data_source, figure_5_panel_abcd_pfc_data_top, figure_5_panel_abcd_dms_data_top, figure_5_panel_abcd_pfc_data_bottom, figure_5_panel_abcd_dms_data_bottom]:
    if not isdir(dir):
        mkdir(dir)

significance_threshold = 0.05

def raster(spikes, cue_times, leftP, session_name, brain_section):
    for ind in range(len(spikes)):
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        pointer = 0
        # left side advantageous
        all_spikes_left = []
        # right side advantageous
        all_spikes_right = []

        trial_indices = []
        relative_spike_times = []

        for tiral_ind, cue in enumerate(cue_times):
            trial_spikes = []
            cue_left = cue - WINDOW_LEFT
            cue_right = cue + WINDOW_RIGHT

            cell = spikes[ind]
            # move the pointer into the trial window
            while pointer < len(cell) and cell[pointer] < cue_left:
                pointer += 1
            # recording all spikes in the window
            while pointer < len(cell) and cell[pointer] <= cue_right:
                relative_time = cell[pointer] - cue
                if leftP[tiral_ind] > 0.5:
                    all_spikes_left.append(relative_time)
                else:
                    all_spikes_right.append(relative_time)
                trial_spikes.append(relative_time)
                pointer += 1
            sns.scatterplot(x=trial_spikes, y=tiral_ind, ax=axes[0], color='black', markers=".", s=5) # type: ignore
            trial_indices.extend(([tiral_ind])*len(trial_spikes))
            relative_spike_times.extend(trial_spikes)
        
        bins = np.arange(start=-WINDOW_LEFT, stop=WINDOW_RIGHT, step=BIN_WIDTH)
        left_y, left_bin_edges = np.histogram(all_spikes_left, bins=bins)
        left_bin_centers = 0.5 * (left_bin_edges[1:] + left_bin_edges[:-1])
        sns.lineplot(x=left_bin_centers, y=(left_y * int(1 / BIN_WIDTH)) / len(cue_times), ax=axes[1], label='Left P high')

        right_y, right_bin_edges = np.histogram(all_spikes_right, bins=bins)
        right_bin_centers = 0.5 * (right_bin_edges[1:] + right_bin_edges[:-1])
        sns.lineplot(x=right_bin_centers, y=right_y * int(1 / BIN_WIDTH) / len(cue_times), ax=axes[1], label='Right P high')

        axes[0].get_xaxis().set_visible(False)
        axes[1].set_xticks([-0.5, 0, 0.5, 1, 1.5])
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['bottom'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        # set axis labels and fig name
        axes[0].set_ylabel('Trial Number')
        axes[1].set_ylabel('Frequency (HZ)')

        axes[1].legend(bbox_to_anchor=(1.15, 0.7))

        fig_name = session_name + '_' + brain_section + '_' + str(ind) + '.png'
        data_file_name = session_name + '_' + brain_section + '_' + str(ind) + '.csv'

        if brain_section == 'pfc':
            fig.savefig(pjoin(figure_5_panel_abcd_pfc_root, fig_name), dpi=100)
        else:
            fig.savefig(pjoin(figure_5_panel_abcd_dms_root, fig_name), dpi=100)

        plt.close(fig)

        # save the firing data
        raster_data = pd.DataFrame({'trial_index': trial_indices, 'relative_spike_time': relative_spike_times})
        if brain_section == 'pfc':
            figure_5_panel_abcd_pfc_data_top_path = pjoin(figure_5_panel_abcd_pfc_data_top, data_file_name)
            raster_data.to_csv(figure_5_panel_abcd_pfc_data_top_path, index=False)
        else:
            figure_5_panel_abcd_dms_data_top_path = pjoin(figure_5_panel_abcd_dms_data_top, data_file_name)
            raster_data.to_csv(figure_5_panel_abcd_dms_data_top_path, index=False)

        left_bin_centers = np.round(left_bin_centers, 2)
        line_data = pd.DataFrame({'bin_centers': left_bin_centers, 'left_p_high': left_y * int(1 / BIN_WIDTH) / len(cue_times), 'right_p_high': right_y * int(1 / BIN_WIDTH) / len(cue_times)})
        if brain_section == 'pfc':
            figure_5_panel_abcd_pfc_data_bottom_path = pjoin(figure_5_panel_abcd_pfc_data_bottom, data_file_name)
            line_data.to_csv(figure_5_panel_abcd_pfc_data_bottom_path, index=False)
        else:
            figure_5_panel_abcd_dms_data_bottom_path = pjoin(figure_5_panel_abcd_dms_data_bottom, data_file_name)
            line_data.to_csv(figure_5_panel_abcd_dms_data_bottom_path, index=False)

def get_figure_5_panel_abcd():
    for dir in tqdm(listdir(spike_data_root)):
        pfc_path = pjoin(spike_data_root, dir, 'pfc.npy')
        dms_path = pjoin(spike_data_root, dir, 'dms.npy')

        if not isfile(pfc_path):
            print(dir)
            continue
        
        pfc_data = np.load(pfc_path, allow_pickle=True)
        dms_data = np.load(dms_path, allow_pickle=True)

        behaviour_path = pjoin('data', 'behaviour_data', dir + '.csv')
        task_info = pd.read_csv(behaviour_path)

        cue_times = task_info['cue_time']
        leftP = task_info['leftP']

        raster(spikes=pfc_data, cue_times=cue_times, leftP=leftP, session_name=dir, brain_section='pfc')
        raster(spikes=dms_data, cue_times=cue_times, leftP=leftP, session_name=dir, brain_section='dms')
 
# calculate the correlation statistics
def calculate_correlation_statistics():
    session_names_prpd = []
    cell_names_prpd = []
    session_names_relative_value = []
    cell_names_relative_value = []

    prpd_correlations_response = []
    prpd_p_values_response = []
    prpd_correlations_bg = []
    prpd_p_values_bg = []
    relative_value_correlations_response = []
    relative_value_p_values_response = []
    relative_value_correlations_bg = []
    relative_value_p_values_bg = []

    # load the spike times for each session
    sessions = glob(pjoin('data', 'spike_times', 'sessions', '*'))
    for session in sessions:
        session_name = basename(session)
        behaviour_data = pd.read_csv(pjoin('data', 'behaviour_data', session_name+'.csv'))
        cue_times = behaviour_data['cue_time'].values
        
        # load the individual cell firing data file paths
        pfc_cells = glob(pjoin(session, 'pfc_*'))
        dms_cells = glob(pjoin(session, 'dms_*'))
        all_cells = pfc_cells + dms_cells

        prpd = np.load(pjoin('data', 'prpd', session_name+'.npy')) 
        prpd[prpd == 1] = 0.999 
        relative = False
        if isfile(pjoin('data', 'relative_values', session_name+'.npy')):
            relative_values = np.load(pjoin('data', 'relative_values', session_name+'.npy'))
            relative_values[relative_values == 1] = 0.999
            relative = True
        
        # calculate the pearson correlation coefficient and p value for prpd and firing rate
        # discretize the prpd into 10 bins from -1 to 1
        bins = np.arange(-1, 1.1, 0.1)
        prpd = np.digitize(prpd, bins, right=True)
        # calculate the bin center for all available prpd values
        prpd_values = np.sort(np.unique(prpd))

        if relative:
            relative_values = np.digitize(relative_values, bins, right=True)
            delta_q_values = np.sort(np.unique(relative_values))

        # load the firing data
        for cell in all_cells:
            cell_name = basename(cell).split('.')[0]
            firing_data = np.load(cell, allow_pickle=False)
            firing_rate_response = np.array(get_firing_rate_window(cue_times, firing_data, RESPONSE_WINDOW_LEFT, RESPONSE_WINDOW_RIGHT))
            firing_rate_bg= np.array(get_firing_rate_window(cue_times, firing_data, ITI_WINDOW_LEFT, ITI_WINDOW_RIGHT))

            if np.std(firing_rate_response) == 0 or np.std(firing_rate_bg) == 0:
                continue

            # calculate the z score for the firing rate
            firing_rate_response = (firing_rate_response - np.mean(firing_rate_response)) / np.std(firing_rate_response)
            firing_rate_bg = (firing_rate_bg - np.mean(firing_rate_bg)) / np.std(firing_rate_bg)

            firing_rate_response_prpd = np.array([np.mean(firing_rate_response[prpd == i]) for i in prpd_values])
            firing_rate_bg_prpd_binned = np.array([np.mean(firing_rate_bg[prpd == i]) for i in prpd_values])

            prpd_correlation_response, prpd_p_value_response = scipy.stats.pearsonr(prpd_values, firing_rate_response_prpd)
            prpd_correlation_bg, prpd_p_value_bg = scipy.stats.pearsonr(prpd_values, firing_rate_bg_prpd_binned)

            # prpd_correlation_response, prpd_p_value_response = scipy.stats.pearsonr(prpd, firing_rate_response)
            # prpd_correlation_bg, prpd_p_value_bg = scipy.stats.pearsonr(prpd, firing_rate_bg)


            session_names_prpd.append(session_name)
            cell_names_prpd.append(cell_name)
            prpd_correlations_response.append(prpd_correlation_response)
            prpd_p_values_response.append(prpd_p_value_response)
            prpd_correlations_bg.append(prpd_correlation_bg)
            prpd_p_values_bg.append(prpd_p_value_bg)

            if relative:
                firing_rate_response_relative_value = np.array([np.mean(firing_rate_response[relative_values == i]) for i in delta_q_values])
                firing_rate_bg_relative_value = np.array([np.mean(firing_rate_bg[relative_values == i]) for i in delta_q_values])

                relative_value_correlation_response, relative_value_p_value_response = scipy.stats.pearsonr(delta_q_values, firing_rate_response_relative_value)
                relative_value_correlation_bg, relative_value_p_value_bg = scipy.stats.pearsonr(delta_q_values, firing_rate_bg_relative_value)

                # relative_value_correlation_response, relative_value_p_value_response = scipy.stats.pearsonr(relative_values, firing_rate_response)
                # relative_value_correlation_bg, relative_value_p_value_bg = scipy.stats.pearsonr(relative_values, firing_rate_bg)

                session_names_relative_value.append(session_name)
                cell_names_relative_value.append(cell_name)
                relative_value_correlations_response.append(relative_value_correlation_response)
                relative_value_p_values_response.append(relative_value_p_value_response)
                relative_value_correlations_bg.append(relative_value_correlation_bg)
                relative_value_p_values_bg.append(relative_value_p_value_bg)
    
    # save the data to csv
    prpd_correlation_data = pd.DataFrame({'session': session_names_prpd, 'cell': cell_names_prpd, 'background_firing_pearson_r': prpd_correlations_bg, 'background_firing_p_values': prpd_p_values_bg, 'response_firing_pearson_r': prpd_correlations_response, 'response_firing_p_values': prpd_p_values_response})
    prpd_correlation_data.to_csv(pjoin('data', 'prpd_correlation.csv'), index=False)

    relative_value_correlation_data = pd.DataFrame({'session': session_names_relative_value, 'cell': cell_names_relative_value, 'background_firing_pearson_r': relative_value_correlations_bg, 'background_firing_p_values': relative_value_p_values_bg, 'response_firing_pearson_r': relative_value_correlations_response, 'response_firing_p_values': relative_value_p_values_response})
    relative_value_correlation_data.to_csv(pjoin('data', 'relative_value_correlation.csv'), index=False)


def get_figure_5_panel_ef_left(prpd=True):
    if prpd:
        correlation_data = pd.read_csv(pjoin('data', 'prpd_correlation.csv'))
    else:
        correlation_data = pd.read_csv(pjoin('data', 'relative_value_correlation.csv'))

    # # remove rows with nan values
    # correlation_data = correlation_data.dropna()

    # correlation_data have following columns:
    # session,background_firing_pearson_r,background_firing_p_values,response_firing_pearson_r,response_firing_p_values,cell
    # get the pfc cells(cell start with pfc) who are not correlated in background firing and correlated in response firing,
    # correlated in background firing only, correlated in response firing only, and correlated in both
    pfc_cells = correlation_data[correlation_data['cell'].str.startswith('pfc')]
    total_pfc_cells = len(glob(pjoin(spike_firing_root, '*', 'pfc_*')))
    
    
    pfc_cells__background_only = pfc_cells[(pfc_cells['background_firing_p_values'] < significance_threshold) & (pfc_cells['response_firing_p_values'] >= significance_threshold)]
    pfc_cells__background_only_percent = len(pfc_cells__background_only) / total_pfc_cells
    pfc_cells_response_only = pfc_cells[(pfc_cells['background_firing_p_values'] >= significance_threshold) & (pfc_cells['response_firing_p_values'] < significance_threshold)]
    pfc_cells_response_only_percent = len(pfc_cells_response_only) / total_pfc_cells
    pfc_cells_both = pfc_cells[(pfc_cells['background_firing_p_values'] < significance_threshold) & (pfc_cells['response_firing_p_values'] < significance_threshold)]
    pfc_cells_both_percent = len(pfc_cells_both) / total_pfc_cells
    pfc_cells_neither = pfc_cells[(pfc_cells['background_firing_p_values'] >= significance_threshold) & (pfc_cells['response_firing_p_values'] >= significance_threshold)]
    pfc_cells_neither_percent = len(pfc_cells_neither) / total_pfc_cells

    # similarly for dms
    dms_cells = correlation_data[correlation_data['cell'].str.startswith('dms')]
    total_dms_cells = len(glob(pjoin(spike_firing_root, '*', 'dms_*')))

    dms_cells__background_only = dms_cells[(dms_cells['background_firing_p_values'] < significance_threshold) & (dms_cells['response_firing_p_values'] >= significance_threshold)]
    dms_cells__background_only_percent = len(dms_cells__background_only) / total_dms_cells
    dms_cells_response_only = dms_cells[(dms_cells['background_firing_p_values'] >= significance_threshold) & (dms_cells['response_firing_p_values'] < significance_threshold)]
    dms_cells_response_only_percent = len(dms_cells_response_only) / total_dms_cells
    dms_cells_both = dms_cells[(dms_cells['background_firing_p_values'] < significance_threshold) & (dms_cells['response_firing_p_values'] < significance_threshold)]
    dms_cells_both_percent = len(dms_cells_both) / total_dms_cells
    dms_cells_neither = dms_cells[(dms_cells['background_firing_p_values'] >= significance_threshold) & (dms_cells['response_firing_p_values'] >= significance_threshold)]
    dms_cells_neither_percent = len(dms_cells_neither) / total_dms_cells

    # save the data to csv with these columns: |cell_location|not-correlated percentage|ITI firing correlated percentage|
    # response magnitude correlated percentage|both ITI firing and response magnitude correlated percentage|
    panel_e_data = pd.DataFrame({'cell_location': ['pfc', 'dms'], 'not-correlated percentage': [pfc_cells_neither_percent, dms_cells_neither_percent], 'ITI firing correlated percentage': [pfc_cells__background_only_percent, dms_cells__background_only_percent], 'response magnitude correlated percentage': [pfc_cells_response_only_percent, dms_cells_response_only_percent], 'both ITI firing and response magnitude correlated percentage': [pfc_cells_both_percent, dms_cells_both_percent]})
    if prpd:
        panel_e_data.to_csv(pjoin(figure_5_data_root, 'figure_5_panel_ef_left_prpd.csv'), index=False) 
    else:
        panel_e_data.to_csv(pjoin(figure_5_data_root, 'figure_5_panel_ef_left_relative_value.csv'), index=False)

    # plot the result as pie charts, show the original counts in the pie chart
    plt.figure(figsize=(4, 4))
    plt.pie([pfc_cells_neither_percent, pfc_cells__background_only_percent, pfc_cells_response_only_percent, pfc_cells_both_percent], labels=[f'not correlated {pfc_cells_neither_percent * total_pfc_cells:.0f}', f'ITI firing only {pfc_cells__background_only_percent * total_pfc_cells:.0f}', f'response magnitude only {pfc_cells_response_only_percent * total_pfc_cells:.0f}', f'both {pfc_cells_both_percent * total_pfc_cells:.0f}'], autopct='%1.1f%%')
    plt.title('PFC')
    

    plt.figure(figsize=(4, 4))
    plt.pie([dms_cells_neither_percent, dms_cells__background_only_percent, dms_cells_response_only_percent, dms_cells_both_percent], labels=[f'not correlated {dms_cells_neither_percent * total_dms_cells:.0f}', f'ITI firing only {dms_cells__background_only_percent * total_dms_cells:.0f}', f'response magnitude only {dms_cells_response_only_percent * total_dms_cells:.0f}', f'both {dms_cells_both_percent * total_dms_cells:.0f}'], autopct='%1.1f%%')
    plt.title('DMS')


def get_figure_5_panel_ef_right(prpd=True):
    if prpd:
        correlation_data = pd.read_csv(pjoin('data', 'prpd_correlation.csv'))
    else:
        correlation_data = pd.read_csv(pjoin('data', 'relative_value_correlation.csv'))

    # # remove rows with nan values
    # correlation_data = correlation_data.dropna()

    # for each session, calculate the percentage cells of strongly positively and negatively correlated with prpd for pfc and dms
    sessions = correlation_data['session'].unique()
    pfc_strongly_positively_correlated = []
    pfc_strongly_negatively_correlated = []
    pfc_strongly_positively_correlated_bg = []
    pfc_strongly_negatively_correlated_bg = []
    dms_strongly_positively_correlated = []
    dms_strongly_negatively_correlated = []
    dms_strongly_negatively_correlated_bg = []
    dms_strongly_positively_correlated_bg = []


    for session in sessions:
        session_cells = correlation_data[correlation_data['session'] == session]
        if len(session_cells) == 0:
            continue
        pfc_cells = session_cells[session_cells['cell'].str.startswith('pfc')]
        dms_cells = session_cells[session_cells['cell'].str.startswith('dms')]

        total_pfc_cells = len(glob(pjoin(spike_firing_root, session, 'pfc_*')))
        total_dms_cells = len(glob(pjoin(spike_firing_root, session, 'dms_*')))

        pfc_strongly_positively_correlated.append(len(pfc_cells[(pfc_cells['response_firing_pearson_r'] > 0) & (pfc_cells['response_firing_p_values'] < significance_threshold)])/ total_pfc_cells)
        pfc_strongly_negatively_correlated.append(len(pfc_cells[(pfc_cells['response_firing_pearson_r'] < 0) & (pfc_cells['response_firing_p_values'] < significance_threshold)])/ total_pfc_cells)
        pfc_strongly_negatively_correlated_bg.append(len(pfc_cells[(pfc_cells['background_firing_pearson_r'] < 0) & (pfc_cells['background_firing_p_values'] < significance_threshold)])/ total_pfc_cells)
        pfc_strongly_positively_correlated_bg.append(len(pfc_cells[(pfc_cells['background_firing_pearson_r'] > 0) & (pfc_cells['background_firing_p_values'] < significance_threshold)])/ total_pfc_cells)
        dms_strongly_positively_correlated.append(len(dms_cells[(dms_cells['response_firing_pearson_r'] > 0) & (dms_cells['response_firing_p_values'] < significance_threshold)])/ total_dms_cells)
        dms_strongly_negatively_correlated.append(len(dms_cells[(dms_cells['response_firing_pearson_r'] < 0) & (dms_cells['response_firing_p_values'] < significance_threshold)])/ total_dms_cells)
        dms_strongly_negatively_correlated_bg.append(len(dms_cells[(dms_cells['background_firing_pearson_r'] < 0) & (dms_cells['background_firing_p_values'] < significance_threshold)])/ total_dms_cells)
        dms_strongly_positively_correlated_bg.append(len(dms_cells[(dms_cells['background_firing_pearson_r'] > 0) & (dms_cells['background_firing_p_values'] < significance_threshold)])/ total_dms_cells)

    # plot the result as boxplots
    pfc_strongly_positively_correlated = np.array(pfc_strongly_positively_correlated)
    pfc_strongly_negatively_correlated = np.array(pfc_strongly_negatively_correlated)
    pfc_strongly_positively_correlated_bg = np.array(pfc_strongly_positively_correlated_bg)
    pfc_strongly_negatively_correlated_bg = np.array(pfc_strongly_negatively_correlated_bg)

    dms_strongly_positively_correlated = np.array(dms_strongly_positively_correlated)
    dms_strongly_negatively_correlated = np.array(dms_strongly_negatively_correlated)
    dms_strongly_negatively_correlated_bg = np.array(dms_strongly_negatively_correlated_bg)
    dms_strongly_positively_correlated_bg = np.array(dms_strongly_positively_correlated_bg)

    # plot pfc data
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    sns.boxplot(data=[pfc_strongly_positively_correlated, pfc_strongly_negatively_correlated], ax=axes[0])
    sns.boxplot(data=[pfc_strongly_positively_correlated_bg, pfc_strongly_negatively_correlated_bg], ax=axes[1])

    axes[0].set_title('PFC response')
    axes[0].set_ylabel('Percentage')
    axes[0].set_xticklabels(['Positively correlated', 'Negatively correlated'])
    axes[1].set_title('PFC background')
    axes[1].set_ylabel('Percentage')
    axes[1].set_xticklabels(['Positively correlated', 'Negatively correlated'])

    # set the y limit of all figures to [0, 1]
    for ax in axes:
        ax.set_ylim([0, 1])

    # plot dms data
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    sns.boxplot(data=[dms_strongly_positively_correlated, dms_strongly_negatively_correlated], ax=axes[0])
    sns.boxplot(data=[dms_strongly_positively_correlated_bg, dms_strongly_negatively_correlated_bg], ax=axes[1])

    axes[0].set_title('DMS response')
    axes[0].set_ylabel('Percentage')
    axes[0].set_xticklabels(['Positively correlated', 'Negatively correlated'])
    axes[1].set_title('DMS background')
    axes[1].set_ylabel('Percentage')
    axes[1].set_xticklabels(['Positively correlated', 'Negatively correlated'])

    # set the y limit of all figures to [0, 1]
    for ax in axes:
        ax.set_ylim([0, 1])

    # do the t test for pfc and dms
    from scipy.stats import ttest_ind
    print('pfc response: ', ttest_ind(pfc_strongly_positively_correlated, pfc_strongly_negatively_correlated))
    print('pfc background: ', ttest_ind(pfc_strongly_positively_correlated_bg, pfc_strongly_negatively_correlated_bg))
    print('dms response: ', ttest_ind(dms_strongly_positively_correlated, dms_strongly_negatively_correlated))
    print('dms background: ', ttest_ind(dms_strongly_positively_correlated_bg, dms_strongly_negatively_correlated_bg))

    # save the data to csv
    panel_ef_right_data = pd.DataFrame({'pfc_strongly_positively_correlated_response': pfc_strongly_positively_correlated, 'pfc_strongly_negatively_correlated_response': pfc_strongly_negatively_correlated, 'pfc_strongly_positively_correlated_bg': pfc_strongly_positively_correlated_bg, 'pfc_strongly_negatively_correlated_bg': pfc_strongly_negatively_correlated_bg, 'dms_strongly_positively_correlated_response': dms_strongly_positively_correlated, 'dms_strongly_negatively_correlated_response': dms_strongly_negatively_correlated, 'dms_strongly_positively_correlated_bg': dms_strongly_positively_correlated_bg, 'dms_strongly_negatively_correlated_bg': dms_strongly_negatively_correlated_bg})

    if prpd:
        panel_ef_right_data.to_csv(pjoin(figure_5_data_root, 'figure_5_panel_ef_right_prpd.csv'), index=False)
    else:
        panel_ef_right_data.to_csv(pjoin(figure_5_data_root, 'figure_5_panel_ef_right_relative_value.csv'), index=False)

significan_p_threshold = 0.05
bin_size = 0.25

def firing_rate_vs_relative_value():
    # instead of spliting to R and L trials
    # plot the firing rate vs relative value
    # for all trials
    # go through each sessions and load up the behaviour data
    # as well as the relative values
    relative_values = []
    firing_rates = []


def get_figure_5_panel_gh():
    # go through each sessions and load up the behaviour data 
    # as well as the relative values
    relative_values_past_R_pfc = []
    relative_values_past_L_pfc = []
    relative_values_future_R_pfc = []
    relative_values_future_L_pfc = []

    relative_values_future_R_pfc_response = []
    relative_values_future_L_pfc_response = []
    relative_values_past_R_pfc_response = []
    relative_values_past_L_pfc_response = []

    relative_values_past_R_dms = []
    relative_values_past_L_dms = []
    relative_values_future_R_dms = []
    relative_values_future_L_dms = []

    relative_values_future_R_dms_response = []
    relative_values_future_L_dms_response = []
    relative_values_past_R_dms_response = []
    relative_values_past_L_dms_response = []

    relative_values_pfc_all = []
    relative_values_dms_all = []

    relative_values_pfc_all_response = []
    relative_values_dms_all_response = []

    pfc_firing_all = []
    dms_firing_all = []

    pfc_firing_all_response = []
    dms_firing_all_response = []

    pfc_firing_rates_past_R_bg = []
    pfc_firing_rates_past_L_bg = []
    pfc_firing_rates_future_R_bg = []
    pfc_firing_rates_future_L_bg = []

    pfc_firing_rates_past_R_response = []
    pfc_firing_rates_past_L_response = []
    pfc_firing_rates_future_R_response = []
    pfc_firing_rates_future_L_response = []

    dms_firing_rates_past_R = []
    dms_firing_rates_past_L = []
    dms_firing_rates_future_R = []
    dms_firing_rates_future_L = []

    dms_firing_rates_past_R_response = []
    dms_firing_rates_past_L_response = []
    dms_firing_rates_future_R_response = []
    dms_firing_rates_future_L_response = []

    for session in glob(pjoin(behaviour_root, '*.csv')):
        session_name = basename(session).split('.')[0]
        relative_values = np.load(pjoin(relative_value_root, session_name+'.npy'))
        # for binning to work correctly, reduce relative value of 1
        # by bin size/2, increase relative value of -1 by bin size/2
        relative_values[relative_values == 1] = relative_values[relative_values == 1] - bin_size/2
        relative_values[relative_values == -1] = relative_values[relative_values == -1] + bin_size/2
        session_data = pd.read_csv(session)
        cue_time = np.array(session_data['cue_time'].values)

        # get the trials where the last trial was a right and left response
        past_R_indices = np.where(session_data['trial_response_side'].values[:-1] == 1)[0]
        past_L_indices = np.where(session_data['trial_response_side'].values[:-1] == -1)[0]
        past_R_indices = past_R_indices + 1
        past_L_indices = past_L_indices + 1

        # get the trials where the next trial was a right and left response
        future_R_indices = np.where(session_data['trial_response_side'].values[1:] == 1)[0]
        future_L_indices = np.where(session_data['trial_response_side'].values[1:] == -1)[0]
        future_R_indices = future_R_indices - 1
        future_L_indices = future_L_indices - 1

        # load up the spike data
        for pfc_cell in glob(pjoin(spike_data_root, session_name, 'pfc_*')):
            pfc_cell_name = basename(pfc_cell).split('.')[0]
            pfc_cell_data = np.load(pfc_cell)

            # get the firing rate of the cell
            firing_rates_bg = get_firing_rate_window(cue_time, pfc_cell_data, window_left=-1, window_right=-0.5)
            firing_rates_bg = np.array(firing_rates_bg)

            firing_rates_response = get_firing_rate_window(cue_time, pfc_cell_data, window_left=0, window_right=1.5)
            firing_rates_response = np.array(firing_rates_response)

            # check if the firing rates and relative values are 
            # strongly correlated using pearson correlation
            # continue if p value is less than threshold
            if np.std(firing_rates_bg) != 0 and scipy.stats.pearsonr(firing_rates_bg, relative_values)[1] < significan_p_threshold:
                firing_rates_bg = (firing_rates_bg - np.mean(firing_rates_bg)) / np.std(firing_rates_bg)
                # if pearson's r < 0 then flip the relative values
                # so that the firing rate is positively correlated with relative values
                if scipy.stats.pearsonr(firing_rates_bg, relative_values)[0] < 0:
                    relative_values = -relative_values

                # get the firing rates for the past and future trials
                pfc_firing_rates_past_R_bg.extend(firing_rates_bg[past_R_indices])
                pfc_firing_rates_past_L_bg.extend(firing_rates_bg[past_L_indices])
                pfc_firing_rates_future_R_bg.extend(firing_rates_bg[future_R_indices])
                pfc_firing_rates_future_L_bg.extend(firing_rates_bg[future_L_indices])

                relative_values_future_L_pfc.extend(relative_values[future_L_indices])
                relative_values_future_R_pfc.extend(relative_values[future_R_indices])
                relative_values_past_L_pfc.extend(relative_values[past_L_indices])
                relative_values_past_R_pfc.extend(relative_values[past_R_indices])

                relative_values_pfc_all.extend(relative_values)
                pfc_firing_all.extend(firing_rates_bg)
            
            if np.std(firing_rates_response) != 0 and scipy.stats.pearsonr(firing_rates_response, relative_values)[1] < significan_p_threshold:
                firing_rates_response = (firing_rates_response - np.mean(firing_rates_response)) / np.std(firing_rates_response)

                if scipy.stats.pearsonr(firing_rates_response, relative_values)[0] < 0:
                    relative_values = -relative_values

                # get the firing rates for the past and future trials
                pfc_firing_rates_past_R_response.extend(firing_rates_response[past_R_indices])
                pfc_firing_rates_past_L_response.extend(firing_rates_response[past_L_indices])
                pfc_firing_rates_future_R_response.extend(firing_rates_response[future_R_indices])
                pfc_firing_rates_future_L_response.extend(firing_rates_response[future_L_indices])

                relative_values_future_L_pfc_response.extend(relative_values[future_L_indices])
                relative_values_future_R_pfc_response.extend(relative_values[future_R_indices])
                relative_values_past_L_pfc_response.extend(relative_values[past_L_indices])
                relative_values_past_R_pfc_response.extend(relative_values[past_R_indices])

                relative_values_pfc_all_response.extend(relative_values)
                pfc_firing_all_response.extend(firing_rates_response)

        for dms_cell in glob(pjoin(spike_data_root, session_name, 'dms_*')):
            dms_cell_name = basename(dms_cell).split('.')[0]
            dms_cell_data = np.load(dms_cell)

            # get the firing rate of the cell
            firing_rates_bg = get_firing_rate_window(cue_time, dms_cell_data, window_left=-1, window_right=-0.5)
            firing_rates_bg = np.array(firing_rates_bg)            

            firing_rates_response = get_firing_rate_window(cue_time, dms_cell_data, window_left=0, window_right=1.5)
            firing_rates_response = np.array(firing_rates_response)

            if np.std(firing_rates_bg) != 0 and scipy.stats.pearsonr(firing_rates_bg, relative_values)[1] < significan_p_threshold:
                if scipy.stats.pearsonr(firing_rates_bg, relative_values)[0] < 0:
                    relative_values = -relative_values

                firing_rates_bg = (firing_rates_bg - np.mean(firing_rates_bg)) / np.std(firing_rates_bg)
                # get the firing rates for the past and future trials
                dms_firing_rates_past_R.extend(firing_rates_bg[past_R_indices])
                dms_firing_rates_past_L.extend(firing_rates_bg[past_L_indices])
                dms_firing_rates_future_R.extend(firing_rates_bg[future_R_indices])
                dms_firing_rates_future_L.extend(firing_rates_bg[future_L_indices])

                relative_values_future_L_dms.extend(relative_values[future_L_indices])
                relative_values_future_R_dms.extend(relative_values[future_R_indices])
                relative_values_past_L_dms.extend(relative_values[past_L_indices])
                relative_values_past_R_dms.extend(relative_values[past_R_indices])

                relative_values_dms_all.extend(relative_values)
                dms_firing_all.extend(firing_rates_bg)

            if np.std(firing_rates_response) != 0 and scipy.stats.pearsonr(firing_rates_response, relative_values)[1] < significan_p_threshold:
                if scipy.stats.pearsonr(firing_rates_response, relative_values)[0] < 0:
                    relative_values = -relative_values

                firing_rates_response = (firing_rates_response - np.mean(firing_rates_response)) / np.std(firing_rates_response)
                # get the firing rates for the past and future trials
                dms_firing_rates_past_R_response.extend(firing_rates_response[past_R_indices])
                dms_firing_rates_past_L_response.extend(firing_rates_response[past_L_indices])
                dms_firing_rates_future_R_response.extend(firing_rates_response[future_R_indices])
                dms_firing_rates_future_L_response.extend(firing_rates_response[future_L_indices])

                relative_values_future_L_dms_response.extend(relative_values[future_L_indices])
                relative_values_future_R_dms_response.extend(relative_values[future_R_indices])
                relative_values_past_L_dms_response.extend(relative_values[past_L_indices])
                relative_values_past_R_dms_response.extend(relative_values[past_R_indices])

                relative_values_dms_all_response.extend(relative_values)
                dms_firing_all_response.extend(firing_rates_response)

    # do paired t test before digitization
    print('pfc firing rate past R vs past L: ', ttest_rel(pfc_firing_rates_past_R_bg, pfc_firing_rates_past_L_bg))
    print('pfc firing rate future R vs future L: ', ttest_rel(pfc_firing_rates_future_R_bg, pfc_firing_rates_future_L_bg))

    # discretize the relative values into 20 bins
    relative_values_past_L_dms = np.digitize(relative_values_past_L_dms, bins=np.arange(-1, 1+bin_size, bin_size), right=False)
    relative_values_past_R_dms = np.digitize(relative_values_past_R_dms, bins=np.arange(-1, 1+bin_size, bin_size), right=False)
    relative_values_future_L_dms = np.digitize(relative_values_future_L_dms, bins=np.arange(-1, 1+bin_size, bin_size), right=False)
    relative_values_future_R_dms = np.digitize(relative_values_future_R_dms, bins=np.arange(-1, 1+bin_size, bin_size), right=False)

    relative_values_past_L_pfc = np.digitize(relative_values_past_L_pfc, bins=np.arange(-1, 1+bin_size, bin_size), right=False)
    relative_values_past_R_pfc = np.digitize(relative_values_past_R_pfc, bins=np.arange(-1, 1+bin_size, bin_size), right=False)
    relative_values_future_L_pfc = np.digitize(relative_values_future_L_pfc, bins=np.arange(-1, 1+bin_size, bin_size), right=False)
    relative_values_future_R_pfc = np.digitize(relative_values_future_R_pfc, bins=np.arange(-1, 1+bin_size, bin_size), right=False)

    relative_values_past_L_dms_response = np.digitize(relative_values_past_L_dms_response, bins=np.arange(-1, 1+bin_size, bin_size), right=False)
    relative_values_past_R_dms_response = np.digitize(relative_values_past_R_dms_response, bins=np.arange(-1, 1+bin_size, bin_size), right=False)
    relative_values_future_L_dms_response = np.digitize(relative_values_future_L_dms_response, bins=np.arange(-1, 1+bin_size, bin_size), right=False)
    relative_values_future_R_dms_response = np.digitize(relative_values_future_R_dms_response, bins=np.arange(-1, 1+bin_size, bin_size), right=False)

    relative_values_past_L_pfc_response = np.digitize(relative_values_past_L_pfc_response, bins=np.arange(-1, 1+bin_size, bin_size), right=False)
    relative_values_past_R_pfc_response = np.digitize(relative_values_past_R_pfc_response, bins=np.arange(-1, 1+bin_size, bin_size), right=False)
    relative_values_future_L_pfc_response = np.digitize(relative_values_future_L_pfc_response, bins=np.arange(-1, 1+bin_size, bin_size), right=False)
    relative_values_future_R_pfc_response = np.digitize(relative_values_future_R_pfc_response, bins=np.arange(-1, 1+bin_size, bin_size), right=False)

    relative_values_pfc_all = np.digitize(relative_values_pfc_all, bins=np.arange(-1, 1+bin_size, bin_size), right=False)
    relative_values_dms_all = np.digitize(relative_values_dms_all, bins=np.arange(-1, 1+bin_size, bin_size), right=False)

    relative_values_pfc_all_response = np.digitize(relative_values_pfc_all_response, bins=np.arange(-1, 1+bin_size, bin_size), right=False)
    relative_values_dms_all_response = np.digitize(relative_values_dms_all_response, bins=np.arange(-1, 1+bin_size, bin_size), right=False)

    x = np.arange(-1+bin_size/2, 1, bin_size)

    pfc_firing_rates_past_R_bg = np.array(pfc_firing_rates_past_R_bg)
    pfc_firing_rates_past_L_bg = np.array(pfc_firing_rates_past_L_bg)
    pfc_firing_rates_future_R_bg = np.array(pfc_firing_rates_future_R_bg)
    pfc_firing_rates_future_L_bg = np.array(pfc_firing_rates_future_L_bg)

    dms_firing_rates_past_R = np.array(dms_firing_rates_past_R)
    dms_firing_rates_past_L = np.array(dms_firing_rates_past_L)
    dms_firing_rates_future_R = np.array(dms_firing_rates_future_R)
    dms_firing_rates_future_L = np.array(dms_firing_rates_future_L)

    pfc_firing_rates_past_R_response = np.array(pfc_firing_rates_past_R_response)
    pfc_firing_rates_past_L_response = np.array(pfc_firing_rates_past_L_response)
    pfc_firing_rates_future_R_response = np.array(pfc_firing_rates_future_R_response)
    pfc_firing_rates_future_L_response = np.array(pfc_firing_rates_future_L_response)

    dms_firing_rates_past_R_response = np.array(dms_firing_rates_past_R_response)
    dms_firing_rates_past_L_response = np.array(dms_firing_rates_past_L_response)
    dms_firing_rates_future_R_response = np.array(dms_firing_rates_future_R_response)
    dms_firing_rates_future_L_response = np.array(dms_firing_rates_future_L_response)

    pfc_firing_rate_past_R_mean, pfc_firing_rate_past_R_sem = get_mean_and_sem(relative_values_past_R_pfc, pfc_firing_rates_past_R_bg)
    pfc_firing_rate_past_L_mean, pfc_firing_rate_past_L_sem = get_mean_and_sem(relative_values_past_L_pfc, pfc_firing_rates_past_L_bg)
    pfc_firing_rate_future_R_mean, pfc_firing_rate_future_R_sem = get_mean_and_sem(relative_values_future_R_pfc, pfc_firing_rates_future_R_bg)
    pfc_firing_rate_future_L_mean, pfc_firing_rate_future_L_sem = get_mean_and_sem(relative_values_future_L_pfc, pfc_firing_rates_future_L_bg)

    dms_firing_rate_past_R_mean, dms_firing_rate_past_R_sem = get_mean_and_sem(relative_values_past_R_dms, dms_firing_rates_past_R)
    dms_firing_rate_past_L_mean, dms_firing_rate_past_L_sem = get_mean_and_sem(relative_values_past_L_dms, dms_firing_rates_past_L)
    dms_firing_rate_future_R_mean, dms_firing_rate_future_R_sem = get_mean_and_sem(relative_values_future_R_dms, dms_firing_rates_future_R)
    dms_firing_rate_future_L_mean, dms_firing_rate_future_L_sem = get_mean_and_sem(relative_values_future_L_dms, dms_firing_rates_future_L)

    pfc_firing_rate_past_R_response_mean, pfc_firing_rate_past_R_response_sem = get_mean_and_sem(relative_values_past_R_pfc_response, pfc_firing_rates_past_R_response)
    pfc_firing_rate_past_L_response_mean, pfc_firing_rate_past_L_response_sem = get_mean_and_sem(relative_values_past_L_pfc_response, pfc_firing_rates_past_L_response)
    pfc_firing_rate_future_R_response_mean, pfc_firing_rate_future_R_response_sem = get_mean_and_sem(relative_values_future_R_pfc_response, pfc_firing_rates_future_R_response)
    pfc_firing_rate_future_L_response_mean, pfc_firing_rate_future_L_response_sem = get_mean_and_sem(relative_values_future_L_pfc_response, pfc_firing_rates_future_L_response)
    
    dms_firing_rate_past_R_response_mean, dms_firing_rate_past_R_response_sem = get_mean_and_sem(relative_values_past_R_dms_response, dms_firing_rates_past_R_response)
    dms_firing_rate_past_L_response_mean, dms_firing_rate_past_L_response_sem = get_mean_and_sem(relative_values_past_L_dms_response, dms_firing_rates_past_L_response)
    dms_firing_rate_future_R_response_mean, dms_firing_rate_future_R_response_sem = get_mean_and_sem(relative_values_future_R_dms_response, dms_firing_rates_future_R_response)
    dms_firing_rate_future_L_response_mean, dms_firing_rate_future_L_response_sem = get_mean_and_sem(relative_values_future_L_dms_response, dms_firing_rates_future_L_response)

    pfc_firing_all_mean, pfc_firing_all_sem = get_mean_and_sem(relative_values_pfc_all, pfc_firing_all)
    dms_firing_all_mean, dms_firing_all_sem = get_mean_and_sem(relative_values_dms_all, dms_firing_all)
    pfc_firing_all_response_mean, pfc_firing_all_response_sem = get_mean_and_sem(relative_values_pfc_all_response, pfc_firing_all_response)
    dms_firing_all_response_mean, dms_firing_all_response_sem = get_mean_and_sem(relative_values_dms_all_response, dms_firing_all_response)

    x = np.arange(-1+bin_size/2, 1, bin_size)

    # plot the firing rates vs relative values as line plots with 
    # shaded error bars for the standard error
    # with R and L trials sharing the same plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].plot(x, pfc_firing_rate_past_R_mean, color='red')
    axes[0, 0].fill_between(x, pfc_firing_rate_past_R_mean-pfc_firing_rate_past_R_sem, pfc_firing_rate_past_R_mean+pfc_firing_rate_past_R_sem, color='red', alpha=0.3)
    axes[0, 0].plot(x, pfc_firing_rate_past_L_mean, color='blue')
    axes[0, 0].fill_between(x, pfc_firing_rate_past_L_mean-pfc_firing_rate_past_L_sem, pfc_firing_rate_past_L_mean+pfc_firing_rate_past_L_sem, color='blue', alpha=0.3)
    axes[0, 1].plot(x, pfc_firing_rate_future_R_mean, color='red')
    axes[0, 1].fill_between(x, pfc_firing_rate_future_R_mean-pfc_firing_rate_future_R_sem, pfc_firing_rate_future_R_mean+pfc_firing_rate_future_R_sem, color='red', alpha=0.3)
    axes[0, 1].plot(x, pfc_firing_rate_future_L_mean, color='blue')
    axes[0, 1].fill_between(x, pfc_firing_rate_future_L_mean-pfc_firing_rate_future_L_sem, pfc_firing_rate_future_L_mean+pfc_firing_rate_future_L_sem, color='blue', alpha=0.3)
    axes[1, 0].plot(x, dms_firing_rate_past_R_mean, color='red')
    axes[1, 0].fill_between(x, dms_firing_rate_past_R_mean-dms_firing_rate_past_R_sem, dms_firing_rate_past_R_mean+dms_firing_rate_past_R_sem, color='red', alpha=0.3)
    axes[1, 0].plot(x, dms_firing_rate_past_L_mean, color='blue')
    axes[1, 0].fill_between(x, dms_firing_rate_past_L_mean-dms_firing_rate_past_L_sem, dms_firing_rate_past_L_mean+dms_firing_rate_past_L_sem, color='blue', alpha=0.3)
    axes[1, 1].plot(x, dms_firing_rate_future_R_mean, color='red')
    axes[1, 1].fill_between(x, dms_firing_rate_future_R_mean-dms_firing_rate_future_R_sem, dms_firing_rate_future_R_mean+dms_firing_rate_future_R_sem, color='red', alpha=0.3)
    axes[1, 1].plot(x, dms_firing_rate_future_L_mean, color='blue')
    axes[1, 1].fill_between(x, dms_firing_rate_future_L_mean-dms_firing_rate_future_L_sem, dms_firing_rate_future_L_mean+dms_firing_rate_future_L_sem, color='blue', alpha=0.3)
    
    axes[0, 0].set_title('Past trials')
    axes[0, 1].set_title('Future trials')
    axes[0, 0].set_ylabel('PFC firing rate')
    axes[1, 0].set_ylabel('DMS firing rate')
    axes[1, 0].set_xlabel('Relative value')
    axes[1, 1].set_xlabel('Relative value')

    # set the x asis to [-1, 1]
    axes[0, 0].set_xticks([-1, 0, 1])
    axes[0, 1].set_xticks([-1, 0, 1])
    axes[1, 0].set_xticks([-1, 0, 1])
    axes[1, 1].set_xticks([-1, 0, 1])

    fig.suptitle('PFC firing rate vs relative value')

    # another figure for the response period
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].plot(x, pfc_firing_rate_past_R_response_mean, color='red')
    axes[0, 0].fill_between(x, pfc_firing_rate_past_R_response_mean-pfc_firing_rate_past_R_response_sem, pfc_firing_rate_past_R_response_mean+pfc_firing_rate_past_R_response_sem, color='red', alpha=0.3)
    axes[0, 0].plot(x, pfc_firing_rate_past_L_response_mean, color='blue')
    axes[0, 0].fill_between(x, pfc_firing_rate_past_L_response_mean-pfc_firing_rate_past_L_response_sem, pfc_firing_rate_past_L_response_mean+pfc_firing_rate_past_L_response_sem, color='blue', alpha=0.3)
    axes[0, 1].plot(x, pfc_firing_rate_future_R_response_mean, color='red')
    axes[0, 1].fill_between(x, pfc_firing_rate_future_R_response_mean-pfc_firing_rate_future_R_response_sem, pfc_firing_rate_future_R_response_mean+pfc_firing_rate_future_R_response_sem, color='red', alpha=0.3)
    axes[0, 1].plot(x, pfc_firing_rate_future_L_response_mean, color='blue')
    axes[0, 1].fill_between(x, pfc_firing_rate_future_L_response_mean-pfc_firing_rate_future_L_response_sem, pfc_firing_rate_future_L_response_mean+pfc_firing_rate_future_L_response_sem, color='blue', alpha=0.3)
    axes[1, 0].plot(x, dms_firing_rate_past_R_response_mean, color='red')
    axes[1, 0].fill_between(x, dms_firing_rate_past_R_response_mean-dms_firing_rate_past_R_response_sem, dms_firing_rate_past_R_response_mean+dms_firing_rate_past_R_response_sem, color='red', alpha=0.3)
    axes[1, 0].plot(x, dms_firing_rate_past_L_response_mean, color='blue')
    axes[1, 0].fill_between(x, dms_firing_rate_past_L_response_mean-dms_firing_rate_past_L_response_sem, dms_firing_rate_past_L_response_mean+dms_firing_rate_past_L_response_sem, color='blue', alpha=0.3)
    axes[1, 1].plot(x, dms_firing_rate_future_R_response_mean, color='red')
    axes[1, 1].fill_between(x, dms_firing_rate_future_R_response_mean-dms_firing_rate_future_R_response_sem, dms_firing_rate_future_R_response_mean+dms_firing_rate_future_R_response_sem, color='red', alpha=0.3)
    axes[1, 1].plot(x, dms_firing_rate_future_L_response_mean, color='blue')
    axes[1, 1].fill_between(x, dms_firing_rate_future_L_response_mean-dms_firing_rate_future_L_response_sem, dms_firing_rate_future_L_response_mean+dms_firing_rate_future_L_response_sem, color='blue', alpha=0.3)

    axes[0, 0].set_title('Past trials')
    axes[0, 1].set_title('Future trials')
    axes[0, 0].set_ylabel('PFC firing rate')
    axes[1, 0].set_ylabel('DMS firing rate')

    axes[1, 0].set_xlabel('Relative value')
    axes[1, 1].set_xlabel('Relative value')

    # set the x asis to [-1, 1]
    axes[0, 0].set_xticks([-1, 0, 1])
    axes[0, 1].set_xticks([-1, 0, 1])
    axes[1, 0].set_xticks([-1, 0, 1])
    axes[1, 1].set_xticks([-1, 0, 1])

    fig.suptitle('firing rate vs relative value (response period)')
    plt.show()

    # another figure for all trials
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].plot(x, pfc_firing_all_mean, color='red')
    axes[0, 0].fill_between(x, pfc_firing_all_mean-pfc_firing_all_sem, pfc_firing_all_mean+pfc_firing_all_sem, color='red', alpha=0.3)
    axes[0, 1].plot(x, pfc_firing_all_mean, color='red')
    axes[0, 1].fill_between(x, pfc_firing_all_mean-pfc_firing_all_sem, pfc_firing_all_mean+pfc_firing_all_sem, color='red', alpha=0.3)
    axes[1, 0].plot(x, dms_firing_all_mean, color='red')
    axes[1, 0].fill_between(x, dms_firing_all_mean-dms_firing_all_sem, dms_firing_all_mean+dms_firing_all_sem, color='red', alpha=0.3)
    axes[1, 1].plot(x, dms_firing_all_mean, color='red')
    axes[1, 1].fill_between(x, dms_firing_all_mean-dms_firing_all_sem, dms_firing_all_mean+dms_firing_all_sem, color='red', alpha=0.3)


    axes[0, 0].set_title('BG')
    axes[0, 1].set_title('Response period')
    axes[0, 0].set_ylabel('PFC firing rate')
    axes[1, 0].set_ylabel('DMS firing rate')
    
    axes[1, 0].set_xlabel('Relative value')
    axes[1, 1].set_xlabel('Relative value')

    # set the x asis to [-1, 1]
    axes[0, 0].set_xticks([-1, 0, 1])
    axes[0, 1].set_xticks([-1, 0, 1])
    axes[1, 0].set_xticks([-1, 0, 1])
    axes[1, 1].set_xticks([-1, 0, 1])

    fig.suptitle('firing rate vs relative value (all trials)')
    plt.show()

    # save all the data to a csv file
    df = pd.DataFrame({'x': x, 'pfc_past_R_bg': pfc_firing_rate_past_R_mean, 'past_R_bg_sem': pfc_firing_rate_past_R_sem, 'pfc_past_L_bg': pfc_firing_rate_past_L_mean, 'past_L_bg_sem': pfc_firing_rate_past_L_sem, 'pfc_future_R_bg': pfc_firing_rate_future_R_mean, 'future_R_bg_sem': pfc_firing_rate_future_R_sem, 'pfc_future_L_bg': pfc_firing_rate_future_L_mean, 'future_L_bg_sem': pfc_firing_rate_future_L_sem, 'dms_past_R_bg': dms_firing_rate_past_R_mean, 'dms_past_R_bg_sem': dms_firing_rate_past_R_sem, 'dms_past_L_bg': dms_firing_rate_past_L_mean, 'dms_past_L_bg_sem': dms_firing_rate_past_L_sem, 'dms_future_R_bg': dms_firing_rate_future_R_mean, 'dms_future_R_bg_sem': dms_firing_rate_future_R_sem, 'dms_future_L_bg': dms_firing_rate_future_L_mean, 'dms_future_L_bg_sem': dms_firing_rate_future_L_sem, 'pfc_past_R_response': pfc_firing_rate_past_R_response_mean, 'pfc_past_R_response_sem': pfc_firing_rate_past_R_response_sem, 'pfc_past_L_response': pfc_firing_rate_past_L_response_mean, 'pfc_past_L_response_sem': pfc_firing_rate_past_L_response_sem, 'pfc_future_R_response': pfc_firing_rate_future_R_response_mean, 'pfc_future_R_response_sem': pfc_firing_rate_future_R_response_sem, 'pfc_future_L_response': pfc_firing_rate_future_L_response_mean, 'pfc_future_L_response_sem': pfc_firing_rate_future_L_response_sem, 'dms_past_R_response': dms_firing_rate_past_R_response_mean, 'dms_past_R_response_sem': dms_firing_rate_past_R_response_sem, 'dms_past_L_response': dms_firing_rate_past_L_response_mean, 'dms_past_L_response_sem': dms_firing_rate_past_L_response_sem, 'dms_future_R_response': dms_firing_rate_future_R_response_mean, 'dms_future_R_response_sem': dms_firing_rate_future_R_response_sem, 'dms_future_L_response': dms_firing_rate_future_L_response_mean, 'dms_future_L_response_sem': dms_firing_rate_future_L_response_sem, 'pfc_all_firing_bg': pfc_firing_all_mean, 'pfc_all_firing_bg_sem': pfc_firing_all_sem, 'dms_all_firing_bg': dms_firing_all_mean, 'dms_all_firing_bg_sem': dms_firing_all_sem, 'pfc_all_firing_response': pfc_firing_all_response_mean, 'pfc_all_firing_response_sem': pfc_firing_all_response_sem, 'dms_all_firing_response': dms_firing_all_response_mean, 'dms_all_firing_response_sem': dms_firing_all_response_sem})
    df.to_csv(pjoin(figure_5_data_root, 'figure_5_panel_gh.csv'), index=False)


def get_mean_and_sem(x, y):
    df = pd.DataFrame({'x': x, 'y': y})
    df_org = df.copy()
    df = df.groupby('x').mean()
    df['sem'] = df_org.groupby('x').sem()['y']

    return np.array(df['y']), np.array(df['sem'])