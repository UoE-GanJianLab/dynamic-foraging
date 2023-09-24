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

from lib.calculation import get_firing_rate_window

WINDOW_LEFT = 0.5
WINDOW_RIGHT = 1.5
# 1/BIN_WIDTH must be an integer
BIN_WIDTH = 0.02

spike_data_root = pjoin('data', 'spike_times', 'sessions')

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
        relative = False
        if isfile(pjoin('data', 'relative_values', session_name+'.npy')):
            relative_values = np.load(pjoin('data', 'relative_values', session_name+'.npy'))
            relative = True

        # load the firing data
        for cell in all_cells:
            cell_name = basename(cell).split('.')[0]
            firing_data = np.load(cell, allow_pickle=False)
            firing_rate_response = np.array(get_firing_rate_window(cue_times, firing_data, RESPONSE_WINDOW_LEFT, RESPONSE_WINDOW_RIGHT))
            firing_rate_bg= np.array(get_firing_rate_window(cue_times, firing_data, ITI_WINDOW_LEFT, ITI_WINDOW_RIGHT))

            # calculate the pearson correlation coefficient and p value for prpd and firing rate
            prpd_correlation_response, prpd_p_value_response = scipy.stats.pearsonr(prpd, firing_rate_response)
            prpd_correlation_bg, prpd_p_value_bg = scipy.stats.pearsonr(prpd, firing_rate_bg)

            session_names_prpd.append(session_name)
            cell_names_prpd.append(cell_name)
            prpd_correlations_response.append(prpd_correlation_response)
            prpd_p_values_response.append(prpd_p_value_response)
            prpd_correlations_bg.append(prpd_correlation_bg)
            prpd_p_values_bg.append(prpd_p_value_bg)

            if relative:
                relative_value_correlation_response, relative_value_p_value_response = scipy.stats.pearsonr(relative_values, firing_rate_response)
                relative_value_correlation_bg, relative_value_p_value_bg = scipy.stats.pearsonr(relative_values, firing_rate_bg)

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

    # remove rows with nan values
    correlation_data = correlation_data.dropna()

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
    dms_cells = correlation_data[correlation_data['cell'].str.startswith('str')]
    total_dms_cells = len(glob(pjoin(spike_firing_root, '*', 'dms_*')))

    dms_cells__background_only = dms_cells[(dms_cells['background_firing_p_values'] < significance_threshold) & (dms_cells['response_firing_p_values'] >= significance_threshold)]
    dms_cells__background_only_percent = len(dms_cells__background_only) / total_dms_cells
    dms_cells_response_only = dms_cells[(dms_cells['background_firing_p_values'] >= significance_threshold) & (dms_cells['response_firing_p_values'] < significance_threshold)]
    dms_cells_response_only_percent = len(dms_cells_response_only) / total_dms_cells
    dms_cells_both = dms_cells[(dms_cells['background_firing_p_values'] < significance_threshold) & (dms_cells['response_firing_p_values'] < significance_threshold)]
    dms_cells_both_percent = len(dms_cells_both) / total_dms_cells
    dms_cells_neither = dms_cells[(dms_cells['background_firing_p_values'] >= significance_threshold) & (dms_cells['response_firing_p_values'] >- significance_threshold)]
    dms_cells_neither_percent = len(dms_cells_neither) / total_dms_cells

    # save the data to csv with these columns: |cell_location|not-correlated percentage|ITI firing correlated percentage|
    # response magnitude correlated percentage|both ITI firing and response magnitude correlated percentage|
    panel_e_data = pd.DataFrame({'cell_location': ['pfc', 'dms'], 'not-correlated percentage': [pfc_cells_neither_percent, dms_cells_neither_percent], 'ITI firing correlated percentage': [pfc_cells__background_only_percent, dms_cells__background_only_percent], 'response magnitude correlated percentage': [pfc_cells_response_only_percent, dms_cells_response_only_percent], 'both ITI firing and response magnitude correlated percentage': [pfc_cells_both_percent, dms_cells_both_percent]})
    panel_e_data.to_csv(pjoin(figure_5_data_root, 'figure_5_panel_e.csv'), index=False)

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

    # remove rows with nan values
    correlation_data = correlation_data.dropna()

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