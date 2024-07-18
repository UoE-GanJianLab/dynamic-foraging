from os.path import join as pjoin, isdir
from os import mkdir
from glob import glob

import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

spike_firing_root = pjoin('data', 'spike_times', 'sessions')
all_mono_pairs = pd.read_csv(pjoin('data', 'mono_pairs.csv'))
prpd_correlated = pd.read_csv(pjoin('data', 'prpd_correlation.csv'))
relative_value_correlated = pd.read_csv(pjoin('data', 'relative_value_correlation.csv'))

figure_6_figure_root = pjoin('figures', 'all_figures', 'figure_6')
figure_6_data_root = pjoin('figure_data', 'figure_6')

significance_level = 0.05

mono_total = len(all_mono_pairs)

def get_figure_6_panel_b():
    response_only_mono_pairs_percent = 0
    background_only_mono_pairs_percent = 0
    both_mono_pairs_percent = 0

    response_only_mono_pairs_percent_relative_value = 0
    background_only_mono_pairs_percent_relative_value = 0
    both_mono_pairs_percent_relative_value = 0

    for ind, pair in all_mono_pairs.iterrows():
        session_name = pair['session']
        dms_cell_name = pair['dms']
        pfc_cell_name = pair['pfc']

        pfc_prpd_correlation_entry = prpd_correlated[(prpd_correlated['session'] == session_name) & (prpd_correlated['cell'] == pfc_cell_name)]
        dms_prpd_correlation_entry = prpd_correlated[(prpd_correlated['session'] == session_name) & (prpd_correlated['cell'] == dms_cell_name)]

        if pfc_prpd_correlation_entry.empty or dms_prpd_correlation_entry.empty:
            continue

        response_correlation = False
        background_correlation = False

        if pfc_prpd_correlation_entry['response_firing_p_values'].values[0] < significance_level and dms_prpd_correlation_entry['response_firing_p_values'].values[0] < significance_level:
            response_correlation = True
        
        if pfc_prpd_correlation_entry['background_firing_p_values'].values[0] < significance_level and dms_prpd_correlation_entry['background_firing_p_values'].values[0] < significance_level:
            background_correlation = True

        if response_correlation and background_correlation:
            both_mono_pairs_percent += 1
        elif response_correlation:
            response_only_mono_pairs_percent += 1
        elif background_correlation:
            background_only_mono_pairs_percent += 1
        
        pfc_relative_value_correlation_entry = relative_value_correlated[(relative_value_correlated['session'] == session_name) & (relative_value_correlated['cell'] == pfc_cell_name)]
        dms_relative_value_correlation_entry = relative_value_correlated[(relative_value_correlated['session'] == session_name) & (relative_value_correlated['cell'] == dms_cell_name)]
        
        if pfc_relative_value_correlation_entry.empty or dms_relative_value_correlation_entry.empty:
            continue
    
        response_correlation = False
        background_correlation = False

        if pfc_relative_value_correlation_entry['response_firing_p_values'].values[0] < significance_level and dms_relative_value_correlation_entry['response_firing_p_values'].values[0] < significance_level:
            response_correlation = True
        
        if pfc_relative_value_correlation_entry['background_firing_p_values'].values[0] < significance_level and dms_relative_value_correlation_entry['background_firing_p_values'].values[0] < significance_level:
            background_correlation = True

        if response_correlation and background_correlation:
            both_mono_pairs_percent_relative_value += 1
        elif response_correlation:
            response_only_mono_pairs_percent_relative_value += 1
        elif background_correlation:
            background_only_mono_pairs_percent_relative_value += 1

    # count the number of mono pairs
    total_mono_pairs = len(all_mono_pairs)
    
    response_only_mono_pairs_percent = response_only_mono_pairs_percent / total_mono_pairs
    background_only_mono_pairs_percent = background_only_mono_pairs_percent / total_mono_pairs
    both_mono_pairs_percent = both_mono_pairs_percent / total_mono_pairs
    neither_mono_pairs_percent = 1 - response_only_mono_pairs_percent - background_only_mono_pairs_percent - both_mono_pairs_percent

    response_only_mono_pairs_percent_relative_value = response_only_mono_pairs_percent_relative_value / total_mono_pairs
    background_only_mono_pairs_percent_relative_value = background_only_mono_pairs_percent_relative_value / total_mono_pairs
    both_mono_pairs_percent_relative_value = both_mono_pairs_percent_relative_value / total_mono_pairs
    neither_mono_pairs_percent_relative_value = 1 - response_only_mono_pairs_percent_relative_value - background_only_mono_pairs_percent_relative_value - both_mono_pairs_percent_relative_value

    # |correlation|percentage|count|
    panel_b_data = pd.DataFrame({'correlation': ['not correlated', 'response only', 'background only', 'both'], 'percentage': [neither_mono_pairs_percent, response_only_mono_pairs_percent, background_only_mono_pairs_percent, both_mono_pairs_percent], 'count': [int(neither_mono_pairs_percent * mono_total), int(response_only_mono_pairs_percent * mono_total), int(background_only_mono_pairs_percent * mono_total), int(both_mono_pairs_percent * mono_total)]})
    panel_b_data.to_csv(pjoin(figure_6_data_root, 'figure_6_panel_b_prpd.csv'), index=False)

    panel_b_data_relative_value = pd.DataFrame({'correlation': ['not correlated', 'response only', 'background only', 'both'], 'percentage': [neither_mono_pairs_percent_relative_value, response_only_mono_pairs_percent_relative_value, background_only_mono_pairs_percent_relative_value, both_mono_pairs_percent_relative_value], 'count': [int(neither_mono_pairs_percent_relative_value * mono_total), int(response_only_mono_pairs_percent_relative_value * mono_total), int(background_only_mono_pairs_percent_relative_value * mono_total), int(both_mono_pairs_percent_relative_value * mono_total)]})
    panel_b_data_relative_value.to_csv(pjoin(figure_6_data_root, 'figure_6_panel_b_relative_value.csv'), index=False)

    # show the result in a pie chart
    plt.figure(figsize=(4, 4))
    plt.pie([neither_mono_pairs_percent, response_only_mono_pairs_percent, background_only_mono_pairs_percent, both_mono_pairs_percent], labels=[f'not correlated {int(neither_mono_pairs_percent * mono_total)}', f'response only {int(response_only_mono_pairs_percent * mono_total)}', f'background only {int(background_only_mono_pairs_percent * mono_total)}', f'both {both_mono_pairs_percent * mono_total}'], autopct='%1.1f%%')
