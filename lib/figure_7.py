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

figure_7_figure_root = pjoin('figures', 'all_figures', 'figure_7')
figure_7_data_root = pjoin('figure_data', 'figure_7')

significance_level = 0.05

def get_figure_7_panel_b(prpd=False):
    response_only_mono_pairs_percent = 0
    background_only_mono_pairs_percent = 0
    both_mono_pairs_percent = 0

    for ind, pair in all_mono_pairs.iterrows():
        session_name = pair['session']
        dms_cell_name = pair['dms']
        pfc_cell_name = pair['pfc']

        pfc_prpd_correlation_entry = prpd_correlated[(prpd_correlated['session'] == session_name) & (prpd_correlated['cell'] == pfc_cell_name)]
        dms_prpd_correlation_entry = prpd_correlated[(prpd_correlated['session'] == session_name) & (prpd_correlated['cell'] == dms_cell_name)]

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

    # count the number of mono pairs
    total_mono_pairs = len(all_mono_pairs)
    
    response_only_mono_pairs_percent = response_only_mono_pairs_percent / total_mono_pairs
    background_only_mono_pairs_percent = background_only_mono_pairs_percent / total_mono_pairs
    both_mono_pairs_percent = both_mono_pairs_percent / total_mono_pairs
    neither_mono_pairs_percent = 1 - response_only_mono_pairs_percent - background_only_mono_pairs_percent - both_mono_pairs_percent

    # |correlation|percentage|
    panel_a_data = pd.DataFrame({'correlation': ['not correlated', 'response only', 'background only', 'both'], 'percentage': [neither_mono_pairs_percent, response_only_mono_pairs_percent, background_only_mono_pairs_percent, both_mono_pairs_percent]})
    panel_a_data.to_csv(pjoin(figure_7_data_root, 'figure_7_panel_a.csv'), index=False)

    # show the result in a pie chart
    plt.figure(figsize=(4, 4))
    plt.pie([neither_mono_pairs_percent, response_only_mono_pairs_percent, background_only_mono_pairs_percent, both_mono_pairs_percent], labels=['not correlated', 'response only', 'background only', 'both'], autopct='%1.1f%%')
