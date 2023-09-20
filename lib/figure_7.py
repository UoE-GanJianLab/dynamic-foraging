from os.path import join as pjoin, isdir
from os import mkdir
from glob import glob

import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

spike_firing_root = pjoin('data', 'spike_times', 'sessions')
all_mono_pairs = pd.read_csv(pjoin('data', 'mono_pairs.csv'))
prpd_correlated_response = pd.read_csv(pjoin('data', 'delta_P_correlated_mono_pairs_response.csv'))
prpd_correlated_background = pd.read_csv(pjoin('data', 'delta_P_correlated_mono_pairs_background.csv'))
prpd_correlated_all_cells = pd.read_csv(pjoin('data', 'delta_P_correlation.csv'))

figure_7_figure_root = pjoin('figures', 'all_figures', 'figure_7')
figure_7_data_root = pjoin('figure_data', 'figure_7')

def get_figure_7_panel_b():
    # find the common entries in prpd_correlated_response and prpd_correlated_background
    common_entries = pd.merge(prpd_correlated_response, prpd_correlated_background, on=['session', 'pfc_name', 'str_name'])
    # find the entries in prpd_correlated_response but not in prpd_correlated_background
    response_only_entries = prpd_correlated_response[~prpd_correlated_response.isin(common_entries)].dropna()
    # find the entries in prpd_correlated_background but not in prpd_correlated_response
    background_only_entries = prpd_correlated_background[~prpd_correlated_background.isin(common_entries)].dropna()

    # count the number of mono pairs
    total_mono_pairs = len(all_mono_pairs)
    # calculate the percentage of not correlated mono pairs, correlated in response firing 
    # only, correlated in background firing only, and correlated in both
    response_only_mono_pairs_percent = len(response_only_entries)/total_mono_pairs
    background_only_mono_pairs_percent = len(background_only_entries)/total_mono_pairs
    both_mono_pairs_percent = len(common_entries)/total_mono_pairs
    neither_mono_pairs_percent = 1 - response_only_mono_pairs_percent - background_only_mono_pairs_percent - both_mono_pairs_percent

    # |correlation|percentage|
    panel_a_data = pd.DataFrame({'correlation': ['not correlated', 'response only', 'background only', 'both'], 'percentage': [neither_mono_pairs_percent, response_only_mono_pairs_percent, background_only_mono_pairs_percent, both_mono_pairs_percent]})
    panel_a_data.to_csv(pjoin(figure_7_data_root, 'figure_7_panel_a.csv'), index=False)

    # show the result in a pie chart
    plt.figure(figsize=(4, 4))
    plt.pie([neither_mono_pairs_percent, response_only_mono_pairs_percent, background_only_mono_pairs_percent, both_mono_pairs_percent], labels=['not correlated', 'response only', 'background only', 'both'], autopct='%1.1f%%')
