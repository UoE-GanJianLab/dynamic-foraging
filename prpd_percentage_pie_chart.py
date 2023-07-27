from os.path import join as pjoin, isdir
from os import mkdir
from glob import glob

import numpy as np
import pandas as pd

spike_firing_root = pjoin('data', 'spike_times', 'sessions')
all_mono_pairs = pd.read_csv(pjoin('data', 'mono_pairs.csv'))
prpd_correlated_response = pd.read_csv(pjoin('data', 'delta_P_correlated_mono_pairs_response.csv'))
prpd_correlated_background = pd.read_csv(pjoin('data', 'delta_P_correlated_mono_pairs_background.csv'))
prpd_correlated_all_cells = pd.read_csv(pjoin('data', 'delta_P_correlation.csv'))

# drop nan values
prpd_correlated_response = prpd_correlated_response.dropna()

figure_5_figure_root = pjoin('figures', 'all_figures', 'figure_5')
figure_5_data_root = pjoin('figure_data', 'figure_5')
figure_7_figure_root = pjoin('figures', 'all_figures', 'figure_7')
figure_7_data_root = pjoin('figure_data', 'figure_7')

for dir in [figure_5_figure_root, figure_5_data_root, figure_7_figure_root, figure_7_data_root]:
    if not isdir(dir):
        mkdir(dir)

significance_threshold = 0.05
 
def get_figure_5_panel_e():
    # prpd_correlated_all_cells have following columns:
    # session,background_firing_pearson_r,background_firing_p_values,response_firing_pearson_r,response_firing_p_values,cell
    # get the pfc cells(cell start with pfc) who are not correlated in background firing and correlated in response firing,
    # correlated in background firing only, correlated in response firing only, and correlated in both
    pfc_cells = prpd_correlated_all_cells[prpd_correlated_all_cells['cell'].str.startswith('pfc')]
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
    dms_cells = prpd_correlated_all_cells[prpd_correlated_all_cells['cell'].str.startswith('str')]
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

def get_figure_5_panel_f():
    # for each session, calculate the percentage cells of strongly positively and negatively correlated with prpd for pfc and dms
    sessions = prpd_correlated_all_cells['session'].unique()
    pfc_strongly_positively_correlated = []
    pfc_strongly_negatively_correlated = []
    pfc_strongly_positively_correlated_bg = []
    pfc_strongly_negatively_correlated_bg = []
    dms_strongly_positively_correlated = []
    dms_strongly_negatively_correlated = []
    dms_strongly_negatively_correlated_bg = []
    dms_strongly_positively_correlated_bg = []

    for session in sessions:
        session_cells = prpd_correlated_all_cells[prpd_correlated_all_cells['session'] == session]
        pfc_cells = session_cells[session_cells['cell'].str.startswith('pfc')]
        dms_cells = session_cells[session_cells['cell'].str.startswith('str')]

        total_pfc_cells = len(glob(pjoin(spike_firing_root, session, 'pfc_*')))
        total_dms_cells = len(glob(pjoin(spike_firing_root, session, 'dms_*')))

        # print('session: ', session)
        # print('total pfc cells: ', total_pfc_cells)
        # print('total dms cells: ', total_dms_cells)
        # print('pfc strongly positively correlated: ', len(pfc_cells[(pfc_cells['response_firing_pearson_r'] >= 0) & (pfc_cells['response_firing_p_values'] < significance_threshold)]))
        # # print(pfc_cells[(pfc_cells['response_firing_pearson_r'] >= 0) & (pfc_cells['response_firing_p_values'] < significance_threshold)])
        # print('pfc strongly negatively correlated: ', len(pfc_cells[(pfc_cells['response_firing_pearson_r'] < 0) & (pfc_cells['response_firing_p_values'] < significance_threshold)]))
        # # print(pfc_cells[(pfc_cells['response_firing_pearson_r'] < 0) & (pfc_cells['response_firing_p_values'] < significance_threshold)])

        pfc_strongly_positively_correlated.append(len(pfc_cells[(pfc_cells['response_firing_pearson_r'] >= 0) & (pfc_cells['response_firing_p_values'] < significance_threshold)])/ total_pfc_cells)
        pfc_strongly_negatively_correlated.append(len(pfc_cells[(pfc_cells['response_firing_pearson_r'] < 0) & (pfc_cells['response_firing_p_values'] < significance_threshold)])/ total_pfc_cells)
        pfc_strongly_negatively_correlated_bg.append(len(pfc_cells[(pfc_cells['background_firing_pearson_r'] < 0) & (pfc_cells['background_firing_p_values'] < significance_threshold)])/ total_pfc_cells)
        pfc_strongly_positively_correlated_bg.append(len(pfc_cells[(pfc_cells['background_firing_pearson_r'] >= 0) & (pfc_cells['background_firing_p_values'] < significance_threshold)])/ total_pfc_cells)
        dms_strongly_positively_correlated.append(len(dms_cells[(dms_cells['response_firing_pearson_r'] >= 0) & (dms_cells['response_firing_p_values'] < significance_threshold)])/ total_dms_cells)
        dms_strongly_negatively_correlated.append(len(dms_cells[(dms_cells['response_firing_pearson_r'] < 0) & (dms_cells['response_firing_p_values'] < significance_threshold)])/ total_dms_cells)
        dms_strongly_negatively_correlated_bg.append(len(dms_cells[(dms_cells['background_firing_pearson_r'] < 0) & (dms_cells['background_firing_p_values'] < significance_threshold)])/ total_dms_cells)
        dms_strongly_positively_correlated_bg.append(len(dms_cells[(dms_cells['background_firing_pearson_r'] >= 0) & (dms_cells['background_firing_p_values'] < significance_threshold)])/ total_dms_cells)
    
    # save the data to csv with these columns: |cell_location|firing_window|positively_correlated_percentage|positive_standard_error|negatively_correlated_percentage|negative_standard_error|
    panel_f_data = pd.DataFrame({'cell_location': ['pfc', 'pfc', 'dms', 'dms'], 'firing_window': ['ITI', 'response', 'ITI', 'response'], 'positively_correlated_percentage': [np.mean(pfc_strongly_positively_correlated_bg), np.mean(pfc_strongly_positively_correlated), np.mean(dms_strongly_positively_correlated_bg), np.mean(dms_strongly_positively_correlated)], 'positive_standard_error': [np.std(pfc_strongly_positively_correlated_bg)/np.sqrt(len(pfc_strongly_positively_correlated_bg)), np.std(pfc_strongly_positively_correlated)/np.sqrt(len(pfc_strongly_positively_correlated)), np.std(dms_strongly_positively_correlated_bg)/np.sqrt(len(dms_strongly_positively_correlated_bg)), np.std(dms_strongly_positively_correlated)/np.sqrt(len(dms_strongly_positively_correlated))], 'negatively_correlated_percentage': [np.mean(pfc_strongly_negatively_correlated_bg), np.mean(pfc_strongly_negatively_correlated), np.mean(dms_strongly_negatively_correlated_bg), np.mean(dms_strongly_negatively_correlated)], 'negative_standard_error': [np.std(pfc_strongly_negatively_correlated_bg)/np.sqrt(len(pfc_strongly_negatively_correlated_bg)), np.std(pfc_strongly_negatively_correlated)/np.sqrt(len(pfc_strongly_negatively_correlated)), np.std(dms_strongly_negatively_correlated_bg)/np.sqrt(len(dms_strongly_negatively_correlated_bg)), np.std(dms_strongly_negatively_correlated)/np.sqrt(len(dms_strongly_negatively_correlated))]})

    panel_f_data.to_csv(pjoin(figure_5_data_root, 'figure_5_panel_f.csv'), index=False)

def get_figure_7_panel_a():
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


get_figure_5_panel_e()
get_figure_5_panel_f()
get_figure_7_panel_a()