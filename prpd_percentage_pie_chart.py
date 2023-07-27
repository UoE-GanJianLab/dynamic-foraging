from os.path import join as pjoin

import pandas as pd

all_mono_pairs = pd.read_csv(pjoin('data', 'mono_pairs.csv'))
prpd_correlated_response = pd.read_csv(pjoin('data', 'delta_P_correlated_mono_pairs_response.csv'))
prpd_correlated_background = pd.read_csv(pjoin('data', 'delta_P_correlated_mono_pairs_background.csv'))

figure_5_figure_root = pjoin('figures', 'all_figures', 'figure_5')
figure_7_figure_root = pjoin('figures', 'all_figures', 'figure_7')

