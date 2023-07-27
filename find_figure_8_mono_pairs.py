from os.path import join as pjoin
from os.path import isfile
from os import rename
import pandas as pd

mono_pairs = pd.read_csv(pjoin('data', 'mono_pairs.csv'))
figure_8_figure_path = pjoin('figures', 'all_figures', 'figure_8', 'panel_abc', 'significant')
figure_8_mono_path = pjoin(figure_8_figure_path, 'mono')

for ind, pair in mono_pairs.iterrows():
    if isfile(pjoin(figure_8_figure_path, '_'.join([pair['session'], pair['pfc'], pair['dms']])+'.png')):
        # move the file to the mono folder
        print('moving')
        rename(pjoin(figure_8_figure_path, '_'.join([pair['session'], pair['pfc'], pair['dms']])+'.png'),
                  pjoin(figure_8_mono_path, '_'.join([pair['session'], pair['pfc'], pair['dms']])+'.png'))