from os.path import join as pjoin
from os import listdir, mkdir
from os.path import basename, isfile, isdir

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
figure_5_panel_abcd_figure_root = pjoin(figure_5_figure_root, 'panel_abcd')
figure_5_panel_abcd_pfc_root = pjoin(figure_5_panel_abcd_figure_root, 'pfc')
figure_5_panel_abcd_dms_root = pjoin(figure_5_panel_abcd_figure_root, 'dms')

for dir in [figure_5_figure_root, figure_5_panel_abcd_figure_root, figure_5_panel_abcd_pfc_root, figure_5_panel_abcd_dms_root, figure_5_data_source, figure_5_panel_abcd_pfc_data_top, figure_5_panel_abcd_dms_data_top, figure_5_panel_abcd_pfc_data_bottom, figure_5_panel_abcd_dms_data_bottom]:
    if not isdir(dir):
        mkdir(dir)

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

def get_figure_4_panel_bc():
    for dir in listdir(spike_data_root):
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
