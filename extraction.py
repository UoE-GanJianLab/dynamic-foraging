from os.path import join as pjoin
from os import listdir

import pandas as pd
import numpy as np

from conversion import one_to_zero_cell, zero_to_one_cell

spike_data_root = pjoin("data", "spike_times")
behaviour_root = pjoin("data", "behaviour_data", "csv", "task_info")
strong_corr_iti_path = "strong_correlation_pairs_ITI.csv"
strong_corr_path = "strong_correlation_pairs.csv"

# using this instead of static path for compatibility with both windows and linux systems
# returns two dataframe of cell pair data, first for iti, then for response window
def get_strong_corr():
    strong_corr_iti = pd.read_csv(strong_corr_iti_path)
    strong_corr = pd.read_csv(strong_corr_path)

    strong_corr_iti["session_name"] = strong_corr_iti["mouse"] + strong_corr_iti["date"].astype(str)
    strong_corr["session_name"] = strong_corr["mouse"] + strong_corr["date"].astype(str)

    strong_corr_iti["session_path"] = strong_corr_iti.apply(lambda row: pjoin(behaviour_root, row['session_name']), axis=1)
    strong_corr["session_path"] = strong_corr.apply(lambda row: pjoin(behaviour_root, row['session_name']), axis=1)
    strong_corr_iti["session_path"] = strong_corr_iti["session_path"] + '.csv'
    strong_corr["session_path"] = strong_corr["session_path"] + '.csv'

    strong_corr_iti["str_path"] = strong_corr_iti["str_name"].apply(one_to_zero_cell) + '.npy'
    strong_corr_iti["pfc_path"] = strong_corr_iti["pfc_name"].apply(one_to_zero_cell) + '.npy'

    strong_corr["str_path"] = strong_corr["str_name"].apply(one_to_zero_cell)+'.npy'
    strong_corr["pfc_path"] = strong_corr["pfc_name"].apply(one_to_zero_cell)+'.npy'

    strong_corr_iti["str_path"] = strong_corr_iti.apply(lambda row: pjoin(spike_data_root, row['session_name'], row['str_path']), axis=1)
    strong_corr_iti["pfc_path"] = strong_corr_iti.apply(lambda row: pjoin(spike_data_root, row['session_name'], row['pfc_path']), axis=1)

    strong_corr["str_path"] = strong_corr.apply(lambda row: pjoin(spike_data_root, row['session_name'], row['str_path']), axis=1)
    strong_corr["pfc_path"] = strong_corr.apply(lambda row: pjoin(spike_data_root, row['session_name'], row['pfc_path']), axis=1)

    return strong_corr_iti[['session_path', 'str_path', 'pfc_path']], strong_corr[['session_path', 'str_path', 'pfc_path']]