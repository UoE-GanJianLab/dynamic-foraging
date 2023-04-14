from os.path import join as pjoin
from os import listdir
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np

from lib.conversion import one_to_zero_cell, zero_to_one_cell

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

def get_session_performances() -> Tuple[Dict[str, float], float]:
    results = {}
    performances = []
    # calculates the xcorr between rightP and proportion of rightward resonse
    for behaviour_path in listdir(behaviour_root):
        session_name = behaviour_path.split('.')[0]
        session_data = pd.read_csv(pjoin(behaviour_root, behaviour_path))
        right_prob = session_data['rightP']
        left_prob = session_data['leftP']
        right_proportion = np.convolve((session_data['trial_response_side']==1).to_numpy(), np.ones(20)/20, mode='same')
        left_proportion = np.convolve((session_data['trial_response_side']==-1).to_numpy(), np.ones(20)/20, mode='same')
        # normalization
        right_prob = np.array(right_prob) / np.linalg.norm(right_prob)
        left_prob = np.array(left_prob) / np.linalg.norm(left_prob)
        right_proportion = right_proportion / np.linalg.norm(right_proportion)
        left_proportion = left_proportion / np.linalg.norm(left_proportion)

        xcorr_right = np.correlate(right_prob, right_proportion)
        xcorr_left = np.correlate(left_prob, left_proportion)
        corrs = [xcorr_left, xcorr_right]
        corrs_averaged = np.mean(corrs, axis=0)
        performance = np.max(corrs_averaged)
        performances.append(performance)
        results[session_name] = performance
    
    return results, np.mean(performances)