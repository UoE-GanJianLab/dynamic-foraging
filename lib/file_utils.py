from glob import glob
from os.path import join as pjoin, basename
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from lib.conversion import one_to_zero_cell

spike_data_root = pjoin('data', 'spike_times', 'sessions')
behaviour_root = pjoin('data', 'behaviour_data')


def get_session_names() -> List[str]:
    sessions = glob(pjoin(behaviour_root, '*.csv'))
    session_names = [s.split('/')[-1].split('.')[0] for s in sessions]
    return session_names

def get_dms_pfc(session_name: str) -> Tuple[Dict, Dict]:
    dms_times = {}
    pfc_times = {}
    session_root = pjoin(spike_data_root, session_name)
    for pfc_cell in glob(pjoin(session_root, 'dms_*')):
        pfc_times[basename(pfc_cell).split('.')[0]] = np.load(pfc_cell)
    for dms_cell in glob(pjoin(session_root, 'dms_*')):
        dms_times[basename(dms_cell).split('.')[0]] = np.load(dms_cell)
    return dms_times, pfc_times

# return the session_name, cue_times, and all pfc_dms pairs' paths from each session
def get_dms_pfc_paths_all(no_nan=False) -> List[Tuple[str, np.ndarray, np.ndarray, List[List[str]]]]:
    session_names = get_session_names()
    dms_pfc_pair_paths = []
    for session_name in session_names:
        session_data_path = pjoin(behaviour_root, session_name+'.csv')
        session_data = pd.read_csv(session_data_path)
        if no_nan:
            session_data = session_data[session_data['trial_response_side'].notna()]
        cue_times = session_data['cue_time'].values
        # fill the nan with 0
        trial_reward = np.array(session_data['trial_reward'].values)
        trial_reward[np.isnan(trial_reward)] = 0
        session_root = pjoin(spike_data_root, session_name)
        cell_pairs = []
        for dms_cell in glob(pjoin(session_root, 'dms_*')):
            for pfc_cell in glob(pjoin(session_root, 'pfc_*')):
                cell_pair = [dms_cell, pfc_cell]
                cell_pairs.append(cell_pair)
        dms_pfc_pair_paths.append([session_name, cue_times, trial_reward, cell_pairs])
    return dms_pfc_pair_paths

# return the session_name, cue_times, and all PMSE pfc_dms pairs' paths from each session
def get_dms_pfc_paths_mono(no_nan=False) -> pd.DataFrame:
    # get all mono pairs
    mono_pairs = pd.read_csv(pjoin('data', 'mono_pairs.csv'))

    session_paths = []
    dms_paths = []
    pfc_paths = []

    # create the session_path colomn using session
    # create the dms_path and pfc_path colomn using dms and pfc
    # do this using apply
    mono_pairs["session_path"] = mono_pairs.apply(lambda row: pjoin(behaviour_root, row['session']), axis=1)
    mono_pairs["session_path"] = mono_pairs["session_path"] + '.csv'
    mono_pairs["dms_path"] = mono_pairs["dms"] + '.npy'
    mono_pairs["pfc_path"] = mono_pairs["pfc"] + '.npy'
    mono_pairs["dms_path"] = mono_pairs.apply(lambda row: pjoin(spike_data_root, row['session'], row['dms_path']), axis=1)
    mono_pairs["pfc_path"] = mono_pairs.apply(lambda row: pjoin(spike_data_root, row['session'], row['pfc_path']), axis=1)
    
    # return the result as a dataframe
    result = mono_pairs[["session_path", "dms_path", "pfc_path"]]
    
    return result

strong_corr_iti_path = pjoin('data', "delta_P_correlated_mono_pairs_background.csv")
strong_corr_path = pjoin('data', "delta_P_correlated_mono_pairs_response.csv")

# using this instead of static path for compatibility with both windows and linux systems
# returns two dataframe of cell pair data, first for iti, then for response window
# instead of the 
def get_dms_pfc_strong_corr_mono():
    strong_corr_iti = pd.read_csv(strong_corr_iti_path)
    strong_corr = pd.read_csv(strong_corr_path)

    strong_corr_iti["session_path"] = strong_corr_iti.apply(lambda row: pjoin(behaviour_root, row['session']), axis=1)
    strong_corr["session_path"] = strong_corr.apply(lambda row: pjoin(behaviour_root, row['session']), axis=1)
    strong_corr_iti["session_path"] = strong_corr_iti["session_path"] + '.csv'
    strong_corr["session_path"] = strong_corr["session_path"] + '.csv'

    strong_corr_iti["dms_path"] = strong_corr_iti["dms_name"] + '.npy'
    strong_corr_iti["pfc_path"] = strong_corr_iti["pfc_name"] + '.npy'

    strong_corr["dms_path"] = strong_corr["dms_name"] + '.npy'
    strong_corr["pfc_path"] = strong_corr["pfc_name"] + '.npy'

    strong_corr_iti["dms_path"] = strong_corr_iti.apply(lambda row: pjoin(spike_data_root, row['session'], row['dms_path']), axis=1)
    strong_corr_iti["pfc_path"] = strong_corr_iti.apply(lambda row: pjoin(spike_data_root, row['session'], row['pfc_path']), axis=1)

    strong_corr["dms_path"] = strong_corr.apply(lambda row: pjoin(spike_data_root, row['session'], row['dms_path']), axis=1)
    strong_corr["pfc_path"] = strong_corr.apply(lambda row: pjoin(spike_data_root, row['session'], row['pfc_path']), axis=1)

    return strong_corr_iti[['session_path', 'dms_path', 'pfc_path']], strong_corr[['session_path', 'dms_path', 'pfc_path']]
