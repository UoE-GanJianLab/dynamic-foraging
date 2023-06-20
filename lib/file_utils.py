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

def get_str_pfc(session_name: str) -> Tuple[Dict, Dict]:
    str_times = {}
    pfc_times = {}
    session_root = pjoin(spike_data_root, session_name)
    for pfc_cell in glob(pjoin(session_root, 'pfc_*')):
        pfc_times[basename(pfc_cell).split('.')[0]] = np.load(pfc_cell)
    for str_cell in glob(pjoin(session_root, 'str_*')):
        str_times[basename(str_cell).split('.')[0]] = np.load(str_cell)
    return str_times, pfc_times

# return the session_name, cue_times, and all pfc_str pairs' paths from each session
def get_str_pfc_paths_all(no_nan=False) -> List[Tuple[str, np.ndarray, np.ndarray, List[List[str]]]]:
    session_names = get_session_names()
    str_pfc_pair_paths = []
    for session_name in session_names:
        session_data_path = pjoin(behaviour_root, session_name+'.csv')
        session_data = pd.read_csv(session_data_path)
        if no_nan:
            session_data = session_data[session_data['trial_response_side'].notna()]
        else:
            session_data.fillna(0, inplace=True)
        cue_times = session_data['cue_time'].values
        trial_reward = session_data['trial_reward'].values
        session_root = pjoin(spike_data_root, session_name)
        cell_pairs = []
        for str_cell in glob(pjoin(session_root, 'str_*')):
            for pfc_cell in glob(pjoin(session_root, 'pfc_*')):
                cell_pair = [str_cell, pfc_cell]
                cell_pairs.append(cell_pair)
        str_pfc_pair_paths.append([session_name, cue_times, trial_reward, cell_pairs])
    return str_pfc_pair_paths

# return the session_name, cue_times, and all PMSE pfc_str pairs' paths from each session
def get_str_pfc_paths_mono(no_nan=False) -> pd.DataFrame:
    # get all mono pairs
    mono_pairs = pd.read_csv(pjoin('data', 'mono_pairs.csv'))

    session_paths = []
    str_paths = []
    pfc_paths = []

    # create the session_path colomn using session
    # create the str_path and pfc_path colomn using str and pfc
    # do this using apply
    mono_pairs["session_path"] = mono_pairs.apply(lambda row: pjoin(behaviour_root, row['session']), axis=1)
    mono_pairs["session_path"] = mono_pairs["session_path"] + '.csv'
    mono_pairs["str_path"] = mono_pairs["str"] + '.npy'
    mono_pairs["pfc_path"] = mono_pairs["pfc"] + '.npy'
    mono_pairs["str_path"] = mono_pairs.apply(lambda row: pjoin(spike_data_root, row['session'], row['str_path']), axis=1)
    mono_pairs["pfc_path"] = mono_pairs.apply(lambda row: pjoin(spike_data_root, row['session'], row['pfc_path']), axis=1)
    
    # return the result as a dataframe
    result = mono_pairs[["session_path", "str_path", "pfc_path"]]
    
    return result

strong_corr_iti_path = pjoin('data', "delta_P_correlated_mono_pairs_background.csv")
strong_corr_path = pjoin('data', "delta_P_correlated_mono_pairs_response.csv")

# using this instead of static path for compatibility with both windows and linux systems
# returns two dataframe of cell pair data, first for iti, then for response window
# instead of the 
def get_str_pfc_strong_corr_mono():
    strong_corr_iti = pd.read_csv(strong_corr_iti_path)
    strong_corr = pd.read_csv(strong_corr_path)

    strong_corr_iti["session_path"] = strong_corr_iti.apply(lambda row: pjoin(behaviour_root, row['session']), axis=1)
    strong_corr["session_path"] = strong_corr.apply(lambda row: pjoin(behaviour_root, row['session']), axis=1)
    strong_corr_iti["session_path"] = strong_corr_iti["session_path"] + '.csv'
    strong_corr["session_path"] = strong_corr["session_path"] + '.csv'

    strong_corr_iti["str_path"] = strong_corr_iti["str_name"] + '.npy'
    strong_corr_iti["pfc_path"] = strong_corr_iti["pfc_name"] + '.npy'

    strong_corr["str_path"] = strong_corr["str_name"] + '.npy'
    strong_corr["pfc_path"] = strong_corr["pfc_name"] + '.npy'

    strong_corr_iti["str_path"] = strong_corr_iti.apply(lambda row: pjoin(spike_data_root, row['session'], row['str_path']), axis=1)
    strong_corr_iti["pfc_path"] = strong_corr_iti.apply(lambda row: pjoin(spike_data_root, row['session'], row['pfc_path']), axis=1)

    strong_corr["str_path"] = strong_corr.apply(lambda row: pjoin(spike_data_root, row['session'], row['str_path']), axis=1)
    strong_corr["pfc_path"] = strong_corr.apply(lambda row: pjoin(spike_data_root, row['session'], row['pfc_path']), axis=1)

    return strong_corr_iti[['session_path', 'str_path', 'pfc_path']], strong_corr[['session_path', 'str_path', 'pfc_path']]
