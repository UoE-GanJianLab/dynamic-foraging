from glob import glob
from os.path import join as pjoin, basename
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from lib.conversion import one_to_zero_cell

def get_session_names() -> List[str]:
    session_root = pjoin('data', 'behaviour_data')
    sessions = glob(pjoin(session_root, '*.csv'))
    session_names = [s.split('/')[-1].split('.')[0] for s in sessions]
    return session_names

def get_str_pfc(session_name: str) -> Tuple[Dict, Dict]:
    str_times = {}
    pfc_times = {}
    session_root = pjoin('data', 'spike_times', session_name)
    for pfc_cell in glob(pjoin(session_root, 'pfc_*')):
        pfc_times[basename(pfc_cell).split('.')[0]] = np.load(pfc_cell)
    for str_cell in glob(pjoin(session_root, 'str_*')):
        str_times[basename(str_cell).split('.')[0]] = np.load(str_cell)
    return str_times, pfc_times

# return the session_name, cue_times, and the pfc_str paths from each session
def get_str_pfc_paths_all(no_nan=False) -> List[Tuple[str, np.ndarray, np.ndarray, List[List[str]]]]:
    spike_dat_root = pjoin('data', 'spike_times')
    session_names = get_session_names()
    str_pfc_pair_paths = []
    for session_name in session_names:
        session_data_path = pjoin('data', 'behaviour_data', session_name+'.csv')
        session_data = pd.read_csv(session_data_path)
        if no_nan:
            session_data = session_data[session_data['trial_response_side'].notna()]
        else:
            session_data.fillna(0, inplace=True)
        cue_times = session_data['cue_time'].values
        trial_reward = session_data['trial_reward'].values
        session_root = pjoin(spike_dat_root, session_name)
        cell_pairs = []
        for str_cell in glob(pjoin(session_root, 'str_*')):
            for pfc_cell in glob(pjoin(session_root, 'pfc_*')):
                cell_pair = [str_cell, pfc_cell]
                cell_pairs.append(cell_pair)
        str_pfc_pair_paths.append([session_name, cue_times, trial_reward, cell_pairs])
    return str_pfc_pair_paths

def get_str_pfc_paths_mono(no_nan=False) -> List[Tuple[str, np.ndarray, np.ndarray, List[List[str]]]]:
    # get all mono pairs
    mono_pairs = pd.read_csv('mono_pairs.csv')

    result = []

    # concatenate mouse and date column of the dataframe to get session name
    # convert 'date' to string
    mono_pairs['date'] = mono_pairs['date'].astype(str)
    mono_pairs['session_names'] = mono_pairs['mouse'] + mono_pairs['date']
    session_names = mono_pairs['session_names'].values
    session_names = np.unique(session_names)

    for session_name in session_names:
        session_data_path = pjoin('data', 'behaviour_data', session_name+'.csv')
        session_data = pd.read_csv(session_data_path)
        if no_nan:
            session_data = session_data[session_data['trial_response_side'].notna()]
        else:
            session_data.fillna(0, inplace=True)
        cue_times = session_data['cue_time'].values
        trial_reward = session_data['trial_reward'].values

        session_pairs = mono_pairs[mono_pairs['mouse']+mono_pairs['date']==session_name]

        str_pfc_paths = []

        for _, row in session_pairs.iterrows():
            str_name = row['str_name']
            pfc_name = row['pfc_name']
            # change index from 1 based to 0 based
            str_name = one_to_zero_cell(str_name)
            pfc_name = one_to_zero_cell(pfc_name)
            str_path = pjoin('data', 'spike_times', session_name, str_name+'.npy')
            pfc_path = pjoin('data', 'spike_times', session_name, pfc_name+'.npy')
            str_pfc_paths.append([str_path, pfc_path])
        
        result.append([session_name, cue_times, trial_reward, str_pfc_paths])
    
    return result


spike_data_root = pjoin("data", "spike_times")
behaviour_root = pjoin("data", "behaviour_data", "csv", "task_info")
strong_corr_iti_path = "strong_correlation_pairs_ITI.csv"
strong_corr_path = "strong_correlation_pairs.csv"

# using this instead of static path for compatibility with both windows and linux systems
# returns two dataframe of cell pair data, first for iti, then for response window
def get_strong_corr():
    strong_corr_iti = pd.read_csv(strong_corr_iti_path)
    strong_corr = pd.read_csv(strong_corr_path)

    strong_corr_iti["session_path"] = strong_corr_iti.apply(lambda row: pjoin(behaviour_root, row['session']), axis=1)
    strong_corr["session_path"] = strong_corr.apply(lambda row: pjoin(behaviour_root, row['session']), axis=1)
    strong_corr_iti["session_path"] = strong_corr_iti["session_path"] + '.csv'
    strong_corr["session_path"] = strong_corr["session_path"] + '.csv'

    strong_corr_iti["str_path"] = strong_corr_iti["str"] + '.npy'
    strong_corr_iti["pfc_path"] = strong_corr_iti["pfc"] + '.npy'

    strong_corr["str_path"] = strong_corr["str_name"] + '.npy'
    strong_corr["pfc_path"] = strong_corr["pfc_name"] + '.npy'

    strong_corr_iti["str_path"] = strong_corr_iti.apply(lambda row: pjoin(spike_data_root, row['session_name'], row['str_path']), axis=1)
    strong_corr_iti["pfc_path"] = strong_corr_iti.apply(lambda row: pjoin(spike_data_root, row['session_name'], row['pfc_path']), axis=1)

    strong_corr["str_path"] = strong_corr.apply(lambda row: pjoin(spike_data_root, row['session_name'], row['str_path']), axis=1)
    strong_corr["pfc_path"] = strong_corr.apply(lambda row: pjoin(spike_data_root, row['session_name'], row['pfc_path']), axis=1)

    return strong_corr_iti[['session_path', 'str_path', 'pfc_path']], strong_corr[['session_path', 'str_path', 'pfc_path']]
