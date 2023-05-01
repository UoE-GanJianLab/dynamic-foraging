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
def get_str_pfc_paths_session(no_nan=False) -> List[Tuple[str, np.ndarray, np.ndarray, List[List[str]]]]:
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
        for str_cell in glob(pjoin(session_root, 'str_*')):
            for pfc_cell in glob(pjoin(session_root, 'pfc_*')):
                str_pfc_pair_paths.append([str_cell, pfc_cell])
        str_pfc_pair_paths.append([session_name, cue_times, trial_reward, str_pfc_pair_paths])
    return str_pfc_pair_paths

def get_str_pfc_paths_mono(no_nan=False) -> List[Tuple[str, np.ndarray, np.ndarray, List[List[str]]]]:
    # get all mono pairs
    mono_pairs = pd.read_csv('mono_pairs.csv')

    result = []

    session_names = mono_pairs['mouse']+mono_pairs['date']
    session_names = session_names.unique()

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
