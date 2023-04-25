from glob import glob
from os.path import join as pjoin, basename
from typing import List, Tuple, Dict

import numpy as np

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