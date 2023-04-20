from glob import glob
from os.path import join as pjoin
from typing import List

def get_session_names() -> List[str]:
    session_root = pjoin('data', 'behaviour_data')
    sessions = glob(pjoin(session_root, '*.csv'))
    session_names = [s.split('/')[-1].split('.')[0] for s in sessions]
    return session_names


