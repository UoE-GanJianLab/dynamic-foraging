# test if our method's result matches the brute force method

from os.path import join as pjoin, isdir, basename
from os import mkdir, listdir
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import find_peaks

from lib.calculation import get_relative_spike_times_flat, get_relative_spike_times_brute_force

LEFT, RIGHT = -1, -0.5

sessions = listdir(pjoin('data', 'spike_times'))

for s in sessions:
    behaviour_path = pjoin('data', 'behaviour_data', s+'.csv')
    print(behaviour_path)
    behaviour_data = pd.read_csv(behaviour_path)
    cue_time = behaviour_data['cue_time']

    session_path = pjoin('data', 'spike_times', s)
    strs = glob(pjoin(session_path, 'str_*'))
    pfcs = glob(pjoin(session_path, 'pfc_*'))
    if not isdir(pjoin('data', 'PMSE', s)):
        mkdir(pjoin('data', 'PMSE', s))

    if not isdir(pjoin('data', 'PMSE', s, 'qualified')):
        mkdir(pjoin('data', 'PMSE', s, 'qualified'))

    for st in strs:
        str_name = basename(st).split('.')[0]
        str_data = np.load(st)
        for pfc in pfcs:
            pfc_name = basename(pfc).split('.')[0]
            pfc_data = np.load(pfc)

            # time the two methods
            t0 = perf_counter()
            relative_times = get_relative_spike_times_flat(spike_times=str_data, cue_times=pfc_data, window_left=LEFT, window_right=RIGHT)
            t1 = perf_counter()
            relative_times_brute_force = get_relative_spike_times_brute_force(spike_times=str_data, cue_times=pfc_data, window_left=LEFT, window_right=RIGHT)
            t2 = perf_counter()

            # flatten the two lists
            relative_times = np.array(relative_times).flatten()
            relative_times_brute_force = np.array(relative_times_brute_force).flatten()

            # check if the two methods' results are the same
            assert np.allclose(relative_times, relative_times_brute_force)
            print('passed')
            print(f'fast: {t1-t0}, brute force: {t2-t1}')
            print(f'fast method is {round((t2-t1)/(t1-t0), 2)} times faster than brute force method')