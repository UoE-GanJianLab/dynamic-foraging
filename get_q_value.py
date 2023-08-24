from glob import glob
from os.path import join as pjoin, basename
from os import cpu_count
import numpy as np
import pandas as pd
import tqdm
from functools import partial
from multiprocessing import Pool

from lib.models import RW
from lib.calculation import moving_window_mean_prior

BEHAVIOUR_ROOT = 'data/behaviour_data/'
RELATIVE_VALUE_ROOT = 'data/relative_values/'

def fit_and_save(session: str, reset=True):
    crainotomy_side = 'R'
    session_name = basename(session).split('.')[0]
    session_data = pd.read_csv(session)

    # get the index of the nan trials
    nan_trials = session_data[session_data['trial_response_side'].isna()].index
    # remove nan trials
    session_data = session_data[~session_data['trial_response_side'].isna()]
    choices = np.array(session_data['trial_response_side'].values)
    # convert choices of -1 to 0
    choices[choices == -1] = 0
    rewards = np.array(session_data['trial_reward'].values)

    if session_name[:6] == "AKED01":
        crainotomy_side = 'L'

    # fit the models
    rw = RW()
    parameters = rw.fit(choices_real=choices, rewards_real=rewards)[0]
    # print the fitted parameters with their names: beta, kappa, b, alpha, accurate to 3 decimal places
    print(f'{session_name}: beta: {parameters[0]:.3f}, b: {parameters[1]:.3f}, alpha: {parameters[2]:.3f}')
    # print out the fitted 
    session_name = session.split('/')[-1].split('.')[0]
    # get the relative values
    # print out the parameters beta kappa b alpha gamma with the session name and parameter name
    relative_values = rw.get_delta_V(parameters, choices, rewards, session_name)
    # remove the last entry for the relative values
    relative_values = relative_values[:-1]

    # for each nan trial, insert a value equal to previous relative value
    # into the relative values
    for nan_trial in nan_trials:
        relative_values = np.insert(relative_values, nan_trial, relative_values[nan_trial-1])

    # smoothen the relative values
    relative_values = moving_window_mean_prior(relative_values, 10)

    # scale relative values to the range of -1 to 1
    relative_values = (relative_values - np.min(relative_values)) / (np.max(relative_values) - np.min(relative_values))


    # if crainotomy_side == 'L':
    #     relative_values = -relative_values

    # save the relative values
    np.save(pjoin(RELATIVE_VALUE_ROOT, session_name+'.npy'), relative_values)

# create a list of session paths to process
sessions = glob(pjoin(BEHAVIOUR_ROOT, '*.csv'))
# sort the session names
sessions = sorted(sessions)

# set the number of processes to use
n_processes = cpu_count() - 1

# create a Pool object with the specified number of processes
pool = Pool(n_processes)

# set the value of the reset parameter
reset = True

# create a new function that has the reset parameter set to the specified value
fit_and_save_reset = partial(fit_and_save, reset=reset)

# use pool.imap to run the fit_and_save_reset function on each session path in parallel
results = list(pool.imap(fit_and_save_reset, sessions))

# close the pool to free up resources
pool.close()