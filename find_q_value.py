from glob import glob
from os.path import join as pjoin
import numpy as np
import pandas as pd
import tqdm
from functools import partial
from multiprocessing import Pool

from lib.models import RW

BEHAVIOUR_ROOT = 'data/behaviour_data/'
RELATIVE_VALUE_ROOT = 'data/relative_values/'

def fit_and_save(session, reset=True):
    session_data = pd.read_csv(session)
    # remove nan trials
    session_data = session_data[~session_data['trial_response_side'].isna()]
    choices = np.array(session_data['trial_response_side'].values)
    # convert choices of -1 to 0
    choices[choices == -1] = 0
    rewards = np.array(session_data['trial_reward'].values)

    # fit the models
    rw = RW()
    parameters = rw.fit(choices_real=choices, rewards_real=rewards)[0]
    
    session_name = session.split('/')[-1].split('.')[0]
    # get the relative values
    relative_values = rw.get_delta_V(parameters, choices, rewards, session_name)
    # remove the last entry for the relative values
    relative_values = relative_values[:-1]

    # save the relative values
    np.save(pjoin(RELATIVE_VALUE_ROOT, session_name+'.npy'), relative_values)

# create a list of session paths to process
sessions = glob(pjoin(BEHAVIOUR_ROOT, '*.csv'))

# set the number of processes to use
n_processes = 4

# create a Pool object with the specified number of processes
pool = Pool(n_processes)

# set the value of the reset parameter
reset = True

# create a new function that has the reset parameter set to the specified value
fit_and_save_reset = partial(fit_and_save, reset=reset)

# use pool.imap to run the fit_and_save_reset function on each session path in parallel
results = list(tqdm.tqdm(pool.imap(fit_and_save_reset, sessions), total=len(sessions)))

# close the pool to free up resources
pool.close()