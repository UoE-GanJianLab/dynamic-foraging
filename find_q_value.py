from glob import glob
from os.path import join as pjoin
import numpy as np
import pandas as pd
import tqdm

from lib.models import RW

BEHAVIOUR_ROOT = 'data/behaviour_data/'
RELATIVE_VALUE_ROOT = 'data/relative_values/'

# load the behaviour data 
# add a progress bar
sessions = glob(pjoin(BEHAVIOUR_ROOT, '*.csv'))
for session in tqdm.tqdm(sessions):
    session_data = pd.read_csv(session)
    # remove nan trials
    session_data = session_data[~session_data['trial_response_side'].isna()]
    choices = session_data['trial_response_side'].values
    # convert choices of -1 to 0
    choices[choices == -1] = 0
    rewards = session_data['trial_reward'].values

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
