# creating poster panel C and D of figure 6 using all pairs
from lib.figure_6 import figure_6_poster_panel_d, figure_6_poster_panel_c
from lib.file_utils import get_str_pfc_paths_mono

import tqdm
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

all_pairs = get_str_pfc_paths_mono()

# pearson correlation
rs = []
ps = []
# spearmans correlation
srs = []
sps = []

pair_size = 0

for session in all_pairs:
    pair_size += len(session[3])

progress_bar = tqdm.tqdm(total=pair_size)

for session in all_pairs:
    rs_session = []
    ps_session = []
    srs_session = []
    sps_session = []
    session_name = session[0]
    cue_times = session[1]
    trial_reward = session[2]
    cell_pairs_session = session[3]

    for cell_pair in cell_pairs_session:
        str_times = np.load(cell_pair[0])
        pfc_times = np.load(cell_pair[1])

        str_name = cell_pair[0].split('/')[-1].split('.')[0]
        pfc_name = cell_pair[1].split('/')[-1].split('.')[0]

        cross_cors, reward_proportion, p, r, sp, sr = figure_6_poster_panel_c(str_times=str_times, pfc_times=pfc_times, cue_times=cue_times, rewarded=trial_reward, str_name=str_name, pfc_name=pfc_name, session_name=session_name, mono=True)

        ps_session.append(p)
        rs_session.append(r)
        srs_session.append(sr)
        sps_session.append(sp)

        progress_bar.update(1)
    
    ps.append(np.array(ps_session))
    rs.append(np.array(rs_session))
    srs.append(np.array(srs_session))
    sps.append(np.array(sps_session))

figure_6_poster_panel_d(rs, ps, mono=True)
# also for spearman
figure_6_poster_panel_d(srs, sps, spearman=True, mono=True)