from os.path import join as pjoin, isdir, basename, isfile
from os import listdir, mkdir
import numpy as np
from glob import glob
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_excel(pjoin('data','PPD_RTW_summary_table_Wei_06may2022_V2.xlsx'))
data.head(10)

mice = []
dates = []
str_names = []
pfc_names = []

Stimulus_related_firing_P_value_str = []
Movement_related_firing_P_value_str = []
Reward_related_firing_P_value_str = []
str_background_firing_pearson_r = []
str_background_firing_p_values= []
str_response_firing_pearson_r = []
str_response_firing_p_values = []

Stimulus_related_firing_P_value_pfc = []
Movement_related_firing_P_value_pfc = []
Reward_related_firing_P_value_pfc = []
pfc_background_firing_pearson_r = []
pfc_background_firing_p_values = []
Response_magnitude_CC_with_PPD_pfc = []
Response_magnitude_CC_with_PPD_P_value_pfc = []

peak = []
peak_width = []
spike_count_in_peak = []


pmse_path = pjoin('data', 'PMSE', 'PMSE.csv')
pmse = pd.read_csv(pmse_path)


for pmse_pair in pmse.iterrows():
    spike_count_data = pd.read_csv(pjoin('data', 'PMSE', f'{session_name}.csv'))

    session_name = pmse_pair['session']
    str_name = pmse_pair['str']
    pfc_name = pmse_pair['pfc']


    str_row = data.loc[(data['session']==session_name) & (data['Cell location']=='STR') & (data['Cell ID']==int(str_code)+1)]
    pfc_row = data.loc[(data['session']==session_name) & (data['Cell location']=='PFC') & (data['Cell ID']==int(pfc_code)+1)]

    if (str_row['Background_firing_correlation_P_ value_with PPD'].item() < 0.01 and 
       pfc_row['Background_firing_correlation_P_ value_with PPD'].item() < 0.01):
        mice.append(mouse)
        dates.append(date)
        str_names.append(f'str_{int(str_code) + 1}')
        pfc_names.append(f'pfc_{int(pfc_code) + 1}')

        str_background_firing_pearson_r.append(str_row['Background_firing_correlation_Coefficient_with_PPD'].item())
        str_background_firing_p_values.append(str_row['Background_firing_correlation_P_ value_with PPD'].item())
        str_response_firing_pearson_r.append(str_row['Response_magnitude_correlation_Coefficient_with_PPD'].item())
        str_response_firing_p_values.append(str_row['Response_magnitude_correlation_P_ value_with PPD'].item())

        pfc_background_firing_pearson_r.append(pfc_row['Background_firing_correlation_Coefficient_with_PPD'].item())
        pfc_background_firing_p_values.append(pfc_row['Background_firing_correlation_P_ value_with PPD'].item())
        Response_magnitude_CC_with_PPD_pfc.append(pfc_row['Response_magnitude_correlation_Coefficient_with_PPD'].item())
        Response_magnitude_CC_with_PPD_P_value_pfc.append(pfc_row['Response_magnitude_correlation_P_ value_with PPD'].item())

    
        spike_count_row = spike_count_data.loc[(spike_count_data['session']==session_name) & (spike_count_data['str']==str_name) & (spike_count_data['pfc']==pfc_name)]
        peak.append(spike_count_row['peak'].item())
        peak_width.append(spike_count_row['peak_width'].item())
        spike_count_in_peak.append(spike_count_row['counts_in_peak'].item())
    else:
        continue

result = pd.DataFrame(data={'mouse': mice, 'date': dates, 'str_name': str_names, 'pfc_name': pfc_names, })

result.to_csv(pjoin('data', 'strong_correlation_pairs_ITI.csv'))