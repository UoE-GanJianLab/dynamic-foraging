from os.path import join as pjoin, isdir, basename, isfile
from os import listdir, mkdir
import numpy as np
from glob import glob
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv(pjoin('data','delta_P_correlation.csv'))
data.head(10)

sessions_backgroud = []
str_names_background = []
pfc_names_backgroud = []

sessions_response = []
str_names_response = []
pfc_names_response = []

str_background_firing_pearson_r = []
str_background_firing_p_values= []
pfc_background_firing_pearson_r = []
pfc_background_firing_p_values = []

str_response_firing_pearson_r = []
str_response_firing_p_values = []
pfc_response_firing_pearson_r = []
pfc_response_firing_p_values = []


pmse_path = pjoin('data', 'PMSE', 'PMSE.csv')
pmse = pd.read_csv(pmse_path)


for pmse_pair in pmse.iterrows():
    pmse_pair = pmse_pair[1]
    session_name = pmse_pair['session']

    str_name = pmse_pair['str']
    pfc_name = pmse_pair['pfc']
    str_code = str_name.split('_')[-1]
    pfc_code = pfc_name.split('_')[-1]


    str_row = data.loc[(data['session']==session_name) & (data['cell_name']==str_name)]
    pfc_row = data.loc[(data['session']==session_name) & (data['cell_name']==pfc_name)]

    if (str_row['Background_firing_correlation_P_ value_with PPD'].item() < 0.05 and 
       pfc_row['Background_firing_correlation_P_ value_with PPD'].item() < 0.05):
        sessions_backgroud.append(session_name)
        str_names_background.append(str_name)
        pfc_names_backgroud.append(pfc_name)

        str_background_firing_pearson_r.append(str_row['Background_firing_correlation_Coefficient_with_PPD'].item())
        str_background_firing_p_values.append(str_row['Background_firing_correlation_P_ value_with PPD'].item())

        pfc_background_firing_pearson_r.append(pfc_row['Background_firing_correlation_Coefficient_with_PPD'].item())
        pfc_background_firing_p_values.append(pfc_row['Background_firing_correlation_P_ value_with PPD'].item())
    
    elif (str_row['Response_magnitude_correlation_P_ value_with PPD'].item() < 0.05 and 
        pfc_row['Response_magnitude_correlation_P_ value_with PPD'].item() < 0.05):
        sessions_response.append(session_name)
        str_names_response.append(str_name)
        pfc_names_response.append(pfc_name)
        
        str_response_firing_pearson_r.append(str_row['Response_magnitude_correlation_Coefficient_with_PPD'].item())
        str_response_firing_p_values.append(str_row['Response_magnitude_correlation_P_ value_with PPD'].item())

        pfc_response_firing_pearson_r.append(pfc_row['Response_magnitude_correlation_Coefficient_with_PPD'].item())
        pfc_response_firing_p_values.append(pfc_row['Response_magnitude_correlation_P_ value_with PPD'].item())
        continue

result_bg = pd.DataFrame(data={'session': sessions_backgroud, 'str_name': str_names_background, 'pfc_name': pfc_names_backgroud, 'str_pearsons_r': str_background_firing_pearson_r, 'str_p_value': str_background_firing_p_values, 'pfc_pearsons_r': pfc_background_firing_pearson_r, 'pfc_p_value': pfc_background_firing_p_values})
result_response = pd.DataFrame(data={'session': sessions_response, 'str_name': str_names_response, 'pfc_name': pfc_names_response, 'str_pearsons_r': str_response_firing_pearson_r, 'str_p_value': str_response_firing_p_values, 'pfc_pearsons_r': pfc_response_firing_pearson_r, 'pfc_p_value': pfc_response_firing_p_values})

result_bg.to_csv(pjoin('data', 'delta_P_correlated_background.csv'))
result_response.to_csv(pjoin('data', 'delta_P_correlated_response.csv'))