import pandas as pd
import numpy as np
import matplotlib
# connect to virtual display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as pjoin
from os import listdir
from os.path import isfile
import glob
from lib.calculation import moving_window_mean, get_session_performances

# directory storing the relative value data
relative_value_dir = pjoin('data', 'relative_values')
# directory storing the prpd data
prpd_dir = pjoin('data', 'prpd')

pearson_correlation = []
performances = get_session_performances()
session_performances = []

# iterate through the npy files in the relative value directory
for relative_value_file in glob.glob(pjoin(relative_value_dir, '*.npy')):
    # get the relative value data
    relative_value_data = np.load(relative_value_file)
    # get the name of the file
    relative_value_filename = relative_value_file.split('/')[-1]
    # get the session name
    session_name = relative_value_filename.split('.')[0]
    # get the prpd data
    prpd_data = np.load(pjoin(prpd_dir,  relative_value_filename))
    # smooth the relative value data
    # relative_value_data = moving_window_mean(relative_value_data, 10)
    # calculate the pearson correlation
    pearson_correlation.append(np.corrcoef(relative_value_data, prpd_data)[0, 1])

    # get the performance of the session
    performance = performances[0][session_name]
    session_performances.append(performance)

# get the absolute value of the pearson correlation
pearson_correlation = np.abs(pearson_correlation)

# plot the two plots in one figure side by side
plt.rcParams.update({'font.size': 20})
# increase line width and axis width
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['lines.linewidth'] = 2

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
axes[0].hist(pearson_correlation, bins=np.arange(0, 1, 0.1))
axes[0].set_xlabel('Pearson Correlation')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Pearson Correlation between\n Relative Value and PRPD')
# set x ticks to -1 to 1 in 0.1 increments
axes[0].set_xticks(np.arange(0, 1, 0.2))
# remove the top and right spines
sns.despine(ax=axes[0])

axes[1].scatter(pearson_correlation, session_performances)
# fit a linear regression line to the data with minimum mse
m, b = np.polyfit(pearson_correlation, session_performances, 1)
axes[1].plot(pearson_correlation, m*np.array(pearson_correlation) + b, color='red') 
axes[1].set_xlabel('Pearson Correlation')
axes[1].set_ylabel('Performance')
axes[1].set_title('Pearson Correlation vs Performance')
# remove the top and right spines
sns.despine(ax=axes[1])

fig.savefig('relative_values_supp.png', dpi=500)
