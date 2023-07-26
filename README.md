# PSRP_RPD
Code base for paper Prefronto-striatal representation of perceived reward probability difference in a two alternative choice dynamic foraging paradigm

## data format

### binary data

The raw experiemental data are stored in binary data format as .dat files

Task data:
- Also $20,000Hz$ sampling rate
- 8 Digital channels stored in 2 bytes
- Each channel 1 bit
- Channels
    - $1$: all state transition 
    - $2$: init 
    - $3$: reward 
    - $5, 6$: rotary $3, 2$($0$ indexed)
        - $1024$ ticks around per $360\degree$

Response time window: $[0s, 1.5s]$

Background firing (intertrial period): $[-1s, -0.5s]$

response magnitude: response time window firing - background firing

PRP: prior reward probability

### figure_data

Organized data used for plotting the figures.

The figure_data are named according to the names on the poster. 

#### Figure 1

1. Panel D

Panel D describes the percentage of choices made to the advantageous side as well as the percentage of rewarded trials relative to each switch from all sessions. The data are organized as follows:

|relative_trial_index|rewarded_percentage|high_reward_percentage|
|---|---|---|
|trial index relative to the switch|percentage of rewarded trials in the 50 trials before and after the switch, averaged across all switches|percentage of choices made to the high reward side in the 50 trials before and after the switch, averaged across all switches|

#### Figure 3
1. Panel A B

Panel A B are for PFC and DMS repectively. These panels contain the firing rates of signal trials, mvt trials and reward trials with the error bar dipicting standard error ploted agains cue time. The data are organized as follows:

|x_values|signal_mean|signal_err|mvt_mean|mvt_err|reward_mean|reward_err|
|---|---|---|---|---|---|---|
|relative time to cue time, center of the 20ms bin|The firing rate line for the signal trials averaged across all PFC/DMS cells| The standard error of the firing rates of signal trials across all PFC/DMS cels|The firing rate line for the mvt trials averaged across all PFC/DMS cells| The standard error of the firing rates of mvt trials across all PFC/DMS cels|The firing rate line for the reward trials averaged across all PFC/DMS cells| The standard error of the firing rates of reward trials across all PFC/DMS cels|

2. Panel C D

Panel C D are for PFC and DMS respectively. These panels contain the signal, mvt and reward regressors ploted against the relative time to cue time in the range of $[-0.5, 1.5]$. The data are organized as follows:

|x_values|signal|mvt|reward|
|---|---|---|---|
|relative time to cue time, center of the 20ms bin|The signal regressor(signal trials firing rate)|The mvt regressor(mvt trials firing rate - signal trials firing rate)|The reward regressor(reward trials firing rate - mvt trials firing rate)| 

3. Panel E F

Again, panel E F are for PFC and DMS respectively. These panels show the result of applying the regressors from panel C D to multiple linear regression to the firing rate during signal, mvt and reward trials of individual PFC/DMS cell. The data are organized as follows:

|coefficient_type|signal_trials|signal_trials_err|mvt_trials|mvt_trials_err|reward_trials|reward_trials_err|
|---|---|---|---|---|---|---|
|The type of coefficient|The coefficients for regression result of signal trials, averaged across all PFC/DMS cells| The standard error for the coefficients for regression result of signal trials across all PFC/DMS cells|The coefficients for regression result of mvt trials, averaged across all PFC/DMS cells| The standard error for the coefficients for regression result of mvt trials across all PFC/DMS cells|The coefficients for regression result of reward trials, averaged across all PFC/DMS cells| The standard error for the coefficients for regression result of reward trials across all PFC/DMS cells|

#### Figure 4 
Panel A B are for PFC and DMS respectively. These panels show the result of applying the regressors from figure 3 panel C D to multiple linear regression of PFC/DMS trials with prior reward probability(prp) $\geq 0.5$ or $<0.5$, each subfigure compares the regressed coefficient for the high and low reward trials. The data are organize as follows:

|trial_type|signal_coeffs|signal_coeffs_err|mvt_coeffs|mvt_coeffs_err|reward_coeffs|reward_coeffs_err|
|---|---|---|---|---|---|---|
|high or low prp trials|regressed coefficient for signal regressor, averaged across all PFC/DMS cells|standard error for regressed coefficient for signal regressor across all PFC/DMS cells|regressed coefficient for mvt regressor, averaged across all PFC/DMS cells|standard error for regressed coefficient for mvt regressor across all PFC/DMS cells|regressed coefficient for reward regressor, averaged across all PFC/DMS cells|standard error for regressed coefficient for reward regressor across all PFC/DMS cells|


#### Figure 5

1. Panel A B C D

The A B panels are for PFC firing that positively or negatively correlated with prpd/relative values. The C D panels are for DMS firing that positively or negatively correlated with prpd/relative values. The top panel is the raster plot with the data organized as follows:

|trial_index|relative_spike_time|
|---|---|
|trial index|relative spike time to cue time|

The bottom panel is the average firing rate(20ms bins) across all trials against the relative time to cue time with the data organized as follows:

|bin_centers|left_p_high|right_p_high|
|---|---|---|
|center of the 20ms bin|average firing rate across trials with hign left reward probability|average firing rate across trials with hign right reward probability|


#### Figure 6

1. Panel B
The data for panel b are split into two groups, one done with prpd, one with relative values from the reinforcement learning models. The data are organized as follows:

|trial_index|pfc_mag_standardized|dms_mag_standardized|prpd_standardized|pfc_mag_filtered|dms_mag_filtered|prpd_filtered|pfc_phase|dms_phase|prpd_phase|
|---|---|---|---|---|---|---|---|---|---|
|trial index|standardized response magnitude for PFC|standardized response magnitude for DMS|standardized prpd|filtered response magnitude for PFC|filtered response magnitude for DMS|filtered prpd|phase of the filtered response magnitude for PFC|phase of the filtered response magnitude for DMS|phase of the filtered prpd|

The pairs here are filtered by removing cells that are suspected to have experienced probe drift to reduce the amount of resulting images to go through. The filtering is done by removing cells with no firing in 10 consecutive trials in the response time window.


#### Figure 8

Figure 8 focus on the interconnectivity strength 

1. Panel A B C

Panel A B C shows the interconnectivity strength between PFC and DMS pairs, prior reward proportion of the corresponding session and the correlation between interconnectivity strength(max absolute normalized cross correlation) and discretized prior reward proportion respectively. The data are organized as follows:

* Panel A

|trial_index|interconnectivity_strength|
|---|---|
|trial index|interconnectivity strength between PFC and DMS(max absolute normalized cross correlation)|

* Panel B

|trial_index|reward_proportion|
|---|---|
|trial index|prior reward proportion of each session|

* Panel C

|discretized_reward_proportion|interconnectivity_strength_mean|interconnectivity_strength_err|
|---|---|---|
|discretized prior reward proportion|mean interconnectivity strength between PFC and DMS pairs|standard error of the mean interconnectivity strength between PFC and DMS pairs|


2. Panel D

This is a barplot comparing the percentage of strongly negatively correlated cell pairs between the PFC and DMS

All pairs:

t: -8.020560608781873, p: 3.576158011492007e-11

Mono:

t: -3.439937909343383, p: 0.0005646506319530828


3. Panel E/ Panel E Extra

This barplot compares the mean interconnectivity strength between rewarded and non-rewarded trials, as well as plateau and transitioning trials

* rewarded vs non-rewarded

All pairs:

t: -116.29928981054866, p: 0.0

Mono:

t: -7.784737816978901, p: 7.019765436743266e-15

* plateau vs transitioning

All pairs:

t: -20.212802029318336, p: 7.59799493448309e-91

Mono:

t: -5.30720280984404, p: 1.116669563832314e-07


## repository structure

## lib

helper functions as well as functions written for producing each figure given formatted session data

### calculation.py

Come extension to the common math libraries, including the moving window mean function as well as relative firing rate calculation

Normalized cross correlation as defined by Wei Xu and the cross correlation metric 

### conversion.py

### extraction.py

### models.py

### file_utils.py



# figures

Figures below are named according to the first draft of the manuscript.

## figure 1

![Alt text](figures/manuscript_figures/figure_1.png?raw=true "Manuscript figure 1")


## figure 2

![Alt text](figures/manuscript_figures/figure_2.png?raw=true "Manuscript figure 2")


## figure 3

![Alt text](figures/manuscript_figures/figure_3.png?raw=true "Manuscript figure 3")


Values are digitized using 20ms bins

## figure 4

![Alt text](figures/manuscript_figures/figure_4.png?raw=true "Manuscript figure 4")


## figure 5

![Alt text](figures/manuscript_figures/figure_5.png?raw=true "Manuscript figure 5")


### fig_5_panel_b

Response magnitude was defined as the mean absolute difference between firing rate in the 0 to 1.5s window and mean intertrial firing rate (-1 to 0s)

We then low-pass filtered these signals using a cut-off frequency that is 10 times the fundamental frequency of the session (the fundamental trial frequency of each session is simply 1 divided by the number of trials).

To calculate the trial-wise phase of the filtered signal we removed its mean and performed a Hilbert transform.   

### fig_5_panel_c, fig_5_panel_d
The circular-mean phase differences between each prefrontal-striatal neuron pair were calculated for all sessions and their distribution plotted in figure 5C (top) for both response magnitudes and inter-trial frequencies.  

The performance of a trial is the xcorr between rightP and proportion of choices made towards the right side. Proportion of rewarded trials and proportion of responses towards the high probability side (advantageous responses) were calculated using a centred moving 20-trial long window.  Perceived reward probability was gauged by calculating, for each side, the proportion of rewarded trials in the previous 10 trials of responses made to that side.  

## figure 6

![Alt text](figures/manuscript_figures/figure_6.png?raw=true "Manuscript figure 6")


