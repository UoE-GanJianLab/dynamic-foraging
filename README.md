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

1. Panel B

Panel B demonstrats the wheel velocity difference between rewarded and non-rewarded trials as well as the reward probability during each 20ms bin between $[-0.5s, 1.5s]$ relative to the cue time. The data are organized as follows:

|time|reward_probability|wheel_velocity_rewarded|wheel_velocity_rewarded_sem|wheel_velocity_unrewarded|wheel_velocity_unrewarded_sem|

2. Panel C

Panel C contains a series of plots that summarizes the behavioral performance of the animal. First figure compares the set reward probability of left and right; Second figure measures the response proportion(20 trial centred window) to left and right side; Third figure shows the comparison between proportion of reward choices and choices made to the high reward probability side; The fourth figure displays the comparison between prpd and relative values. The data are organized as follows:

|trial_index|leftP|rightP|responses|left_response_proportion|right_response_proportion|perceived_left|perceived_right|prpd|relative_value|chosen_high_reward_side|high_reward_proportion|reward_proportion|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|trial index|left reward probability|right reward probability|response side|proportion of left responses in the 20 trial window centred at the trial|proportion of right responses in the 20 trial window centred at the trial|perceived left reward probability|perceived right reward probability|perceived reward probability difference(contra - ipsi)|relative value($Q_l-Q_r$)|chosen high reward side|proportion of choices made to the high reward side in the 20 trial window centred at the trial|proportion of rewarded trials in the 20 trial window centred at the trial|

3. Panel D

Panel D describes the percentage of choices made to the advantageous side as well as the percentage of rewarded trials relative to each switch from all sessions. The data are organized as follows:

|relative_trial_index|rewarded_percentage|high_reward_percentage|
|---|---|---|
|trial index relative to the switch|percentage of rewarded trials in the 50 trials before and after the switch, averaged across all switches|percentage of choices made to the high reward side in the 50 trials before and after the switch, averaged across all switches|


#### Figure 2

* Panel G

Panel G is an example trial, showing angular velocity of wheel (blue), total angle traversed (red) and simultaneous multiple PFC single-unit recordings (black). The data are organized as follows:

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

2. Panel E

Panel E has a pair of pie charts showing the percentage of PFC and DMS cells that are correlated with PRPD/relative values. Each pie chart is split into four parts: not-correlated, only ITI firing correlated, only response magnitude correlated, both ITI firing and response magnitude correlated. The data are organized as follows:

|cell_location|not-correlated percentage|ITI firing correlated percentage|response magnitude correlated percentage|both ITI firing and response magnitude correlated percentage|
|---|---|---|---|---|
|PFC or DMS|percentage of cells that are not correlated|percentage of cells that are only ITI firing correlated|percentage of cells that are only response magnitude correlated|percentage of cells that are both ITI firing and response magnitude correlated|

3. Panel F

Panel F makes further distinction between positive and negatively correlated cells. The mean percentage are calculated across all sessions. The data are organized as follows:

|cell_location|firing window|positively correlated percentage|positive standard error|negatively correlated percentage|negative standard error|
|---|---|---|---|---|---|
|PFC or DMS|response magnitude or ITI firing|mean percentage of cells that are positively correlated|standard error of positively correlated percentage across all sessions|mean percentage of cells that are negatively correlated|standard error of negatively correlated percentage across all sessions|

#### Figure 6

1. Panel B
The data for panel b are split into two groups, one done with prpd, one with relative values from the reinforcement learning models. The data are organized as follows:

|trial_index|pfc_mag_standardized|dms_mag_standardized|prpd_standardized|pfc_mag_filtered|dms_mag_filtered|prpd_filtered|pfc_phase|dms_phase|prpd_phase|
|---|---|---|---|---|---|---|---|---|---|
|trial index|standardized response magnitude for PFC|standardized response magnitude for DMS|standardized prpd|filtered response magnitude for PFC|filtered response magnitude for DMS|filtered prpd|phase of the filtered response magnitude for PFC|phase of the filtered response magnitude for DMS|phase of the filtered prpd|

The pairs here are filtered by removing cells that are suspected to have experienced probe drift to reduce the amount of resulting images to go through. The filtering is done by removing cells with no firing in 10 consecutive trials in the response time window.

2. Panel C

Panel C shows the discretized mean phase difference between PFC and DMS pairs split by session performance. The bins range from $[-\pi, \pi]$ with 36 bins.

|bin_center|good response count|good bg count|bad response count|bad bg count|
|---|---|---|---|---|
|center of the bin|number of good session PFC DMS pairs whose mean phase difference of response window firing falls into the bin|number of good session PFC DMS pairs whose mean phase difference of background firing falls into the bin|number of bad session PFC DMS pairs whose mean phase difference of response window firing falls into the bin|number of bad session PFC DMS pairs whose mean phase difference of background firing falls into the bin|

3. Panel D

Panel C shows the discretized mean phase difference between PFC/DMS cells during reponse/ITI windows and PRPD. The data are organized as follows:

|bin_center|pfc_response_count|pfc_bg_count|dms_response_count|dms_bg_count|
|---|---|---|---|---|
|center of the bin|number of PFC cells whose mean phase difference with PRPD during response window falls into the bin|number of PFC cells whose mean phase difference with PRPD during ITI window falls into the bin|number of DMS cells whose mean phase difference with PRPD during response window falls into the bin|number of DMS cells whose mean phase difference with PRPD during ITI window falls into the bin|


#### Figure 7

* Panel B

This is a pie chart showing the percentage of PRPD modulated cell pairs out of all the mono cell pairs. Similar to figure 5 panel E, the pie chart is split into four parts: not-correlated, only ITI firing correlated, only response magnitude correlated, both ITI firing and response magnitude correlated. The data are organized as follows:

|correlation|percentage|
|---|---|
|one of the four types of correlation|percentage of cell pairs that are correlated with the corresponding type|

#### Figure 8

Figure 8 focus on the interconnectivity strength, calculated using the maximum absolute deviation from 0 of the normalized cross correlation between the pair of DMS and PFC firring during ITI.

The normalized cross correlation $\'x(\tau)$ between two time series $g(t), h(t)$ is calculated using:

$$\'x(\tau)=\frac{x(\tau)-E[x(\tau)]}{E[x(\tau)]}$$

Where:

$$x(\tau)=\sum^N_{t=-N}g(t)h(t+\tau)$$

$$E[x(\tau)]=\sum^N_{t=-N}\bar{g}(t)\bar{h}(t+\tau)$$

$\bar{g}(t)$ is a time series with equal length to $g(t)$, where each time point equals $\bar{g(t)}$, the mean of $g(t)$. Same applies to $h(t)$.

To adjust for the sudden variations in the firing rates, the interconnectivity stength for each trial is calculated using a centered 20-trial long window. The binned firing rates(10ms) from 1 second before the first trial initiation till the last trial initiation in the 20-trial window of the corresponding DMS and PFC cells are the $g(t)$ and $h(t)$ in the above equations.

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

4. Panel F



#### Beri-Cohen Extra

To rebate the reviewer's comment on influence of past and future choices, we plotted the firing rate of PFC and DMS cells against the relative value/prpd, split into past and future choices. The data are organized as follows:

|x|pfc_past_R_bg|past_R_bg_sem|pfc_past_L_bg|past_L_bg_sem|pfc_future_R_bg|future_R_bg_sem|pfc_future_L_bg|future_L_bg_sem|dms_past_R_bg|dms_past_R_bg_sem|dms_past_L_bg|dms_past_L_bg_sem|dms_future_R_bg|dms_future_R_bg_sem|dms_future_L_bg|dms_future_L_bg_sem|pfc_past_R_response|pfc_past_R_response_sem|pfc_past_L_response|pfc_past_L_response_sem|pfc_future_R_response|pfc_future_R_response_sem|pfc_future_L_response|pfc_future_L_response_sem|dms_past_R_response|dms_past_R_response_sem|dms_past_L_response|dms_past_L_response_sem|dms_future_R_response|dms_future_R_response_sem|dms_future_L_response|dms_future_L_response_sem|pfc_all_firing_bg|pfc_all_firing_bg_sem|dms_all_firing_bg|dms_all_firing_bg_sem|pfc_all_firing_response|pfc_all_firing_response_sem|dms_all_firing_response|dms_all_firing_response_sem|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|center of 0.25 bin|mean firing rate of PFC cells whose past choice is right during ITI|standard error of the mean firing rate of PFC cells whose past choice is right during ITI|mean firing rate of PFC cells whose past choice is left during ITI|standard error of the mean firing rate of PFC cells whose past choice is left during ITI|mean firing rate of PFC cells whose future choice is right during ITI|standard error of the mean firing rate of PFC cells whose future choice is right during ITI|mean firing rate of PFC cells whose future choice is left during ITI|standard error of the mean firing rate of PFC cells whose future choice is left during ITI|mean firing rate of DMS cells whose past choice is right during ITI|standard error of the mean firing rate of DMS cells whose past choice is right during ITI|mean firing rate of DMS cells whose past choice is left during ITI|standard error of the mean firing rate of DMS cells whose past choice is left during ITI|mean firing rate of DMS cells whose future choice is right during ITI|standard error of the mean firing rate of DMS cells whose future choice is right during ITI|mean firing rate of DMS cells whose future choice is left during ITI|standard error of the mean firing rate of DMS cells whose future choice is left during ITI|mean firing rate of PFC cells whose past choice is right during response|standard error of the mean firing rate of PFC cells whose past choice is right during response|mean firing rate of PFC cells whose past choice is left during response|standard error of the mean firing rate of PFC cells whose past choice is left during response|mean firing rate of PFC cells whose future choice is right during response|standard error of the mean firing rate of PFC cells whose future choice is right during response|mean firing rate of PFC cells whose future choice is left during response|standard error of the mean firing rate of PFC cells whose future choice is left during response|mean firing rate of DMS cells whose past choice is right during response|standard error of the mean firing rate of DMS cells whose past choice is right during response|mean firing rate of DMS cells whose past choice is left during response|standard error of the mean firing rate of DMS cells whose past choice is left during response|mean firing rate of DMS cells whose future choice is right during response|standard error of the mean firing rate of DMS cells whose future choice is right during response|mean firing rate of DMS cells whose future choice is left during response|standard error of the mean firing rate of DMS cells whose future choice is left during response|mean firing rate of PFC cells during ITI|standard error of the mean firing rate of PFC cells during ITI|mean firing rate of DMS cells during ITI|standard error of the mean firing rate of DMS cells during ITI|mean firing rate of PFC cells during response|standard error of the mean firing rate of PFC cells during response|mean firing rate of DMS cells during response|standard error of the mean firing rate of DMS cells during response|

## Reinforcement Learning Modelling

In addition to the intuitive PRPD. reinforcement learning modelling was also utilized in this study. The Rescorla Wagner model common in non-stationary two-arm bandit tasks was chosen to be a suitable candidate.

The model (sometimes referred to as the Q-learning model in other publications)'s decision is guided by two decision variables, associated with two choices given to the animals. In our case, $Q_l$ corresponds to leftward choices, while $Q_r$ corresponds to the right. They were fed into the softmax function, along with a bias term $b$ and inverse temperature parameter $\beta$ to produce the probability of making a rightward choice.

$$
\begin{align*}P_r&=\frac{e^{\beta\cdot Q_r}}{e^{\beta\cdot Q_r}+e^{\beta\cdot Q_l+b}}\\&=\frac{1}{1+e^{-\beta(Q_r-Q_l)+b}}\end{align*}
$$

The second equation is the result of dividing both the numerator and denominator by $e^{\beta\cdot Q_r}$, and is actually used in the optimization process.

After each trial, the model updates the decision variables using the following equations:

$$
\begin{align*}\delta&=r-Q_{choice}\\Q_{choice}&=Q_{choice}+\alpha\cdot\delta\\Q_{non-choice}&=Q_{non-choice}\end{align*}
$$

Where $\delta$ is the prediction error, $r$ is the reward received, $Q_{choice}$ is the decision variable associated with the chosen side, $Q_{non-choice}$ is the decision variable associated with the chosen by the animals, and $\alpha$ is the learning rate.

The optimization process aims to minimize the negative log-likelihood of the model's choice probability given the actual choices made by the animals. The negative log-likelihood is calculated using the following equation:

$$
NLL=-\sum_{i=1}^{n}log(P_{i})
$$

Where $n$ is the number of trials, $P_{r_i}$ is the probability of making the animal's choice at trial $i$. The optimization process is handled by Python's Scipy package, using the Nelder-Mead method of the minimize function. The optimization process was repeated 100 times for each session, with the initial parameters randomly sampled from a uniform distribution to avoid local minima as much as possible.

The parameters were bounded to the following ranges:

|Parameter|Lower Bound|Upper Bound|
|---|---|---|
|$\beta$|0.1|$\infty$|
|$\alpha$|0.001|1|
|$b$|$-\infty$|$\infty$|



## repository structure

## lib

helper functions as well as functions written for producing each figure given formatted session data

### calculation.py

Come extension to the common math libraries, including the moving window mean function as well as relative firing rate calculation

Normalized cross-correlation as defined by Wei Xu and the cross-correlation metric 

### conversion.py

### extraction.py

### models.py

### file_utils.py


