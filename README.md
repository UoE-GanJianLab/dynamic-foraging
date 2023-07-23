# PSRP_RPD
Code base for paper Prefronto-striatal representation of perceived reward probability difference in a two alternative choice dynamic foraging paradigm

## data format

### binary data

The raw experiemental data are stored in binary data format as .dat files

Task data:
- Also 20,000Hz
- 8 Digital channels stored in 2 bytes
- Each channel 1 bit
- Channels
    - 1: all state transition 
    - 2: init 
    - 3: reward 
    - 5, 6: rotary 3, 2(0 indexed)
        - 1024 ticks around per $360\deg$

Response time window: $[0s, 1.5s]$

Background firing (intertrial period): $[-1s, -0.5s]$

response magnitude: response time window firing - background firing

PRP: prior reward probability

## figure_data

The figure_data are named according to the names on the poster. 

### Figure 3
#### Panel A B

Panel A B are for PFC and DMS repectively. These panels contain the firing rates of signal trials, mvt trials and reward trials with the error bar dipicting standard error ploted agains cue time. The Data is organized as follows:

|x_values|signal_mean|signal_err|mvt_mean|mvt_err|reward_mean|reward_err|
|---|---|---|---|---|---|---|
|relative time to cue time, center of the 20ms bin|The firing rate line for the signal trials averaged across all PFC/DMS cells| The standard error of the firing rates of signal trials across all PFC/DMS cels|The firing rate line for the mvt trials averaged across all PFC/DMS cells| The standard error of the firing rates of mvt trials across all PFC/DMS cels|The firing rate line for the reward trials averaged across all PFC/DMS cells| The standard error of the firing rates of reward trials across all PFC/DMS cels|

#### Panel C D

Panel C D are for PFC and DMS respectively. These panels contain the signal, mvt and reward regressors ploted against the relative time to cue time in the range of $[-0.5, 1.5]$. The data is organized as follows:

|x_values|signal|mvt|reward|
|---|---|---|---|
|relative time to cue time, center of the 20ms bin|The signal regressor(signal trials firing rate)|The mvt regressor(mvt trials firing rate - signal trials firing rate)|The reward regressor(reward trials firing rate - mvt trials firing rate)| 

#### Panel E F

Again, panel E F are for PFC and DMS respectively. 


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

## figure 2

![Alt text](figures/manuscript_figures/figure_2.jpg?raw=true "Manuscript figure 2")


## figure 3

![Alt text](figures/manuscript_figures/figure_3.jpg?raw=true "Manuscript figure 3")


Values are digitized using 20ms bins

## figure 4

![Alt text](figures/manuscript_figures/figure_4.jpg?raw=true "Manuscript figure 4")


## figure 5

![Alt text](figures/manuscript_figures/figure_5.jpg?raw=true "Manuscript figure 5")


### fig_5_panel_b

Response magnitude was defined as the mean absolute difference between firing rate in the 0 to 1.5s window and mean intertrial firing rate (-1 to 0s)

We then low-pass filtered these signals using a cut-off frequency that is 10 times the fundamental frequency of the session (the fundamental trial frequency of each session is simply 1 divided by the number of trials).

To calculate the trial-wise phase of the filtered signal we removed its mean and performed a Hilbert transform.   

### fig_5_panel_c, fig_5_panel_d
The circular-mean phase differences between each prefrontal-striatal neuron pair were calculated for all sessions and their distribution plotted in figure 5C (top) for both response magnitudes and inter-trial frequencies.  

The performance of a trial is the xcorr between rightP and proportion of choices made towards the right side. Proportion of rewarded trials and proportion of responses towards the high probability side (advantageous responses) were calculated using a centred moving 20-trial long window.  Perceived reward probability was gauged by calculating, for each side, the proportion of rewarded trials in the previous 10 trials of responses made to that side.  

## figure 6

![Alt text](figures/manuscript_figures/figure_6.jpg?raw=true "Manuscript figure 6")


