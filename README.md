# PSRP_RPD
Code base for paper Prefronto-striatal representation of perceived reward probability difference in a two alternative choice dynamic foraging paradigm

## data format

All of the data uses 0 based indexing, while the two csv files specifying strong corralation pairs uses 1 based indexing from matlab, for example, str_1 in the strong_corr csv files points to the str_0.npy file storing the spike firing data.

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

## figure 3

## figure 4

## figure 5

### fig_5_panel_b

Response magnitude was defined as the mean absolute difference between firing rate in the 0 to 1.5s window and mean intertrial firing rate (-1 to 0s)

We then low-pass filtered these signals using a cut-off frequency that is 10 times the fundamental frequency of the session (the fundamental trial frequency of each session is simply 1 divided by the number of trials).

To calculate the trial-wise phase of the filtered signal we removed its mean and performed a Hilbert transform.   

### fig_5_panel_c, fig_5_panel_d
The circular-mean phase differences between each prefrontal-striatal neuron pair were calculated for all sessions and their distribution plotted in figure 5C (top) for both response magnitudes and inter-trial frequencies.  

The performance of a trial is the xcorr between rightP and proportion of choices made towards the right side. Proportion of rewarded trials and proportion of responses towards the high probability side (advantageous responses) were calculated using a centred moving 20-trial long window.  Perceived reward probability was gauged by calculating, for each side, the proportion of rewarded trials in the previous 10 trials of responses made to that side.  

## figure 6


