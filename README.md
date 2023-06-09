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



## figures

### figure 2

### figure 3

### figure 4

### figure 6


