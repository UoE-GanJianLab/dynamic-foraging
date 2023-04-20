import numpy as np

def moving_window_mean(data: np.ndarray, window_size=5) -> np.ndarray:
    data_len = data.size
    output = np.zeros(data_len)

    for i in range(data_len):
        if i < window_size // 2:
            output[i] = np.mean(data[:i+window_size//2+1])
        elif i > data_len - window_size // 2 - 1:
            output[i] = np.mean(data[i-window_size//2:])
        else:
            output[i] = np.mean(data[i-window_size//2:i+window_size//2+1])

    return output

def get_firing_rate_window(cue_times: np.ndarray, spike_times:np.ndarray, window_left: float, window_right: float) -> np.ndarray:
    spike_ptr = 0

    firing_rates = []

    for cue in cue_times:
        cur_count = 0

        window_left_cur = cue + window_left
        window_right_cur = cue + window_right

        # move the pointers into window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_left_cur:
            spike_ptr += 1
        
        # count the amount of spikes in window
        while spike_ptr < len(spike_times) and spike_times[spike_ptr] < window_right_cur:
            spike_ptr += 1
            cur_count += 1
        
        
        firing_rates.append(cur_count / (window_right - window_left))       

    return firing_rates