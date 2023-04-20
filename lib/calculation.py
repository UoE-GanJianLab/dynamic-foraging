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