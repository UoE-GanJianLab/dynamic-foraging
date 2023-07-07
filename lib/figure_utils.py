import matplotlib.pyplot as plt

# helper functions for figure processing
def remove_top_and_right_spines(ax: plt.Axes):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)