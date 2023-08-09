import matplotlib.pyplot as plt

# helper functions for figure processing
def remove_top_and_right_spines(ax: plt.Axes):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def plot_with_sem_error_bar(ax: plt.Axes, x, mean, sem, label=None, color='blue'):
    if label:
        ax.plot(x, mean, label=label, c=color)
    else:
        ax.plot(x, mean, c=color)

    ax.fill_between(x=x, y1=mean-sem, y2=mean+sem, alpha=0.5, color=color)