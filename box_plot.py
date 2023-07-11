import matplotlib.pyplot as plt
import numpy as np
import constants


def generate_plots():
    max_vals = [2.1, 1.3, 2.92]
    min_vals = [0.8, 0.68, 0.94]
    error_vals = [0.65, 0.31, 0.99]
    sensor_data = [1.45, 0.99, 1.93]

    #create
    fig, ax = plt.subplots()
    ax.boxplot(sensor_data, sym='o')
    ax.errorbar([1, 2, 3], max_vals, yerr=error_vals, fmt='none', c='r')
    ax.errorbar([1, 2, 3], min_vals, yerr=error_vals, fmt='none', c='r')

    # Add labels for the maximum, minimum, and error values
    for j, (max_val, min_val, error_val) in enumerate(zip(max_vals, min_vals, error_vals)):
        ax.text(j + 1, max_val, f'Max = {max_val:.2f}', va='center')
        ax.text(j + 1, min_val, f'Min = {min_val:.2f}', va='center')
        ax.text(j + 1, (max_val + min_val) / 2, f'Error = {error_val:.2f}', va='center', ha='center')

    #ax.set_xticklabels(['0', '2', '40'])
    #locs, labels = plt.xticks()  # Get the current locations and labels.
    plt.xticks([1, 2, 3], ['0', '2', '40'])
    ax.set_ylabel('Sensor Normalized Voltages')
    ax.set_xlabel('Acetone Conc. in ml')
    ax.set_title('TGS 2602: Volt. vs Acetone Conc.')
    plt.show()
    fig.savefig(constants.DATA_ANALYSIS_FIGS + "TGS2602_all_conc" + '.png', dpi=600, bbox_inches='tight')
    return