import matplotlib.pyplot as plt
import numpy as np

import constants, commons


def generate_sensor_wise_plot(comb_relevant_sensors_data, comb_acetone_conc_list, column_names):
    for i in range(0, len(comb_relevant_sensors_data)):
        sensor_name = column_names[3+i]
        data1 = comb_relevant_sensors_data[i][:comb_acetone_conc_list[i].count(0)]
        endInd = len(data1)+comb_acetone_conc_list[i].count(2)
        data2 = comb_relevant_sensors_data[i][len(data1):endInd]
        endInd = len(data1) + len(data2) + comb_acetone_conc_list[i].count(40)
        stInd = len(data1)+len(data2)
        data3 = comb_relevant_sensors_data[i][stInd:endInd]
        # Calculate the maximum, minimum, and error values for each dataset
        max_vals = [np.max(data1), np.max(data2), np.max(data3)]
        min_vals = [np.min(data1), np.min(data2), np.min(data3)]
        error_vals = [(max_val - min_val) / 2 for max_val, min_val in zip(max_vals, min_vals)]

        # Create the box plot
        fig, ax = plt.subplots()
        #plt.boxplot(y)
        #plt.errorbar(x, np.mean(y, axis=0), yerr=np.std(y, axis=0))
        ax.boxplot([data1, data2, data3], sym='o')
        ax.errorbar([1, 2, 3], max_vals, yerr=error_vals, fmt='none', c='r')
        ax.errorbar([1, 2, 3], min_vals, yerr=error_vals, fmt='none', c='r')

        # Add labels for the maximum, minimum, and error values
        for j, (max_val, min_val, error_val) in enumerate(zip(max_vals, min_vals, error_vals)):
            ax.text(j + 1, max_val, f'Max = {max_val:.2f}', va='center')
            ax.text(j + 1, min_val, f'Min = {min_val:.2f}', va='center')
            ax.text(j + 1, (max_val + min_val) / 2, f'Error = {error_val:.2f}', va='center', ha='center')

        ax.set_xticklabels(['0ml', '2ml', '40ml'])
        ax.set_ylabel('Sensor Values')
        ax.set_title(sensor_name)
        plt.show()
        fig.savefig(constants.RAW_DATA_ANALYSIS_FIGS + sensor_name + '.png', dpi=600, bbox_inches='tight')
    return


def get_max_min_err_sensor_vals(sensors_data, acetone_data, reqd_acetone_conc):
    max_vals = []
    min_vals = []
    reqd_sensor_data = commons.get_sensor_data_per_conc(sensors_data, acetone_data, reqd_acetone_conc)
    for i in range(0, len(reqd_sensor_data)):
        reqd_data = reqd_sensor_data[i]
        # Calculate the maximum, minimum, and error values for each dataset
        max_vals.append(np.max(reqd_data))
        min_vals.append(np.min(reqd_data))
    error_vals = [(max_val - min_val) / 2 for max_val, min_val in zip(max_vals, min_vals)]
    return [max_vals, min_vals, error_vals, reqd_sensor_data]


def generate_plots_spec_conc(sensors_data, acetone_data, column_names, acetone_conc):
    max_vals, min_vals, error_vals, reqd_sensor_data = get_max_min_err_sensor_vals(sensors_data, acetone_data, acetone_conc)
    # Create the box plot
    fig, ax = plt.subplots()
    ax.boxplot(reqd_sensor_data, sym='o')
    ax.errorbar([1, 2, 3, 4, 5, 6, 7], max_vals, yerr=error_vals, fmt='none', c='r')
    ax.errorbar([1, 2, 3, 4, 5, 6, 7], min_vals, yerr=error_vals, fmt='none', c='r')

    # Add labels for the maximum, minimum, and error values
    for j, (max_val, min_val, error_val) in enumerate(zip(max_vals, min_vals, error_vals)):
        ax.text(j + 1, max_val, f'Max = {max_val:.2f}', va='center')
        ax.text(j + 1, min_val, f'Min = {min_val:.2f}', va='center')
        ax.text(j + 1, (max_val + min_val) / 2, f'Error = {error_val:.2f}', va='center', ha='center')

    ax.set_xticklabels(column_names[3:])
    ax.set_ylabel('Raw Sensor Voltages')
    ax.set_title('Performance at '+str(acetone_conc)+'ml conc.')
    #plt.show()
    fig.savefig(constants.RAW_DATA_ANALYSIS_FIGS + str(acetone_conc)+"ml_Performance" + '.png', dpi=600, bbox_inches='tight')
    return


def analyze_data_quality(comb_relevant_sensors_data, comb_acetone_conc_list, column_names):
    generate_sensor_wise_plot(comb_relevant_sensors_data, comb_acetone_conc_list, column_names)
    for acetone_conc in constants.ACETONE_CONC:
        generate_plots_spec_conc(comb_relevant_sensors_data, comb_acetone_conc_list, column_names, acetone_conc)
    return


