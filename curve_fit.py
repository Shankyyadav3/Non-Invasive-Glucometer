import constants, commons
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd


# Acetone conc vs Voltage equations in the form of different non-linear regression functions
def step(x, a, b):
    return a * (x >= b)

# Define the function to be fitted
def step_func(x, a, b, c):
    return step(x, a, b) + c * (x < b)



def linear(x, a, b):
    return a * x + b


def power_log(volt, a, b, c, d):
    print(volt)
    return a * np.power(b * np.log(volt) + c, d)


def exponential(volt, a, b, c, d):
    return a * np.exp(-b * volt) + c * np.exp(-d * volt)


def gaussian(x, a, b, c, d):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d


def hill(x, a, b, n):
    return (a * x ** n) / (b ** n + x ** n)


def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d


def power_law(x, a, b, c):
    return a * x ** b + c


def log_func1(x, a, b):
    return a * np.log(x) + b


def log_func2(x, a, b, c):
    return a * np.log(b * x) + c


def get_norm_sensor_data(sensors_data, humidity, temp):
    norm_temp = commons.get_norm_values(temp)
    norm_humidity = commons.get_norm_values(humidity)
    norm_sensor_data = []
    for item in sensors_data:
        norm_sensor_data.append(commons.get_norm_voltage_values(item, norm_temp, norm_humidity))
    return norm_sensor_data


def obtain_relevant_data(sample_id_list, norm_sensor_data):
    relevant_sensors_data = []
    unique_sample_ids = list(set(sample_id_list))
    for sensors_data in norm_sensor_data:
        relevant_sensor_data = []  # for a particular sensor
        for sample_id in unique_sample_ids:
            sample_sensor_volt_data = commons.obtain_sample_specific_sensor_data(sensors_data, sample_id_list,
                                                                                 sample_id)
            relevant_voltages = sorted(sample_sensor_volt_data, reverse=True)
            relevant_sensor_data.append(np.max(relevant_voltages[:5]))
        #print(relevant_sensor_data)
        relevant_sensors_data.append(relevant_sensor_data)
        #print(relevant_sensors_data)
    return relevant_sensors_data


def obtain_relevant_norm_sensor_data(data_sheet, column_names):
    #print(column_names)
    sensor_column_vals = commons.get_csv_column_vals_using_names(data_sheet, column_names)
    sample_id_list, temp, humidity, t_2600, t_2602, t_2603, t_2610, t_2620, t_826, t_822 = sensor_column_vals.T #, m_138
    sensors_data = [t_2600, t_2602, t_2603, t_2610, t_2620, t_826, t_822 ] #m_138
    norm_sensor_data = get_norm_sensor_data(sensors_data, humidity, temp)
    relevant_norm_sensors_data = obtain_relevant_data(sample_id_list, norm_sensor_data)
    relevant_raw_sensors_data = obtain_relevant_data(sample_id_list, sensors_data)
    return [relevant_raw_sensors_data, relevant_norm_sensors_data]


def plot_fit_results(comb_relevant_sensors_data, comb_acetone_conc_list, column_names):
    funcs = [step_func, linear, power_log, exponential, gaussian, sigmoid, power_law, hill, log_func1, log_func2]
    # empty list to store results
    for i in range(0, len(comb_relevant_sensors_data)):  # obtain equations for each sensor
        results = []
        column_name = column_names[3+i]
        sensor_volt_data = comb_relevant_sensors_data[i]  # for a particular acetone conc
        x = np.array(sensor_volt_data)
        y = np.array(comb_acetone_conc_list[i])  # first acetone_conc
        fig1 = plt.gcf()
        plot_once_flag = False
        for j, func in enumerate(funcs):
            try:
                popt, pcov = curve_fit(func, x, y, maxfev=50000)  # p0=initial_param_guess[j],
            except:
                continue
            y_fit = func(x, *popt)

            # calculate R-squared, AIC, and BIC
            r2, aic, bic = commons.compute_eval_metrics(y, y_fit, popt)

            # add results to list
            results.append([func.__name__, *popt, r2, aic, bic])
            # plot data and fitted curves
            if plot_once_flag == False:
                plot_once_flag = True
                plt.plot(y, x, 'o', label=column_name[3:] + "_before_fit")
            plt.plot(y_fit, x, label=column_name[3:] + "_" + func.__name__ + "_fit")
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.xlabel('Acetone Conc.')
        plt.ylabel('Sensor Voltages')
        plt.show()
        fig1.savefig(constants.RAW_FIGS + column_name + '.png', dpi=600, bbox_inches='tight')
        # convert results list to dataframe and write to CSV
        df = pd.DataFrame(results,
                          columns=['Function', 'Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4', 'R-squared',
                                   'AIC', 'BIC'])
        df.to_csv(constants.RAW_CURVE_FIT_RESULTS + column_name[3:] + '.csv', index=False)
        return


def obtain_fitted_eq(data_sheets, acetone_conc_list):
    # Get Data
    column_names = constants.ACETONE_DATA_COL_NAMES
    relevant_raw_sensors_data = []
    relevant_norm_sensors_data = []

    for i in range(len(data_sheets)):
        result = obtain_relevant_norm_sensor_data(data_sheets[i], column_names)
        relevant_raw_sensors_data.append(result[0])
        relevant_norm_sensors_data.append(result[1])
    comb_relevant_raw_sensors_data = commons.obtain_combined_sensors_volt_data(relevant_raw_sensors_data)
    comb_relevant_norm_sensors_data = commons.obtain_combined_sensors_volt_data(relevant_norm_sensors_data)
    comb_raw_acetone_conc_list = commons.obtain_combined_acetone_conc_data(relevant_raw_sensors_data, acetone_conc_list)
    comb_norm_acetone_conc_list = commons.obtain_combined_acetone_conc_data(relevant_norm_sensors_data, acetone_conc_list)
    return [comb_relevant_raw_sensors_data, comb_relevant_norm_sensors_data, comb_raw_acetone_conc_list, comb_norm_acetone_conc_list, column_names]
