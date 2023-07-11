import commons, constants
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.stats import skew, kurtosis, entropy
import pywt
from statsmodels.tsa.ar_model import AutoReg as AR
import numpy.fft as fft
from scipy.integrate import trapz
import commons, statistics


def compute_magnitude_features(sample):
    #print(sample)
    mag_list = list(np.abs(sample))
    max_mag = np.max(mag_list)
    mean_mag = np.mean(mag_list)
    mag_features = [max_mag, mean_mag]
    for mag in mag_list:
        mag_features.append(np.abs(mag) / max_mag)
    return mag_features


def compute_shape_features(sample):
    skewness = skew(sample)
    kurt = kurtosis(sample)
    ent = entropy(sample)
    return [skewness, kurt, ent]


def compute_derivative_features(sample):
    deriv_list = list(np.gradient(sample))
    s_deriv_list = list(np.gradient(deriv_list))
    max_deriv = np.max(deriv_list)
    deriv_features = [max_deriv, np.min(deriv_list), np.max(s_deriv_list), np.min(s_deriv_list)]
    if max_deriv > 0:
        denom = max_deriv
    else:
        denom = 1
    for deriv in deriv_list:
        if not(denom == 0):
            deriv_features.append(deriv / denom)
        else:
            deriv_features.append(deriv)
    return deriv_features


def compute_slope_integral_features(sample):
    # Split the curve into 5 intervals based on the difference feature
    intervals = np.array_split(sample, 5)
    # Compute the integral and slope of each interval
    integrals = []
    slopes = []
    for interval in intervals:
        integrals.append(np.trapz(interval))
        interval_difference = np.diff(interval)
        slope = np.mean(interval_difference)
        slopes.append(slope)
    return integrals + slopes


def compute_phase_features(sample):
    magnitude = np.abs(sample)
    derivative = np.gradient(sample)
    phase_features = np.cumsum(derivative) * magnitude[-1] - np.cumsum(derivative * magnitude)
    return list(phase_features)


def compute_ar_features(sample, fileId):
    model = AR(sample, lags=1)
    model_fit = model.fit()
    ar_coefficients = model_fit.params
    prediction_error = model_fit.resid
    return list(ar_coefficients) + list(prediction_error)


def compute_fft_features(sample):
    fft_signal = fft.fft(sample)
    phase = np.angle(fft_signal)
    spectrum = np.abs(fft_signal)
    power_spectrum = np.abs(spectrum) ** 2
    spectral_entropy = entropy(power_spectrum)
    return list(phase) + list(power_spectrum) + [spectral_entropy]


def compute_wavelet_features(sample):
    wavelet_name = 'db4'
    magnitude = np.abs(sample)
    wavelet_coeffs = pywt.wavedec(magnitude, wavelet_name)
    wavelet_features = []
    for level in range(len(wavelet_coeffs)):
        wavelet_features.extend(wavelet_coeffs[level])
    return wavelet_features


def compute_peak_features(sample):
    # print(sample)
    peaks, _ = find_peaks(np.array(sample))
    #print(peaks)
    if len(peaks) > 0:
        peak_index = peaks[0]
        peak_height = sample[peak_index]
        # Find the indices where the signal crosses the half-height
        half_height = peak_height / 2.0
        left_index = np.argmax(np.array(sample[:peak_index]) <= half_height)
        right_index = peak_index + np.argmax(np.array(sample[peak_index:]) <= half_height)
        # Calculate the width as the distance between the indices
        width = right_index - left_index
        # Calculate the peak area using the trapezoidal rule
        area = trapz(sample[left_index:right_index + 1], dx=1)
        return [peak_height, width, area]
    else:
        return []


def obtain_spatial_freq_features_sensor_data(signal, fileId):
    comb_feature_set = []
    #mag_features = compute_magnitude_features(signal)
    #deriv_features = compute_derivative_features(signal)
    # slope_integral_features = compute_slope_integral_features(signal)
    # phase_features = compute_phase_features(signal)
    fft_features = compute_fft_features(signal)
    # wvt_features = compute_wavelet_features(signal)
    # peak_features = compute_peak_features(signal)
    # shape_features = compute_shape_features(signal)
    # ar_features = compute_ar_features(signal, fileId)
    comb_feature_set= comb_feature_set  + fft_features 
                                        # + mag_features
                            # + deriv_features + slope_integral_features + phase_features + fft_features
                            # + wvt_features + peak_features + shape_features + ar_features
    return comb_feature_set


def obtain_spatial_frequency_feature_set_acetone_data(comb_sensors_data, comb_acetone_conc_list, column_names):
    feature_set = []
    # print(len(comb_sensors_data))
    # print(comb_sensors_data[0])
    # print(len(comb_acetone_conc_list))
    # print(comb_acetone_conc_list[0])
    # reduced_pca_feature_set, variance = apply_PCA(comb_sensors_data)
    # print(comb_sensors_data[0])
    # print(comb_acetone_conc_list[0])
    acetone_vals = list(set(comb_acetone_conc_list)).sort()
    max_mag_list = []
    min_mag_list = []

    for sensor_data in comb_sensors_data:
        for conc in acetone_vals:
            sensor_data_per_conc = commons.get_sensor_data_per_conc(sensor_data, comb_acetone_conc_list, conc)
            max_mag_list.append(np.max(sensor_data_per_conc))
            min_mag_list.append(np.min(sensor_data_per_conc))
    return feature_set


def get_patient_singular_data_and_labels(patient_df):
    duration = commons.compute_duration_from_timestamps(patient_df["Timestamps"])
    singular_features = [duration]
    for column_name in constants.SINGULAR_DATA_COLUMN_NAMES:
        singular_features.append(list(set(patient_df[column_name]))[0])
    #print(singular_features)
    return [singular_features[:-1], singular_features[-1:]]  # last 2 values are Diabetes and BGL


def get_patient_wise_aggre_data(df):
    columns_to_extract = constants.PATIENT_COLUMN_NAMES
    # Initialize a dictionary of empty lists for data
    keys = constants.AGGREGATE_FUNC_DICT.keys()
    patient_wise_data = {key: [] for key in keys}
    patient_ids = list(set(df['Sample_No']))
    for patient_id in patient_ids:
        condition = df["Sample_No"] == patient_id  # define the condition
        patient_df = df.loc[condition, columns_to_extract]
        singular_features, labels = get_patient_singular_data_and_labels(patient_df)
        #print(labels)
        for key in keys:
            patient_wise_data[key].append(
                singular_features + commons.get_aggregate_vals(patient_df, constants.AGGREGATE_FUNC_DICT[key]) + labels)
    return patient_wise_data


def get_stat_features(sensor_vals):
    # Sort the list in descending order
    sorted_list = sorted(sensor_vals, reverse=True)
    # Get the top 5 values
    stat_features = sorted_list[:5]
    return stat_features


def get_patient_wise_elem_data(df, task):
    columns_to_extract = constants.PATIENT_COLUMN_NAMES
    # Initialize a dictionary of empty lists for data
    patient_ids = list(set(df['Sample_No']))
    feature_set = []
    for patient_id in patient_ids:
        #print("Patient ID:" + str(patient_id))
        condition = df["Sample_No"] == patient_id  # define the condition
        patient_df = df.loc[condition, columns_to_extract]
        singular_features, labels = get_patient_singular_data_and_labels(patient_df)
        # Fetch values of multiple columns by a list of column names
        feature_vector = []
        for col in constants.SENSOR_COLS:
            patient_values = patient_df.loc[:, col]
            patient_sensor_data = np.array(patient_values.values.tolist())
            
            stat_features = get_stat_features(patient_sensor_data)
            #spatial features
            sensor_features = obtain_spatial_freq_features_sensor_data(stat_features,
                                                                       task + '_' + str(patient_id))
            feature_vector = feature_vector + sensor_features
        feature_set.append(singular_features + feature_vector + labels)
    return feature_set


def extract_patient_features(data_sheet, task):
    df = commons.get_data_as_data_frame(data_sheet, None)
    # patient_wise_data = get_patient_wise_aggre_data(df)
    patient_wise_data = get_patient_wise_elem_data(df, task)
    #print(patient_wise_data)
    if task == 'classification':
        labels = [sublist[-1] for sublist in patient_wise_data]  # Extract the last element from each sublist
        feature_set = [sublist[:-1] for sublist in patient_wise_data]
    else:
        labels = [sublist[-2] for sublist in patient_wise_data]  # Extract the last element from each sublist
        feature_set = patient_wise_data
        for row in feature_set:
            del row[-2]
        feature_set = feature_set
        feature_set, labels = commons.get_balanced_dataset(feature_set, labels)
    return feature_set, labels
    #return patient_wise_data
