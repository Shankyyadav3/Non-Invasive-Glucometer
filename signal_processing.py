import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import commons, constants


def apply_PCA(sensors_data):
    # Apply PCA to the feature data
    pca = PCA()
    pca.fit(sensors_data)
    commons.save_model(pca, constants.MODELS+'PCA.pkl')
    X_pca = pca.transform(sensors_data)
    return X_pca, pca.explained_variance_ratio_


def apply_chebyshev1_filter(signal_data, fs, cutoff_freq, filter_order, filter_type):
    b, a = signal.cheby1(filter_order, 1, cutoff_freq / (fs / 2), btype=filter_type, analog=False)
    return signal.lfilter(b, a, signal_data)


def apply_butterworth_filter(signal_data, fs, cutoff_freq, filter_order, filter_type):
    b, a = signal.butter(filter_order, cutoff_freq / (fs / 2), btype=filter_type, analog=False)
    return signal.lfilter(b, a, signal_data)


def apply_chebyshev2_filter(signal_data, fs, cutoff_freq, filter_order, filter_type):
    b, a = signal.cheby2(filter_order, 1, cutoff_freq / (fs / 2), btype=filter_type, analog=False)
    return signal.lfilter(b, a, signal_data)


def apply_elliptic_filter(signal_data, fs, cutoff_freq, filter_order, filter_type):
    b, a = signal.ellip(filter_order, 1, 1, cutoff_freq / (fs / 2), btype=filter_type, analog=False)
    return signal.lfilter(b, a, signal_data)


def perform_high_pass_filtering(signal_data, filter_key):
    # Sample frequency and desired cutoff frequencies
    fs = 1000
    cutoff_freq = 10
    filter_order = 4
    filter_type = 'high'
    if filter_key == 1:
        filtered_signal = apply_butterworth_filter(signal_data, fs, cutoff_freq, filter_order, filter_type)
    elif filter_key == 2:
        filtered_signal = apply_chebyshev1_filter(signal_data, fs, cutoff_freq, filter_order, filter_type)
    elif filter_key == 3:
        filtered_signal = apply_chebyshev2_filter(signal_data, fs, cutoff_freq, filter_order, filter_type)
    else:
        filtered_signal = apply_elliptic_filter(signal_data, fs, cutoff_freq, filter_order, filter_type)
    return filtered_signal


def perform_low_pass_filtering(signal_data, filter_key):
    # Sample frequency and desired cutoff frequencies
    fs = 1000
    cutoff_freq = 100
    filter_order = 4
    filter_type = 'low'
    if filter_key == 1:
        filtered_signal = apply_butterworth_filter(signal_data, fs, cutoff_freq, filter_order, filter_type)
    elif filter_key == 2:
        filtered_signal = apply_chebyshev1_filter(signal_data, fs, cutoff_freq, filter_order, filter_type)
    elif filter_key == 3:
        filtered_signal = apply_chebyshev2_filter(signal_data, fs, cutoff_freq, filter_order, filter_type)
    else:
        filtered_signal = apply_elliptic_filter(signal_data, fs, cutoff_freq, filter_order, filter_type)
    return filtered_signal


def perform_min_max_scaling(signal_data):
    # Create a scaler object and fit it to the data
    scaler = MinMaxScaler()
    scaler.fit(signal_data)

    # Transform the data using the scaler object
    scaled_data = scaler.transform(signal_data)
    return scaled_data

def preprocess_data(signal_data_list, filter_key):
    signal_data = np.array(signal_data_list)
    baseline_corrected = signal.detrend(signal_data)
    high_pass_filtered_data = perform_high_pass_filtering(baseline_corrected, filter_key)
    low_pass_filtered_data = perform_low_pass_filtering(high_pass_filtered_data, filter_key)
    #scaled_data = perform_min_max_scaling(low_pass_filtered_data)
    return low_pass_filtered_data
