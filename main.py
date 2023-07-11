#%%
import commons, constants# regressors, classifiers, signal_processing
import curve_fit, data_analysis, box_plot, feature_extractor, math
import numpy as np
import hypertuneCode
import stackMetaModel
#import improvedStackMeta
import serial
import csv, time
from datetime import datetime, timedelta
import keyboard
import feature_extractor
def perform_supervised_learning(data_sheet):
    for task in ['classification']: # regression
        feature_set, labels = feature_extractor.extract_patient_features(data_sheet, task)
        #print(feature_set)
        uniform_data = commons.perform_padding(feature_set)
        dataset = commons.remove_nan_values(uniform_data)
        commons.array_to_csv(dataset, constants.RED_FEATURES_DIR + task + '_features.csv')
        commons.array_to_csv(labels, constants.RED_FEATURES_DIR + task + '_labels.csv')
        #results_file = task + '_results.csv'
        '''
        if task == "regression":
            results, regression_folder_path = regressors.perform_regression(dataset, labels, task)
            commons.array_to_csv(results, regression_folder_path + results_file)
        else:
            results, classification_folder_path = classifiers.perform_classification(dataset, labels, task)
            commons.array_to_csv(results, classification_folder_path + results_file)
        
        for filter_key in constants.FILTERS:
            filter = constants.FILTERS[filter_key]
            file_filter_Id = task + '_' + filter
            results_file = file_filter_Id + '_' + filter +'_results.csv'
            preprocessed_data = signal_processing.preprocess_data(dataset, filter_key)
            # reduced_features, variance = signal_processing.apply_PCA(preprocess_data)
            prep_uniform_data = commons.perform_padding(preprocessed_data)
            commons.array_to_csv(prep_uniform_data, task+'Data/'+ task + '_' + filter +'_features.csv')
            if task == "regression":
                results, regression_folder_path = regressors.perform_regression(prep_uniform_data, labels, task + '_' + filter)
                commons.array_to_csv(results, regression_folder_path + results_file)
            else:
                results, classification_folder_path = classifiers.perform_classification(prep_uniform_data, labels, task + '_' + filter)
                commons.array_to_csv(results, classification_folder_path + results_file)
            '''
    return

def perform_plot_fitting_analysis(sensor_data_sheets, acetone_conc_list):
    proc_data = curve_fit.obtain_fitted_eq(sensor_data_sheets, acetone_conc_list)
    comb_relevant_raw_sensors_data = proc_data[0]
    comb_relevant_norm_sensors_data = proc_data[1]
    comb_raw_acetone_conc_list = proc_data[2]
    comb_norm_acetone_conc_list = proc_data[3]
    column_names = proc_data[4]
    curve_fit.plot_fit_results(comb_relevant_raw_sensors_data, comb_raw_acetone_conc_list, column_names)
    #data_analysis.analyze_data_quality(comb_relevant_raw_sensors_data, comb_raw_acetone_conc_list, column_names)
    #curve_fit.plot_fit_results(comb_relevant_norm_sensors_data, comb_norm_acetone_conc_list, column_names)
    #data_analysis.analyze_data_quality(comb_relevant_norm_sensors_data, comb_norm_acetone_conc_list, column_names)
    return


def perform_hyper_tuning(data_dir):
    fileId = data_dir + data_dir.replace('reduced', '').replace('Data','').replace('/','').lower()
    featuresFile = fileId + '_features.csv' #fileId + filterNm + '_features.csv'
    print(featuresFile)
    labelsFile = fileId + '_labels.csv'
    print("performing hypertuning of: "+featuresFile.strip('.csv'))
    # Load data from CSV file
    data = commons.get_data_as_data_frame(featuresFile, 0).values
    # Get the number of columns
    num_columns = data.shape[1]
    X = data[:, 1:num_columns]
    y = commons.get_data_as_data_frame(labelsFile, 0).values[:, 1]
    #print(y)
    hypertuneCode.get_best_model(X, y, constants.HYPERTUNE_RESULTS, featuresFile) 
    return


def perform_bgl_classification(data_dir):
    fileId = data_dir + '/' + data_dir.strip('Data') + '_'
    for filterNm in list(constants.FILTERS.values()) + ['noFilter']:
        featuresFile = fileId + filterNm + '_features.csv'
        labelsFile = fileId + 'labels.csv'
        # Load data from CSV file
        #print(featuresFile)
        data = commons.get_data_as_data_frame(featuresFile, 1)
        labels = commons.get_data_as_data_frame(labelsFile, 1)
        # Get the number of columns
        y = labels.values[:, 1].tolist()
        num_columns = data.shape[1]
        X = data.values[:, 1:num_columns]
        three_class_labels = commons.get_three_class_labels(y)
        print("performing hypertuning of: " + featuresFile.strip('.csv'))
        hypertuneCode.get_best_model(X, three_class_labels, constants.HYPERTUNE_RESULTS, 'bgl_classification_features')
    return



def generate_data(test_data_dir):
    # Get user input for age, gender, and heart rate
    test_data_sheet = commons.fetch_test_data_sheet(test_data_dir)
    modified_csv = commons.remove_irrelevant_data_test(test_data_sheet)

    body_vitals = commons.get_body_vitals_info()
    commons.add_body_vitals_data_to_sensors_data(modified_csv, body_vitals, constants.TEST_DATA_SHEET)
    with open(modified_csv, 'r') as file:
        reader = csv.reader(file)
        existing_data = list(reader)
    file.close()
    return body_vitals

def main():
    #X, Y = commons.get_data_as_X_Y(constants.PIMA_DIABETES_MODIFIED_DATASET)
    #classifiers.perform_classification(X, Y)
    #perform_supervised_learning(constants.AIIMS_COMP_DATA)
    #perform_plot_fitting_analysis(constants.SENSORS_VOLTAGE_DATA, constants.ACETONE_CONC)
    #box_plot.generate_plots() # onw time generation
    #raw_sensor_feature_set = feature_extractor.obtain_spatial_frequency_feature_set_acetone_data(comb_relevant_raw_sensors_data, comb_raw_acetone_conc_list, column_names)
    #norm_sensor_feature_set = feature_extractor.obtain_spatial_frequency_feature_set_acetone_data(comb_relevant_norm_sensors_data, comb_norm_acetone_conc_list, column_names)
    #perform_bgl_classification('regressionData')
    #perform_hyper_tuning(constants.RED_FEATURES_DIR)
    #hypertuneCode.perform_hyperparameter_tuning(constants.FEATURES_DATA, constants.FEATURE_LABELS)
    #hypertuneCode.perform_hyperparameter_tuning('reducedClassificationData/sp_classification_features.csv', 'reducedClassificationData/sp_classification_labels.csv')
    stackMetaModel.perform_meta_modelling_classification(constants.FEATURES_DATA, constants.FEATURE_LABELS)
    #stackMetaModel.perform_meta_modelling_regression(constants.REGRESSION_FEATURES, constants.REGRESSION_LABELS)
    name = input("Enter Name: ")
    body_vitals = generate_data(constants.SENSOR_TEST_DATA_DIR)
    bgl = commons.bgl_test()
    decision = stackMetaModel.perform_testing(constants.TEST_DATA_SHEET, constants.BEST_MODELS_DIR) #ata sheet as csv
    commons.project_decision_in_app(decision, name, body_vitals, bgl)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# %%
