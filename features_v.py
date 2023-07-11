import constants
import numpy as np
import pandas as pd
import commons

# loading datset
df = pd.read_csv(constants.AIIMS_COMP_DATA)

# # extracting required columns

# patient_ids = list(set(df['Sample_No']))
# feature_set = []
# for patient_id in patient_ids:

#     condition = df["Sample_No"] == patient_id  # define the condition
#     patient_df = df.loc[condition, columns_to_extract]
def get_patient_singular_data_and_labels(patient_df, patient_id):
    duration = commons.compute_duration_from_timestamps(patient_df["Timestamps"])
    singular_features = [duration]
    for column_name in constants.SINGULAR_DATA_COLUMN_NAMES:
        singular_features.append(list(set(patient_df[column_name]))[0])
    return [singular_features[:-2], singular_features[-2:]]  # last 2 values are Diabetes and BGL




def get_patient_wise_elem_data(df, task):
    columns_to_extract = constants.PATIENT_COLUMN_NAMES
    # Initialize a dictionary of empty lists for data
    patient_ids = list(set(df['Sample_No']))
    feature_set = []
    for patient_id in patient_ids:
        #print("Patient ID:" + str(patient_id))
        condition = df["Sample_No"] == patient_id  # define the condition
        patient_df = df.loc[condition, columns_to_extract]
        singular_features, labels = get_patient_singular_data_and_labels(patient_df, patient_id)
        # Fetch values of multiple columns by a list of column names
        feature_vector = []
        for col in constants.SENSOR_COLS:
            patient_values = patient_df.loc[:, col]
            patient_sensor_data = np.array(patient_values.values.tolist())

            feature_vector = feature_vector + patient_sensor_data
        feature_set.append(singular_features + feature_vector[0] + labels)
    return feature_set




def extract_patient_features(data_sheet, task):
    df = commons.get_data_as_data_frame(data_sheet, None)

    # patient_wise_data = get_patient_wise_aggre_data(df)
    patient_wise_data = get_patient_wise_elem_data(df, task)
    #print(patient_wise_data)
    labels = [sublist[-2] for sublist in patient_wise_data]  # Extract the last element from each sublist
    feature_set = patient_wise_data
    for row in feature_set:
        del row[-2]
    feature_set = feature_set
    return feature_set, labels