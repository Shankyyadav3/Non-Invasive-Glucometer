import pandas as pd
import commons
import constants

df = pd.read_csv(constants.AIIMS_COMP_DATA)


columns_to_extract = constants.PATIENT_COLUMN_NAMES
condition = df["Sample_No"] == 49  # define the condition
patient_df = df.loc[condition, columns_to_extract]


# duration = commons.compute_duration_from_timestamps(patient_df["Timestamps"])

time_stamps = patient_df["Timestamps"]
timestamps = pd.to_datetime(time_stamps)
print((timestamps[1796] - timestamps[1795]).total_seconds() / 60)

print(timestamps)