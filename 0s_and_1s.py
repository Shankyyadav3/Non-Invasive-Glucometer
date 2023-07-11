import constants
import pandas as pd


_df= pd.read_csv(constants.AIIMS_COMP_DATA)

# print(_df['0'])
# length= len(_df['0'])
count_1 = 0
count_0 = 0
patient_ids = list(set(_df['Sample_No']))
for patient_id in patient_ids:
    #print("Patient ID:" + str(patient_id))
    condition = _df["Sample_No"] == patient_id  # define the condition
    patient_df = _df.loc[condition, 'Diabetes']
    # print(patient_df.head())
    if(patient_df.iloc[0]==1):
        count_1+=1
    elif(patient_df.iloc[0]==0):
        count_0+=1

print("Count of 1s = ",count_1)
print("Count of 0s = ",count_0)
