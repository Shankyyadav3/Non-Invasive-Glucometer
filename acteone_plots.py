#%%
#Load libraries
import constants
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv


# reading dataset
data_2ml = pd.read_csv(constants.SENSORS_DATA_2_ML)
data_10ml = pd.read_csv(constants.SENSORS_DATA_10_ML)
data_20ml = pd.read_csv(constants.SENSORS_DATA_20_ML)
data_40ml = pd.read_csv(constants.SENSORS_DATA_40_ML)
print(data_40ml.head())


#extracting required attributes
df_2ml = data_2ml.loc[data_2ml['Sample_No']==3,"TGS2600"]
df_10ml = data_10ml.loc[data_10ml['Sample_No']==3,"TGS2600"]
df_20ml = data_20ml.loc[data_20ml['Sample_No']==3,"TGS2600"]
df_40ml = data_40ml.loc[data_40ml['Sample_No']==3,"TGS2600"]
print(df_2ml)
print(df_10ml)
print(df_20ml)
print(df_40ml)
x1 = np.arange(106)
x2 = np.arange(207)
x3 = np.arange(40)
x4 = np.arange(144)


# plt.figure(figsize=(12, 8)) 

plt.plot(x1,df_2ml, label='Acetone 2ml')
# plt.plot(x2,df_10ml, label='Acetone 10ml')
# plt.plot(x3,df_20ml, label='Acetone 20ml')
plt.plot(x4[:124],df_40ml[20:], label= 'Acetone 40ml')

plt.title('TGS2600 sensor readings plot for Sample 3', fontsize= 12, weight= 'bold')
plt.ylabel('Sensor output',fontsize= 10, weight='bold')
plt.legend()
plt.show()



# plt.subplot(1,3,1)
# df_2ml.plot(x='Timestamps',y='TGS2600',ax=plt.gca())
# plt.xticks(rotation=45)
# plt.ylim(0.3,1)
# plt.title('Acetone 2ml(Sample 0)',weight='bold')
# plt.xlabel('Timestamps')


# plt.subplot(1,3,2)
# df_20ml.plot(x='Time',y='TGS2600',ax=plt.gca())
# plt.xticks(rotation=45)
# plt.ylim(0.3,1)
# plt.title('Acetone 20ml(Sample 1)',weight='bold')
# plt.xlabel('Timestamps')



# plt.subplot(1,3,3)
# df_40ml.plot(x='Time',y='TGS2600',ax=plt.gca())
# plt.xticks(rotation=45)
# plt.ylim(0.3,1)
# plt.title('Acetone 40ml(Sample 0)',weight='bold')
# plt.xlabel('Timestamps')

# plt.show()
# %%
