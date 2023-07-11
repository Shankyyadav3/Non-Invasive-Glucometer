#Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import constants
from pandas import read_csv

#reading dataset
_data = pd.read_csv(constants.HYPERTUNE_RESULTS+"classification_noFilter_tune_results.csv")
print(_data.head())

#Extracting classifiers with significant accuracies
_df = _data.loc[((_data['Model']=='DecisionTreeClassifier')|(_data['Model']=='GradientBoostingClassifier')|(_data['Model']=='RandomForestClassifier')|(_data['Model']=='XGBClassifier')) & (_data['K-Fold']==5)]


models = list(_df.iloc[:,0])
eval_values = _df.iloc[:,1:4]


x = np.arange(len(models))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize= (10,6))

for attribute, measurement in eval_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, round(measurement,3), width, label=attribute)
    ax.bar_label(rects, padding=3,rotation=0)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title('Two Class Classification results',fontsize=20,weight='bold')
ax.set_ylabel('Performance evaluation metrics values', fontsize=13, weight='bold')
ax.set_xlabel('ML Algorithms',fontsize=13,weight='bold')
ax.set_xticks(x + width, models)
ax.legend(loc='upper right')
ax.set_ylim(0,1)

plt.show()





