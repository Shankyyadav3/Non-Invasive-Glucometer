#Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import constants


# reading dataset
df= pd.read_csv('results1/final_evaluation_results.csv')

#Extracting classifiers with significant accuracies
_df = df.loc[((df['Classifier']=='Gradient Boosting')|(df['Classifier']=='XGBoost')|(df['Classifier']=='Gradient Boosting - XGBoost')|(df['Classifier']=='Random Forest - XGBoost')|(df['Classifier']=='Decision Tree - AdaBoost')) & (df['SMOTE Technique']=='ADASYN')]


models = list(_df.iloc[:,1])
eval_values = _df.iloc[:,2:]


# x1 = np.arange(len(models))  # the label locations

x = range(len(models))
new_x = np.array([1.2*i for i in x])
# print(new_x)
width = 0.3  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize= (15,7))

for attribute, measurement in eval_values.items():
    offset = width * multiplier
    measurement_1 = np.array([round(float(i),3) for i in measurement])
    rects = ax.bar(new_x + offset, measurement_1, width, label=attribute)
    ax.bar_label(rects, padding=3,rotation=0)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title('Performance Comparison of Diabetes Prediction Models',fontsize=20,weight='bold')
ax.set_ylabel('Performance evaluation metrics values', fontsize=13, weight='bold')
ax.set_xlabel('ML Algorithms',fontsize=15,weight='bold')
ax.set_xticks(new_x + width, models)
ax.legend(loc='upper right')
ax.set_ylim(0,1.2)
# plt.grid(axis = 'y')

plt.show()


# x = _df.columns[3:6]

# # print(x)
# y = _df.iloc[0,3:6]

# fig = plt.figure(figsize = (10, 6))
# color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c']
# plt.bar(x, y, width = 0.4, color= color_palette)

# # Add values at the top of each bar
# for i, v in enumerate(y):
#     plt.text(i, v, str(round(v, 3)), ha='center', va='bottom')
    
# plt.xlabel("Evaluation Metrics for Meta Model Stack", fontsize=13, weight='bold')
# plt.ylabel("Evaluation Metrics Values",fontsize=13,weight='bold')
# plt.title("Two Class Classifications results",fontsize=20,weight='bold')
# plt.show()
