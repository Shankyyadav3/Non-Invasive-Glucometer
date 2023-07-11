#load libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import constants
import plotly.express as px


def plot(x):
    #reading dataset
    _data = pd.read_csv(constants.HYPERTUNE_RESULTS+x)
    print(_data.head())

    #Extracting classifiers with significant accuracies
    _df = _data.loc[((_data['Model']=='DecisionTreeClassifier')|(_data['Model']=='GradientBoostingClassifier')|(_data['Model']=='RidgeClassifier')) ]
    print(_df)

    #violin plot for f1 score
    plt.figure(figsize=(8,6))
    sns.violinplot(data=_df, x='Model', y='F1 Score', palette='tab10', linewidth=2)

    #formatting
    plt.title(x[:-4],fontsize=15,weight='bold')
    plt.xlabel('ML Algorithms',fontsize=13,weight='bold')
    plt.ylabel("F1 Score",fontsize=13,weight='bold')
    plt.show()

li=["classification_Butterworth_tune_results.csv","classification_Chebyshev_1_tune_results.csv","classification_Chebyshev_2_tune_results.csv","classification_Elliptic_tune_results.csv","classification_noFilter_tune_results.csv"]

for i in li:
    plot(i)
