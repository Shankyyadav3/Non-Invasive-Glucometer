import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter
# import commons

# readind dataset
_df = pd.read_csv('/Users/sangeeta/Downloads/classification_labels_try.csv', skiprows=0)
print(_df.head())
# extracting required dataset and columns

num_columns = _df.shape[1]
X = _df.iloc[:,1:]
print(X)
print(X.shape)
