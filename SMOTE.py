
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter
import pandas as pd

# Generate an imbalanced dataset
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5, n_redundant=5, weights=[0.9], random_state=42
)

# Count the class distribution before SMOTE
print("Before SMOTE:", Counter(y))

# Print the dataset before SMOTE
print("Dataset before SMOTE:")
_df = pd.DataFrame(X)
_df= _df.assign(y=y)
print(_df.head())
print(_df.shape)


# Perform SMOTE with a desired ratio
smote = SMOTE(sampling_strategy=1)  # Set the desired ratio here
X_resampled, y_resampled = smote.fit_resample(X, y)

# Count the class distribution after SMOTE
print("After SMOTE:", Counter(y_resampled))

# # Print the dataset after SMOTE
_df = pd.DataFrame(X_resampled)
_df = _df.assign(y=y_resampled)
print(_df.shape)

