# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint


import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#importing data
data = pd.read_csv('Data.csv')

data['Offered Food'] = data['Offered Food'].map({'Yes': 1, 'No': 0})

# Perform one-hot encoding on 'Size of Organization' and 'Type of Event'
one_hot_cols = ['Org Size', 'Type']
data = pd.get_dummies(data, columns=one_hot_cols)

def split_multiselect_to_columns(data, column_name, separator=","):
    multiselect = data[column_name].str.get_dummies(sep=separator)
    multiselect.columns = [f"{column_name} - {col.strip()}" for col in multiselect.columns]
    return pd.concat([data, multiselect], axis=1).drop(columns=[column_name])

# Apply the function to your multi-select columns
data = split_multiselect_to_columns(data, "Days")
data = split_multiselect_to_columns(data, "Time")
data = split_multiselect_to_columns(data, "Advertisements")

# new label:
data["Popular"] = data["Actual"] >= data["Expected"]
data = data * 1
# estimate the ratio of expected to organization size
small_er = (data['Org Size_Small (1-50 members)'][data['Org Size_Small (1-50 members)'] == 1] * data['Expected']/25).fillna(0)
mid_er =  (data['Org Size_Medium (50-200 members)'][data['Org Size_Medium (50-200 members)'] == 1] * data['Expected']/125).fillna(0)
large_er = (data['Org Size_Large (200+ members)'][data['Org Size_Large (200+ members)'] == 1] * data['Expected']/200).fillna(0)
data["Expected Ratio"] = small_er + mid_er + large_er
data.describe()
data["Expected Ratio"].quantile(.05)
# clip outliers
data["Expected Ratio"].quantile(.95)
data["Expected Ratio"] = data["Expected Ratio"].clip(lower=0.2)
data["Expected Ratio"] = data["Expected Ratio"].clip(upper=4.0)

# new label:
labels = data["Popular"]

# Remove the labels from the features
data= data.drop(['Actual', 'Popular', 'Expected'], axis = 1)

# Saving feature names for later use
feature_list = list(data.columns)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

# just do train/test because not a lot of data to go off of
X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.70, random_state=42)
rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)
print(rf.feature_names_in_)

y_pred_train = rf.predict(X_train)
accuracy_score(y_pred_train, y_train)

y_pred_test = rf.predict(X_test)
accuracy_score(y_pred_test, y_test)

# Baseline classifier: always guess underestimate=1
y_test.mean()