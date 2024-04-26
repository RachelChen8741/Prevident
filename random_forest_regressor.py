from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

import pandas as pd
import numpy as np


# Load data
data = pd.read_csv('Data.csv')

data['Offered Food'] = data['Offered Food'].map({'Yes': 1, 'No': 0})
one_hot_cols = ['Org Size', 'Type', 'Days', 'Time', 'Advertisements']
data = pd.get_dummies(data, columns=one_hot_cols)

# Separate features and the target variable
features = data.drop('Actual', axis=1)
labels = data['Actual']

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, train_size=0.70, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# baseline prediction: expected values
baseline_preds = X_test[:, list(data.columns).index('Expected')]
baseline_errors = abs(baseline_preds - y_test)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# Initialize and train the RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10000, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model using the test set
predictions = rf.predict(X_test)
test_error = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error on Test set: {test_error:.2f} attendees')



