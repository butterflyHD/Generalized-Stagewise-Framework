import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]
#print(diabetes_X)
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
XX = np.ones(422).reshape(-1, 1)
print(XX)
print(diabetes_X_train)
print(type(XX))
print(type(diabetes_X_train))
print(diabetes_X_train.size)
diabetes_X_train = XX
#print(len(diabetes_X_train))
#diabetes_X_train = pd.DataFrame({"intercept", XX})
#print(diabetes_X_train)
#diabetes_X_train = np.ones(diabetes_X_train.size(),1)


# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]
print(diabetes_y_train.mean())

# Create linear regression object
regr = linear_model.LinearRegression(fit_intercept=False)

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

print(regr.coef_)