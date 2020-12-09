import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

X = [1,1,1,1,1]
y = [1,2,3,4,5]
lr = LinearRegression(fit_intercept=False)
X = np.array(X).reshape(-1, 1)
y = np.array(y).reshape(-1,1)
print(y.mean())
lr.fit(X,y)
print(lr.coef_)