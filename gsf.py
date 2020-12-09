import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import example1 as ex1
from sklearn.linear_model import LinearRegression

def logloptimizor(args):
    # TweedieRegression
    # min_w 1/2n \sum_i d(yi yi^hat) + \alpha/2||w||_2
    # normal distribution case
    # simple array
    model = args.modelPara.model
    X     = pd.DataFrame(args.data.X.ix[:,7].reshape(-1,1))
    y     = pd.DataFrame(args.data.y.reshape(-1,1))
    reg   = LinearRegression(fit_intercept=False) 
    reg.fit(X, y)
    return reg.coef_
    

if __name__ == "__main__":
    ex1.example1()



