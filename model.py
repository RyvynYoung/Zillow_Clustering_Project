import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import explore

import math
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def linear_reg_model(x_scaleddf, target):
    lm = LinearRegression()
    lm.fit(x_scaleddf, target)
    y_hat = lm.predict(x_scaleddf)

    LM_MSE = math.sqrt(mean_squared_error(target, y_hat))
    return LM_MSE

def get_baseline(y_train, target):
    # determine Baseline to beat
    rows_needed = y_train.shape[0]
    # create array of predictions of same size as y_train.logerror based on the mean
    y_hat = np.full(rows_needed, np.mean(target))
    # calculate the MSE for these predictions, this is our baseline to beat
    baseline = math.sqrt(mean_squared_error(target, y_hat))
    print("Baseline:", baseline)
    return baseline

