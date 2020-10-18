import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression, SelectKBest, RFE 
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
import explore


def linear_reg_model(x_scaleddf, target):
    lm = LinearRegression()
    lm.fit(x_scaleddf, target)
    y_hat = lm.predict(x_scaleddf)

    LM_MSE = sqrt(mean_squared_error(target, y_hat))
    return LM_MSE

def get_baseline(y_train, target):
    # determine Baseline to beat
    rows_needed = y_train.shape[0]
    # create array of predictions of same size as y_train.logerror based on the mean
    y_hat = np.full(rows_needed, np.mean(target))
    # calculate the MSE for these predictions, this is our baseline to beat
    baseline = sqrt(mean_squared_error(target, y_hat))
    print("Baseline:", baseline)
    return baseline

def lasso_lars(x_scaleddf, target):
    # Make a model
    lars = LassoLars(alpha=1)
    # Fit a model
    lars.fit(x_scaleddf, target)
    # Make Predictions
    lars_pred = lars.predict(x_scaleddf)
    # Computer root mean squared error
    lars_rmse = sqrt(mean_squared_error(target, lars_pred))
    return lars_rmse