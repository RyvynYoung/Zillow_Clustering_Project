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

def polynomial(X_trainsdf, target):
    # Make a model
    pf = PolynomialFeatures(degree=2)
    # note: tried increasing degree to 4 but took forever to run and would probably overfit, retest if time permits
    # Fit and Transform model
    # to get a new set of features..which are the original features squared
    X_train_squared = pf.fit_transform(X_trainsdf)
    
    # Feed new features in to linear model. 
    lm_squared = LinearRegression(normalize=True)
    lm_squared.fit(X_train_squared, target)
    # Make predictions
    lm_squared_pred = lm_squared.predict(X_train_squared)
    # Compute root mean squared error
    lm_squared_rmse = sqrt(mean_squared_error(target, lm_squared_pred))
    return lm_squared_rmse


def poly_val_test(X_train_scaled, X_validate_scaled, y_train, y_validate):
    # Make a model
    pf = PolynomialFeatures(degree=2)
    X_train_squared = pf.fit_transform(X_train_scaled)
    X_validate_squared = pf.transform(X_validate_scaled)
    #X_test_squared = pf.transform(X_test_scaled)
    # Feed new features in to linear model. 
    lm_squared = LinearRegression(normalize=True)
    lm_squared.fit(X_train_squared, y_train)
    #lm_squared.fit(X_test_squared, y_test)
    # Make Predictions
    lm_pred_val = lm_squared.predict(X_validate_squared)
    #lm_pred_test = lm_squared.predict(X_test_squared)

    # Compute root mean squared error
    lm_rmse_val = sqrt(mean_squared_error(y_validate, lm_pred_val))
    #lm_rmse_test = sqrt(mean_squared_error(y_test, lm_pred_test))
    return lm_rmse_val #lm_rmse_test


def linear_reg_vt(X_train_scaled, X_validate_scaled, y_train, y_validate):
    lm = LinearRegression()
    lm.fit(X_train_scaled, y_train)

    y_hat = lm.predict(X_validate_scaled)

    LM_MSE = sqrt(mean_squared_error(y_validate, y_hat))
    return LM_MSE    