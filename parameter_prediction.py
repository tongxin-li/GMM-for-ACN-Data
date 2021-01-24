from sklearn.mixture import GaussianMixture
import pymongo
import bson
import numpy as np
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from scipy.stats import multivariate_normal
from visualization import *
from data_collection import *
from parameter_clustering import expectation_conditioned_on_arrival

def calc_errors(x_eval, x, y, model, covariance_type='full'):
    errors = []
    input_errors = []
    percentile_errors = []
    percentile_input_errors = []
    percentile_mean_errors = []
    for i in range(len(x_eval)):
        prediction = expectation_conditioned_on_arrival(x_eval[i, 0], model, covariance_type=covariance_type)
        percentile_errors.append([abs((abs(prediction[0]) - x_eval[i, 1])/(x_eval[i, 1]+abs(prediction[0]))), abs((abs(prediction[1]) - x_eval[i, 2])/(x_eval[i, 2]+abs(prediction[1])))])
        percentile_input_errors.append([abs((y[i, 1] - x_eval[i, 1]) / (x_eval[i, 1]+y[i, 1])), abs((y[i, 2] - x_eval[i, 2]) / (x_eval[i, 2]+y[i, 2]))])
        percentile_mean_errors.append([abs((x[:, 1].mean(axis=0) - x_eval[i, 1]) / (x_eval[i, 1]+x[:, 1].mean(axis=0))), abs((x[:, 2].mean(axis=0) - x_eval[i, 2]) / (x_eval[i, 2]+x[:, 2].mean(axis=0)))])
        errors.append(prediction - x_eval[i, 1:])
        input_errors.append(y[i, 1:] - x_eval[i, 1:])
    return np.array(errors), np.array(input_errors), np.array(percentile_errors), np.array(percentile_input_errors), np.array(percentile_mean_errors)

def show_prediction(x_eval, model, covariance_type='full'):
    prediction = []
    for i in range(len(x_eval)):
        prediction.append(expectation_conditioned_on_arrival(x_eval[i, 0], model, covariance_type=covariance_type))
    np.save('prediction',np.array(prediction))
    return np.array(prediction)

#np.array(percentile_errors)

def mse(e):
    return np.square(e).mean(axis=0)


def ase(e):
    return np.abs(e).mean(axis=0)


def display_result(x, x_eval, y, y_eval, model, cov_type):
    # errors, input_errors, precentile_errors, precentile_input_errors, percentile_mean_errors = calc_errors(x, y, model, covariance_type=cov_type)
    # mean_err = x[:, 1:].mean(axis=0) - x[:, 1:]
    eval_err, eval_input_err, eval_percentile_errors, eval_percentile_input_errors, eval_percentile_mean_errors = calc_errors(x_eval, x, y_eval, model, covariance_type=cov_type)
    eval_mean_err = x[:, 1:].mean(axis=0) - x_eval[:, 1:]
    return ase(eval_err), ase(eval_input_err), ase(eval_mean_err), eval_percentile_errors.mean(axis=0), eval_percentile_input_errors.mean(axis=0), eval_percentile_mean_errors.mean(axis=0)

    #  eval_percentile_input_errors.mean(axis=0)
    #  eval_percentile_errors.mean(axis=0)

    # print('Training')
    # print(mse(errors))
    # print(mse(input_errors))
    # print(mse(mean_err))
    # print('---')
    # print('Testing')
    # print(mse(eval_err))
    # print(mse(eval_input_err))
    # print(mse(eval_mean_err))

def test_pipeline(id, n, X, X_eval, Y, Y_eval, test_case ='Overall', cov_type = 'full', cond = 'All'):

    x = X[id]
    y = Y[id]
    x_eval = X_eval[id]
    y_eval = Y_eval[id]

    if test_case == 'Individual_ini':

        # Get Initialization from User Input Data
        # This works pretty well if the user input data is somehow accurate. Example: id = 560; cond = 'ALL'

        model_ini = GaussianMixture(n_components=n, max_iter=1000, n_init=25, covariance_type=cov_type).fit(y_eval)

        # Train Model_C using only User Input Data
        # Unused: (weights_init=model_ini.weights_,)
        for i in range(len(model_ini.weights_)):
            if model_ini.weights_[i] > 1:
                model_ini.weights_[i] = 1
        # , weights_init=model_ini.weights_
        model = GaussianMixture(n_components=n, max_iter=1000, n_init=25, means_init=model_ini.means_, weights_init=model_ini.weights_, covariance_type=cov_type).fit(x)
        error, input_error, mean_error, percentile_error, percentile_input_error, percentile_mean_error = display_result(x, x_eval, y, y_eval, model, cov_type)
        return error, input_error, mean_error, percentile_error, percentile_input_error, percentile_mean_error

    if test_case == 'Individual':

        # Train Model_D w/ random or k-mean initialization
        model = GaussianMixture(n_components=n, max_iter=1000, n_init=25, covariance_type=cov_type).fit(x)
        error, input_error, mean_error, percentile_error, percentile_input_error, percentile_mean_error = display_result(x, x_eval, y, y_eval, model, cov_type)
        show_prediction(x_eval, model, covariance_type='full')
        return error, input_error, mean_error, percentile_error, percentile_input_error, percentile_mean_error