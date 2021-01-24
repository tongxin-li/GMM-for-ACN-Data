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
from parameter_clustering import expectation_conditioned_on_arrival_and_departure


def calc_errors(x, y, model, covariance_type='full'):
    errors = []
    input_errors = []
    for i in range(len(x)):
        prediction = expectation_conditioned_on_arrival_and_departure(x[i, 0:1], model, covariance_type=covariance_type)
        errors.append(prediction - x[i, 2])
        input_errors.append(y[i, 2] - x[i, 2])
    return np.array(errors), np.array(input_errors)


def mse(e):
    return np.square(e).mean(axis=0)


def display_result(x, x_eval, y, y_eval, model, cov_type):
    errors, input_errors = calc_errors(x, y, model, covariance_type=cov_type)
    mean_err = x[:, 2].mean() - x[:, 2]
    eval_err, eval_input_err = calc_errors(x_eval, y_eval, model, covariance_type=cov_type)
    eval_mean_err = x[:, 2].mean() - x_eval[:, 2]

    print('Training')
    print(mse(errors))
    print(mse(input_errors))
    print(mse(mean_err))
    print('---')
    print('Testing')
    print(mse(eval_err))
    print(mse(eval_input_err))
    print(mse(eval_mean_err))

def test_pipeline_1d(user_id, n, test_case, cov_type = 'full', acn = 'CaltechACN', cond = 'All'):
    # acn = 'JplACN'
    # Prepare User Input Data
    X, Y = get_data_by_user(acn, datetime(2018, 5, 1).astimezone(), datetime(2018, 12, 1).astimezone(), cond)
    X_eval, Y_eval = get_data_by_user(acn, datetime(2018, 12, 1).astimezone(), datetime(2019, 1, 1).astimezone(), cond)

    x = X[user_id]
    y = Y[user_id]
    x_eval = X_eval[user_id]
    y_eval = Y_eval[user_id]

    # Prepare Overall Data
    z = get_data(acn, datetime(2018, 5, 1).astimezone(), datetime(2018, 12, 1).astimezone(), cond)

    if __name__ == "__main__":
        if test_case == 'Overall_int':

            # Get Initialization from User Input Data
            # This works pretty well if the user input data is somehow accurate. Example: user_id = 560; cond = 'ALL'

            model_ini = GaussianMixture(n_components=n, max_iter=1000, n_init=25, covariance_type=cov_type).fit(y)

            # Train Model_A using Overall Data

            model_pre = GaussianMixture(n_components=n, max_iter=1000, n_init=25, means_init=model_ini.means_, covariance_type=cov_type).fit(z)
            predict_label = model_pre.predict(x)

            # Change the weights based on User Data

            unique, counts = np.unique(predict_label, return_counts=True)
            count_numbers = dict(zip(unique, counts))

            for components in range(n):
                if components in count_numbers:
                    model_pre.weights_[components] = count_numbers[components] / len(predict_label)
                else:
                    model_pre.weights_[components] = 0

            display_result(x, x_eval, y, y_eval, model_pre, cov_type)

        if test_case == 'Overall':

            # Train Model_B using Overall Data

            model_pre = GaussianMixture(n_components=n, max_iter=1000, n_init=25, covariance_type=cov_type).fit(z)
            predict_label = model_pre.predict(x)

            # Change the weights based on User Data

            unique, counts = np.unique(predict_label, return_counts=True)
            count_numbers = dict(zip(unique, counts))

            for components in range(n):
                if components in count_numbers:
                    model_pre.weights_[components] = count_numbers[components] / len(predict_label)
                else:
                    model_pre.weights_[components] = 0

            display_result(x, x_eval, y, y_eval, model_pre, cov_type)

        if test_case == 'Individual_int':

            # Get Initialization from User Input Data
            # This works pretty well if the user input data is somehow accurate. Example: user_id = 560; cond = 'ALL'

            model_ini = GaussianMixture(n_components=n, max_iter=1000, n_init=25, covariance_type=cov_type).fit(y)

            # Train Model_C using only User Input Data
            # Unused: (weights_init=model_ini.weights_,)

            model = GaussianMixture(n_components=n, max_iter=1000, n_init=25, means_init=model_ini.means_, covariance_type=cov_type).fit(x)
            display_result(x, x_eval, y, y_eval, model, cov_type)

        if test_case == 'Individual':

            # Train Model_D w/ random or kmean initialization

            model = GaussianMixture(n_components=n, max_iter=1000, n_init=25, init_params='random', covariance_type=cov_type).fit(x)
            display_result(x, x_eval, y, y_eval, model, cov_type)


# Example
test_pipeline_1d(560, 4, test_case='Individual', cov_type='full', acn='CaltechACN')