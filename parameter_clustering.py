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
from numpy.linalg import inv


def n_component_comparison(X, n_comps, X_eval=None):
    bics = {}
    aics = {}
    scores = {}
    eval_scores = {}

    models = {}
    for n in n_comps:
        gmm = GaussianMixture(n_components=n).fit(X)
        bics[n] = gmm.bic(X)
        aics[n] = gmm.aic(X)
        scores[n] = gmm.score(X)
        if X_eval is not None:
            eval_scores[n] = gmm.score(X_eval)
        models[n] = gmm

    pd.Series(scores).plot()
    pd.Series(eval_scores).plot()
    plt.figure()
    pd.Series(bics).plot()
    plt.figure()
    pd.Series(aics).plot()
    return models


def eval_energy_del(model, start, end):
    x = get_data(start, end)
    energy_del = sum(x[:, 2])
    x_pred = model.sample(len(x))[0]
    energy_pred = sum(x_pred[:, 2])
    return energy_del, energy_pred


def weights_conditioned_on_arrival(a, gmm, covariance_type='full'):
    K = gmm.n_components
    w = np.zeros(K)
    for k in range(K):
        if covariance_type == 'full':
            w[k] = multivariate_normal.pdf(a, mean=gmm.means_[k, 0], cov=gmm.covariances_[k, 0, 0]) * gmm.weights_[k]
        elif covariance_type == 'tied':
            w[k] = multivariate_normal.pdf(a, mean=gmm.means_[k, 0], cov=gmm.covariances_[0, 0]) * gmm.weights_[k]
        elif covariance_type == 'diag':
            w[k] = multivariate_normal.pdf(a, mean=gmm.means_[k, 0], cov=gmm.covariances_[k, 0]) * gmm.weights_[k]
    return w / sum(w)


def weights_conditioned_on_arrival_and_departure(a, gmm, covariance_type='full'):
    K = gmm.n_components
    w = np.zeros(K)
    for k in range(K):
        if covariance_type == 'full':
            w[k] = multivariate_normal.pdf(a, mean=gmm.means_[k, 0:1], cov=gmm.covariances_[k, 0:1, 0:1]) * gmm.weights_[k]
    return w / sum(w)


def expectation_conditioned_on_arrival(a, gmm, covariance_type='full'):
    K = gmm.n_components
    mu = np.zeros([K, 2])
    for k in range(K):
        if covariance_type == 'full':
            mu[k] = gmm.means_[k, 1:] + ((gmm.covariances_[k, 1:, 0]) * (1/gmm.covariances_[k, 0, 0]) * (a - gmm.means_[k, 0]))
        elif covariance_type == 'tied':
            mu[k] = gmm.means_[k, 1:] + gmm.covariances_[0, 1:] * 1/gmm.covariances_[0, 0] * (a - gmm.means_[k, 0])
        elif covariance_type == 'diag':
            mu[k] = gmm.means_[k, 1:] + gmm.covariances_[k, 1:] * 1 / gmm.covariances_[k, 0] * (a - gmm.means_[k, 0])
    cond_weights = weights_conditioned_on_arrival(a, gmm, covariance_type=covariance_type)
    # hard-margin
    # label = np.argmax(cond_weights)
    # soft-margin
    return sum(cond_weights[k] * mu[k] for k in range(K))
    # return mu[label]

def expectation_conditioned_on_arrival_and_departure(a, gmm, covariance_type='full'):
    K = gmm.n_components
    mu = np.zeros([K, 1])
    for k in range(K):
        if covariance_type == 'full':
            vector = gmm.covariances_[k, 2, 0:1].dot(inv(gmm.covariances_[k, 0:1, 0:1]))
            mu[k] = gmm.means_[k, 2] +  vector.dot((a - gmm.means_[k, 0:1]))
    cond_weights = weights_conditioned_on_arrival_and_departure(a, gmm, covariance_type=covariance_type)
    # hard-margin
    # label = np.argmax(cond_weights)
    # soft-margin
    return sum(cond_weights[k] * mu[k] for k in range(K))
    # return mu[label]


if __name__ == "__main__":
    acn = 'JplACN'
    X = get_data(acn, datetime(2018, 11, 1).astimezone(), datetime(2019, 1, 1).astimezone(), cond='WEEKDAY')
    model = GaussianMixture(n_components=10, max_iter=1000, n_init=25, covariance_type='full').fit(X)
    visualization(X, model)
    print('---')
