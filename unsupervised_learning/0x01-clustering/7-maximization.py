#!/usr/bin/env python3
"""
    Maximization - maximization step in the EM algorithm
"""
import numpy as np


def maximization(X, g):
    """
    Method to calculate the maximization step in the EM algorithm
    for a GMM.
    Parameters:
        X (numpy.ndarray of shape (n, d)):
         containing the data set.
        g (numpy.ndarray of shape (k, n)):
         containing the posterior probabilities for each data
         point in each cluster.
    returns: pi, m, S, or None, None, None on failure.
        pi (numpy.ndarray of shape (k,)):
         containing the *priors* for each cluster.
        m (numpy.ndarray of shape (k, d)):
         containing the *centroid means* for each cluster.
        S (numpy.ndarray of shape (k, d, d)):
         containing the *covariance* matrices for each cluster.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    if not np.isclose(np.sum(g, axis=0), 1).all():
        return (None, None, None)

    n, d = X.shape
    k, n = g.shape

    # the sum of posterior probabilities
    posteriors_sum = np.sum(g, axis=1)

    pi = posteriors_sum / n

    mean = np.zeros((k, d))
    cov = np.zeros((k, d, d))
    for j in range(k):
        mean[j] = np.matmul(g[j], X) / posteriors_sum[j]
        diff = X - mean[j]
        cov[j] = np.matmul(g[j] * diff.T, diff) / posteriors_sum[j]

    return pi, mean, cov
