#!/usr/bin/env python3
"""
    Script that calculates the expectation step in the EM algorithm for a GMM.
"""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
        Calculates the expectation step of the EM algorithm for a GMM.

        Args:
            X: numpy.ndarray of shape (n, d) containing the dataset
            pi: numpy.ndarray of shape (k, 1) containing the prior
                probabilities
            m: numpy.ndarray of shape (k, d) containing the centroids
            S: numpy.ndarray of shape (k, d, d) containing the covariances

        Returns:
            exp: numpy.ndarray of shape (n, k) containing the expectation of
                each cluster for each data point
            lg_likelihood: float containing the log likelihood of the model
    """

    try:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray")

        if X.ndim != 2:
            raise TypeError("X must be a 2D array")

        if not isinstance(pi, np.ndarray):
            raise TypeError("pi must be a numpy.ndarray")

        if pi.ndim != 1:
            raise TypeError("pi must be a 1D array")

        if not isinstance(m, np.ndarray):
            raise TypeError("m must be a numpy.ndarray")

        if m.ndim != 2:
            raise TypeError("m must be a 2D array")

        if not isinstance(S, np.ndarray):
            raise TypeError("S must be a numpy.ndarray")

        if S.ndim != 3:
            raise TypeError("S must be a 3D array")

        if not np.isclose([np.sum(pi)], [1])[0]:
            raise ValueError("pi must sum to 1")

        n, d = X.shape
        k = pi.shape[0]

        if m.shape[0] != k or m.shape[1] != d:
            raise ValueError("m and pi must have the same shape")

        if (
            S.shape[0] != k or
            S.shape[1] != d or
            S.shape[2] != d
        ):
            raise ValueError("S must have the same shape as pi")
        posterior = np.zeros((k, n))

        for i in range(k):
            posterior[i] = np.dot(pi[i], pdf(X, m[i], S[i]))

        marginal = np.sum(posterior, axis=0)
        posterior /= marginal
        log_likelihood = np.sum(np.log(marginal))

        return posterior, log_likelihood

    except Exception as e:
        return None, None
