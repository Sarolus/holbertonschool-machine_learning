#!/usr/bin/env python3
"""
    Script that find the best number of clusters for a GMM
    using the Bayesian Information Criterion.
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000,
        tol=1e-5, verbose=False):
    """
        Finds the best number of clusters for a GMM using the Bayesian
        Information Criterion.

        Args:
            X: numpy.ndarray of shape (n, d) containing the dataset
            kmin: minimum number of clusters
            kmax: maximum number of clusters
            iterations: maximum number of iterations for the algorithm
            tol: tolerance to declare convergence
            verbose: boolean to print or not the loglikelihoods during the
                iterations

        Returns:
            best_k: best number of clusters
            best_loglikelihood: best loglikelihood
            loglikelihoods: list containing the loglikelihoods during the
                iterations
    """
    try:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray")

        if X.ndim != 2:
            raise TypeError("X must be a 2D array")

        if not isinstance(kmin, int):
            raise TypeError("kmin must be an integer")

        if kmin <= 0:
            raise ValueError("kmin must be greater than 0")

        data_points, dimensions = X.shape

        if kmax is None:
            kmax = data_points

        if not isinstance(kmax, int):
            raise TypeError("kmax must be an integer")

        if kmax <= 0:
            raise ValueError("kmax must be greater than 0")

        if kmax < kmin:
            raise ValueError("kmax must be greater than kmin")

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        if iterations <= 0:
            raise ValueError("iterations must be greater than 0")

        if not isinstance(tol, float):
            raise TypeError("tol must be a float")

        if tol < 0:
            raise ValueError("tol must be greater than 0")

        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")

        parameters = []
        log_likelihoods = []
        BICs = []

        for cluster in range(kmin, kmax + 1):
            priors, centroids, covariances, _, log_likelihood = \
                expectation_maximization(
                    X, cluster, iterations, tol, verbose
                )

            parameters.append((priors, centroids, covariances))
            log_likelihoods.append(log_likelihood)

            parameter_count = (
                cluster * (dimensions + 2) * (dimensions + 1) / 2
                - 1
            )

            BICs.append(
                np.log(data_points) * parameter_count -
                np.dot(2, log_likelihood)
            )

        best_k = kmin + np.argmin(BICs)
        best_parameters = parameters[np.argmin(BICs)]

        return (
            best_k,
            best_parameters,
            np.array(log_likelihoods),
            np.array(BICs)
        )
    except Exception as e:
        return (None, None, None, None)
