#!/usr/bin/env python3
"""
    Script that performs the expectation maximization algorithm for a GMM.
"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
        Performs the expectation maximization algorithm for a GMM.
        Args:
            X: numpy.ndarray of shape (n, d) containing the dataset
            k: number of clusters
            iterations: maximum number of iterations for the algorithm
            tol: tolerance to declare convergence
            verbose: boolean to print or not the loglikelihoods during the
                iterations
        Returns:
            pi: numpy.ndarray of shape (k, 1) containing the probabilities
                for each cluster
            mean: numpy.ndarray of shape (k, d) containing the centroids
            covariance: numpy.ndarray of shape (k, d, d) containing
                        the covariances
            g: numpy.ndarray of shape (n, k) containing the probabilities
               for each cluster for each sample
            loglikelihoods: list containing the loglikelihoods during the
                iterations
    """
    try:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray")

        if X.ndim != 2:
            raise TypeError("X must be a 2D array")

        if not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k <= 0:
            raise ValueError("k must be greater than 0")

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

        iteration = 0
        pi, mean, covariance = initialize(X, k)
        g, c_loglikelihood = expectation(X, pi, mean, covariance)
        p_loglikelihood = 0

        while (
            iteration < iterations and
            np.abs(p_loglikelihood - c_loglikelihood) > tol
        ):
            p_loglikelihood = c_loglikelihood

            if verbose and iteration % 10 == 0:
                rounded = c_loglikelihood.round(5)
                print("Log Likelihood after {} iterations: {}".format(
                    iteration, rounded
                ))

            pi, mean, covariance = maximization(X, g)
            g, c_loglikelihood = expectation(X, pi, mean, covariance)

            iteration += 1

        if verbose:
            rounded = c_loglikelihood.round(5)
            print("Log Likelihood after {} iterations: {}".format(
                iteration, rounded
            ))

        return pi, mean, covariance, g, c_loglikelihood

    except Exception as exception:
        return None, None, None, None, None
