#!/usr/bin/env python3
"""
    Data Set Mean And Covariance Calculation
"""
import numpy as np


def mean_cov(X):
    """
        Calculates the mean and covariance of a data set.

        Args:
            X (np.ndarray): The given data set.

        Returns:
            np.ndarray: The np.ndarray containing the mean of the
            data set and the np.ndarray containing the covariance
            matrix of the data set.
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    data_points, dimensions = np.shape(X)

    if data_points < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0).reshape(1, dimensions)
    X -= mean
    covariance = np.dot(X.T, X) / (data_points - 1)

    return mean, covariance
