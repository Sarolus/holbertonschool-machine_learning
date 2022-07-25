#!/usr/bin/env python3
"""
    Multivariate Normal Distribution Representation
"""
import numpy as np

mean_cov = __import__('0-mean_cov').mean_cov


class MultiNormal:
    """
        Multivariate Normal Distribution Class
    """

    mean = None
    cov = None

    def __init__(self, data):
        """
            Constructor Method

            Args:
                data (np.ndarray): np.ndarray containing the data set.
        """

        self.mean, self.cov = self.mean_cov(data)

    def mean_cov(self, X):
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
            raise TypeError("data must be a 2D numpy.ndarray")

        data_points, dimensions = np.shape(X)

        if data_points < 2:
            raise ValueError("data must contain multiple data points")

        mean = np.mean(X, axis=1).reshape(data_points, 1)
        X -= mean
        covariance = np.dot(X, X.T) / (dimensions - 1)

        return mean, covariance
