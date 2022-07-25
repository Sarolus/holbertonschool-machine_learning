#!/usr/bin/env python3
"""
    Multivariate Normal Distribution Representation
"""
import numpy as np


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

        if dimensions < 2:
            raise ValueError("data must contain multiple data points")

        mean = np.mean(X.T, axis=0).reshape(1, data_points).T
        X = X.T - mean.T
        covariance = np.dot(X.T, X) / (dimensions - 1)

        return mean, covariance

    def pdf(self, x):
        """
            Calculates the Probability Density Function

            Args:
                x (np.ndarray): np.ndarray containing the data points.

            Returns:
                pdf: Float containing the probability density
                function
        """

        if not type(x) is np.ndarray:
            raise TypeError("x must be a numpy.ndarray")

        if x.ndim != 2 or x.shape != (self.cov.shape[0], 1):
            raise ValueError(
                "x must have the shape ({}, 1)".format(self.cov.shape[0]))

        k = self.cov.shape[0]
        det = np.linalg.det(self.cov)
        x -= self.mean

        mul = np.dot(np.dot(x.T, np.linalg.inv(self.cov)), x)
        exp = np.exp(-0.5 * mul)
        pdf = exp / np.sqrt((np.pi * 2) ** k * det)

        return np.asscalar(pdf)
