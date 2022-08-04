#!/usr/bin/env python3
"""
    Script that calculates the probability density function of a Gaussian.
"""

import numpy as np


def pdf(X, m, S):
    """
        Calculates the probability density function of a Gaussian distribution.

        Args:
            X: numpy.ndarray of shape (n, d) containing the dataset
            m: numpy.ndarray of shape (k, d) containing the centroids
            S: numpy.ndarray of shape (k, d, d) containing the covariances

        Returns:
            pdfs: numpy.ndarray of shape (n, k) containing the pdfs of each
                cluster for each data point
    """

    try:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray")

        if X.ndim != 2:
            raise TypeError("X must be a 2D array")

        if not isinstance(m, np.ndarray):
            raise TypeError("m must be a numpy.ndarray")

        if m.ndim != 1:
            raise TypeError("m must be a 1D array")

        if not isinstance(S, np.ndarray):
            raise TypeError("S must be a numpy.ndarray")

        if S.ndim != 2:
            raise TypeError("S must be a 3D array")

        if S.shape[0] != S.shape[1]:
            raise ValueError("S must be a square matrix")

        _, dimensions = X.shape

        X_mean = X - m
        inverse = np.linalg.inv(S)
        determinant = np.linalg.det(S)

        # Calculate the jointly gaussian pdf
        power = np.power(np.dot(2, np.pi), dimensions)
        density = np.sqrt(np.dot(power, determinant))
        mul = X_mean * np.matmul(inverse, X_mean.T).T
        sum = np.sum(mul, axis=1)
        exp = np.exp(np.dot(-0.5, sum))
        pdf = np.divide(exp, density)

        return np.where(pdf > 1e-300, pdf, 1e-300)
    except Exception as e:
        return None
