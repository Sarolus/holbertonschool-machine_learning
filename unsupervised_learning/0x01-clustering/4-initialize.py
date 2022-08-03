#!/usr/bin/env python3
"""
    Script that initializes a Gaussian Mixture Model.
"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
        Initializes variables for a Gaussian Mixture Model.

        Args:
            X: numpy.ndarray of shape (n, d) containing the dataset
            k: int representing the number of clusters
        Returns:
            probs: numpy.ndarray of shape (k, 1) containing the probabilities
            centroids: numpy.ndarray of shape (k, d) containing the centroids
            sigma: numpy.ndarray of shape (k, d) containing the covariances
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

        _, dimensions = X.shape

        probs = np.ones(k) / k

        centroids, _ = kmeans(X, k)

        # Initialize sigma
        sigma = np.tile(np.identity(dimensions), (k, 1)
                        ).reshape(k, dimensions, dimensions)

        return probs, centroids, sigma

    except Exception as e:
        return None, None, None
