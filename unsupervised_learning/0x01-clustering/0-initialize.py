#!/usr/bin/env python3
"""
    K-means Cluster Centroids Initialization
"""

import numpy as np


def initialize(X, k):
    """
        Initializes cluster centroids for K-means

        Args:
            X: numpy.ndarray of shape (n, d) containing the dataset
            k: int representing the number of clusters
        Returns:
            centroids: numpy.ndarray of shape (k, d) containing the centroids
    """

    try:

        if not isinstance(k, int):
            raise TypeError

        if k <= 0:
            raise ValueError

        _, dimensions = X.shape
        min_value = np.ndarray.min(X, axis=0)
        max_value = np.ndarray.max(X, axis=0)
        # Draw samples from a uniform distribution.
        # Samples are uniformly distributed over the
        # half-open interval [low, high)
        # (includes low, but excludes high)
        centroids = np.random.uniform(
            low=min_value, high=max_value, size=(k, dimensions))

        return centroids

    except Exception as exception:
        return None
