#!/usr/bin/env python3
"""
    Principal Component Analysis
"""
import numpy as np


def pca(X, ndim):
    """
        Performs a PCA on a dataset.

        Args:
            X (np.ndarray): The data set.
            ndim (np.ndarray): new dimensionality of the
            transformed X.

        Returns:
            np.ndarray: The transformed version of X.
    """

    variance_matrix = X - np.mean(X, axis=0)
    _, _, V = np.linalg.svd(variance_matrix)

    weight = V.T[:, :ndim]

    return np.dot(variance_matrix, weight)
