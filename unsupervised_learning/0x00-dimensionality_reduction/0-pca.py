#!/usr/bin/env python3
"""
    Principal Component Analysis
"""
import numpy as np


def pca(X, var=0.95):
    """
        Performs PCA on a dataset.

        Args:
            X (np.ndarray): Data set
            var (float, optional): Fraction of the variance. Defaults to 0.95.

        Returns:
            np.ndarray: The weights matrix.
    """
    _, Sigma, V = np.linalg.svd(X)

    variance_components = np.cumsum(Sigma) / np.sum(Sigma)

    for index, variance_fraction in enumerate(variance_components):
        if variance_fraction >= var:
            break

    weight = V.T[:, :index + 1]

    return weight
