#!/usr/bin/env python3
"""
    Inter-cluster Variance Calculation
"""

import numpy as np


def variance(X, C):
    """
        Calculates the total intra-cluster variance for
        a data-set.

        Args:
            X (np.ndarray): The data-set.
            C (np.ndarray): The centroid means for each
            cluster.

        Returns:
            float: The variance.
    """
    try:
        if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
            raise TypeError

        if X.ndim != 2 or C.ndim != 2:
            raise ValueError

        variance_matrix = np.apply_along_axis(np.subtract, 1, X, C)
        cluster_matrix = np.square(variance_matrix).sum(
            axis=2).min(axis=1).sum()

        return cluster_matrix
    except Exception as exception:
        return None
