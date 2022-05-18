#!/usr/bin/env python3
"""
    Normalization Constants Caculation Module
"""
import numpy as np


def normalization_constants(X):
    """
        Calculates the normalization constants of a matrix.

        Args:
            X (np.ndarray): np.ndarray to normalize.

        Returns:
            int: the mean and standard deviation of each feature.
    """
    mean = np.mean(X, axis=0)
    std_deviation = np.std(X, axis=0)

    return mean, std_deviation
