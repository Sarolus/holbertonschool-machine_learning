#!/usr/bin/env python3
"""
    Matrix Normalization Module
"""
import numpy as np


def normalize(X, mean, std_deviation):
    """
    Normalizes a matrix.

    Args:
        X (np.ndarray): np.ndarray to normalize.
        mean (np.ndarray): np.ndarray that contains the
        mean of all features of X.
        std_deviation (np.ndarray): np.ndarray that contains
        the standard deviation of all features of X.

    Returns:
        np.ndarray: The normalized X matrix.
    """

    return (X - mean) / std_deviation
