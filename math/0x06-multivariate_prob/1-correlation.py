#!/usr/bin/env python3
"""
    Matrix Correlation Calculation
"""
import numpy as np


def correlation(C):
    """
    Calculates the correlation Matrix.

    Args:
        C (np.ndarray): np.ndarray containing a covariance
        matrix.

    Returns:
        np.ndarray: np.ndarray containing the correlation
        matrix.
    """

    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")

    d1, d2 = np.shape(C)

    if len(C.shape) != 2 or d1 != d2:
        raise ValueError("C must be a 2D square matrix")

    correlation_matrix = np.zeros((d1, d1))

    for row in range(d1):
        for column in range(d1):
            correlation_matrix[row, column] = C[row, column] / \
                np.sqrt(C[row, row] * C[column, column])

    return correlation_matrix