#!/usr/bin/env python3
"""
    Matrix Definiteness Calculation Module
"""

import numpy as np


def is_pos_def(matrix):
    """
        Check if the matrix is positive definite.
    """
    return np.all(np.linalg.eigvals(matrix) > 0)


def is_neg_def(matrix):
    """
        Check if the matrix is negative definite.
    """
    return np.all(np.linalg.eigvals(matrix) < 0)


def is_pos_semi_def(matrix):
    """
        Check if the matrix is positive semi-definite.
    """
    return np.all(np.linalg.eigvals(matrix) >= 0)


def is_neg_semi_def(matrix):
    """
        Check if the matrix is negative semi-definite.
    """
    return np.all(np.linalg.eigvals(matrix) <= 0)


def definiteness(matrix):
    """
        Determines the definiteness of the given matrix.

        Args:
            matrix (list): The given matrix.

        Returns:
            list: The definiteness.
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    try:
        row, column = matrix.shape

        if row != column:
            raise TypeError

        matrix_transpose = matrix.copy().T

        if not np.array_equal(matrix, matrix_transpose):
            raise TypeError

    except Exception as exception:
        return None

    if is_pos_def(matrix):
        msg = "Positive definite"
    elif is_neg_def(matrix):
        msg = "Negative definite"
    elif is_pos_semi_def(matrix):
        msg = "Positive semi-definite"
    elif is_neg_semi_def(matrix):
        msg = "Negative semi-definite"
    else:
        msg = "Indefinite"

    return msg
