#!/usr/bin/env python3
"""
    Shuffle Data Module
"""
import numpy as np


def shuffle_data(X, Y):
    """
        Shuffles the data points in two matrices the same way.

        Args:
            X (np.ndarray): The first matrix to shuffle.
            Y (np.ndarray): The second matrix to shuffle.

        Returns:
            np.ndarray: The shuffled X and Y matrices.
    """
    permutation = np.random.permutation(X.shape[0])

    return X[permutation], Y[permutation]
