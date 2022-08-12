#!/usr/bin/env python3
"""
    Script that calculates the Q affinities of the data points in Y.
"""

import numpy as np


def Q_affinities(Y):
    """
        Calculates the Q affinities of the data points in Y.

        Args:
            Y: numpy.ndarray of shape (n, n_dim) containing the data set

        Returns:
            Q: numpy.ndarray of shape (n, n) representing the Q affinities
               matrix
            numerator: numpy.ndarray of shape (n, n) representing the numerator
    """

    sum_Y = np.sum(np.square(Y), axis=1)
    distances = ((-2 * np.dot(Y, Y.T)) + sum_Y).T + sum_Y
    np.fill_diagonal(distances, 0)
    numerator = np.power(1 + distances, -1)
    np.fill_diagonal(numerator, 0)
    Q = numerator / numerator.sum()

    return Q, numerator
