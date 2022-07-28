#!/usr/bin/env python3
"""
    Likelihood Calculation
"""
import numpy as np


def likelihood(x, n, P):
    """
        Calculates the likelihood of a data point in a Gaussian distribution.

        Args:
            x: numpy.ndarray of shape (n, 1) representing a data point
            n: int representing the number of dimensions
            P: numpy.ndarray of shape (n, n) representing the P affinities
                matrix

        Returns:
            likelihood: float representing the likelihood of a data point
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "n must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    for p in P:
        if p < 0 or p > 1:
            raise ValueError("All values in P must be in the range [0, 1]")

    factorial_p = np.math.factorial(
        n) / (np.math.factorial(x) * np.math.factorial(n - x))

    return factorial_p * np.power(P, x) * np.power((1 - P), (n - x))
