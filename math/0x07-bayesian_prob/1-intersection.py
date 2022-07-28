#!/usr/bin/env python3
"""
    Intersection Calculation
"""

import numpy as np


def intersection(x, n, P, Pr):
    """
        Calculates the intersection of the posterior and prior distributions

        Args:
            x: number of patients that develop severe side effects
            n: total number of patients
            P: vector of probabilities of developing severe side effects
            Pr: vector of prior probabilities of developing severe side effects

        Returns:
            intersection: intersection of the posterior and prior distributions
    """

    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    for p in P:
        if p < 0 or p > 1:
            raise ValueError(
                "All values in P must be in the range [0, 1]"
            )

    for prior in Pr:
        if prior < 0 or prior > 1:
            raise ValueError(
                "All values in Pr must be in the range [0, 1]"
            )

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    factorial_p = np.math.factorial(
        n) / (np.math.factorial(x) * np.math.factorial(n - x))

    L = factorial_p * np.power(P, x) * np.power((1 - P), (n - x))

    return Pr * L
