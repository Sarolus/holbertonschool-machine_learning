#!/usr/bin/env python3
"""
    Intersection Calculation
"""

import numpy as np

likelihood = __import__('0-likelihood').likelihood


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

    # Calculate the likelihood of the data given the model
    L = likelihood(x, n, P)

    if not isinstance(Pr, int) and P.shape != Pr.shape:
        raise TypeError("Pr must bbe a numpy.ndarray with the same shape as P")

    if any(i < 0 or i > 1 for i in Pr):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if np.isclose(np.sum(Pr), 1) is False:
        raise ValueError("Pr must sum to 1")

    return Pr * L
