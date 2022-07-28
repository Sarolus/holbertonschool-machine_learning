#!/usr/bin/env python3
"""
    Marginal Probability Calculation
"""

import numpy as np

intersection = __import__('1-intersection').intersection


def marginal(x, n, P, Pr):
    """
        Calculates the marginal probability of x,
        given the class priors P and Pr.

        Args:
            x: number of patients that develop severe side effects
            n: total number of patients
            P: vector of probabilities of developing severe side effects
            Pr: vector of prior probabilities of developing severe side effects

        Returns:
            marginal: marginal probability of x
    """

    intersect = intersection(x, n, P, Pr)

    return np.sum(intersect)
