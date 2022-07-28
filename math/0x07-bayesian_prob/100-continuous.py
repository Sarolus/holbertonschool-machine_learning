#!/usr/bin/env python3
"""
    Continuous Prosterior Probability Calculation
"""

from scipy import special


def posterior(x, n, p1, p2):
    """
        Calculates the posterior probability of x,
        given the class priors P and Pr.

        Args:
            x: number of patients that develop severe side effects
            n: total number of patients
            p1: probability of the first group
            p2: probability of the second group

        Returns:
            posterior: posterior probability of x
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        text = "x must be an integer that is greater than or equal to 0"
        raise ValueError(text)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")

    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    alpha = x + 1
    beta = n - x + 1
    beta_dist_1 = special.btdtr(alpha, beta, p1)
    beta_dist_2 = special.btdtr(alpha, beta, p2)
    posterior = beta_dist_2 - beta_dist_1

    return posterior
