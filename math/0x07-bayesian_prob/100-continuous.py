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

    alpha = x + 1
    beta = n - x + 1
    beta_dist_1 = special.btdtr(alpha, beta, p1)
    beta_dist_2 = special.btdtr(alpha, beta, p2)
    posterior = beta_dist_2 - beta_dist_1

    return posterior
