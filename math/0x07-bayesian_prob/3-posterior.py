#!/usr/bin/env python3
"""
    Posterior Probability Calculation
"""

import numpy as np

marginal = __import__('2-marginal').marginal
intersection = __import__('1-intersection').intersection


def posterior(x, n, P, Pr):
    """
        Calculates the posterior probability of x,
        given the class priors P and Pr.

        Args:
            x: number of patients that develop severe side effects
            n: total number of patients
            P: vector of probabilities of developing severe side effects
            Pr: vector of prior probabilities of developing severe side effects

        Returns:
            posterior: posterior probability of x
    """

    # Calculate the intersection of the posterior and prior distributions
    intersect = intersection(x, n, P, Pr)

    # Calculate the marginal probability of x
    marg = marginal(x, n, P, Pr)

    return intersect / marg
