#!/usr/bin/env python3
"""
    Markov Chain Probability Determination
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
        Determines the probability of a markov chain being in a particular
        state after a specified number of iterations.

        Args:
            P (np.ndarray): Transition matrix.
            s (np.ndarray): Probability of starting in each state.
            t (int, optional): Number of iterations. Defaults to 1.

        Returns:
            np.ndarray: The probability of being in a specific state after t
            iterations, or None on failure.
    """
    try:
        if not isinstance(P, np.ndarray) or P.ndim != 2:
            raise TypeError

        if not isinstance(s, np.ndarray) or s.ndim != 2:
            raise TypeError

        if P.shape[0] != s.shape[1] or s.shape[0] != 1:
            raise TypeError

        if not isinstance(t, int) or t < 1:
            raise TypeError

        for _ in range(t):
            s = np.dot(s, P)

        return s
    except Exception as e:
        return None
