#!/usr/bin/env python3
"""
    Regular Markov Chain Steady State Probability Determination
"""
import numpy as np


def regular(P):
    """
        Determines the steady state probabilities of
        a regular markov chain.

        Args:
            P (np.ndarray): The transition matrix;.

        Returns:
            np.ndarray: Steady state probabilities or
            None on failure.
    """
    try:
        if not isinstance(P, np.ndarray):
            raise TypeError

        if P.ndim != 2:
            raise TypeError

        if P.shape[0] != P.shape[1]:
            raise TypeError

        if np.all(np.square(P * P)) == 0:
            return None

        n, _ = P.shape

        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        closed_values = np.isclose(eigenvalues, 1)
        steady_state = eigenvectors[:, closed_values]

        return (steady_state / np.sum(steady_state)).reshape(1, n)
    except Exception as exception:
        return None
