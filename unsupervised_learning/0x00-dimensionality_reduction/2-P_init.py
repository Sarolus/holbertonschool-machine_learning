#!/usr/bin/env python3
"""
    t-SNE Variables Initialization
"""
import numpy as np


def P_init(X, perplexity):
    """
        Initializes all variables required to calculate
        the P affinities in t-SNE.

        Args:
            X (np.ndarray): The dataset to transform.
            perplexity (float): The perplexity that all
            Gaussian distributions.

        Returns:
            np.ndarray: The D, P, betas and H
    """
    n = X.shape[0]

    D = np.square(X[:, None, :] - X[None, :, :]).sum(axis=-1)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))

    H = np.log2(perplexity)

    return D, P, betas, H
