#!/usr/bin/env python3
"""
    Symmetric P Affinities Calculation
"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def pairwise_distance_with_binary_search(
        perplexity_entropy, calculated_entropy,
        distance, beta, tol, high, low
):
    """
        Binary search to find the optimal P affinities.

        Args:
            perplexity_entropy: float representing the perplexity entropy
            calculated_entropy: float representing the calculated entropy
            distance: numpy.ndarray of shape (n, n) containing the pairwise
                      distance matrix
            beta: float representing the beta value
            tol: float representing the tolerance
            high: float representing the high value
            low: float representing the low value

        Returns:
            Pi: numpy.ndarray of shape (n, n) representing the P affinities
    """
    if np.abs(perplexity_entropy - calculated_entropy) >= tol:
        if perplexity_entropy < calculated_entropy:
            low = beta.copy()

            if high is None:
                beta *= 2
            else:
                beta = (high + beta) / 2
        else:
            high = beta.copy()

            if low is None:
                beta /= 2
            else:
                beta = (low + beta) / 2

        calculated_entropy, _ = HP(distance, beta)

        return pairwise_distance_with_binary_search(
            perplexity_entropy, calculated_entropy,
            distance, beta, tol, high, low
        )

    _, Pi = HP(distance, beta)

    return Pi


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
        Calculates the P affinities in t-SNE.

        Args:
            X: numpy.ndarray of shape (n, n_features) containing the dataset
            tol: float representing the tolerance
            perplexity: float representing the perplexity value

        Returns:
            Pi: numpy.ndarray of shape (n, n) representing the P affinities
    """

    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        Di = np.delete(D[i], i)  # excludes itself point
        beta = betas[i]
        Hi, Pi = HP(Di, beta)

        # Binary search:
        high = low = None
        Pi = pairwise_distance_with_binary_search(
            H, Hi, Di, beta, tol, high, low)

        P[i][:i] = Pi[:i].copy()
        P[i][i + 1:] = Pi[i:].copy()

    return (P + P.T) / (2 * n)
